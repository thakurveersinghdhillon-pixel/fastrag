"""FastRAG Evaluation Runner.

Orchestrates batch calculation of retrieval metrics across an evaluation query set
using pre-computed or synthetic rankings and ground-truth relevance judgments (qrels).
Produces structured JSON reports saved to benchmarks/results/.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.experiment import ExperimentMetadata, create_experiment_metadata
from evaluation.metrics import evaluate_query_metrics


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
DEFAULT_DATASETS_DIR = PROJECT_ROOT / "datasets"


@dataclass
class EvaluationReport:
    """Complete evaluation report containing metadata, aggregate scores, and per-query details."""

    experiment_id: str
    timestamp_utc: str
    dataset_version: str
    dataset_manifest_checksum: str
    num_queries: int
    metric_configuration: list[str]
    aggregate_metrics: dict[str, dict[str, float]]
    per_query_results: dict[str, dict[str, Any]]
    output_path: str | None = None
    experiment_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to standard dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> str:
        """Produce formatted console summary table of evaluation metrics."""
        lines = [
            f"============================================================",
            f" FastRAG Retrieval Evaluation Report: {self.experiment_id}",
            f" Timestamp (UTC): {self.timestamp_utc}",
            f" Queries Evaluated: {self.num_queries} | Dataset Version: {self.dataset_version}",
            f" Manifest Hash: {self.dataset_manifest_checksum[:12]}...",
            f"============================================================",
            f" {'Metric':<16} | {'Mean':<8} | {'Min':<8} | {'Max':<8} | {'StdDev':<8}",
            f"------------------------------------------------------------",
        ]
        for metric, stats in self.aggregate_metrics.items():
            lines.append(
                f" {metric:<16} | {stats['mean']:<8.4f} | {stats['min']:<8.4f} | "
                f"{stats['max']:<8.4f} | {stats.get('std', 0.0):<8.4f}"
            )
        lines.append(f"============================================================")
        if self.output_path:
            lines.append(f" Full results written to: {self.output_path}")
        return "\n".join(lines)


def load_qrels(qrels_path: Path | str) -> dict[str, dict[str, int]]:
    """Parse qrels.tsv file into structured mapping: query_id -> {doc_id: relevance}."""
    qrels_file = Path(qrels_path)
    if not qrels_file.is_file():
        raise FileNotFoundError(f"Qrels file not found at: {qrels_file}")

    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 4:
                continue
            q_id, _, d_id, rel_str = parts
            try:
                rel = int(rel_str)
            except ValueError:
                continue
            if q_id not in qrels:
                qrels[q_id] = {}
            qrels[q_id][d_id] = rel

    return qrels


class EvaluationRunner:
    """Executes metric evaluations across ranked candidate lists and qrels."""

    DEFAULT_METRICS = ["recall@5", "recall@10", "mrr@10", "ndcg@5", "ndcg@10"]

    def __init__(
        self,
        qrels: Mapping[str, Mapping[str, int]] | Path | str,
        metrics: Sequence[str] | None = None,
        datasets_dir: Path | str = DEFAULT_DATASETS_DIR,
        results_dir: Path | str = DEFAULT_RESULTS_DIR,
    ) -> None:
        if isinstance(qrels, (str, Path)):
            self.qrels = load_qrels(qrels)
        else:
            self.qrels = {q: dict(docs) for q, docs in qrels.items()}

        self.metrics = list(metrics or self.DEFAULT_METRICS)
        self.datasets_dir = Path(datasets_dir)
        self.results_dir = Path(results_dir)

    def run(
        self,
        rankings: Mapping[str, Sequence[str]],
        experiment_id: str = "baseline_run",
        description: str = "FastRAG evaluation run",
        retrieval_configuration: dict[str, Any] | None = None,
        save_results: bool = True,
    ) -> EvaluationReport:
        """Compute metrics for provided rankings against qrels.

        Args:
            rankings: Mapping of query_id to ordered sequence of retrieved doc_ids.
            experiment_id: Unique identifier for this evaluation run.
            description: Narrative description of experiment / configuration.
            retrieval_configuration: Details of retriever used (e.g. dense, bm25, hybrid).
            save_results: Whether to serialize output report to JSON in results_dir.

        Returns:
            EvaluationReport containing per-query metrics and dataset aggregates.
        """
        manifest_path = self.datasets_dir / "dataset_manifest.json"
        experiment_meta = create_experiment_metadata(
            experiment_id=experiment_id,
            description=description,
            manifest_path=manifest_path,
            retrieval_configuration=retrieval_configuration or {},
            metric_configuration=self.metrics,
        )

        per_query_results: dict[str, dict[str, Any]] = {}
        metric_values_by_name: dict[str, list[float]] = {m: [] for m in self.metrics}

        for query_id, query_qrels in self.qrels.items():
            retrieved = rankings.get(query_id, [])
            computed_metrics = evaluate_query_metrics(retrieved, query_qrels, self.metrics)

            num_relevant = sum(1 for rel in query_qrels.values() if rel > 0)
            per_query_results[query_id] = {
                "metrics": computed_metrics,
                "retrieved_count": len(retrieved),
                "relevant_count": num_relevant,
            }

            for m_name, val in computed_metrics.items():
                metric_values_by_name[m_name].append(val)

        # Compute aggregate metrics
        aggregate_metrics: dict[str, dict[str, float]] = {}
        for m_name, vals in metric_values_by_name.items():
            if not vals:
                aggregate_metrics[m_name] = {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
                continue

            n = len(vals)
            mean_val = sum(vals) / n
            variance = sum((x - mean_val) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
            std_val = math.sqrt(variance)

            aggregate_metrics[m_name] = {
                "mean": round(mean_val, 6),
                "min": round(min(vals), 6),
                "max": round(max(vals), 6),
                "std": round(std_val, 6),
                "count": n,
            }

        timestamp = datetime.now(timezone.utc).isoformat()
        output_filepath: str | None = None

        if save_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            safe_ts = timestamp.replace(":", "-").replace(".", "_")
            out_file = self.results_dir / f"{experiment_id}_{safe_ts}.json"
            output_filepath = str(out_file)

        report = EvaluationReport(
            experiment_id=experiment_id,
            timestamp_utc=timestamp,
            dataset_version=experiment_meta.dataset_version,
            dataset_manifest_checksum=experiment_meta.dataset_manifest_checksum,
            num_queries=len(self.qrels),
            metric_configuration=self.metrics,
            aggregate_metrics=aggregate_metrics,
            per_query_results=per_query_results,
            output_path=output_filepath,
            experiment_metadata=experiment_meta.to_dict(),
        )

        if save_results and output_filepath:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(report.to_json())

        return report
