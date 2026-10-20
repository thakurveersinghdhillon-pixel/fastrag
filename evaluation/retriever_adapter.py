"""FastRAG Retriever Evaluation Adapter.

Executes a BaseRetriever against evaluation queries, records cold-start and warm query latencies,
extracts ranked candidate documents, and bridges results into the Phase 2 EvaluationRunner.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from app.core.logging_config import get_logger
from app.retrieval.base import BaseRetriever
from evaluation.runner import EvaluationReport, EvaluationRunner, load_qrels

logger = get_logger("fastrag.evaluation.adapter")


@dataclass
class RetrievalLatencyStats:
    """Latency profile separating cold-start from warm-query execution."""

    cold_start_ms: float
    warm_mean_ms: float
    warm_p50_ms: float
    warm_p95_ms: float
    warm_p99_ms: float
    warm_query_count: int
    total_duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cold_start_ms": round(self.cold_start_ms, 3),
            "warm_mean_ms": round(self.warm_mean_ms, 3),
            "warm_p50_ms": round(self.warm_p50_ms, 3),
            "warm_p95_ms": round(self.warm_p95_ms, 3),
            "warm_p99_ms": round(self.warm_p99_ms, 3),
            "warm_query_count": self.warm_query_count,
            "total_duration_seconds": round(self.total_duration_seconds, 4),
        }


def load_evaluation_queries(queries_path: Path | str) -> list[dict[str, Any]]:
    """Parse queries from queries.jsonl into a list of query dicts."""
    path = Path(queries_path)
    if not path.is_file():
        raise FileNotFoundError(f"Queries file not found at: {path}")

    queries: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                queries.append(json.loads(stripped))
    return queries


def evaluate_retriever(
    retriever: BaseRetriever,
    queries_path: Path | str,
    qrels_path: Path | str,
    experiment_id: str,
    top_k: int = 10,
    metrics: Sequence[str] | None = None,
    description: str | None = None,
    save_results: bool = True,
    results_dir: Path | str | None = None,
    datasets_dir: Path | str | None = None,
    search_kwargs: dict[str, Any] | None = None,
) -> tuple[EvaluationReport, RetrievalLatencyStats]:
    """Execute evaluation queries against a retriever and produce Phase 2 EvaluationReport with latency stats."""
    queries = load_evaluation_queries(queries_path)
    qrels = load_qrels(qrels_path)

    extra_kwargs = search_kwargs or {}
    rankings: dict[str, list[str]] = {}
    latencies_ms: list[float] = []

    # Diagnostics accumulation for hybrid and reranking pipelines
    dense_branch_ms: list[float] = []
    bm25_branch_ms: list[float] = []
    fusion_branch_ms: list[float] = []
    rerank_branch_ms: list[float] = []
    degraded_count = 0

    overall_start = time.perf_counter()

    for query_item in queries:
        q_id = query_item["query_id"]
        q_text = query_item["query"]

        q_start = time.perf_counter()
        results = retriever.search(q_text, top_k=top_k, **extra_kwargs)
        q_elapsed_ms = (time.perf_counter() - q_start) * 1000.0
        latencies_ms.append(q_elapsed_ms)

        # Collect diagnostics if available
        last_diag = getattr(retriever, "last_diagnostics", None)
        if last_diag is not None:
            if hasattr(last_diag, "rerank_ms"):
                rerank_branch_ms.append(last_diag.rerank_ms)
                if last_diag.hybrid_breakdown:
                    h = last_diag.hybrid_breakdown
                    dense_branch_ms.append(h.get("dense", {}).get("duration_ms", 0.0))
                    bm25_branch_ms.append(h.get("bm25", {}).get("duration_ms", 0.0))
                    fusion_branch_ms.append(h.get("fusion_duration_ms", 0.0))
                    if h.get("degraded", False):
                        degraded_count += 1
            elif hasattr(last_diag, "dense"):
                dense_branch_ms.append(last_diag.dense.duration_ms)
                bm25_branch_ms.append(last_diag.bm25.duration_ms)
                fusion_branch_ms.append(last_diag.fusion_duration_ms)
                if last_diag.degraded:
                    degraded_count += 1

        # Extract ranked document IDs preserving order, deduplicating consecutive chunk hits
        seen_docs: set[str] = set()
        ranked_doc_ids: list[str] = []
        for r in results:
            if r.document_id not in seen_docs:
                seen_docs.add(r.document_id)
                ranked_doc_ids.append(r.document_id)

        rankings[q_id] = ranked_doc_ids

    total_elapsed = time.perf_counter() - overall_start

    # Compute cold vs warm latency statistics
    cold_start_ms = latencies_ms[0] if latencies_ms else 0.0
    warm_latencies = latencies_ms[1:] if len(latencies_ms) > 1 else latencies_ms

    if warm_latencies:
        warm_mean = statistics.mean(warm_latencies)
        warm_sorted = sorted(warm_latencies)
        n_warm = len(warm_sorted)
        p50 = warm_sorted[int(n_warm * 0.50)]
        p95 = warm_sorted[min(int(n_warm * 0.95), n_warm - 1)]
        p99 = warm_sorted[min(int(n_warm * 0.99), n_warm - 1)]
    else:
        warm_mean = cold_start_ms
        p50 = cold_start_ms
        p95 = cold_start_ms
        p99 = cold_start_ms

    latency_stats = RetrievalLatencyStats(
        cold_start_ms=cold_start_ms,
        warm_mean_ms=warm_mean,
        warm_p50_ms=p50,
        warm_p95_ms=p95,
        warm_p99_ms=p99,
        warm_query_count=len(warm_latencies),
        total_duration_seconds=total_elapsed,
    )

    index_info = retriever.get_index_info()
    retriever_config: dict[str, Any] = {
        "retriever_type": index_info.retriever_type if index_info else "unknown",
        "index_info": index_info.to_dict() if index_info else {},
        "top_k": top_k,
        "search_kwargs": extra_kwargs,
        "latency_stats": latency_stats.to_dict(),
    }

    if dense_branch_ms and bm25_branch_ms:
        retriever_config["hybrid_breakdown"] = {
            "dense_mean_ms": round(statistics.mean(dense_branch_ms), 3),
            "dense_p50_ms": round(statistics.median(dense_branch_ms), 3),
            "bm25_mean_ms": round(statistics.mean(bm25_branch_ms), 3),
            "bm25_p50_ms": round(statistics.median(bm25_branch_ms), 3),
            "fusion_mean_ms": round(statistics.mean(fusion_branch_ms), 3),
            "degraded_queries": degraded_count,
        }

    if rerank_branch_ms:
        retriever_config["reranker_breakdown"] = {
            "rerank_mean_ms": round(statistics.mean(rerank_branch_ms), 3),
            "rerank_p50_ms": round(statistics.median(rerank_branch_ms), 3),
            "rerank_query_count": len(rerank_branch_ms),
        }

    if datasets_dir is not None:
        target_datasets_dir = Path(datasets_dir)
    elif (Path(queries_path).parent / "dataset_manifest.json").is_file():
        target_datasets_dir = Path(queries_path).parent
    else:
        target_datasets_dir = Path(__file__).resolve().parent.parent / "datasets"

    runner_kwargs: dict[str, Any] = {
        "qrels": qrels,
        "metrics": metrics or ["recall@5", "recall@10", "mrr@10", "ndcg@5", "ndcg@10"],
        "datasets_dir": target_datasets_dir,
    }
    if results_dir:
        runner_kwargs["results_dir"] = results_dir

    runner = EvaluationRunner(**runner_kwargs)
    report = runner.run(
        rankings=rankings,
        experiment_id=experiment_id,
        description=description or f"Evaluation of {retriever_config['retriever_type']} retriever",
        retrieval_configuration=retriever_config,
        save_results=save_results,
    )

    return report, latency_stats
