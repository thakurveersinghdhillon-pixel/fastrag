"""Tests for FastRAG Evaluation Runner, experiment tracking, and fixtures."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from evaluation.experiment import ExperimentMetadata, create_experiment_metadata
from evaluation.fixtures import (
    get_synthetic_empty_rankings,
    get_synthetic_inverse_rankings,
    get_synthetic_partial_rankings,
    get_synthetic_perfect_rankings,
)
from evaluation.runner import EvaluationRunner, load_qrels


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def test_load_qrels_format() -> None:
    """Verify load_qrels parses the TSV file into nested query_id -> doc_id mapping."""
    qrels_path = DATASETS_DIR / "qrels.tsv"
    qrels = load_qrels(qrels_path)

    assert len(qrels) == 24
    assert "q_001" in qrels
    assert qrels["q_001"]["doc_001"] == 2
    assert qrels["q_001"]["doc_033"] == 0


def test_experiment_metadata_creation_and_checksum(tmp_path: Path) -> None:
    """Verify experiment metadata locks the manifest checksum and serializes to JSON."""
    manifest_path = DATASETS_DIR / "dataset_manifest.json"
    meta = create_experiment_metadata(
        experiment_id="test_exp_01",
        description="Unit test experiment",
        manifest_path=manifest_path,
        retrieval_configuration={"model": "synthetic"},
        metric_configuration=["recall@5", "mrr@10"],
    )

    assert meta.experiment_id == "test_exp_01"
    assert meta.dataset_version == "1.0.0"
    assert len(meta.dataset_manifest_checksum) == 64

    # Save and reload
    save_file = tmp_path / "experiment.json"
    meta.save(save_file)
    assert save_file.is_file()

    loaded = ExperimentMetadata.from_json(save_file.read_text(encoding="utf-8"))
    assert loaded.experiment_id == meta.experiment_id
    assert loaded.dataset_manifest_checksum == meta.dataset_manifest_checksum


def test_runner_perfect_fixture_yields_perfect_scores(tmp_path: Path) -> None:
    """Verify that the synthetic perfect ranking fixture scores 1.0 on all metrics."""
    qrels = load_qrels(DATASETS_DIR / "qrels.tsv")
    rankings = get_synthetic_perfect_rankings(qrels)

    runner = EvaluationRunner(
        qrels=qrels,
        metrics=["recall@5", "recall@10", "mrr@10", "ndcg@5", "ndcg@10"],
        datasets_dir=DATASETS_DIR,
        results_dir=tmp_path,
    )

    report = runner.run(
        rankings=rankings,
        experiment_id="perfect_test",
        save_results=True,
    )

    for metric_name in ["recall@5", "recall@10", "mrr@10", "ndcg@5", "ndcg@10"]:
        assert report.aggregate_metrics[metric_name]["mean"] == 1.0

    assert report.output_path is not None
    assert Path(report.output_path).is_file()

    # Verify JSON structure
    report_json = json.loads(Path(report.output_path).read_text(encoding="utf-8"))
    assert report_json["experiment_id"] == "perfect_test"
    assert report_json["num_queries"] == 24
    assert len(report_json["per_query_results"]) == 24


def test_runner_empty_rankings_yields_zero_scores(tmp_path: Path) -> None:
    """Verify that completely empty rankings safely yield 0.0 without errors."""
    qrels = load_qrels(DATASETS_DIR / "qrels.tsv")
    rankings = get_synthetic_empty_rankings(qrels)

    runner = EvaluationRunner(
        qrels=qrels,
        metrics=["recall@5", "mrr@10", "ndcg@10"],
        datasets_dir=DATASETS_DIR,
        results_dir=tmp_path,
    )

    report = runner.run(
        rankings=rankings,
        experiment_id="empty_test",
        save_results=False,
    )

    for metric_name in ["recall@5", "mrr@10", "ndcg@10"]:
        assert report.aggregate_metrics[metric_name]["mean"] == 0.0


def test_runner_summary_table_formatting() -> None:
    """Verify human-readable summary output generation."""
    qrels = {"q1": {"d1": 2, "d2": 1}}
    rankings = {"q1": ["d1", "d2"]}

    runner = EvaluationRunner(
        qrels=qrels,
        metrics=["recall@5", "mrr@5"],
        datasets_dir=DATASETS_DIR,
    )

    report = runner.run(rankings=rankings, experiment_id="summary_test", save_results=False)
    summary_text = report.summary()
    assert "FastRAG Retrieval Evaluation Report: summary_test" in summary_text
    assert "recall@5" in summary_text
    assert "mrr@5" in summary_text
