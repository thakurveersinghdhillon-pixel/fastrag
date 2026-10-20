"""Unit tests for FastRAG Research Evaluation Dataset v2 integrity and validation module."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from evaluation.dataset_validation_v2 import DatasetV2Validator, validate_dataset_v2


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_V2_DIR = PROJECT_ROOT / "datasets" / "v2"


def test_production_dataset_v2_is_valid() -> None:
    """Verify that the actual FastRAG Dataset v2 passes all validation checks."""
    result = validate_dataset_v2(DATASETS_V2_DIR)
    assert result.is_valid, f"Dataset v2 validation failed: {result.errors}"
    assert result.stats["counts"]["documents"] == 36
    assert result.stats["counts"]["queries_total"] == 120
    assert result.stats["counts"]["queries_dev"] == 60
    assert result.stats["counts"]["queries_test"] == 60

    # Verify category distribution: 20 per category
    expected_categories = {
        "factual_direct": 20,
        "multi_hop": 20,
        "technical_identifiers": 20,
        "unanswerable_negative": 20,
        "ambiguous_underspecified": 20,
        "citation_grounding": 20,
    }
    assert result.stats["categories"] == expected_categories

    # Verify provenance statuses are tracked
    assert "human_reviewed" in result.stats["provenance"]
    assert "heuristic_candidate" in result.stats["provenance"]
    assert "pending_review" in result.stats["provenance"]


def _copy_dataset_v2_to_tmp(tmp_path: Path) -> None:
    for fname in [
        "corpus.jsonl",
        "queries.jsonl",
        "queries_dev.jsonl",
        "queries_test.jsonl",
        "qrels.tsv",
        "qrels_dev.tsv",
        "qrels_test.tsv",
        "dataset_manifest.json",
    ]:
        (tmp_path / fname).write_text((DATASETS_V2_DIR / fname).read_text(encoding="utf-8"), encoding="utf-8")


def test_missing_required_file_v2(tmp_path: Path) -> None:
    """Verify validation fails when a required v2 dataset file is missing."""
    _copy_dataset_v2_to_tmp(tmp_path)
    (tmp_path / "queries_dev.jsonl").unlink()

    validator = DatasetV2Validator(tmp_path)
    result = validator.validate()
    assert not result.is_valid
    assert any("Missing required dataset file" in err and "queries_dev.jsonl" in err for err in result.errors)


def test_missing_required_fields_v2(tmp_path: Path) -> None:
    """Verify validation detects missing required fields in queries.jsonl."""
    _copy_dataset_v2_to_tmp(tmp_path)

    # Mutate line 1 of queries.jsonl to omit 'expected_evidence'
    lines = (tmp_path / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    del rec["expected_evidence"]
    lines[0] = json.dumps(rec)
    (tmp_path / "queries.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = validate_dataset_v2(tmp_path)
    assert not result.is_valid
    assert any("missing field 'expected_evidence'" in err for err in result.errors)


def test_invalid_split_v2(tmp_path: Path) -> None:
    """Verify validation detects an invalid split label."""
    _copy_dataset_v2_to_tmp(tmp_path)

    lines = (tmp_path / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["split"] = "validation_split"
    lines[0] = json.dumps(rec)
    (tmp_path / "queries.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = validate_dataset_v2(tmp_path)
    assert not result.is_valid
    assert any("invalid split 'validation_split'" in err for err in result.errors)


def test_duplicate_query_ids_v2(tmp_path: Path) -> None:
    """Verify validation detects duplicate query identifiers in queries.jsonl."""
    _copy_dataset_v2_to_tmp(tmp_path)

    lines = (tmp_path / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    lines.append(lines[0])
    (tmp_path / "queries.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = validate_dataset_v2(tmp_path)
    assert not result.is_valid
    assert any("duplicate query_id" in err for err in result.errors)


def test_duplicate_query_text_leakage_v2(tmp_path: Path) -> None:
    """Verify validation detects duplicate query text between dev and test queries."""
    _copy_dataset_v2_to_tmp(tmp_path)

    lines = (tmp_path / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    rec_dev = json.loads(lines[0])  # dev query
    # Find a test query and set its text identical to dev query
    for i in range(1, len(lines)):
        rec_test = json.loads(lines[i])
        if rec_test["split"] == "test":
            rec_test["query"] = rec_dev["query"]
            lines[i] = json.dumps(rec_test)
            break
    (tmp_path / "queries.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = validate_dataset_v2(tmp_path)
    assert not result.is_valid
    assert any("Duplicate query text" in err for err in result.errors)


def test_unanswerable_negative_inconsistency_v2(tmp_path: Path) -> None:
    """Verify validation detects when an unanswerable query has non-empty source_docs or positive relevance."""
    _copy_dataset_v2_to_tmp(tmp_path)

    lines = (tmp_path / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if rec["category"] == "unanswerable_negative":
            rec["source_doc_ids"] = ["doc_001"]  # Invalid for unanswerable
            lines[i] = json.dumps(rec)
            break
    (tmp_path / "queries.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = validate_dataset_v2(tmp_path)
    assert not result.is_valid
    assert any("unanswerable_negative query" in err and "must have empty source_doc_ids" in err for err in result.errors)


def test_invalid_doc_reference_v2(tmp_path: Path) -> None:
    """Verify validation detects qrels referencing non-existent documents."""
    _copy_dataset_v2_to_tmp(tmp_path)

    with open(tmp_path / "qrels.tsv", "a", encoding="utf-8") as f:
        f.write("q2_dev_001\t0\tdoc_9999\t2\n")

    result = validate_dataset_v2(tmp_path)
    assert not result.is_valid
    assert any("references non-existent doc_id 'doc_9999'" in err for err in result.errors)


def test_invalid_relevance_grade_v2(tmp_path: Path) -> None:
    """Verify validation detects out-of-scale relevance grades."""
    _copy_dataset_v2_to_tmp(tmp_path)

    with open(tmp_path / "qrels.tsv", "a", encoding="utf-8") as f:
        f.write("q2_dev_001\t0\tdoc_001\t5\n")

    result = validate_dataset_v2(tmp_path)
    assert not result.is_valid
    assert any("invalid relevance grade '5'" in err for err in result.errors)


def test_manifest_checksum_mismatch_v2(tmp_path: Path) -> None:
    """Verify validation catches content alteration via SHA-256 checksum mismatch."""
    _copy_dataset_v2_to_tmp(tmp_path)

    with open(tmp_path / "qrels.tsv", "a", encoding="utf-8") as f:
        f.write("\n")  # Modify file content

    result = validate_dataset_v2(tmp_path)
    assert not result.is_valid
    assert any("Checksum mismatch for 'qrels.tsv'" in err for err in result.errors)
