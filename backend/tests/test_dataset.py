"""Tests for FastRAG Dataset integrity and validation module."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from evaluation.dataset_validation import DatasetValidator, validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def test_production_dataset_is_valid() -> None:
    """Verify that the actual FastRAG dataset passes all validation checks."""
    result = validate_dataset(DATASETS_DIR)
    assert result.is_valid, f"Dataset validation failed: {result.errors}"
    assert result.stats["counts"]["documents"] == 36
    assert result.stats["counts"]["queries"] == 24
    assert result.stats["counts"]["qrels"] == 88


def test_missing_required_file(tmp_path: Path) -> None:
    """Verify validation fails when a required dataset file is missing."""
    validator = DatasetValidator(tmp_path)
    result = validator.validate()
    assert not result.is_valid
    assert any("Missing required dataset file" in err for err in result.errors)


def test_invalid_jsonl_parsing(tmp_path: Path) -> None:
    """Verify validation fails when a JSONL file contains invalid JSON syntax."""
    # Copy valid dataset to tmp_path
    for fname in ["corpus.jsonl", "queries.jsonl", "qrels.tsv", "dataset_manifest.json"]:
        (tmp_path / fname).write_text((DATASETS_DIR / fname).read_text(encoding="utf-8"), encoding="utf-8")

    # Corrupt corpus.jsonl line 2
    corpus_lines = (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    corpus_lines[1] = '{"doc_id": "bad", corrupt json'
    (tmp_path / "corpus.jsonl").write_text("\n".join(corpus_lines) + "\n", encoding="utf-8")

    result = validate_dataset(tmp_path)
    assert not result.is_valid
    assert any("invalid JSON" in err for err in result.errors)


def test_missing_required_fields_in_corpus(tmp_path: Path) -> None:
    """Verify validation detects missing required fields in corpus."""
    for fname in ["corpus.jsonl", "queries.jsonl", "qrels.tsv", "dataset_manifest.json"]:
        (tmp_path / fname).write_text((DATASETS_DIR / fname).read_text(encoding="utf-8"), encoding="utf-8")

    # Replace with record missing 'text'
    bad_record = {"doc_id": "doc_bad", "title": "No text", "source": "test"}
    (tmp_path / "corpus.jsonl").write_text(json.dumps(bad_record) + "\n", encoding="utf-8")

    result = validate_dataset(tmp_path)
    assert not result.is_valid
    assert any("missing field 'text'" in err for err in result.errors)


def test_duplicate_document_ids(tmp_path: Path) -> None:
    """Verify validation detects duplicate document identifiers."""
    for fname in ["corpus.jsonl", "queries.jsonl", "qrels.tsv", "dataset_manifest.json"]:
        (tmp_path / fname).write_text((DATASETS_DIR / fname).read_text(encoding="utf-8"), encoding="utf-8")

    lines = (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    lines.append(lines[0])  # Duplicate first doc
    (tmp_path / "corpus.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = validate_dataset(tmp_path)
    assert not result.is_valid
    assert any("duplicate doc_id" in err for err in result.errors)


def test_duplicate_query_ids(tmp_path: Path) -> None:
    """Verify validation detects duplicate query identifiers."""
    for fname in ["corpus.jsonl", "queries.jsonl", "qrels.tsv", "dataset_manifest.json"]:
        (tmp_path / fname).write_text((DATASETS_DIR / fname).read_text(encoding="utf-8"), encoding="utf-8")

    lines = (tmp_path / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    lines.append(lines[0])  # Duplicate first query
    (tmp_path / "queries.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = validate_dataset(tmp_path)
    assert not result.is_valid
    assert any("duplicate query_id" in err for err in result.errors)


def test_invalid_qrels_references(tmp_path: Path) -> None:
    """Verify validation detects qrels referencing non-existent docs or queries."""
    for fname in ["corpus.jsonl", "queries.jsonl", "qrels.tsv", "dataset_manifest.json"]:
        (tmp_path / fname).write_text((DATASETS_DIR / fname).read_text(encoding="utf-8"), encoding="utf-8")

    with open(tmp_path / "qrels.tsv", "a", encoding="utf-8") as f:
        f.write("q_nonexistent\t0\tdoc_001\t2\n")
        f.write("q_001\t0\tdoc_nonexistent\t2\n")

    result = validate_dataset(tmp_path)
    assert not result.is_valid
    assert any("references non-existent query_id 'q_nonexistent'" in err for err in result.errors)
    assert any("references non-existent doc_id 'doc_nonexistent'" in err for err in result.errors)


def test_invalid_relevance_labels(tmp_path: Path) -> None:
    """Verify validation catches invalid relevance scale scores (e.g. 5 or negative)."""
    for fname in ["corpus.jsonl", "queries.jsonl", "qrels.tsv", "dataset_manifest.json"]:
        (tmp_path / fname).write_text((DATASETS_DIR / fname).read_text(encoding="utf-8"), encoding="utf-8")

    with open(tmp_path / "qrels.tsv", "a", encoding="utf-8") as f:
        f.write("q_001\t0\tdoc_002\t9\n")

    result = validate_dataset(tmp_path)
    assert not result.is_valid
    assert any("invalid relevance '9'" in err for err in result.errors)


def test_incorrect_manifest_counts(tmp_path: Path) -> None:
    """Verify validation catches mismatches between manifest counts and actual files."""
    for fname in ["corpus.jsonl", "queries.jsonl", "qrels.tsv", "dataset_manifest.json"]:
        (tmp_path / fname).write_text((DATASETS_DIR / fname).read_text(encoding="utf-8"), encoding="utf-8")

    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["counts"]["documents"] = 999
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_dataset(tmp_path)
    assert not result.is_valid
    assert any("Manifest document count mismatch" in err for err in result.errors)


def test_incorrect_manifest_checksum(tmp_path: Path) -> None:
    """Verify validation detects modified file contents failing manifest SHA-256."""
    for fname in ["corpus.jsonl", "queries.jsonl", "qrels.tsv", "dataset_manifest.json"]:
        (tmp_path / fname).write_text((DATASETS_DIR / fname).read_text(encoding="utf-8"), encoding="utf-8")

    # Modify corpus without updating manifest
    with open(tmp_path / "corpus.jsonl", "a", encoding="utf-8") as f:
        f.write("\n")

    result = validate_dataset(tmp_path)
    assert not result.is_valid
    assert any("Checksum mismatch" in err for err in result.errors)
