"""FastRAG Dataset Validation Module.

Validates the integrity, schema, consistency, and checksums of the evaluation dataset:
- Existence of corpus.jsonl, queries.jsonl, qrels.tsv, and dataset_manifest.json
- Valid JSON and required field schemas for each record
- Uniqueness of document and query identifiers
- Referential integrity of qrels against corpus and queries
- Valid relevance scale membership (0, 1, 2)
- Programmatic agreement with dataset manifest counts and SHA-256 hashes
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ValidationResult:
    """Structured report produced by the dataset validator."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Produce human-readable diagnostic summary."""
        lines = [
            f"Dataset Validation: {'PASSED' if self.is_valid else 'FAILED'}",
            f"  Errors:   {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        if self.stats:
            lines.append("  Counts:")
            for k, v in self.stats.get("counts", {}).items():
                lines.append(f"    - {k}: {v}")
        if self.errors:
            lines.append("  Error Details:")
            for err in self.errors:
                lines.append(f"    - {err}")
        return "\n".join(lines)


def sha256_file(filepath: Path) -> str:
    """Compute the SHA-256 hexadecimal hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class DatasetValidator:
    """Validates FastRAG dataset files for integrity, schema adherence, and consistency."""

    REQUIRED_FILES = ("corpus.jsonl", "queries.jsonl", "qrels.tsv", "dataset_manifest.json")
    CORPUS_REQUIRED_FIELDS = {"doc_id": str, "title": str, "text": str, "source": str}
    QUERY_REQUIRED_FIELDS = {"query_id": str, "query": str, "category": str, "expected_behavior": str}
    VALID_RELEVANCE_LEVELS = {0, 1, 2}

    def __init__(self, dataset_dir: Path | str) -> None:
        self.dataset_dir = Path(dataset_dir)

    def validate(self) -> ValidationResult:
        """Execute complete validation suite and return structured ValidationResult."""
        errors: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {}

        # 1. Verify existence of required files
        for filename in self.REQUIRED_FILES:
            file_path = self.dataset_dir / filename
            if not file_path.is_file():
                errors.append(f"Missing required dataset file: {file_path}")

        if errors:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # 2. Validate manifest JSON
        manifest_path = self.dataset_dir / "dataset_manifest.json"
        manifest_data = {}
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except json.JSONDecodeError as exc:
            errors.append(f"dataset_manifest.json is not valid JSON: {exc}")
        except Exception as exc:
            errors.append(f"Failed to read dataset_manifest.json: {exc}")

        # 3. Validate corpus.jsonl
        corpus_path = self.dataset_dir / "corpus.jsonl"
        doc_ids: set[str] = set()
        corpus_count = 0

        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    raw_line = line.strip()
                    if not raw_line:
                        errors.append(f"corpus.jsonl line {line_num}: empty line encountered")
                        continue
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        errors.append(f"corpus.jsonl line {line_num}: invalid JSON ({exc})")
                        continue

                    if not isinstance(record, dict):
                        errors.append(f"corpus.jsonl line {line_num}: record is not a JSON object")
                        continue

                    # Check required fields and types
                    for field_name, expected_type in self.CORPUS_REQUIRED_FIELDS.items():
                        if field_name not in record:
                            errors.append(f"corpus.jsonl line {line_num}: missing field '{field_name}'")
                        elif not isinstance(record[field_name], expected_type):
                            errors.append(
                                f"corpus.jsonl line {line_num}: field '{field_name}' must be of type {expected_type.__name__}"
                            )
                        elif expected_type is str and not record[field_name].strip():
                            errors.append(f"corpus.jsonl line {line_num}: field '{field_name}' must not be empty")

                    # Check metadata field if present
                    if "metadata" in record and not isinstance(record["metadata"], dict):
                        errors.append(f"corpus.jsonl line {line_num}: 'metadata' must be a JSON object")

                    doc_id = record.get("doc_id")
                    if isinstance(doc_id, str) and doc_id:
                        if doc_id in doc_ids:
                            errors.append(f"corpus.jsonl line {line_num}: duplicate doc_id '{doc_id}'")
                        doc_ids.add(doc_id)
                    corpus_count += 1
        except Exception as exc:
            errors.append(f"Error reading corpus.jsonl: {exc}")

        # 4. Validate queries.jsonl
        queries_path = self.dataset_dir / "queries.jsonl"
        query_ids: set[str] = set()
        queries_count = 0

        try:
            with open(queries_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    raw_line = line.strip()
                    if not raw_line:
                        errors.append(f"queries.jsonl line {line_num}: empty line encountered")
                        continue
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        errors.append(f"queries.jsonl line {line_num}: invalid JSON ({exc})")
                        continue

                    if not isinstance(record, dict):
                        errors.append(f"queries.jsonl line {line_num}: record is not a JSON object")
                        continue

                    for field_name, expected_type in self.QUERY_REQUIRED_FIELDS.items():
                        if field_name not in record:
                            errors.append(f"queries.jsonl line {line_num}: missing field '{field_name}'")
                        elif not isinstance(record[field_name], expected_type):
                            errors.append(
                                f"queries.jsonl line {line_num}: field '{field_name}' must be of type {expected_type.__name__}"
                            )
                        elif expected_type is str and not record[field_name].strip():
                            errors.append(f"queries.jsonl line {line_num}: field '{field_name}' must not be empty")

                    query_id = record.get("query_id")
                    if isinstance(query_id, str) and query_id:
                        if query_id in query_ids:
                            errors.append(f"queries.jsonl line {line_num}: duplicate query_id '{query_id}'")
                        query_ids.add(query_id)
                    queries_count += 1
        except Exception as exc:
            errors.append(f"Error reading queries.jsonl: {exc}")

        # 5. Validate qrels.tsv
        qrels_path = self.dataset_dir / "qrels.tsv"
        qrels_count = 0
        seen_pairs: set[tuple[str, str]] = set()

        try:
            with open(qrels_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    raw_line = line.strip()
                    if not raw_line:
                        errors.append(f"qrels.tsv line {line_num}: empty line encountered")
                        continue

                    parts = raw_line.split("\t")
                    if len(parts) != 4:
                        errors.append(
                            f"qrels.tsv line {line_num}: expected 4 tab-separated columns "
                            f"(query_id, 0, doc_id, relevance), found {len(parts)}"
                        )
                        continue

                    q_id, iter_zero, d_id, rel_str = parts

                    if iter_zero != "0":
                        errors.append(
                            f"qrels.tsv line {line_num}: column 2 must be '0' per TREC convention, found '{iter_zero}'"
                        )

                    if q_id not in query_ids:
                        errors.append(
                            f"qrels.tsv line {line_num}: references non-existent query_id '{q_id}'"
                        )

                    if d_id not in doc_ids:
                        errors.append(
                            f"qrels.tsv line {line_num}: references non-existent doc_id '{d_id}'"
                        )

                    try:
                        rel = int(rel_str)
                        if rel not in self.VALID_RELEVANCE_LEVELS:
                            errors.append(
                                f"qrels.tsv line {line_num}: invalid relevance '{rel}'. "
                                f"Must be one of {sorted(self.VALID_RELEVANCE_LEVELS)}"
                            )
                    except ValueError:
                        errors.append(f"qrels.tsv line {line_num}: relevance '{rel_str}' is not an integer")

                    pair = (q_id, d_id)
                    if pair in seen_pairs:
                        errors.append(
                            f"qrels.tsv line {line_num}: duplicate judgment for query '{q_id}' and doc '{d_id}'"
                        )
                    seen_pairs.add(pair)
                    qrels_count += 1
        except Exception as exc:
            errors.append(f"Error reading qrels.tsv: {exc}")

        # 6. Validate manifest counts and checksums
        if manifest_data:
            manifest_counts = manifest_data.get("counts", {})
            if manifest_counts.get("documents") != corpus_count:
                errors.append(
                    f"Manifest document count mismatch: manifest says {manifest_counts.get('documents')}, "
                    f"actual is {corpus_count}"
                )
            if manifest_counts.get("queries") != queries_count:
                errors.append(
                    f"Manifest query count mismatch: manifest says {manifest_counts.get('queries')}, "
                    f"actual is {queries_count}"
                )
            if manifest_counts.get("qrels") != qrels_count:
                errors.append(
                    f"Manifest qrels count mismatch: manifest says {manifest_counts.get('qrels')}, "
                    f"actual is {qrels_count}"
                )

            manifest_files = manifest_data.get("files", {})
            for fname, expected_info in manifest_files.items():
                target_file = self.dataset_dir / fname
                if not target_file.is_file():
                    errors.append(f"Manifest references file that does not exist: {target_file}")
                    continue

                actual_sha = sha256_file(target_file)
                expected_sha = expected_info.get("sha256")
                if actual_sha != expected_sha:
                    errors.append(
                        f"Checksum mismatch for '{fname}': expected {expected_sha}, computed {actual_sha}"
                    )

        stats["counts"] = {
            "documents": corpus_count,
            "queries": queries_count,
            "qrels": qrels_count,
        }
        stats["unique_docs_in_corpus"] = len(doc_ids)
        stats["unique_queries_in_dataset"] = len(query_ids)
        stats["unique_judged_pairs"] = len(seen_pairs)

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, stats=stats)


def validate_dataset(dataset_dir: Path | str) -> ValidationResult:
    """Convenience functional wrapper for dataset validation."""
    validator = DatasetValidator(dataset_dir)
    return validator.validate()
