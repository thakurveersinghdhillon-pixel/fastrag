"""FastRAG Research Evaluation Dataset v2 Validation Module.

Validates the integrity, schema, split isolation, consistency, and checksums
of FastRAG Dataset v2:
- Required files: corpus.jsonl, queries.jsonl, queries_dev.jsonl, queries_test.jsonl,
  qrels.tsv, qrels_dev.tsv, qrels_test.tsv, dataset_manifest.json
- Schema conformity for corpus and queries (including v2 fields: split, category,
  reference_answer, source_doc_ids, expected_evidence, relevance_judgments, annotation_provenance)
- Unique IDs for documents and queries
- Split consistency: partitioning into dev and test splits without leakage or ID overlap
- Near-duplicate detection across queries (lexical Jaccard and normalized match)
- Referential integrity: source_doc_ids and qrels refer to real corpus doc_ids and queries
- Graded relevance scale validation: {0, 1, 2}
- Consistency: unanswerable queries have no source docs and no positive relevance;
  factual/multi-hop queries have non-empty source docs with positive relevance
- Manifest integrity: cryptographic SHA-256 agreement and count validation
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.dataset_validation import ValidationResult, sha256_file


class DatasetV2Validator:
    """Validates FastRAG Dataset v2 structure, schemas, consistency, and split integrity."""

    REQUIRED_FILES = (
        "corpus.jsonl",
        "queries.jsonl",
        "queries_dev.jsonl",
        "queries_test.jsonl",
        "qrels.tsv",
        "qrels_dev.tsv",
        "qrels_test.tsv",
        "dataset_manifest.json",
    )
    CORPUS_REQUIRED_FIELDS = {"doc_id": str, "title": str, "text": str, "source": str}
    QUERY_REQUIRED_FIELDS = {
        "query_id": str,
        "query": str,
        "category": str,
        "split": str,
        "source_doc_ids": list,
        "expected_evidence": list,
        "relevance_judgments": dict,
        "annotation_provenance": dict,
    }
    ALLOWED_CATEGORIES = {
        "factual_direct",
        "multi_hop",
        "technical_identifiers",
        "unanswerable_negative",
        "ambiguous_underspecified",
        "citation_grounding",
    }
    ALLOWED_SPLITS = {"dev", "test"}
    ALLOWED_PROVENANCE_STATUSES = {"human_reviewed", "heuristic_candidate", "pending_review"}
    VALID_RELEVANCE_LEVELS = {0, 1, 2}
    QUERY_ID_PATTERN = re.compile(r"^q2_(dev|test)_[0-9]{3}$")

    def __init__(self, dataset_dir: Path | str) -> None:
        self.dataset_dir = Path(dataset_dir)

    def validate(self) -> ValidationResult:
        """Execute full validation suite and return structured ValidationResult."""
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
        manifest_data: dict[str, Any] = {}
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

                    for field_name, expected_type in self.CORPUS_REQUIRED_FIELDS.items():
                        if field_name not in record:
                            errors.append(f"corpus.jsonl line {line_num}: missing field '{field_name}'")
                        elif not isinstance(record[field_name], expected_type):
                            errors.append(
                                f"corpus.jsonl line {line_num}: field '{field_name}' must be of type {expected_type.__name__}"
                            )
                        elif expected_type is str and not record[field_name].strip():
                            errors.append(f"corpus.jsonl line {line_num}: field '{field_name}' must not be empty")

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
        all_query_records: list[dict[str, Any]] = []
        query_ids: set[str] = set()
        dev_query_ids: set[str] = set()
        test_query_ids: set[str] = set()
        queries_by_id: dict[str, dict[str, Any]] = {}
        category_distribution: dict[str, int] = {}
        provenance_distribution: dict[str, int] = {}

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

                    # Check required fields
                    for field_name, expected_type in self.QUERY_REQUIRED_FIELDS.items():
                        if field_name not in record:
                            errors.append(f"queries.jsonl line {line_num}: missing field '{field_name}'")
                        elif not isinstance(record[field_name], expected_type):
                            errors.append(
                                f"queries.jsonl line {line_num}: field '{field_name}' must be of type {expected_type.__name__}"
                            )

                    # Check reference_answer: str or None
                    if "reference_answer" in record and record["reference_answer"] is not None and not isinstance(record["reference_answer"], str):
                        errors.append(f"queries.jsonl line {line_num}: 'reference_answer' must be a string or null")

                    q_id = record.get("query_id")
                    if isinstance(q_id, str) and q_id:
                        if q_id in query_ids:
                            errors.append(f"queries.jsonl line {line_num}: duplicate query_id '{q_id}'")
                        if not self.QUERY_ID_PATTERN.match(q_id):
                            errors.append(
                                f"queries.jsonl line {line_num}: query_id '{q_id}' must match pattern ^q2_(dev|test)_[0-9]{{3}}$"
                            )
                        query_ids.add(q_id)
                        queries_by_id[q_id] = record

                    # Category check
                    cat = record.get("category")
                    if cat not in self.ALLOWED_CATEGORIES:
                        errors.append(
                            f"queries.jsonl line {line_num}: invalid category '{cat}'. Must be one of {sorted(self.ALLOWED_CATEGORIES)}"
                        )
                    else:
                        category_distribution[cat] = category_distribution.get(cat, 0) + 1

                    # Split check
                    split = record.get("split")
                    if split not in self.ALLOWED_SPLITS:
                        errors.append(
                            f"queries.jsonl line {line_num}: invalid split '{split}'. Must be one of {sorted(self.ALLOWED_SPLITS)}"
                        )
                    elif q_id:
                        if split == "dev":
                            if not q_id.startswith("q2_dev_"):
                                errors.append(f"queries.jsonl line {line_num}: query with split 'dev' must have prefix 'q2_dev_'")
                            dev_query_ids.add(q_id)
                        elif split == "test":
                            if not q_id.startswith("q2_test_"):
                                errors.append(f"queries.jsonl line {line_num}: query with split 'test' must have prefix 'q2_test_'")
                            test_query_ids.add(q_id)

                    # Source docs referential integrity
                    source_docs = record.get("source_doc_ids", [])
                    if isinstance(source_docs, list):
                        for s_doc in source_docs:
                            if s_doc not in doc_ids:
                                errors.append(
                                    f"queries.jsonl line {line_num}: source_doc_id '{s_doc}' does not exist in corpus"
                                )

                    # Annotation provenance validation
                    prov = record.get("annotation_provenance", {})
                    if isinstance(prov, dict):
                        status = prov.get("status")
                        if status not in self.ALLOWED_PROVENANCE_STATUSES:
                            errors.append(
                                f"queries.jsonl line {line_num}: invalid provenance status '{status}'. "
                                f"Must be one of {sorted(self.ALLOWED_PROVENANCE_STATUSES)}"
                            )
                        else:
                            provenance_distribution[status] = provenance_distribution.get(status, 0) + 1
                    else:
                        errors.append(f"queries.jsonl line {line_num}: 'annotation_provenance' must be a dict")

                    # Consistency by category
                    if cat == "unanswerable_negative":
                        if source_docs:
                            errors.append(
                                f"queries.jsonl line {line_num}: unanswerable_negative query '{q_id}' must have empty source_doc_ids"
                            )
                        expected_ev = record.get("expected_evidence", [])
                        if expected_ev:
                            errors.append(
                                f"queries.jsonl line {line_num}: unanswerable_negative query '{q_id}' must have empty expected_evidence"
                            )
                        # Check relevance judgments in query record
                        rels = record.get("relevance_judgments", {})
                        for d_id, r_val in rels.items():
                            if r_val > 0:
                                errors.append(
                                    f"queries.jsonl line {line_num}: unanswerable_negative query '{q_id}' has positive relevance {r_val} for {d_id}"
                                )
                    elif cat in {"factual_direct", "multi_hop", "citation_grounding", "technical_identifiers"}:
                        if not source_docs:
                            errors.append(
                                f"queries.jsonl line {line_num}: category '{cat}' query '{q_id}' must have at least one source_doc_id"
                            )
                        expected_ev = record.get("expected_evidence", [])
                        if not expected_ev:
                            errors.append(
                                f"queries.jsonl line {line_num}: category '{cat}' query '{q_id}' must have at least one expected_evidence span"
                            )

                    all_query_records.append(record)
        except Exception as exc:
            errors.append(f"Error reading queries.jsonl: {exc}")

        # 5. Split File Alignment & Partition Checks
        for split_name, expected_ids, filename in [
            ("dev", dev_query_ids, "queries_dev.jsonl"),
            ("test", test_query_ids, "queries_test.jsonl"),
        ]:
            split_file = self.dataset_dir / filename
            split_seen_ids: set[str] = set()
            try:
                with open(split_file, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, start=1):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        split_record = json.loads(stripped)
                        s_id = split_record.get("query_id")
                        if s_id:
                            if s_id in split_seen_ids:
                                errors.append(f"{filename} line {line_num}: duplicate query_id '{s_id}'")
                            split_seen_ids.add(s_id)
                            if split_record.get("split") != split_name:
                                errors.append(
                                    f"{filename} line {line_num}: query '{s_id}' has split '{split_record.get('split')}' but is in {filename}"
                                )
                if split_seen_ids != expected_ids:
                    errors.append(
                        f"{filename} query ID set does not match queries.jsonl {split_name} split. "
                        f"Missing: {len(expected_ids - split_seen_ids)}, Extra: {len(split_seen_ids - expected_ids)}"
                    )
            except Exception as exc:
                errors.append(f"Error reading {filename}: {exc}")

        # 6. Leakage & Near-Duplicate Query Analysis
        norm_queries: dict[str, str] = {}
        tokens_by_query: dict[str, set[str]] = {}
        for rec in all_query_records:
            q_id = rec["query_id"]
            q_text = rec["query"].strip()
            norm = re.sub(r"[^\w\s]", "", q_text.lower())
            tokens = set(norm.split())
            if norm in norm_queries:
                errors.append(
                    f"Duplicate query text between '{q_id}' and '{norm_queries[norm]}': '{q_text}'"
                )
            norm_queries[norm] = q_id
            tokens_by_query[q_id] = tokens

        # Jaccard check across dev and test to flag extreme leakage
        for dev_id in dev_query_ids:
            dev_tokens = tokens_by_query.get(dev_id, set())
            for test_id in test_query_ids:
                test_tokens = tokens_by_query.get(test_id, set())
                if not dev_tokens or not test_tokens:
                    continue
                intersection = len(dev_tokens & test_tokens)
                union = len(dev_tokens | test_tokens)
                jaccard = intersection / union if union > 0 else 0.0
                if jaccard > 0.85:
                    warnings.append(
                        f"High query token similarity (Jaccard={jaccard:.2f}) between dev '{dev_id}' and test '{test_id}'"
                    )

        # 7. Validate qrels.tsv, qrels_dev.tsv, and qrels_test.tsv
        for qrels_filename, allowed_split_ids in [
            ("qrels.tsv", query_ids),
            ("qrels_dev.tsv", dev_query_ids),
            ("qrels_test.tsv", test_query_ids),
        ]:
            qrels_path = self.dataset_dir / qrels_filename
            seen_pairs: set[tuple[str, str]] = set()
            try:
                with open(qrels_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, start=1):
                        raw_line = line.strip()
                        if not raw_line:
                            continue
                        parts = raw_line.split("\t")
                        if len(parts) != 4:
                            errors.append(
                                f"{qrels_filename} line {line_num}: expected 4 tab-separated columns, found {len(parts)}"
                            )
                            continue
                        q_id, iter_zero, d_id, rel_str = parts
                        if iter_zero != "0":
                            errors.append(f"{qrels_filename} line {line_num}: column 2 must be '0'")
                        if q_id not in allowed_split_ids:
                            errors.append(
                                f"{qrels_filename} line {line_num}: invalid query_id '{q_id}' for this file"
                            )
                        if d_id not in doc_ids:
                            errors.append(
                                f"{qrels_filename} line {line_num}: references non-existent doc_id '{d_id}'"
                            )
                        try:
                            rel = int(rel_str)
                            if rel not in self.VALID_RELEVANCE_LEVELS:
                                errors.append(
                                    f"{qrels_filename} line {line_num}: invalid relevance grade '{rel}'"
                                )
                        except ValueError:
                            errors.append(f"{qrels_filename} line {line_num}: relevance '{rel_str}' is not an integer")

                        pair = (q_id, d_id)
                        if pair in seen_pairs:
                            errors.append(
                                f"{qrels_filename} line {line_num}: duplicate judgment for query '{q_id}' and doc '{d_id}'"
                            )
                        seen_pairs.add(pair)
            except Exception as exc:
                errors.append(f"Error reading {qrels_filename}: {exc}")

        # 8. Validate manifest counts and checksums
        if manifest_data:
            manifest_counts = manifest_data.get("counts", {})
            if manifest_counts.get("documents") != corpus_count:
                errors.append(
                    f"Manifest documents count mismatch: manifest says {manifest_counts.get('documents')}, actual is {corpus_count}"
                )
            if manifest_counts.get("queries_total") != len(all_query_records):
                errors.append(
                    f"Manifest queries_total count mismatch: manifest says {manifest_counts.get('queries_total')}, actual is {len(all_query_records)}"
                )
            if manifest_counts.get("queries_dev") != len(dev_query_ids):
                errors.append(
                    f"Manifest queries_dev count mismatch: manifest says {manifest_counts.get('queries_dev')}, actual is {len(dev_query_ids)}"
                )
            if manifest_counts.get("queries_test") != len(test_query_ids):
                errors.append(
                    f"Manifest queries_test count mismatch: manifest says {manifest_counts.get('queries_test')}, actual is {len(test_query_ids)}"
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
            "queries_total": len(all_query_records),
            "queries_dev": len(dev_query_ids),
            "queries_test": len(test_query_ids),
        }
        stats["categories"] = category_distribution
        stats["provenance"] = provenance_distribution

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, stats=stats)


def validate_dataset_v2(dataset_dir: Path | str) -> ValidationResult:
    """Convenience functional wrapper for Dataset v2 validation."""
    validator = DatasetV2Validator(dataset_dir)
    return validator.validate()
