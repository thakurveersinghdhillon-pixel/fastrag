"""FastRAG Ingestion and Chunk Validation.

Provides rigorous pre-ingestion and post-chunking verification to prevent
corrupt documents, broken offsets, duplicate identifiers, or malformed payloads
from entering retrieval index preparation.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from app.ingestion.models import Chunk, Document


class IngestionValidationError(Exception):
    """Raised when document or chunk records violate structural validation rules."""
    pass


def validate_document(document: Document) -> list[str]:
    """Validate a single Document record. Returns a list of error strings (empty if valid)."""
    errors: list[str] = []

    if not isinstance(document, Document):
        return [f"Expected Document instance, got {type(document).__name__}"]

    if not document.document_id or not document.document_id.strip():
        errors.append("document_id must be a non-empty string")

    if not isinstance(document.text, str):
        errors.append(f"document text must be a string, got {type(document.text).__name__}")

    if not isinstance(document.metadata, dict):
        errors.append("document metadata must be a dictionary")
    else:
        try:
            json.dumps(document.metadata)
        except (TypeError, OverflowError) as exc:
            errors.append(f"document metadata is not JSON-serializable: {exc}")

    return errors


def validate_documents_batch(documents: Iterable[Document]) -> tuple[list[str], set[str]]:
    """Validate a batch of documents for schema correctness and ID uniqueness.

    Returns:
        tuple of (errors: list[str], duplicate_ids: set[str])
    """
    errors: list[str] = []
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    for idx, doc in enumerate(documents, start=1):
        doc_errors = validate_document(doc)
        for err in doc_errors:
            errors.append(f"Document #{idx} ('{getattr(doc, 'document_id', 'unknown')}'): {err}")

        doc_id = getattr(doc, "document_id", None)
        if doc_id:
            if doc_id in seen_ids:
                duplicate_ids.add(doc_id)
                errors.append(f"Document #{idx}: duplicate document_id '{doc_id}' detected")
            seen_ids.add(doc_id)

    return errors, duplicate_ids


def validate_chunk(chunk: Chunk, original_text: str | None = None) -> list[str]:
    """Validate a Chunk record for schema and character offset alignment.

    If original_text is provided, verifies that:
        original_text[chunk.start_offset:chunk.end_offset] == chunk.text
    """
    errors: list[str] = []

    if not isinstance(chunk, Chunk):
        return [f"Expected Chunk instance, got {type(chunk).__name__}"]

    if not chunk.chunk_id or not chunk.chunk_id.strip():
        errors.append("chunk_id must be a non-empty string")

    if not chunk.document_id or not chunk.document_id.strip():
        errors.append("document_id must be a non-empty string")

    if not isinstance(chunk.chunk_index, int) or chunk.chunk_index < 0:
        errors.append(f"chunk_index must be a non-negative integer, got {chunk.chunk_index}")

    if not chunk.text or not chunk.text.strip():
        errors.append("chunk text must not be empty")

    if chunk.start_offset < 0:
        errors.append(f"start_offset must be non-negative, got {chunk.start_offset}")

    if chunk.end_offset <= chunk.start_offset:
        errors.append(
            f"end_offset ({chunk.end_offset}) must be strictly greater than start_offset ({chunk.start_offset})"
        )

    if original_text is not None:
        if chunk.end_offset > len(original_text):
            errors.append(
                f"end_offset ({chunk.end_offset}) exceeds document text length ({len(original_text)})"
            )
        else:
            slice_text = original_text[chunk.start_offset:chunk.end_offset]
            if slice_text != chunk.text:
                errors.append(
                    f"Offset mismatch for chunk '{chunk.chunk_id}': "
                    f"text[{chunk.start_offset}:{chunk.end_offset}] does not equal chunk.text"
                )

    return errors


def validate_chunks_batch(chunks: Iterable[Chunk]) -> list[str]:
    """Validate a batch of chunks for unique chunk_ids and sequential indexing."""
    errors: list[str] = []
    seen_chunk_ids: set[str] = set()
    doc_chunk_indices: dict[str, list[int]] = {}

    for idx, chunk in enumerate(chunks, start=1):
        chunk_errors = validate_chunk(chunk)
        for err in chunk_errors:
            errors.append(f"Chunk #{idx} ('{getattr(chunk, 'chunk_id', 'unknown')}'): {err}")

        c_id = getattr(chunk, "chunk_id", None)
        if c_id:
            if c_id in seen_chunk_ids:
                errors.append(f"Chunk #{idx}: duplicate chunk_id '{c_id}' detected")
            seen_chunk_ids.add(c_id)

        d_id = getattr(chunk, "document_id", None)
        c_idx = getattr(chunk, "chunk_index", None)
        if d_id and c_idx is not None:
            if d_id not in doc_chunk_indices:
                doc_chunk_indices[d_id] = []
            doc_chunk_indices[d_id].append(c_idx)

    # Check for sequential chunk indices per document (0, 1, 2, ...)
    for doc_id, indices in doc_chunk_indices.items():
        expected_indices = list(range(len(indices)))
        if indices != expected_indices:
            errors.append(
                f"Non-sequential chunk indices for document '{doc_id}': "
                f"expected {expected_indices}, found {indices}"
            )

    return errors
