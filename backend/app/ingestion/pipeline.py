"""FastRAG Document Ingestion Pipeline.

Orchestrates the complete document ingestion lifecycle:
Loading -> Validation -> Text Normalization -> Chunking -> Chunk Validation -> Atomic Export
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable, Literal

from app.ingestion.chunking import Chunker
from app.ingestion.loaders import load_documents
from app.ingestion.models import Chunk, Document, IngestionSummary
from app.ingestion.normalization import normalize_text
from app.ingestion.validation import (
    IngestionValidationError,
    validate_chunk,
    validate_document,
)


ErrorPolicy = Literal["collect", "fail_fast"]


class IngestionPipeline:
    """Ingestion pipeline orchestrating loading, normalization, chunking, and export."""

    def __init__(
        self,
        chunk_size: int = Chunker.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = Chunker.DEFAULT_CHUNK_OVERLAP,
        preserve_paragraphs: bool = True,
        on_error: ErrorPolicy = "collect",
    ) -> None:
        self.chunker = Chunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_paragraphs=preserve_paragraphs,
        )
        self.on_error = on_error

    def process_documents(
        self,
        documents: Iterable[Document],
        output_path: Path | str | None = None,
    ) -> tuple[list[Chunk], IngestionSummary]:
        """Execute the ingestion pipeline on an iterable of Document objects.

        Args:
            documents: Sequence or generator of Document instances.
            output_path: Optional path to export chunks as JSONL.

        Returns:
            tuple of (all_chunks: list[Chunk], summary: IngestionSummary)
        """
        start_time = time.perf_counter()

        all_chunks: list[Chunk] = []
        errors: list[str] = []
        total_docs = 0
        processed_docs = 0
        rejected_docs = 0
        empty_docs = 0
        seen_document_ids: set[str] = set()

        for idx, doc in enumerate(documents, start=1):
            total_docs += 1

            # 1. Validate Document schema
            doc_errors = validate_document(doc)
            if doc_errors:
                msg = f"Document #{idx} ('{getattr(doc, 'document_id', 'unknown')}'): {'; '.join(doc_errors)}"
                if self.on_error == "fail_fast":
                    raise IngestionValidationError(msg)
                errors.append(msg)
                rejected_docs += 1
                continue

            # Check for duplicate document ID across the batch
            if doc.document_id in seen_document_ids:
                msg = f"Document #{idx}: duplicate document_id '{doc.document_id}'"
                if self.on_error == "fail_fast":
                    raise IngestionValidationError(msg)
                errors.append(msg)
                rejected_docs += 1
                continue
            seen_document_ids.add(doc.document_id)

            # Check for empty document
            if doc.is_empty:
                empty_docs += 1
                processed_docs += 1
                continue

            # 2. Normalize text conservatively (preserving original raw_text)
            original_raw = doc.raw_text if doc.raw_text is not None else doc.text
            normalized = normalize_text(
                doc.text,
                preserve_paragraphs=self.chunker.preserve_paragraphs,
            )

            # Update document with normalized text for chunking
            normalized_doc = Document(
                document_id=doc.document_id,
                text=normalized,
                title=doc.title,
                source=doc.source,
                metadata=doc.metadata,
                raw_text=original_raw,
            )

            # 3. Chunk Document
            doc_chunks = self.chunker.chunk_document(normalized_doc)

            # 4. Validate output chunks against normalized document text
            chunk_batch_errors: list[str] = []
            for c in doc_chunks:
                c_errs = validate_chunk(c, original_text=normalized)
                chunk_batch_errors.extend(c_errs)

            if chunk_batch_errors:
                msg = f"Chunk validation failed for doc '{doc.document_id}': {'; '.join(chunk_batch_errors)}"
                if self.on_error == "fail_fast":
                    raise IngestionValidationError(msg)
                errors.append(msg)
                rejected_docs += 1
                continue

            all_chunks.extend(doc_chunks)
            processed_docs += 1

        # 5. Export to JSONL if requested
        output_file_str: str | None = None
        if output_path is not None:
            output_file = Path(output_path)
            self._export_chunks_atomic(all_chunks, output_file)
            output_file_str = str(output_file)

        elapsed = time.perf_counter() - start_time
        avg_chunks = (
            len(all_chunks) / processed_docs if processed_docs > 0 else 0.0
        )

        summary = IngestionSummary(
            total_documents_input=total_docs,
            documents_processed=processed_docs,
            documents_rejected=rejected_docs,
            empty_documents=empty_docs,
            total_chunks_output=len(all_chunks),
            average_chunks_per_document=avg_chunks,
            chunking_config={
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.chunk_overlap,
                "preserve_paragraphs": self.chunker.preserve_paragraphs,
            },
            output_file=output_file_str,
            duration_seconds=elapsed,
            errors=errors,
        )

        return all_chunks, summary

    def process_file(
        self,
        input_path: Path | str,
        output_path: Path | str | None = None,
    ) -> tuple[list[Chunk], IngestionSummary]:
        """Load documents from input file or directory and run the ingestion pipeline."""
        docs = load_documents(input_path)
        return self.process_documents(docs, output_path=output_path)

    def _export_chunks_atomic(self, chunks: list[Chunk], destination: Path) -> None:
        """Write chunk records atomically to a temporary file before renaming.

        Guarantees that a failed or aborted write does not corrupt or leave partial output.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_file = destination.with_suffix(f"{destination.suffix}.tmp_{os.getpid()}")

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    line = json.dumps(chunk.to_dict(), ensure_ascii=False)
                    f.write(line + "\n")
            # Atomic file swap
            os.replace(temp_file, destination)
        except Exception:
            if temp_file.is_file():
                temp_file.unlink(missing_ok=True)
            raise
