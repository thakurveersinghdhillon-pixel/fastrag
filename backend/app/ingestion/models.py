"""FastRAG Document Ingestion Data Models.

Defines validated, typed data structures for raw documents, normalized chunks,
and ingestion pipeline execution summaries.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Document:
    """Represents a validated input document prior to chunking."""

    document_id: str
    text: str
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_text: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError(f"document_id must be a non-empty string, got {self.document_id!r}")
        self.document_id = self.document_id.strip()

        if not isinstance(self.text, str):
            raise TypeError(f"Document text must be a string, got {type(self.text).__name__}")

        if not isinstance(self.title, str):
            self.title = str(self.title)

        if not isinstance(self.source, str):
            self.source = str(self.source)

        if not isinstance(self.metadata, dict):
            raise TypeError(f"Document metadata must be a dictionary, got {type(self.metadata).__name__}")

        # Ensure metadata is JSON-serializable
        try:
            json.dumps(self.metadata)
        except (TypeError, OverflowError) as exc:
            raise ValueError(f"Document metadata must be JSON-serializable: {exc}") from exc

    @property
    def is_empty(self) -> bool:
        """Return True if the document text is empty or contains only whitespace."""
        return len(self.text.strip()) == 0

    @property
    def character_count(self) -> int:
        """Return character length of document text."""
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        """Serialize Document to standard dictionary."""
        data = {
            "document_id": self.document_id,
            "title": self.title,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
        }
        if self.raw_text is not None:
            data["raw_text"] = self.raw_text
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Construct Document from dictionary, supporting both doc_id and document_id keys."""
        doc_id = data.get("document_id") or data.get("doc_id")
        if not doc_id:
            raise ValueError("Record missing required 'document_id' or 'doc_id' field")

        return cls(
            document_id=str(doc_id),
            text=data.get("text", ""),
            title=data.get("title", ""),
            source=data.get("source", ""),
            metadata=data.get("metadata") or {},
            raw_text=data.get("raw_text"),
        )


@dataclass
class Chunk:
    """Represents a discrete text passage produced by chunking a Document."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    source: str
    start_offset: int
    end_offset: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string")
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.chunk_index, int) or self.chunk_index < 0:
            raise ValueError(f"chunk_index must be a non-negative integer, got {self.chunk_index}")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Chunk text must be a non-empty string")
        if not isinstance(self.start_offset, int) or self.start_offset < 0:
            raise ValueError(f"start_offset must be non-negative, got {self.start_offset}")
        if not isinstance(self.end_offset, int) or self.end_offset <= self.start_offset:
            raise ValueError(
                f"end_offset ({self.end_offset}) must be strictly greater than start_offset ({self.start_offset})"
            )

    @property
    def character_count(self) -> int:
        """Return the character count of this chunk."""
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        """Serialize Chunk to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        """Construct Chunk from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            chunk_index=data["chunk_index"],
            text=data["text"],
            source=data.get("source", ""),
            start_offset=data["start_offset"],
            end_offset=data["end_offset"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class IngestionSummary:
    """Execution statistics and summary of an ingestion pipeline run."""

    total_documents_input: int
    documents_processed: int
    documents_rejected: int
    empty_documents: int
    total_chunks_output: int
    average_chunks_per_document: float
    chunking_config: dict[str, Any]
    output_file: str | None = None
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize summary to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> str:
        """Format human-readable summary table."""
        lines = [
            "============================================================",
            " FastRAG Document Ingestion Summary",
            f" Timestamp (UTC): {self.timestamp_utc}",
            "============================================================",
            f" Input Documents:        {self.total_documents_input}",
            f" Successfully Processed: {self.documents_processed}",
            f" Rejected Documents:     {self.documents_rejected}",
            f" Empty Documents:        {self.empty_documents}",
            f" Total Chunks Output:    {self.total_chunks_output}",
            f" Avg Chunks / Document:  {self.average_chunks_per_document:.2f}",
            f" Ingestion Duration:     {self.duration_seconds:.4f}s",
            "------------------------------------------------------------",
            f" Chunking Config:        size={self.chunking_config.get('chunk_size')}, "
            f"overlap={self.chunking_config.get('chunk_overlap')}, "
            f"preserve_paragraphs={self.chunking_config.get('preserve_paragraphs')}",
        ]
        if self.output_file:
            lines.append(f" Output Location:        {self.output_file}")
        if self.errors:
            lines.append("------------------------------------------------------------")
            lines.append(f" Errors Encountered ({len(self.errors)}):")
            for err in self.errors[:10]:
                lines.append(f"  - {err}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more errors.")
        lines.append("============================================================")
        return "\n".join(lines)
