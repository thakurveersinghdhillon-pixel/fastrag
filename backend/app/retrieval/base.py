"""FastRAG Shared Retrieval Interfaces and Result Schemas.

Defines the common abstraction, result representation, and metadata structures
for both dense semantic retrieval and BM25 lexical retrieval.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from app.ingestion.models import Chunk


def validate_top_k(top_k: int) -> None:
    """Validate that top_k is a strictly positive integer."""
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}")
    if top_k <= 0:
        raise ValueError(f"top_k must be strictly positive (>= 1), got {top_k}")


def compute_corpus_fingerprint(chunks: Sequence[Chunk]) -> str:
    """Compute deterministic SHA-256 fingerprint for a sequence of chunks.

    Sorts chunks by chunk_id to ensure order invariance of input collection.
    """
    hasher = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda c: c.chunk_id):
        entry = f"{chunk.chunk_id}:{chunk.document_id}:{chunk.start_offset}:{chunk.end_offset}:{chunk.text}\n"
        hasher.update(entry.encode("utf-8"))
    return hasher.hexdigest()


@dataclass
class RetrievalResult:
    """Represents a single ranked candidate retrieved for a query.

    All scores are non-calibrated raw system scores (e.g. cosine similarity in [-1, 1],
    or BM25 score in [0, inf)). Scores from different retrievers must NOT be directly compared.
    """

    chunk_id: str
    document_id: str
    rank: int  # 1-indexed rank
    score: float
    score_type: str  # e.g., 'cosine_similarity', 'bm25'
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string")
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError(f"rank must be a positive 1-based integer, got {self.rank}")
        if not isinstance(self.score, (int, float)):
            raise TypeError(f"score must be numeric, got {type(self.score).__name__}")
        self.score = float(self.score)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalResult:
        """Construct RetrievalResult from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            rank=data["rank"],
            score=data["score"],
            score_type=data["score_type"],
            text=data["text"],
            source=data.get("source", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class IndexInfo:
    """Metadata describing an indexing artifact, its configuration, and corpus provenance."""

    retriever_type: str
    corpus_fingerprint: str
    chunk_count: int
    document_count: int
    schema_version: str = "1.0.0"
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize IndexInfo to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexInfo:
        """Construct IndexInfo from dictionary."""
        return cls(
            retriever_type=data["retriever_type"],
            corpus_fingerprint=data["corpus_fingerprint"],
            chunk_count=data["chunk_count"],
            document_count=data.get("document_count", 0),
            schema_version=data.get("schema_version", "1.0.0"),
            created_at_utc=data.get("created_at_utc", datetime.now(timezone.utc).isoformat()),
            hyperparameters=data.get("hyperparameters", {}),
        )


class BaseRetriever(ABC):
    """Abstract base class for independent retrieval engines."""

    @abstractmethod
    def build_index(self, chunks: Sequence[Chunk]) -> IndexInfo:
        """Construct the search index from an ordered sequence of Chunk records."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Execute top-K ranked search for a user query string."""
        pass

    @abstractmethod
    def save_index(self, directory_path: Path | str) -> None:
        """Persist index structures and metadata atomically to local storage."""
        pass

    @abstractmethod
    def load_index(self, directory_path: Path | str) -> IndexInfo:
        """Load and validate index structures and metadata from local storage."""
        pass

    @abstractmethod
    def get_index_info(self) -> IndexInfo | None:
        """Return metadata for the currently loaded/built index, or None if empty."""
        pass

    @property
    @abstractmethod
    def is_indexed(self) -> bool:
        """Return True if index is built/loaded and ready to serve queries."""
        pass
