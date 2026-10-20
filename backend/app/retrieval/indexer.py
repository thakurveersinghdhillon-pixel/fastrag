"""FastRAG Index Management Utilities.

Coordinates index building and safe persistence for Dense and BM25 retrievers
from processed JSONL chunk corpora.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from app.core.logging_config import get_logger
from app.ingestion.models import Chunk
from app.retrieval.base import IndexInfo
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever, EmbeddingEncoder

logger = get_logger("fastrag.retrieval.indexer")


def load_chunks_for_indexing(chunks_file: Path | str) -> list[Chunk]:
    """Load and parse processed chunk records from JSONL."""
    path = Path(chunks_file)
    if not path.is_file():
        raise FileNotFoundError(f"Chunks file not found at: {path}")

    chunks: list[Chunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                chunks.append(Chunk.from_dict(record))
            except Exception as exc:
                raise ValueError(f"{path} line {line_num}: failed to parse Chunk ({exc})") from exc

    if not chunks:
        raise ValueError(f"No chunks found in {path}")

    return chunks


def build_and_save_bm25_index(
    chunks: Sequence[Chunk],
    output_dir: Path | str,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[BM25Retriever, IndexInfo]:
    """Build and persist BM25 index to target directory."""
    retriever = BM25Retriever(k1=k1, b=b)
    info = retriever.build_index(chunks)
    retriever.save_index(output_dir)
    return retriever, info


def build_and_save_dense_index(
    chunks: Sequence[Chunk],
    output_dir: Path | str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize: bool = True,
    encoder: EmbeddingEncoder | None = None,
) -> tuple[DenseRetriever, IndexInfo]:
    """Build and persist Dense index to target directory."""
    retriever = DenseRetriever(
        encoder=encoder,
        model_name=model_name,
        batch_size=batch_size,
        normalize=normalize,
    )
    info = retriever.build_index(chunks)
    retriever.save_index(output_dir)
    return retriever, info
