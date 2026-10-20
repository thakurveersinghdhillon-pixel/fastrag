"""FastRAG BM25 Lexical Retrieval Engine.

Implements standalone BM25 Okapi lexical retrieval with domain-aware tokenization
preserving technical identifiers, acronyms, code terms, and hyperparameters.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from app.core.logging_config import get_logger
from app.ingestion.models import Chunk
from app.retrieval.base import (
    BaseRetriever,
    IndexInfo,
    RetrievalResult,
    compute_corpus_fingerprint,
    validate_top_k,
)

logger = get_logger("fastrag.retrieval.bm25")

# Tokenizer preserving technical phrases, dotted names, hyphenated terms, and identifiers
_TOKEN_PATTERN = re.compile(
    r"[a-z0-9_]+(?:\.[a-z0-9_]+)+|"  # Dotted identifiers (e.g. asyncio.gather)
    r"[a-z0-9_]+-[a-z0-9_]+|"        # Hyphenated terms (e.g. bi-encoder)
    r"[a-z0-9_]+|"                    # Alphanumeric identifiers (e.g. efconstruction, doc_001)
    r"[^\s\w]"                       # Important isolated symbols
)


def default_bm25_tokenizer(text: str) -> list[str]:
    """Tokenize text into lowercase lexical terms while preserving technical identifiers."""
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever(BaseRetriever):
    """Standalone lexical search engine implementing BM25 Okapi scoring."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        if k1 < 0:
            raise ValueError(f"k1 parameter must be non-negative, got {k1}")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"b parameter must be between 0.0 and 1.0, got {b}")

        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = tokenizer or default_bm25_tokenizer

        # Index state
        self._chunks: list[dict[str, Any]] = []
        self._chunk_ids: list[str] = []
        self._corpus_fingerprint: str | None = None
        self._index_info: IndexInfo | None = None

        # BM25 internal statistics
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._doc_freqs: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._term_freqs: list[dict[str, int]] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._is_indexed: bool = False

    @property
    def is_indexed(self) -> bool:
        return self._is_indexed

    def get_index_info(self) -> IndexInfo | None:
        return self._index_info

    def build_index(self, chunks: Sequence[Chunk]) -> IndexInfo:
        """Construct inverted frequencies and compute IDF tables from chunks."""
        start_time = time.perf_counter()

        if not chunks:
            raise ValueError("Cannot build BM25 index from an empty chunk sequence.")

        self._chunks = [c.to_dict() for c in chunks]
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._corpus_fingerprint = compute_corpus_fingerprint(chunks)

        n_docs = len(chunks)
        self._doc_lens = []
        self._term_freqs = []
        self._doc_freqs = {}

        # Tokenize and compute term frequencies per document
        for chunk in chunks:
            tokens = self.tokenizer(chunk.text)
            doc_len = len(tokens)
            self._doc_lens.append(doc_len)

            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._term_freqs.append(tf)

            for t in tf.keys():
                self._doc_freqs[t] = self._doc_freqs.get(t, 0) + 1

        self._avgdl = sum(self._doc_lens) / n_docs if n_docs > 0 else 0.0

        # Compute Robertson-Spärck Jones IDF with floor to avoid negative weights
        self._idf = {}
        for term, df in self._doc_freqs.items():
            # Standard Okapi IDF formulation: log((N - df + 0.5) / (df + 0.5) + 1.0)
            idf_val = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            self._idf[term] = max(idf_val, 0.0)

        elapsed = time.perf_counter() - start_time
        unique_docs = len({c.document_id for c in chunks})

        self._index_info = IndexInfo(
            retriever_type="bm25",
            corpus_fingerprint=self._corpus_fingerprint,
            chunk_count=n_docs,
            document_count=unique_docs,
            hyperparameters={
                "k1": self.k1,
                "b": self.b,
                "avgdl": self._avgdl,
                "vocab_size": len(self._doc_freqs),
                "build_duration_seconds": elapsed,
            },
        )
        self._is_indexed = True
        self._rebuild_postings()

        logger.info(
            f"Built BM25 index for {n_docs} chunks across {unique_docs} documents in {elapsed*1000:.2f}ms",
            extra={"operation": "build_bm25_index", "latency_ms": elapsed * 1000},
        )
        return self._index_info

    def _rebuild_postings(self) -> None:
        """Construct inverted index postings mapping term -> list of (doc_idx, term_frequency)."""
        self._postings = {}
        for doc_idx, tf_dict in enumerate(self._term_freqs):
            for term, count in tf_dict.items():
                if term not in self._postings:
                    self._postings[term] = []
                self._postings[term].append((doc_idx, count))

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Execute top-K BM25 scoring for the provided query string using inverted index.

        Ties are broken deterministically by chunk_id ascending.
        """
        validate_top_k(top_k)

        if not self._is_indexed or not self._chunks:
            return []

        if not query or not query.strip():
            return []

        tokens = self.tokenizer(query)
        if not tokens:
            return []

        # Filter query tokens to those present in vocabulary
        query_terms = [t for t in tokens if t in self._idf]
        if not query_terms:
            return []

        if not self._postings and self._term_freqs:
            self._rebuild_postings()

        # Accumulate BM25 scores over matching inverted postings
        doc_scores: dict[int, float] = {}
        for term in query_terms:
            idf = self._idf[term]
            postings = self._postings.get(term, [])
            for doc_idx, freq in postings:
                doc_len = self._doc_lens[doc_idx]
                len_norm = 1.0 - self.b + self.b * (doc_len / self._avgdl) if self._avgdl > 0 else 1.0
                tf_component = (freq * (self.k1 + 1.0)) / (freq + self.k1 * len_norm)
                doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + (idf * tf_component)

        scored_candidates = [
            (score, self._chunk_ids[doc_idx], doc_idx)
            for doc_idx, score in doc_scores.items()
            if score > 0.0
        ]

        # Deterministic ranking: score descending, then chunk_id ascending for ties
        scored_candidates.sort(key=lambda item: (-item[0], item[1]))

        results: list[RetrievalResult] = []
        for rank, (score, chunk_id, doc_idx) in enumerate(scored_candidates[:top_k], start=1):
            chunk_data = self._chunks[doc_idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=chunk_data["document_id"],
                    rank=rank,
                    score=round(score, 6),
                    score_type="bm25",
                    text=chunk_data["text"],
                    source=chunk_data.get("source", ""),
                    metadata=chunk_data.get("metadata", {}),
                )
            )

        return results

    def save_index(self, directory_path: Path | str) -> None:
        """Persist BM25 index tables and chunk metadata safely to JSON files."""
        if not self._is_indexed or self._index_info is None:
            raise ValueError("Cannot save uninitialized or empty BM25 index.")

        target_dir = Path(directory_path)
        target_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "index_info": self._index_info.to_dict(),
            "k1": self.k1,
            "b": self.b,
            "avgdl": self._avgdl,
            "doc_lens": self._doc_lens,
            "doc_freqs": self._doc_freqs,
            "idf": self._idf,
            "term_freqs": self._term_freqs,
            "chunks": self._chunks,
            "chunk_ids": self._chunk_ids,
        }

        temp_file = target_dir / "bm25_index.json.tmp"
        final_file = target_dir / "bm25_index.json"

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        os.replace(temp_file, final_file)
        logger.info(f"Saved BM25 index to {final_file}")

    def load_index(self, directory_path: Path | str) -> IndexInfo:
        """Load and validate BM25 index structures from local directory."""
        target_dir = Path(directory_path)
        index_file = target_dir / "bm25_index.json"

        if not index_file.is_file():
            raise FileNotFoundError(f"BM25 index file not found at: {index_file}")

        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._index_info = IndexInfo.from_dict(data["index_info"])
        if self._index_info.retriever_type != "bm25":
            raise ValueError(f"Invalid retriever type in index: {self._index_info.retriever_type}")

        self.k1 = data["k1"]
        self.b = data["b"]
        self._avgdl = data["avgdl"]
        self._doc_lens = data["doc_lens"]
        self._doc_freqs = data["doc_freqs"]
        self._idf = data["idf"]
        self._term_freqs = data["term_freqs"]
        self._chunks = data["chunks"]
        self._chunk_ids = data["chunk_ids"]
        self._corpus_fingerprint = self._index_info.corpus_fingerprint
        self._is_indexed = True
        self._rebuild_postings()

        logger.info(f"Loaded BM25 index ({self._index_info.chunk_count} chunks) from {index_file}")
        return self._index_info
