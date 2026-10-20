"""FastRAG Dense Semantic Retrieval Engine.

Implements dense vector indexing and cosine-similarity search using NumPy matrix operations,
lazy model loading, batched inference, deterministic tie-breaking, and safe NPZ/JSON persistence.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from app.core.logging_config import get_logger
from app.ingestion.models import Chunk
from app.retrieval.base import (
    BaseRetriever,
    IndexInfo,
    RetrievalResult,
    compute_corpus_fingerprint,
    validate_top_k,
)
import numpy as np

logger = get_logger("fastrag.retrieval.dense")


class EmbeddingEncoder(ABC):
    """Abstract interface for dense embedding generation."""

    @abstractmethod
    def encode(self, texts: Sequence[str], batch_size: int = 32, normalize: bool = True) -> Any:
        """Encode a sequence of text strings into a 2D float32 array (N, D)."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier name of the underlying embedding model."""
        pass


class MockEmbeddingEncoder(EmbeddingEncoder):
    """Deterministic, zero-dependency embedding encoder for unit tests and offline fixtures.

    Generates reproducible float32 vectors derived from text SHA-256 digests.
    """

    def __init__(self, dimension: int = 64, model_name: str = "mock-embedding-v1") -> None:
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: Sequence[str], batch_size: int = 32, normalize: bool = True) -> Any:
        import numpy as np

        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        vectors = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            # Expand digest deterministically across dimensions
            seed_vals = [
                float((digest[j % len(digest)] * (j + 1)) % 100 - 50)
                for j in range(self._dimension)
            ]
            vec = np.array(seed_vals, dtype=np.float32)
            if normalize:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors[i] = vec

        return vectors


class SentenceTransformerEncoder(EmbeddingEncoder):
    """Dense encoder powered by sentence-transformers with lazy loading."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        import threading
        self._model_name = model_name
        if device is not None:
            self._device = device
        else:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        self._device_lock = threading.Lock()
        self._model: Any = None
        self._dimension: int | None = None

    def _ensure_model(self) -> Any:
        """Lazily load SentenceTransformer model on first usage."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed. Install it via "
                    "'pip install sentence-transformers' to use dense semantic retrieval."
                ) from exc

            start_t = time.perf_counter()
            logger.info(f"Loading embedding model '{self._model_name}'...")
            try:
                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                    local_files_only=True,
                )
            except Exception:
                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                )
            load_elapsed = time.perf_counter() - start_t
            logger.info(
                f"Loaded embedding model '{self._model_name}' in {load_elapsed:.2f}s",
                extra={"operation": "load_embedding_model", "latency_ms": load_elapsed * 1000},
            )
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_model()
        return self._dimension  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        return self._model_name

    def warmup(self) -> None:
        """Pre-warm model weights and execution engine to eliminate runtime cold-start."""
        _ = self._ensure_model()
        _ = self.encode(["warmup query"], batch_size=1, normalize=True)

    def encode(self, texts: Sequence[str], batch_size: int = 32, normalize: bool = True) -> Any:
        if not texts:
            dim = self.dimension
            return np.empty((0, dim), dtype=np.float32)

        model = self._ensure_model()

        def _do_encode() -> Any:
            try:
                import torch
                with torch.inference_mode():
                    return model.encode(
                        list(texts),
                        batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=normalize,
                        convert_to_numpy=True,
                    )
            except (ImportError, AttributeError):
                return model.encode(
                    list(texts),
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=normalize,
                    convert_to_numpy=True,
                )

        if getattr(self, "_device", "") == "mps":
            with self._device_lock:
                embeddings = _do_encode()
        else:
            embeddings = _do_encode()
        return embeddings.astype(np.float32)


class DenseRetriever(BaseRetriever):
    """Dense semantic retrieval engine executing exact cosine similarity in NumPy."""

    def warmup(self) -> None:
        """Pre-warm underlying encoder model."""
        if hasattr(self.encoder, "warmup"):
            self.encoder.warmup()

    def __init__(
        self,
        encoder: EmbeddingEncoder | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize: bool = True,
        embedding_cache: Any | None = None,
    ) -> None:
        self.encoder = encoder or SentenceTransformerEncoder(model_name=model_name)
        self.batch_size = batch_size
        self.normalize = normalize
        self.embedding_cache = embedding_cache

        # Index state
        self._vectors: Any = None  # np.ndarray shape (N, D)
        self._chunks: list[dict[str, Any]] = []
        self._chunk_ids: list[str] = []
        self._corpus_fingerprint: str | None = None
        self._index_info: IndexInfo | None = None
        self._is_indexed: bool = False

    @property
    def is_indexed(self) -> bool:
        return self._is_indexed

    def get_index_info(self) -> IndexInfo | None:
        return self._index_info

    def build_index(self, chunks: Sequence[Chunk]) -> IndexInfo:
        """Generate batched embeddings and construct the in-memory cosine index."""
        start_time = time.perf_counter()

        if not chunks:
            raise ValueError("Cannot build Dense index from an empty chunk sequence.")

        self._chunks = [c.to_dict() for c in chunks]
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._corpus_fingerprint = compute_corpus_fingerprint(chunks)

        texts = [c.text for c in chunks]
        embed_start = time.perf_counter()
        vectors = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            normalize=self.normalize,
        )
        embed_elapsed = time.perf_counter() - embed_start

        if not np.isfinite(vectors).all():
            raise ValueError("Computed embedding matrix contains non-finite values (NaN/Inf).")

        self._vectors = vectors
        total_elapsed = time.perf_counter() - start_time
        unique_docs = len({c.document_id for c in chunks})

        self._index_info = IndexInfo(
            retriever_type="dense",
            corpus_fingerprint=self._corpus_fingerprint,
            chunk_count=len(chunks),
            document_count=unique_docs,
            hyperparameters={
                "model_name": self.encoder.model_name,
                "dimension": int(vectors.shape[1]),
                "batch_size": self.batch_size,
                "normalized": self.normalize,
                "embedding_time_seconds": embed_elapsed,
                "total_build_seconds": total_elapsed,
            },
        )
        self._is_indexed = True

        logger.info(
            f"Built Dense index ({len(chunks)} chunks, dim={vectors.shape[1]}) in {total_elapsed:.2f}s "
            f"(embedding compute: {embed_elapsed:.2f}s)",
            extra={"operation": "build_dense_index", "latency_ms": total_elapsed * 1000},
        )
        return self._index_info

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Compute cosine similarity of query against all indexed chunk vectors.

        Ties are broken deterministically by chunk_id ascending.
        """
        validate_top_k(top_k)

        if not self._is_indexed or self._vectors is None or len(self._chunks) == 0:
            return []

        if not query or not query.strip():
            return []

        # Encode query string (with Tier 2 embedding cache if configured)
        q_vec = None
        if self.embedding_cache is not None and getattr(self.embedding_cache, "enabled", False):
            try:
                q_vec = self.embedding_cache.get(query=query, model_name=self.encoder.model_name)
            except Exception as exc:
                logger.warning(f"Embedding cache read failed safely: {exc}", exc_info=True)
                q_vec = None

        if q_vec is None:
            q_vec = self.encoder.encode([query], batch_size=1, normalize=self.normalize)[0]
            if not np.isfinite(q_vec).all():
                return []
            if self.embedding_cache is not None and getattr(self.embedding_cache, "enabled", False):
                try:
                    self.embedding_cache.set(query=query, model_name=self.encoder.model_name, vector=q_vec)
                except Exception as exc:
                    logger.warning(f"Embedding cache write failed safely: {exc}", exc_info=True)
        else:
            if not np.isfinite(q_vec).all():
                return []

        # When vectors are L2-normalized, cosine similarity is equivalent to dot product
        similarities = np.dot(self._vectors, q_vec)

        # Rank candidates deterministically: similarity descending, then chunk_id ascending
        scored_candidates = [
            (float(similarities[i]), self._chunk_ids[i], i)
            for i in range(len(self._chunk_ids))
        ]
        scored_candidates.sort(key=lambda item: (-item[0], item[1]))

        results: list[RetrievalResult] = []
        for rank, (score, chunk_id, idx) in enumerate(scored_candidates[:top_k], start=1):
            chunk_data = self._chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=chunk_data["document_id"],
                    rank=rank,
                    score=round(score, 6),
                    score_type="cosine_similarity",
                    text=chunk_data["text"],
                    source=chunk_data.get("source", ""),
                    metadata=chunk_data.get("metadata", {}),
                )
            )

        return results

    def save_index(self, directory_path: Path | str) -> None:
        """Persist dense index vectors to NPZ and metadata to JSON atomically."""
        import numpy as np

        if not self._is_indexed or self._vectors is None or self._index_info is None:
            raise ValueError("Cannot save uninitialized or empty dense index.")

        target_dir = Path(directory_path)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata and chunk mappings
        meta_payload = {
            "index_info": self._index_info.to_dict(),
            "model_name": self.encoder.model_name,
            "dimension": int(self._vectors.shape[1]),
            "normalized": self.normalize,
            "chunk_ids": self._chunk_ids,
            "chunks": self._chunks,
        }

        meta_temp = target_dir / "dense_meta_tmp.json"
        meta_file = target_dir / "dense_meta.json"
        with open(meta_temp, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False)
        os.replace(meta_temp, meta_file)

        # Save embeddings safely as NPZ array
        vectors_temp = target_dir / "dense_vectors_tmp.npz"
        vectors_file = target_dir / "dense_vectors.npz"
        np.savez_compressed(vectors_temp, vectors=self._vectors)
        os.replace(vectors_temp, vectors_file)

        logger.info(f"Saved dense index to {target_dir}")

    def load_index(self, directory_path: Path | str) -> None:
        """Load and validate dense index from NPZ and JSON files."""
        import numpy as np

        target_dir = Path(directory_path)
        meta_file = target_dir / "dense_meta.json"
        vectors_file = target_dir / "dense_vectors.npz"

        if not meta_file.is_file():
            raise FileNotFoundError(f"Dense index metadata not found: {meta_file}")
        if not vectors_file.is_file():
            raise FileNotFoundError(f"Dense index vectors not found: {vectors_file}")

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._index_info = IndexInfo.from_dict(data["index_info"])
        if self._index_info.retriever_type != "dense":
            raise ValueError(f"Invalid retriever type: {self._index_info.retriever_type}")

        npz_archive = np.load(vectors_file)
        self._vectors = npz_archive["vectors"]
        self._chunks = data["chunks"]
        self._chunk_ids = data["chunk_ids"]
        self._corpus_fingerprint = self._index_info.corpus_fingerprint
        self.normalize = data["normalized"]

        if len(self._vectors) != len(self._chunk_ids):
            raise ValueError(
                f"Mismatch between vector rows ({len(self._vectors)}) and chunk count ({len(self._chunk_ids)})"
            )

        self._is_indexed = True
        logger.info(
            f"Loaded dense index ({len(self._chunk_ids)} chunks, dim={self._vectors.shape[1]}) from {target_dir}"
        )
        return self._index_info
