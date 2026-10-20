"""FastRAG Query Embedding Vector Cache (Tier 2).

Caches normalized dense query embedding vectors in-memory to eliminate redundant
Transformer encoder forward passes for recurring query strings.

Memory Accounting:
    A standard 384-dimensional dense vector stored in float32 consumes:
        384 dimensions * 4 bytes = 1,536 bytes (~1.5 KB).
    - Default capacity: 10,000 entries consumes ~15.36 MB of RAM.
    - Extended capacity: 50,000 entries consumes ~76.80 MB of RAM.
    This bounded memory consumption easily fits within standard container/process limits.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable
import numpy as np

from app.caching.base import BaseLRUCache, CacheStats
from app.core.logging_config import get_logger

logger = get_logger("fastrag.caching.embedding")


class QueryEmbeddingCache:
    """Thread-safe bounded in-memory cache for query embedding vectors.

    Features:
    - Key Isolation: Keys are strictly bound to (encoder_model_name, normalized_query).
    - Query-Only Boundary: Exclusively intended for search queries; document chunks
      must never be cached under query-cache keys.
    - Immutability Protection: Cached NumPy arrays are marked read-only and deep-copied
      upon retrieval, preventing callers from mutating internal state.
    - Feature Switch: Can be disabled via configuration without modifying callers.
    """

    def __init__(
        self,
        max_size: int = 10000,
        enabled: bool = True,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.enabled = enabled
        self._max_size = max_size
        self._cache: BaseLRUCache[np.ndarray] = BaseLRUCache(
            max_size=max_size,
            ttl_seconds=None,  # Query embeddings are deterministic functions of weights; no TTL needed
            clock=clock,
        )

    @property
    def max_size(self) -> int:
        return self._max_size

    def __len__(self) -> int:
        return len(self._cache)

    @staticmethod
    def build_cache_key(query: str, model_name: str) -> str:
        """Construct deterministic SHA-256 key from model identifier and normalized query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string must be a non-empty string")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Model name must be a non-empty string")

        norm_query = query.strip().lower()
        norm_model = model_name.strip().lower()
        payload = f"query_embed::{norm_model}::{norm_query}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, query: str, model_name: str) -> np.ndarray | None:
        """Retrieve cached query embedding vector if present and enabled.

        Returns a writeable copy to ensure internal cache state cannot be corrupted.
        """
        if not self.enabled:
            return None

        if not query or not query.strip() or not model_name or not model_name.strip():
            return None

        key = self.build_cache_key(query, model_name)
        cached_vec = self._cache.get(key)
        if cached_vec is None:
            return None

        # Return independent writeable copy to caller
        return cached_vec.copy()

    def set(self, query: str, model_name: str, vector: np.ndarray) -> None:
        """Store query embedding vector into cache.

        Validates vector finiteness, converts to float32, and stores an immutable copy.
        """
        if not self.enabled:
            return

        if not query or not query.strip() or not model_name or not model_name.strip():
            return

        if not isinstance(vector, np.ndarray):
            raise TypeError(f"vector must be a numpy.ndarray, got {type(vector).__name__}")

        # Ensure 1D flat vector representation
        flat_vec = vector.squeeze()
        if flat_vec.ndim != 1:
            raise ValueError(f"Vector must be 1-dimensional after squeeze, got shape {vector.shape}")

        if not np.isfinite(flat_vec).all():
            logger.warning("Refusing to cache embedding vector containing NaN or Inf values")
            return

        # Prepare immutable cached copy
        cached_copy = np.array(flat_vec, dtype=np.float32, copy=True)
        cached_copy.flags.writeable = False

        key = self.build_cache_key(query, model_name)
        self._cache.set(key, cached_copy)

    def delete(self, query: str, model_name: str) -> bool:
        """Remove cached query vector."""
        if not self.enabled or not query or not model_name:
            return False
        key = self.build_cache_key(query, model_name)
        return self._cache.delete(key)

    def clear(self) -> None:
        """Purge all cached embeddings."""
        self._cache.clear()

    def stats(self) -> CacheStats:
        """Return cache performance statistics."""
        return self._cache.stats()

    def reset_stats(self) -> None:
        """Reset cache performance statistics."""
        self._cache.reset_stats()
