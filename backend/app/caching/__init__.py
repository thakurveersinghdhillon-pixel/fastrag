"""FastRAG In-Memory Caching Package (Phase 8.1).

Exports:
- BaseLRUCache: Thread-safe bounded LRU cache with monotonic TTL expiration.
- CacheStats: Cache telemetry metrics snapshot.
- QueryEmbeddingCache: Tier 2 query vector cache for dense semantic retrieval.
- QueryResponseCache: Tier 1 full-response cache with corpus fingerprint and tenant isolation.
"""

from app.caching.base import BaseLRUCache, CacheStats
from app.caching.embedding_cache import QueryEmbeddingCache
from app.caching.query_cache import QueryResponseCache

__all__ = [
    "BaseLRUCache",
    "CacheStats",
    "QueryEmbeddingCache",
    "QueryResponseCache",
]
