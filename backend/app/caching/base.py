"""FastRAG Core In-Memory Caching Primitives.

Provides a thread-safe, bounded Least Recently Used (LRU) cache with per-entry Time-To-Live (TTL),
deterministic passive eviction, microsecond-accurate monotonic clock tracking, and telemetry counters.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CacheStats:
    """Thread-safe snapshot of cache telemetry counters and capacity."""

    hits: int
    misses: int
    expirations: int
    evictions: int
    size: int
    max_size: int
    ttl_seconds: float | None

    @property
    def total_lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_lookups
        return round(self.hits / total, 4) if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "expirations": self.expirations,
            "evictions": self.evictions,
            "size": self.size,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "total_lookups": self.total_lookups,
            "hit_rate": self.hit_rate,
        }


@dataclass
class _CacheEntry(Generic[T]):
    """Internal entry wrapper storing value and expiration timestamp."""

    value: T
    created_at: float
    expires_at: float | None


class BaseLRUCache(Generic[T]):
    """Thread-safe bounded in-memory LRU cache with monotonic TTL expiration.

    Design Principles:
    - Bounded Memory: Enforces strict capacity limit; evicts least recently accessed on overflow.
    - Passive Eviction: Expired entries are discarded upon access; no background daemon threads.
    - Concurrency Safe: Uses an RLock to guard all mutations and reads.
    - Monotonic Time: Employs time.monotonic to remain immune to system clock jumps.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(max_size, int) or isinstance(max_size, bool):
            raise TypeError(f"max_size must be an integer, got {type(max_size).__name__}")
        if max_size <= 0:
            raise ValueError(f"max_size must be strictly positive (>= 1), got {max_size}")

        if ttl_seconds is not None:
            if not isinstance(ttl_seconds, (int, float)):
                raise TypeError(f"ttl_seconds must be numeric, got {type(ttl_seconds).__name__}")
            if ttl_seconds <= 0:
                raise ValueError(f"ttl_seconds must be strictly positive (> 0), got {ttl_seconds}")

        self._max_size = max_size
        self._default_ttl = float(ttl_seconds) if ttl_seconds is not None else None
        self._clock = clock or time.monotonic

        self._lock = threading.RLock()
        self._entries: OrderedDict[str, _CacheEntry[T]] = OrderedDict()

        # Telemetry counters
        self._hits: int = 0
        self._misses: int = 0
        self._expirations: int = 0
        self._evictions: int = 0

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def default_ttl(self) -> float | None:
        return self._default_ttl

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _is_expired(self, entry: _CacheEntry[T], now: float) -> bool:
        return entry.expires_at is not None and now >= entry.expires_at

    def get(self, key: str, default: T | None = None) -> T | None:
        """Retrieve value for key, updating LRU order if valid.

        Returns default on cache miss or TTL expiration.
        """
        now = self._clock()
        with self._lock:
            if key not in self._entries:
                self._misses += 1
                return default

            entry = self._entries[key]
            if self._is_expired(entry, now):
                del self._entries[key]
                self._expirations += 1
                self._misses += 1
                return default

            # Refresh LRU access position
            self._entries.move_to_end(key, last=True)
            self._hits += 1
            return entry.value

    def set(
        self,
        key: str,
        value: T,
        ttl_seconds: float | None = None,
    ) -> None:
        """Store value for key with optional per-item TTL override."""
        now = self._clock()
        effective_ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expires_at = (now + effective_ttl) if effective_ttl is not None else None

        with self._lock:
            if key in self._entries:
                # Update existing entry and touch access order
                self._entries[key] = _CacheEntry(
                    value=value,
                    created_at=now,
                    expires_at=expires_at,
                )
                self._entries.move_to_end(key, last=True)
                return

            # Check capacity before inserting new key
            if len(self._entries) >= self._max_size:
                # Evict least recently used entry (first in OrderedDict)
                self._entries.popitem(last=False)
                self._evictions += 1

            self._entries[key] = _CacheEntry(
                value=value,
                created_at=now,
                expires_at=expires_at,
            )

    def delete(self, key: str) -> bool:
        """Remove key from cache if present. Returns True if removed, False otherwise."""
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                return True
            return False

    def clear(self) -> None:
        """Purge all entries from the cache."""
        with self._lock:
            self._entries.clear()

    def contains(self, key: str) -> bool:
        """Check if active unexpired key exists in cache without updating LRU order."""
        now = self._clock()
        with self._lock:
            if key not in self._entries:
                return False
            entry = self._entries[key]
            if self._is_expired(entry, now):
                del self._entries[key]
                self._expirations += 1
                return False
            return True

    def stats(self) -> CacheStats:
        """Return snapshot of cache metrics and capacity."""
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                expirations=self._expirations,
                evictions=self._evictions,
                size=len(self._entries),
                max_size=self._max_size,
                ttl_seconds=self._default_ttl,
            )

    def reset_stats(self) -> None:
        """Reset telemetry counters without clearing cached entries."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._expirations = 0
            self._evictions = 0
