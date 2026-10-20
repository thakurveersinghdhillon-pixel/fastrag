"""FastRAG Query Response Cache (Tier 1).

Caches end-to-end grounded query responses (assembled context, synthesized answer,
citations, and diagnostics) to bypass retrieval and LLM generation entirely for identical queries.

Tenant & Fingerprint Architecture Note:
    - Multi-Tenancy: FastRAG's public QueryRequest currently does not expose a tenant_id field.
      To ensure security and future multi-tenant compatibility, the cache key generator supports
      an explicit tenant_id parameter (defaulting to 'default'). All keys are strictly namespaced.
    - Corpus Invalidation: FastRAG computes a deterministic SHA-256 corpus fingerprint
      (IndexInfo.corpus_fingerprint) over all indexed chunks during index build/load.
      This fingerprint is incorporated into every Tier 1 cache key. Rebuilding or modifying
      the index changes the fingerprint, which instantly invalidates stale cached answers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any, Callable

from app.caching.base import BaseLRUCache, CacheStats
from app.core.logging_config import get_logger
from app.retrieval.pipeline import PipelineDiagnostics, PipelineResponse

logger = get_logger("fastrag.caching.query")


class QueryResponseCache:
    """Thread-safe bounded in-memory cache for complete RAG pipeline responses.

    Guarantees:
    - Successful Generations Only: Responses without completed LLM generation or
      those with abnormal finish reasons are never cached.
    - Strict Parameter Isolation: Differentiates queries by retrieval top_k, reranking flag,
      model name, prompt variation, temperature, tenant ID, and corpus fingerprint.
    - Stochastic Bypass: Automatically bypasses caching when temperature > 0.0 unless
      allow_stochastic is explicitly enabled.
    - Mutation Protection: Returns a deep-copied response object so callers cannot mutate
      internal cache entries.
    """

    def __init__(
        self,
        max_size: int = 2000,
        ttl_seconds: float = 3600.0,
        enabled: bool = True,
        allow_stochastic: bool = False,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.enabled = enabled
        self.allow_stochastic = allow_stochastic
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: BaseLRUCache[PipelineResponse] = BaseLRUCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            clock=clock,
        )

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    def __len__(self) -> int:
        return len(self._cache)

    @staticmethod
    def build_cache_key(
        query: str,
        candidate_k: int = 20,
        rerank_top_k: int = 10,
        context_max_tokens: int = 1000,
        context_max_chunks: int = 5,
        bypass_reranking: bool = False,
        model: str = "default",
        temperature: float = 0.0,
        system_prompt: str | None = None,
        corpus_fingerprint: str = "",
        tenant_id: str | None = None,
        max_tokens: int | None = None,
        min_relevance_score: float | None = None,
        ordering_strategy: str = "descending",
        rerank_blending_alpha: float = 0.0,
    ) -> str:
        """Construct deterministic SHA-256 key covering all response-affecting parameters."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string must be a non-empty string")

        prompt_digest = (
            hashlib.sha256(system_prompt.strip().encode("utf-8")).hexdigest()
            if system_prompt and system_prompt.strip()
            else "default_prompt"
        )

        canonical = {
            "tenant_id": (tenant_id or "default").strip(),
            "query": query.strip().lower(),
            "candidate_k": int(candidate_k),
            "rerank_top_k": int(rerank_top_k),
            "context_max_tokens": int(context_max_tokens),
            "context_max_chunks": int(context_max_chunks),
            "bypass_reranking": bool(bypass_reranking),
            "model": (model or "default").strip().lower(),
            "temperature": round(float(temperature or 0.0), 4),
            "max_tokens": int(max_tokens) if max_tokens is not None else None,
            "prompt_digest": prompt_digest,
            "corpus_fingerprint": (corpus_fingerprint or "unindexed").strip(),
            "min_relevance_score": round(float(min_relevance_score), 4) if min_relevance_score is not None else None,
            "ordering_strategy": str(ordering_strategy or "descending").lower(),
            "rerank_blending_alpha": round(float(rerank_blending_alpha or 0.0), 4),
        }

        serialized = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(
        self,
        query: str,
        candidate_k: int = 20,
        rerank_top_k: int = 10,
        context_max_tokens: int = 1000,
        context_max_chunks: int = 5,
        bypass_reranking: bool = False,
        model: str = "default",
        temperature: float = 0.0,
        system_prompt: str | None = None,
        corpus_fingerprint: str = "",
        tenant_id: str | None = None,
        max_tokens: int | None = None,
        min_relevance_score: float | None = None,
        ordering_strategy: str = "descending",
        rerank_blending_alpha: float = 0.0,
    ) -> PipelineResponse | None:
        """Retrieve cached pipeline response if valid, unexpired, and enabled."""
        if not self.enabled:
            return None

        # By default, bypass cache when non-deterministic sampling is requested
        eff_temp = temperature if temperature is not None else 0.0
        if eff_temp > 0.0 and not self.allow_stochastic:
            return None

        if not query or not query.strip():
            return None

        key = self.build_cache_key(
            query=query,
            candidate_k=candidate_k,
            rerank_top_k=rerank_top_k,
            context_max_tokens=context_max_tokens,
            context_max_chunks=context_max_chunks,
            bypass_reranking=bypass_reranking,
            model=model,
            temperature=eff_temp,
            system_prompt=system_prompt,
            corpus_fingerprint=corpus_fingerprint,
            tenant_id=tenant_id,
            max_tokens=max_tokens,
            min_relevance_score=min_relevance_score,
            ordering_strategy=ordering_strategy,
            rerank_blending_alpha=rerank_blending_alpha,
        )

        cached = self._cache.get(key)
        if cached is None:
            return None

        # Return a deepcopy to prevent caller mutation of internal cache data
        cloned = copy.deepcopy(cached)
        return cloned

    def set(
        self,
        response: PipelineResponse,
        candidate_k: int = 20,
        rerank_top_k: int = 10,
        context_max_tokens: int = 1000,
        context_max_chunks: int = 5,
        bypass_reranking: bool = False,
        model: str = "default",
        temperature: float = 0.0,
        system_prompt: str | None = None,
        corpus_fingerprint: str = "",
        tenant_id: str | None = None,
        max_tokens: int | None = None,
        ttl_seconds: float | None = None,
        min_relevance_score: float | None = None,
        ordering_strategy: str = "descending",
        rerank_blending_alpha: float = 0.0,
    ) -> None:
        """Cache a successful, fully generated pipeline response."""
        if not self.enabled:
            return

        eff_temp = temperature if temperature is not None else 0.0
        if eff_temp > 0.0 and not self.allow_stochastic:
            return

        # Verification of generation completeness
        gen = response.generation
        if gen is None:
            # Never cache retrieval-only results in the response cache
            return

        if not gen.answer or not gen.answer.strip():
            # Refuse to cache empty answers
            return

        if gen.finish_reason not in ("stop", "complete"):
            # Refuse to cache responses aborted due to length, errors, or timeouts
            logger.debug(f"Refusing to cache response with finish_reason='{gen.finish_reason}'")
            return

        key = self.build_cache_key(
            query=response.query,
            candidate_k=candidate_k,
            rerank_top_k=rerank_top_k,
            context_max_tokens=context_max_tokens,
            context_max_chunks=context_max_chunks,
            bypass_reranking=bypass_reranking,
            model=model or gen.model,
            temperature=eff_temp,
            system_prompt=system_prompt,
            corpus_fingerprint=corpus_fingerprint,
            tenant_id=tenant_id,
            max_tokens=max_tokens,
            min_relevance_score=min_relevance_score,
            ordering_strategy=ordering_strategy,
            rerank_blending_alpha=rerank_blending_alpha,
        )

        # Store a deepcopy to isolate cache from outside modifications
        self._cache.set(key, copy.deepcopy(response), ttl_seconds=ttl_seconds)

    def delete(
        self,
        query: str,
        candidate_k: int = 20,
        rerank_top_k: int = 10,
        context_max_tokens: int = 1000,
        context_max_chunks: int = 5,
        bypass_reranking: bool = False,
        model: str = "default",
        temperature: float = 0.0,
        system_prompt: str | None = None,
        corpus_fingerprint: str = "",
        tenant_id: str | None = None,
        max_tokens: int | None = None,
        min_relevance_score: float | None = None,
        ordering_strategy: str = "descending",
        rerank_blending_alpha: float = 0.0,
    ) -> bool:
        """Explicitly invalidate a cached response by key."""
        if not self.enabled:
            return False

        key = self.build_cache_key(
            query=query,
            candidate_k=candidate_k,
            rerank_top_k=rerank_top_k,
            context_max_tokens=context_max_tokens,
            context_max_chunks=context_max_chunks,
            bypass_reranking=bypass_reranking,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            corpus_fingerprint=corpus_fingerprint,
            tenant_id=tenant_id,
            max_tokens=max_tokens,
            min_relevance_score=min_relevance_score,
            ordering_strategy=ordering_strategy,
            rerank_blending_alpha=rerank_blending_alpha,
        )
        return self._cache.delete(key)

    def clear(self) -> None:
        """Purge all cached responses."""
        self._cache.clear()

    def stats(self) -> CacheStats:
        """Return cache performance statistics."""
        return self._cache.stats()

    def reset_stats(self) -> None:
        """Reset cache telemetry counters."""
        self._cache.reset_stats()
