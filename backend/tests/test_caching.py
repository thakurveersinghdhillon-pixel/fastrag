"""FastRAG Caching Subsystem Test Suite.

Tests:
1. BaseLRUCache: LRU eviction, deterministic TTL expiration, thread safety, telemetry counters.
2. QueryEmbeddingCache: Model key isolation, array immutability, finiteness validation, enabled switch.
3. QueryResponseCache: Parameter key isolation, tenant isolation, corpus fingerprint invalidation,
   temperature bypass, rejection of incomplete generations, response immutability.
4. Pipeline Integration: End-to-end caching in DenseRetriever and RAGPipeline (synchronous, async, and streaming).
"""

from __future__ import annotations

import concurrent.futures
import copy
import numpy as np
import pytest

from app.caching.base import BaseLRUCache, CacheStats
from app.caching.embedding_cache import QueryEmbeddingCache
from app.caching.query_cache import QueryResponseCache
from app.generation.generator import RAGGenerator
from app.generation.models import GenerationResponse, StreamEvent, UsageStats
from app.generation.providers import MockLLMProvider
from app.ingestion.models import Chunk
from app.retrieval.base import IndexInfo, RetrievalResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.context_assembler import AssembledContext, ContextAssembler, ContextChunk
from app.retrieval.dense import DenseRetriever, MockEmbeddingEncoder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pipeline import PipelineDiagnostics, PipelineResponse, RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker, MockCrossEncoderScorer


# -----------------------------------------------------------------------------
# Fixtures & Helpers
# -----------------------------------------------------------------------------

class FakeClock:
    """Deterministic monotonic clock for TTL unit testing without real-time sleep."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self._current_time = start_time

    def __call__(self) -> float:
        return self._current_time

    def advance(self, seconds: float) -> None:
        self._current_time += seconds


def _create_mock_response(query: str = "Test query", finish_reason: str = "stop") -> PipelineResponse:
    context_chunk = ContextChunk(
        chunk_id="c_001",
        document_id="doc_001",
        citation_id="[Doc-1]",
        rank=1,
        text="Sample passage text.",
        source="test.txt",
        approximate_tokens=10,
    )
    context = AssembledContext(
        context_text="[Doc-1] Sample passage text.",
        selected_chunks=[context_chunk],
        selected_chunk_ids=["c_001"],
        selected_document_ids=["doc_001"],
        total_chunks_selected=1,
        total_chunks_considered=1,
        approximate_token_count=10,
        token_budget=500,
        max_chunks_budget=5,
        budget_exceeded=False,
        truncation_applied=False,
        truncation_reasons=[],
        assembly_duration_ms=0.05,
    )
    diagnostics = PipelineDiagnostics(
        retrieval_ms=1.5,
        rerank_ms=0.5,
        assembly_ms=0.05,
        total_ms=10.0,
        candidate_count=1,
        reranked_count=1,
        selected_context_count=1,
        generation_ms=8.0,
    )
    gen = GenerationResponse(
        query=query,
        answer="This is a test answer [Doc-1].",
        model="mock-v1",
        provider="mock",
        finish_reason=finish_reason,
        cited_doc_ids=["doc_001"],
        context_insufficient=False,
        diagnostics=None,
        usage=UsageStats(prompt_tokens=10, completion_tokens=8, total_tokens=18),
    )
    return PipelineResponse(
        query=query,
        candidates=[],
        reranked_results=[],
        context=context,
        diagnostics=diagnostics,
        generation=gen,
    )


def _build_test_pipeline(
    enable_embedding_cache: bool = True,
    enable_query_cache: bool = True,
) -> tuple[RAGPipeline, QueryEmbeddingCache, QueryResponseCache]:
    chunks = [
        Chunk(
            chunk_id="c1",
            document_id="d1",
            chunk_index=0,
            text="Reciprocal rank fusion combines multiple search methods.",
            source="src1.txt",
            start_offset=0,
            end_offset=56,
        ),
        Chunk(
            chunk_id="c2",
            document_id="d2",
            chunk_index=0,
            text="BM25 is a probabilistic lexical ranking algorithm.",
            source="src2.txt",
            start_offset=0,
            end_offset=51,
        ),
    ]

    bm25 = BM25Retriever()
    bm25.build_index(chunks)

    emb_cache = QueryEmbeddingCache(max_size=100, enabled=enable_embedding_cache)
    dense = DenseRetriever(
        encoder=MockEmbeddingEncoder(dimension=16),
        embedding_cache=emb_cache,
    )
    dense.build_index(chunks)

    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)
    reranker = CrossEncoderReranker(scorer=MockCrossEncoderScorer())
    assembler = ContextAssembler(max_tokens=300, max_chunks=3)
    generator = RAGGenerator(provider=MockLLMProvider())

    query_cache = QueryResponseCache(
        max_size=100,
        ttl_seconds=3600.0,
        enabled=enable_query_cache,
    )

    pipeline = RAGPipeline(
        retriever=hybrid,
        reranker=reranker,
        context_assembler=assembler,
        generator=generator,
        query_cache=query_cache,
    )
    return pipeline, emb_cache, query_cache


# -----------------------------------------------------------------------------
# 1. BaseLRUCache Unit Tests
# -----------------------------------------------------------------------------

def test_base_cache_get_set_and_stats():
    cache: BaseLRUCache[str] = BaseLRUCache(max_size=3)
    assert cache.get("k1") is None
    assert cache.stats().misses == 1
    assert cache.stats().hits == 0

    cache.set("k1", "v1")
    assert cache.get("k1") == "v1"
    assert cache.stats().hits == 1
    assert len(cache) == 1
    assert cache.contains("k1") is True


def test_base_cache_lru_eviction():
    cache: BaseLRUCache[int] = BaseLRUCache(max_size=3)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)

    # Access 'a' so 'b' becomes least recently used
    assert cache.get("a") == 1

    # Insert 'd', triggering eviction of 'b'
    cache.set("d", 4)
    assert len(cache) == 3
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4
    assert cache.stats().evictions == 1


def test_base_cache_ttl_expiration_with_fake_clock():
    clock = FakeClock(100.0)
    cache: BaseLRUCache[str] = BaseLRUCache(max_size=10, ttl_seconds=50.0, clock=clock)

    cache.set("item1", "val1")
    assert cache.get("item1") == "val1"

    # Advance clock to 140s (within 50s TTL)
    clock.advance(40.0)
    assert cache.get("item1") == "val1"

    # Advance clock past 150s (expired)
    clock.advance(15.0)
    assert cache.get("item1") is None
    assert cache.stats().expirations == 1
    assert len(cache) == 0


def test_base_cache_thread_safety():
    cache: BaseLRUCache[int] = BaseLRUCache(max_size=50)

    def worker(worker_id: int):
        for i in range(100):
            key = f"key_{i % 30}"
            cache.set(key, i)
            _ = cache.get(key)
            if i % 10 == 0:
                cache.delete(key)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, w) for w in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert len(cache) <= 50
    stats = cache.stats()
    assert stats.hits + stats.misses > 0


# -----------------------------------------------------------------------------
# 2. QueryEmbeddingCache Unit Tests
# -----------------------------------------------------------------------------

def test_embedding_cache_model_isolation():
    cache = QueryEmbeddingCache(max_size=10)
    vec1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    vec2 = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    cache.set("what is rrf", "model_mini", vec1)
    cache.set("what is rrf", "model_large", vec2)

    cached_mini = cache.get("what is rrf", "model_mini")
    cached_large = cache.get("what is rrf", "model_large")

    assert np.allclose(cached_mini, vec1)
    assert np.allclose(cached_large, vec2)
    assert not np.allclose(cached_mini, cached_large)


def test_embedding_cache_immutability():
    cache = QueryEmbeddingCache(max_size=10)
    original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cache.set("test query", "model_v1", original)

    retrieved = cache.get("test query", "model_v1")
    assert retrieved is not None
    # Mutating returned array must not alter internal cache
    retrieved[0] = 999.0

    retrieved_again = cache.get("test query", "model_v1")
    assert retrieved_again[0] == 1.0


def test_embedding_cache_finiteness_check():
    cache = QueryEmbeddingCache(max_size=10)
    nan_vec = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    cache.set("nan query", "model_v1", nan_vec)
    assert cache.get("nan query", "model_v1") is None

    inf_vec = np.array([1.0, np.inf, 3.0], dtype=np.float32)
    cache.set("inf query", "model_v1", inf_vec)
    assert cache.get("inf query", "model_v1") is None


def test_embedding_cache_disabled_switch():
    cache = QueryEmbeddingCache(max_size=10, enabled=False)
    vec = np.array([1.0, 2.0], dtype=np.float32)
    cache.set("query", "m", vec)
    assert cache.get("query", "m") is None
    assert len(cache) == 0


# -----------------------------------------------------------------------------
# 3. QueryResponseCache Unit Tests
# -----------------------------------------------------------------------------

def test_response_cache_key_isolation_parameters():
    """Verify distinct cache keys across all answer-affecting parameters."""
    k_base = QueryResponseCache.build_cache_key(
        query="what is fastrag",
        candidate_k=20,
        rerank_top_k=10,
        context_max_tokens=1000,
        context_max_chunks=5,
        bypass_reranking=False,
        model="gemini",
        temperature=0.0,
        system_prompt="sys1",
        corpus_fingerprint="fp1",
        tenant_id="tenant_a",
    )

    # 1. candidate_k change
    k_cand = QueryResponseCache.build_cache_key(
        query="what is fastrag", candidate_k=30, rerank_top_k=10,
        context_max_tokens=1000, context_max_chunks=5, bypass_reranking=False,
        model="gemini", temperature=0.0, system_prompt="sys1", corpus_fingerprint="fp1", tenant_id="tenant_a",
    )
    assert k_base != k_cand

    # 2. rerank_top_k change
    k_top = QueryResponseCache.build_cache_key(
        query="what is fastrag", candidate_k=20, rerank_top_k=5,
        context_max_tokens=1000, context_max_chunks=5, bypass_reranking=False,
        model="gemini", temperature=0.0, system_prompt="sys1", corpus_fingerprint="fp1", tenant_id="tenant_a",
    )
    assert k_base != k_top

    # 3. bypass_reranking change
    k_bypass = QueryResponseCache.build_cache_key(
        query="what is fastrag", candidate_k=20, rerank_top_k=10,
        context_max_tokens=1000, context_max_chunks=5, bypass_reranking=True,
        model="gemini", temperature=0.0, system_prompt="sys1", corpus_fingerprint="fp1", tenant_id="tenant_a",
    )
    assert k_base != k_bypass

    # 4. model change
    k_model = QueryResponseCache.build_cache_key(
        query="what is fastrag", candidate_k=20, rerank_top_k=10,
        context_max_tokens=1000, context_max_chunks=5, bypass_reranking=False,
        model="gpt-4o", temperature=0.0, system_prompt="sys1", corpus_fingerprint="fp1", tenant_id="tenant_a",
    )
    assert k_base != k_model

    # 5. system_prompt change
    k_prompt = QueryResponseCache.build_cache_key(
        query="what is fastrag", candidate_k=20, rerank_top_k=10,
        context_max_tokens=1000, context_max_chunks=5, bypass_reranking=False,
        model="gemini", temperature=0.0, system_prompt="sys2", corpus_fingerprint="fp1", tenant_id="tenant_a",
    )
    assert k_base != k_prompt

    # 6. corpus_fingerprint change
    k_corpus = QueryResponseCache.build_cache_key(
        query="what is fastrag", candidate_k=20, rerank_top_k=10,
        context_max_tokens=1000, context_max_chunks=5, bypass_reranking=False,
        model="gemini", temperature=0.0, system_prompt="sys1", corpus_fingerprint="fp2", tenant_id="tenant_a",
    )
    assert k_base != k_corpus

    # 7. tenant_id change
    k_tenant = QueryResponseCache.build_cache_key(
        query="what is fastrag", candidate_k=20, rerank_top_k=10,
        context_max_tokens=1000, context_max_chunks=5, bypass_reranking=False,
        model="gemini", temperature=0.0, system_prompt="sys1", corpus_fingerprint="fp1", tenant_id="tenant_b",
    )
    assert k_base != k_tenant


def test_response_cache_temperature_bypass():
    cache = QueryResponseCache(max_size=10, allow_stochastic=False)
    resp = _create_mock_response("stochastic test")

    # Set with temperature 0.7 -> should not cache
    cache.set(resp, temperature=0.7)
    assert cache.get("stochastic test", temperature=0.7) is None
    assert len(cache) == 0

    # Set with temperature 0.0 -> caches cleanly
    cache.set(resp, temperature=0.0)
    assert len(cache) == 1
    # Get with temperature 0.7 -> bypasses cache even if key exists
    assert cache.get("stochastic test", temperature=0.7) is None
    # Get with temperature 0.0 -> returns cached answer
    assert cache.get("stochastic test", temperature=0.0) is not None


def test_response_cache_incomplete_generation_never_cached():
    cache = QueryResponseCache(max_size=10)

    # 1. Response without generation
    resp_no_gen = _create_mock_response("no gen")
    resp_no_gen.generation = None
    cache.set(resp_no_gen)
    assert cache.get("no gen") is None

    # 2. Response with abnormal finish_reason
    resp_truncated = _create_mock_response("truncated", finish_reason="length")
    cache.set(resp_truncated)
    assert cache.get("truncated") is None

    # 3. Response with empty answer
    resp_empty = _create_mock_response("empty")
    resp_empty.generation.answer = "   "
    cache.set(resp_empty)
    assert cache.get("empty") is None

    assert len(cache) == 0


def test_response_cache_immutability():
    cache = QueryResponseCache(max_size=10)
    original = _create_mock_response("immutability test")
    cache.set(original)

    retrieved = cache.get("immutability test")
    assert retrieved is not None
    # Modify retrieved response answer
    retrieved.generation.answer = "MUTATED"

    # Second fetch must still have the original answer
    second = cache.get("immutability test")
    assert second.generation.answer == "This is a test answer [Doc-1]."


def test_response_cache_corpus_invalidation():
    cache = QueryResponseCache(max_size=10)
    resp = _create_mock_response("query about rrf")

    cache.set(resp, corpus_fingerprint="fingerprint_v1")
    assert cache.get("query about rrf", corpus_fingerprint="fingerprint_v1") is not None

    # If documents change, corpus fingerprint changes -> cache miss
    assert cache.get("query about rrf", corpus_fingerprint="fingerprint_v2") is None


# -----------------------------------------------------------------------------
# 4. Pipeline End-to-End Integration Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_embedding_cache_hit():
    """Verify that DenseRetriever queries hit the embedding cache on repeated calls."""
    pipeline, emb_cache, _ = _build_test_pipeline()

    # Pass 1: Cache Miss
    resp1 = await pipeline.execute_async("reciprocal rank fusion", generate_answer=False)
    assert emb_cache.stats().misses >= 1
    assert emb_cache.stats().hits == 0

    # Pass 2: Cache Hit
    resp2 = await pipeline.execute_async("reciprocal rank fusion", generate_answer=False)
    assert emb_cache.stats().hits >= 1

    # Retrieved chunk IDs must be identical
    assert [c.chunk_id for c in resp1.reranked_results] == [c.chunk_id for c in resp2.reranked_results]


@pytest.mark.asyncio
async def test_pipeline_query_response_cache_hit():
    """Verify end-to-end response caching drops latency and marks cache_hit in diagnostics."""
    pipeline, _, q_cache = _build_test_pipeline()

    # Pass 1: Cold execution
    resp1 = await pipeline.execute_async("what is reciprocal rank fusion?", generate_answer=True)
    assert q_cache.stats().misses == 1
    assert q_cache.stats().hits == 0
    assert resp1.generation is not None
    assert len(q_cache) == 1

    # Pass 2: Warm execution (cache hit)
    resp2 = await pipeline.execute_async("what is reciprocal rank fusion?", generate_answer=True)
    assert q_cache.stats().hits == 1
    assert resp2.diagnostics.generation_diagnostics.get("cache_hit") is True
    assert resp2.generation.answer == resp1.generation.answer
    assert resp2.generation.cited_doc_ids == resp1.generation.cited_doc_ids


@pytest.mark.asyncio
async def test_pipeline_stream_response_cache_hit():
    """Verify execute_stream_async replays cached tokens over SSE on cache hit."""
    pipeline, _, q_cache = _build_test_pipeline()

    # Populate cache via execute_async
    _ = await pipeline.execute_async("streaming test", generate_answer=True)
    assert len(q_cache) == 1

    # Stream query that is cached
    events: list[StreamEvent] = []
    async for event in pipeline.execute_stream_async("streaming test"):
        events.append(event)

    event_types = [e.event for e in events]
    assert event_types[0] == "context"
    assert "token" in event_types
    assert event_types[-1] == "done"

    # Context event and done event should record cache_hit
    assert events[0].data.get("cache_hit") is True
    assert events[-1].data.get("diagnostics", {}).get("cache_hit") is True


@pytest.mark.asyncio
async def test_pipeline_both_caches_disabled():
    """Verify pipeline functions properly when both caches are disabled."""
    pipeline, emb_cache, q_cache = _build_test_pipeline(
        enable_embedding_cache=False,
        enable_query_cache=False,
    )

    resp1 = await pipeline.execute_async("uncached query", generate_answer=True)
    resp2 = await pipeline.execute_async("uncached query", generate_answer=True)

    assert len(emb_cache) == 0
    assert len(q_cache) == 0
    assert resp1.generation is not None
    assert resp2.generation is not None


# -----------------------------------------------------------------------------
# 5. Correctness Audit Tests (Phase 8.1 Audit)
# -----------------------------------------------------------------------------

def test_audit_response_cache_key_includes_max_tokens():
    """Requirement 7: Response-cache keys include every answer-affecting parameter, including max_tokens."""
    k_100 = QueryResponseCache.build_cache_key(
        query="test query",
        candidate_k=20,
        rerank_top_k=10,
        context_max_tokens=1000,
        context_max_chunks=5,
        bypass_reranking=False,
        model="gemini",
        temperature=0.0,
        system_prompt="sys",
        corpus_fingerprint="fp1",
        tenant_id="tenant_a",
        max_tokens=100,
    )
    k_500 = QueryResponseCache.build_cache_key(
        query="test query",
        candidate_k=20,
        rerank_top_k=10,
        context_max_tokens=1000,
        context_max_chunks=5,
        bypass_reranking=False,
        model="gemini",
        temperature=0.0,
        system_prompt="sys",
        corpus_fingerprint="fp1",
        tenant_id="tenant_a",
        max_tokens=500,
    )
    assert k_100 != k_500


def test_audit_embedding_cache_numerical_identity_and_immutability():
    """Requirement 3: Embedding-cache hits return the same numerical embedding as fresh computation and cannot corrupt cached arrays."""
    encoder = MockEmbeddingEncoder(dimension=16)
    cache = QueryEmbeddingCache(max_size=10)

    query = "semantic vector search"
    fresh_vector = encoder.encode([query], batch_size=1, normalize=True)[0]
    cache.set(query, encoder.model_name, fresh_vector)

    cached_vector = cache.get(query, encoder.model_name)
    assert cached_vector is not None
    assert np.array_equal(cached_vector, fresh_vector)
    assert np.allclose(cached_vector, fresh_vector, atol=1e-7)

    # Verify that mutating the returned array does not corrupt the cached array
    cached_vector[0] = 99999.0
    second_fetch = cache.get(query, encoder.model_name)
    assert second_fetch[0] == fresh_vector[0]
    assert np.array_equal(second_fetch, fresh_vector)


@pytest.mark.asyncio
async def test_audit_streaming_cache_hit_event_sequence():
    """Requirement 2: Streaming cache hits emit valid events in correct order (context -> token* -> done)."""
    pipeline, _, q_cache = _build_test_pipeline()

    # Seed the cache
    _ = await pipeline.execute_async("streaming sequence test", generate_answer=True)
    assert len(q_cache) == 1

    events: list[StreamEvent] = []
    async for event in pipeline.execute_stream_async("streaming sequence test"):
        events.append(event)

    # 1. Context event must be first
    assert len(events) >= 3
    assert events[0].event == "context"
    assert "query" in events[0].data
    assert events[0].data["cache_hit"] is True
    assert "selected_chunks" in events[0].data

    # 2. Middle events must be 'token'
    tokens = [e for e in events if e.event == "token"]
    assert len(tokens) > 0
    assert tokens[0].data["is_first"] is True
    for t in tokens[1:]:
        assert t.data["is_first"] is False

    reconstructed_text = "".join(t.data["delta"] for t in tokens)
    assert len(reconstructed_text) > 0

    # 3. Final event must be 'done'
    assert events[-1].event == "done"
    assert events[-1].data["finish_reason"] == "stop"
    assert events[-1].data["diagnostics"]["cache_hit"] is True
    assert events[-1].data["total_pipeline_ms"] >= 0.0


@pytest.mark.asyncio
async def test_audit_failed_stream_never_cached():
    """Requirement 2: Failed or incomplete streams are never cached."""
    from unittest.mock import AsyncMock
    pipeline, _, q_cache = _build_test_pipeline()

    # Configure generator.stream_answer to fail halfway
    async def failing_stream(*args, **kwargs):
        from app.generation.models import StreamChunk
        yield StreamChunk(delta="Started", is_first=True, is_final=False)
        raise RuntimeError("Stream connection dropped abruptly")

    pipeline.generator.stream_answer = failing_stream

    with pytest.raises(RuntimeError, match="Stream connection dropped abruptly"):
        async for _ in pipeline.execute_stream_async("failing query"):
            pass

    # Verify query cache remains completely empty
    assert len(q_cache) == 0


@pytest.mark.asyncio
async def test_audit_pipeline_tenant_isolation():
    """Requirement 6: Tenant namespaces are enforced wherever tenant-aware calls are supported."""
    pipeline, _, q_cache = _build_test_pipeline()

    # Query with tenant_alpha
    resp_alpha = await pipeline.execute_async(
        "shared knowledge question",
        generate_answer=True,
        tenant_id="tenant_alpha",
    )
    assert q_cache.stats().hits == 0
    assert q_cache.stats().misses == 1

    # Query identical string with tenant_beta -> must be a cache miss!
    resp_beta = await pipeline.execute_async(
        "shared knowledge question",
        generate_answer=True,
        tenant_id="tenant_beta",
    )
    assert q_cache.stats().hits == 0
    assert q_cache.stats().misses == 2

    # Query again with tenant_alpha -> hits tenant_alpha cache
    resp_alpha_repeat = await pipeline.execute_async(
        "shared knowledge question",
        generate_answer=True,
        tenant_id="tenant_alpha",
    )
    assert q_cache.stats().hits == 1
    assert resp_alpha_repeat.diagnostics.generation_diagnostics.get("cache_hit") is True


@pytest.mark.asyncio
async def test_audit_corpus_fingerprint_invalidation_end_to_end():
    """Requirement 1: Corpus/index fingerprint changes invalidate response-cache entries end-to-end."""
    pipeline, _, q_cache = _build_test_pipeline()

    # Pass 1: Original corpus
    resp1 = await pipeline.execute_async("what is rrf", generate_answer=True)
    assert q_cache.stats().hits == 0

    # Pass 2: Repeat with same corpus -> Cache hit
    resp2 = await pipeline.execute_async("what is rrf", generate_answer=True)
    assert q_cache.stats().hits == 1
    assert resp2.diagnostics.generation_diagnostics.get("cache_hit") is True

    # Mutate index with an additional chunk to update corpus fingerprint
    new_chunks = [
        Chunk(
            chunk_id="c_new",
            document_id="d_new",
            chunk_index=0,
            text="Brand new knowledge inserted into corpus.",
            source="new_doc.txt",
            start_offset=0,
            end_offset=42,
        ),
    ]
    pipeline.retriever.build_index(new_chunks)

    # Pass 3: Query after index mutation -> Must miss cache due to fingerprint change!
    resp3 = await pipeline.execute_async("what is rrf", generate_answer=True)
    assert q_cache.stats().misses == 2
    assert resp3.diagnostics.generation_diagnostics.get("cache_hit") is not True


def test_audit_concurrent_access_does_not_corrupt_state():
    """Requirement 5: Concurrent requests do not corrupt entries, counters, or returned objects."""
    cache = QueryResponseCache(max_size=20)
    mock_resp = _create_mock_response("concurrent test")

    def worker(idx: int):
        for i in range(50):
            q = f"query_{i % 10}"
            cache.set(mock_resp, model=f"model_{idx}", candidate_k=idx)
            retrieved = cache.get(mock_resp.query, model=f"model_{idx}", candidate_k=idx)
            if retrieved is not None:
                assert retrieved.generation.answer == mock_resp.generation.answer
                # Attempt to mutate returned object in thread
                retrieved.generation.answer = f"mutated_by_{idx}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, w) for w in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    stats = cache.stats()
    assert stats.hits + stats.misses > 0
    assert len(cache) <= 20


def test_audit_dense_retriever_fails_safely_on_cache_error():
    """Requirement 8: Cache errors fail safely without hiding retrieval or generation errors."""
    from unittest.mock import MagicMock
    broken_cache = MagicMock()
    broken_cache.enabled = True
    broken_cache.get.side_effect = RuntimeError("Embedding cache disk failure")
    broken_cache.set.side_effect = RuntimeError("Embedding cache write failure")

    dense = DenseRetriever(
        encoder=MockEmbeddingEncoder(dimension=16),
        embedding_cache=broken_cache,
    )
    chunks = [
        Chunk(chunk_id="c1", document_id="d1", chunk_index=0, text="Sample text", source="s", start_offset=0, end_offset=11),
    ]
    dense.build_index(chunks)

    # search must not crash despite cache throwing exceptions
    results = dense.search("query")
    assert len(results) == 1
    assert results[0].chunk_id == "c1"


@pytest.mark.asyncio
async def test_audit_pipeline_fails_safely_on_cache_error():
    """Requirement 8: Pipeline continues and produces answer even if query cache raises exceptions."""
    from unittest.mock import MagicMock
    pipeline, _, _ = _build_test_pipeline()

    broken_query_cache = MagicMock()
    broken_query_cache.enabled = True
    broken_query_cache.get.side_effect = RuntimeError("Query cache memory fault")
    broken_query_cache.set.side_effect = RuntimeError("Query cache storage fault")
    pipeline.query_cache = broken_query_cache

    # Pipeline execution must succeed cleanly
    resp = await pipeline.execute_async("test query", generate_answer=True)
    assert resp.generation is not None
    assert resp.generation.answer is not None


@pytest.mark.asyncio
async def test_audit_pipeline_preserves_retrieval_and_generation_errors():
    """Requirement 8: Real retrieval and generation exceptions are preserved and not swallowed by cache."""
    from app.core.exceptions import ProviderTimeoutError, RetrievalError
    from unittest.mock import AsyncMock
    pipeline, _, _ = _build_test_pipeline()

    # Generator raises ProviderTimeoutError -> must propagate
    pipeline.generator.generate_answer = AsyncMock(side_effect=ProviderTimeoutError("LLM gateway timeout"))

    with pytest.raises(ProviderTimeoutError, match="LLM gateway timeout"):
        await pipeline.execute_async("test query", generate_answer=True)

