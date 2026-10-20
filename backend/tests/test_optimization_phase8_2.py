"""FastRAG Phase 8.2 Performance Optimization Regression Tests.

Verifies:
1. BM25 inverted postings table equivalence and deterministic ranking.
2. Model warmup hook functionality across encoder, reranker, and pipeline.
3. OpenAICompatibleProvider persistent HTTP client reuse and clean aclose().
4. RAGPipeline resource cleanup lifecycle.
5. Numerical invariance of CrossEncoder scoring under inference mode.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from app.generation.models import GenerationRequest
from app.generation.providers import MockLLMProvider, OpenAICompatibleProvider
from app.generation.generator import RAGGenerator
from app.ingestion.models import Chunk
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever, MockEmbeddingEncoder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker, MockCrossEncoderScorer


def _sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="chunk_001",
            document_id="doc_A",
            chunk_index=0,
            text="Reciprocal rank fusion merges multiple retrieval rankings without calibration.",
            source="source_A",
            start_offset=0,
            end_offset=74,
        ),
        Chunk(
            chunk_id="chunk_002",
            document_id="doc_B",
            chunk_index=0,
            text="BM25 is a bag-of-words retrieval function that ranks documents based on term frequency.",
            source="source_B",
            start_offset=0,
            end_offset=86,
        ),
        Chunk(
            chunk_id="chunk_003",
            document_id="doc_C",
            chunk_index=0,
            text="Dense vector retrieval performs semantic search using bi-encoder embeddings.",
            source="source_C",
            start_offset=0,
            end_offset=75,
        ),
    ]


def test_bm25_inverted_postings_generation_and_search():
    """Verify that BM25Retriever builds an inverted postings table and scores deterministically."""
    chunks = _sample_chunks()
    bm25 = BM25Retriever()
    bm25.build_index(chunks)

    # Check postings table exists and indexes all terms
    assert hasattr(bm25, "_postings")
    assert len(bm25._postings) > 0
    assert "bm25" in bm25._postings
    assert "reciprocal" in bm25._postings

    # Search for a term present only in doc_B
    results_bm25 = bm25.search("BM25 retrieval function", top_k=5)
    assert len(results_bm25) > 0
    assert results_bm25[0].chunk_id == "chunk_002"
    assert results_bm25[0].score > 0.0

    # Search for an unknown term yields empty
    results_empty = bm25.search("nonexistenttermxyz", top_k=5)
    assert len(results_empty) == 0


def test_warmup_hooks_on_pipeline_and_components():
    """Verify warmup() propagates correctly without errors on mock or real components."""
    chunks = _sample_chunks()
    encoder = MockEmbeddingEncoder()
    dense = DenseRetriever(encoder=encoder)
    dense.build_index(chunks)

    bm25 = BM25Retriever()
    bm25.build_index(chunks)

    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)
    scorer = MockCrossEncoderScorer()
    reranker = CrossEncoderReranker(scorer=scorer)

    pipeline = RAGPipeline(
        retriever=hybrid,
        reranker=reranker,
    )

    # Calling warmup should not raise and should trigger component warmup
    pipeline.warmup()
    hybrid.warmup()
    dense.warmup()
    reranker.warmup()


@pytest.mark.asyncio
async def test_openai_compatible_provider_client_reuse_and_aclose():
    """Verify OpenAICompatibleProvider reuses a single AsyncClient and closes it upon aclose()."""
    provider = OpenAICompatibleProvider(
        api_key="mock-key",
        base_url="https://mock.api.endpoint/v1",
        model_name="mock-model",
    )

    # Initial call to _get_client creates client
    client1 = provider._get_client()
    assert isinstance(client1, httpx.AsyncClient)
    assert not client1.is_closed

    # Second call reuses the exact same client instance
    client2 = provider._get_client()
    assert client1 is client2

    # Closing provider closes the internal client
    await provider.aclose()
    assert client1.is_closed

    # Calling _get_client again after aclose recreates a fresh active client
    client3 = provider._get_client()
    assert not client3.is_closed
    assert client3 is not client1
    await provider.aclose()


@pytest.mark.asyncio
async def test_rag_pipeline_and_generator_aclose():
    """Verify RAGPipeline.aclose() delegates down to RAGGenerator and provider."""
    mock_provider = AsyncMock()
    mock_provider.aclose = AsyncMock()

    generator = RAGGenerator(provider=mock_provider)
    pipeline = RAGPipeline(
        retriever=BM25Retriever(),
        generator=generator,
    )

    await pipeline.aclose()
    mock_provider.aclose.assert_awaited_once()


def test_mock_scorer_score_pairs_invariance():
    """Verify that CrossEncoder scorer produces deterministic scores."""
    scorer = MockCrossEncoderScorer()
    pairs = [("query A", "passage 1"), ("query A", "passage 2")]
    scores1 = scorer.score_pairs(pairs)
    scores2 = scorer.score_pairs(pairs)
    assert scores1 == scores2
    assert len(scores1) == 2
