"""FastRAG End-to-End Pipeline Generation and API Endpoint Tests.

Tests RAGPipeline coordination with LLM answer synthesis, SSE streaming,
and FastAPI REST/SSE endpoints.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_pipeline
from app.core.config import Settings
from app.core.exceptions import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderTimeoutError,
)
from app.generation.generator import RAGGenerator
from app.generation.models import StreamEvent
from app.generation.providers import MockLLMProvider
from app.ingestion.models import Chunk
from app.main import create_app
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.context_assembler import ContextAssembler
from app.retrieval.pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker, MockCrossEncoderScorer


def _build_test_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="c_rrf",
            document_id="doc_001",
            chunk_index=0,
            text="FastRAG uses reciprocal rank fusion for hybrid retrieval.",
            source="test_corpus.jsonl",
            start_offset=0,
            end_offset=56,
            metadata={"title": "RRF Fusion"},
        ),
        Chunk(
            chunk_id="c_rerank",
            document_id="doc_002",
            chunk_index=0,
            text="CrossEncoder reranking improves top-k candidate precision.",
            source="test_corpus.jsonl",
            start_offset=0,
            end_offset=58,
            metadata={"title": "CrossEncoder Reranking"},
        ),
        Chunk(
            chunk_id="c_stream",
            document_id="doc_003",
            chunk_index=0,
            text="Streaming pushes incremental tokens via Server-Sent Events.",
            source="test_corpus.jsonl",
            start_offset=0,
            end_offset=59,
            metadata={"title": "Streaming SSE"},
        ),
    ]


def build_test_pipeline(mock_provider: MockLLMProvider | None = None) -> RAGPipeline:
    chunks = _build_test_chunks()
    bm25 = BM25Retriever()
    bm25.build_index(chunks)

    reranker = CrossEncoderReranker(scorer=MockCrossEncoderScorer(), default_top_k=5)
    assembler = ContextAssembler(max_tokens=800, max_chunks=3)
    provider = mock_provider or MockLLMProvider(model_name="test-mock-llm")
    generator = RAGGenerator(provider=provider)

    return RAGPipeline(
        retriever=bm25,
        reranker=reranker,
        context_assembler=assembler,
        generator=generator,
    )


# -------------------------------------------------------------------------
# RAGPipeline Generation Tests
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_execute_async_with_generation():
    pipeline = build_test_pipeline()
    response = await pipeline.execute_async(
        query="What is reciprocal rank fusion?",
        candidate_k=5,
        rerank_top_k=3,
        generate_answer=True,
    )

    assert response.query == "What is reciprocal rank fusion?"
    assert len(response.candidates) > 0
    assert len(response.reranked_results) > 0
    assert response.context.total_chunks_selected > 0

    gen = response.generation
    assert gen is not None
    assert gen.model == "test-mock-llm"
    assert gen.provider == "mock"
    assert gen.finish_reason == "stop"
    assert len(gen.cited_doc_ids) > 0
    assert "doc_001" in gen.cited_doc_ids
    assert gen.context_insufficient is False

    diag = response.diagnostics
    assert diag.generation_ms > 0.0
    assert diag.ttft_ms is not None
    assert diag.total_ms > 0.0
    assert diag.usage is not None


@pytest.mark.asyncio
async def test_pipeline_execute_async_without_generation():
    pipeline = build_test_pipeline()
    response = await pipeline.execute_async(
        query="What is reciprocal rank fusion?",
        generate_answer=False,
    )

    assert response.generation is None
    assert response.diagnostics.generation_ms == 0.0
    assert response.diagnostics.ttft_ms is None


@pytest.mark.asyncio
async def test_pipeline_execute_stream_async():
    pipeline = build_test_pipeline()
    events: list[StreamEvent] = []

    async for event in pipeline.execute_stream_async(
        query="Explain streaming in FastRAG.",
        candidate_k=5,
        rerank_top_k=3,
    ):
        events.append(event)

    assert len(events) >= 3

    # Event 1: Context
    assert events[0].event == "context"
    assert "selected_chunks" in events[0].data
    assert "approximate_tokens" in events[0].data

    # Intermediate events: Tokens
    token_events = [e for e in events if e.event == "token"]
    assert len(token_events) >= 1
    assert token_events[0].data["is_first"] is True

    # Final event: Done
    assert events[-1].event == "done"
    done_data = events[-1].data
    assert "finish_reason" in done_data
    assert "diagnostics" in done_data
    assert "usage" in done_data
    assert "total_pipeline_ms" in done_data


def test_pipeline_execute_sync():
    pipeline = build_test_pipeline()
    response = pipeline.execute(
        query="What is reciprocal rank fusion?",
        generate_answer=True,
    )
    assert response.generation is not None
    assert response.generation.answer != ""


# -------------------------------------------------------------------------
# FastAPI Endpoints Tests
# -------------------------------------------------------------------------

@pytest.fixture
def test_app():
    settings = Settings(environment="test")
    app = create_app(settings)
    pipeline = build_test_pipeline()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return app


def test_api_query_endpoint(test_app):
    client = TestClient(test_app)
    payload = {
        "query": "What is reciprocal rank fusion?",
        "candidate_k": 5,
        "rerank_top_k": 3,
        "temperature": 0.0,
        "max_tokens": 128,
    }

    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["query"] == payload["query"]
    assert data["answer"] != ""
    assert data["provider"] == "mock"
    assert "doc_001" in data["cited_doc_ids"]
    assert data["context_insufficient"] is False
    assert "total_chunks_selected" in data["context"]
    assert "retrieval_ms" in data["diagnostics"]
    assert "generation_ms" in data["diagnostics"]
    assert "prompt_tokens" in data["usage"]


def test_api_query_stream_endpoint(test_app):
    client = TestClient(test_app)
    payload = {
        "query": "Explain FastRAG features.",
        "candidate_k": 5,
        "rerank_top_k": 3,
    }

    with client.stream("POST", "/api/v1/query/stream", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = [line for line in response.iter_lines() if line]

    # Verify SSE structure
    event_lines = [l for l in lines if l.startswith("event: ")]
    data_lines = [l for l in lines if l.startswith("data: ")]

    assert len(event_lines) >= 3
    event_types = [l.replace("event: ", "").strip() for l in event_lines]
    assert "context" in event_types
    assert "token" in event_types
    assert "done" in event_types


def test_api_query_auth_error(test_app):
    error_pipeline = build_test_pipeline(
        mock_provider=MockLLMProvider(simulate_error=ProviderAuthenticationError("Invalid API key"))
    )
    test_app.dependency_overrides[get_pipeline] = lambda: error_pipeline

    client = TestClient(test_app)
    response = client.post("/api/v1/query", json={"query": "reciprocal rank fusion"})
    assert response.status_code == 401
    assert "authentication failed" in response.json()["detail"].lower()


def test_api_query_timeout_error(test_app):
    error_pipeline = build_test_pipeline(
        mock_provider=MockLLMProvider(simulate_error=ProviderTimeoutError("Upstream timeout"))
    )
    test_app.dependency_overrides[get_pipeline] = lambda: error_pipeline

    client = TestClient(test_app)
    response = client.post("/api/v1/query", json={"query": "reciprocal rank fusion"})
    assert response.status_code == 504
    assert "timeout" in response.json()["detail"].lower()


def test_api_query_provider_error(test_app):
    error_pipeline = build_test_pipeline(
        mock_provider=MockLLMProvider(simulate_error=ProviderError("Rate limit exceeded"))
    )
    test_app.dependency_overrides[get_pipeline] = lambda: error_pipeline

    client = TestClient(test_app)
    response = client.post("/api/v1/query", json={"query": "reciprocal rank fusion"})
    assert response.status_code == 502
    assert "provider error" in response.json()["detail"].lower()


# -------------------------------------------------------------------------
# Phase 9.4 Integration Tests: Thresholding, Boundary Layout, and Abstention
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_execute_async_abstention_via_threshold():
    """Verify end-to-end pipeline abstention when candidates fall below relevance threshold."""
    # Configure provider that would error if invoked
    unreachable_provider = MockLLMProvider(simulate_error=RuntimeError("Provider should not be called!"))
    pipeline = build_test_pipeline(mock_provider=unreachable_provider)

    # Set threshold higher than any candidate score
    response = await pipeline.execute_async(
        query="What is quantum gravity?",
        candidate_k=5,
        rerank_top_k=3,
        min_relevance_score=999.0,
        generate_answer=True,
    )

    assert response.context.context_insufficient is True
    assert response.context.total_chunks_selected == 0
    assert response.context.context_text == ""
    assert response.generation is not None
    assert response.generation.context_insufficient is True
    assert response.generation.answer == "I do not have sufficient information in the provided context to answer this question."
    assert response.generation.cited_doc_ids == []


@pytest.mark.asyncio
async def test_pipeline_execute_async_boundary_ordering():
    """Verify end-to-end boundary ordering layout in pipeline execution."""
    pipeline = build_test_pipeline()
    response = await pipeline.execute_async(
        query="reciprocal reranking streaming",
        candidate_k=5,
        rerank_top_k=3,
        ordering_strategy="boundary",
        generate_answer=True,
    )

    assert response.context.context_insufficient is False
    assert len(response.context.selected_chunks) == 3
    # Rank 1 is first, Rank 2 is last, Rank 3 is in middle
    assert response.context.selected_chunks[0].rank == 1
    assert response.context.selected_chunks[1].rank == 3
    assert response.context.selected_chunks[2].rank == 2


@pytest.mark.asyncio
async def test_pipeline_execute_stream_async_abstention():
    """Verify streaming pipeline yields deterministic abstention without calling provider."""
    unreachable_provider = MockLLMProvider(simulate_error=RuntimeError("Provider should not be called!"))
    pipeline = build_test_pipeline(mock_provider=unreachable_provider)

    events: list[StreamEvent] = []
    async for event in pipeline.execute_stream_async(
        query="What is dark energy?",
        min_relevance_score=999.0,
    ):
        events.append(event)

    assert len(events) >= 3
    assert events[0].event == "context"
    assert events[0].data["selected_chunks"] == []

    # Token event contains exact abstention string
    token_events = [e for e in events if e.event == "token"]
    full_streamed = "".join(e.data["delta"] for e in token_events)
    assert full_streamed == "I do not have sufficient information in the provided context to answer this question."

    # Done event
    done_events = [e for e in events if e.event == "done"]
    assert len(done_events) == 1
    assert done_events[0].data["finish_reason"] == "stop"


def test_api_query_endpoint_abstention_via_threshold(test_app):
    """Verify API endpoint returns context_insufficient=True and abstention response."""
    unreachable_provider = MockLLMProvider(simulate_error=RuntimeError("Provider should not be called!"))
    pipeline = build_test_pipeline(mock_provider=unreachable_provider)
    test_app.dependency_overrides[get_pipeline] = lambda: pipeline

    client = TestClient(test_app)
    response = client.post(
        "/api/v1/query",
        json={"query": "test query", "min_relevance_score": 999.0},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["context_insufficient"] is True
    assert data["answer"] == "I do not have sufficient information in the provided context to answer this question."
    assert data["cited_doc_ids"] == []

