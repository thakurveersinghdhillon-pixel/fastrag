"""FastRAG LLM Generation Unit Tests.

Covers models, prompt construction, mock and OpenAI-compatible providers,
grounding/citation resolution, and generator coordination.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from app.core.exceptions import (
    GenerationError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    StreamInterruptedError,
)
from app.generation.models import (
    GenerationDiagnostics,
    GenerationRequest,
    GenerationResponse,
    StreamChunk,
    StreamEvent,
    UsageStats,
)
from app.generation.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    build_grounded_prompt,
)
from app.generation.providers import (
    MockLLMProvider,
    OpenAICompatibleProvider,
)
from app.generation.generator import RAGGenerator
from app.retrieval.context_assembler import ContextAssembler
from app.retrieval.base import RetrievalResult


# -------------------------------------------------------------------------
# Test Models
# -------------------------------------------------------------------------

def test_usage_stats_and_diagnostics_to_dict():
    usage = UsageStats(prompt_tokens=10, completion_tokens=20, total_tokens=30, is_estimated=False)
    d = usage.to_dict()
    assert d["prompt_tokens"] == 10
    assert d["completion_tokens"] == 20
    assert d["total_tokens"] == 30
    assert d["is_estimated"] is False

    diag = GenerationDiagnostics(
        ttft_ms=12.5,
        generation_ms=45.2,
        total_duration_ms=57.7,
        chunk_count=18,
        model_name="test-model",
        provider_name="test-provider",
    )
    dd = diag.to_dict()
    assert dd["ttft_ms"] == 12.5
    assert dd["generation_ms"] == 45.2
    assert dd["total_duration_ms"] == 57.7
    assert dd["chunk_count"] == 18
    assert dd["model_name"] == "test-model"
    assert dd["provider_name"] == "test-provider"


def test_stream_event_sse_serialization():
    event = StreamEvent(event="token", data={"delta": "Hello", "is_first": True, "is_final": False})
    sse = event.to_sse()
    assert sse.startswith("event: token\n")
    assert '"delta": "Hello"' in sse
    assert sse.endswith("\n\n")


# -------------------------------------------------------------------------
# Test Grounded Prompts
# -------------------------------------------------------------------------

def test_build_grounded_prompt_basic():
    query = "What is Reciprocal Rank Fusion?"
    context_text = "[Doc-1] (doc_001): RRF combines ranked lists without score calibration."

    system_prompt, user_prompt = build_grounded_prompt(query=query, context_text=context_text)

    # Verification of strict system instructions
    assert "[Doc-1]" in system_prompt
    assert "ONLY the factual evidence" in system_prompt
    assert "Prompt Injection Defense" in system_prompt
    assert "insufficient" in system_prompt.lower()

    # Verification of user message formatting
    assert f"User Question:\n{query}" in user_prompt
    assert "Retrieved Context:" in user_prompt
    assert "[Doc-1] (doc_001)" in user_prompt


def test_build_grounded_prompt_empty_context():
    query = "What is quantum computing?"
    system_prompt, user_prompt = build_grounded_prompt(query=query, context_text="")

    assert "[NO RETRIEVED CONTEXT AVAILABLE]" in user_prompt
    assert query in user_prompt


# -------------------------------------------------------------------------
# Test MockLLMProvider
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mock_llm_provider_generate():
    provider = MockLLMProvider(model_name="mock-model")
    req = GenerationRequest(
        query="Explain RRF.",
        context_text="[Doc-1] (doc_001): RRF is reciprocal rank fusion.",
        temperature=0.0,
        max_tokens=100,
    )
    resp = await provider.generate(req)

    assert resp.model == "mock-model"
    assert resp.provider == "mock"
    assert "[Doc-1]" in resp.answer
    assert resp.finish_reason == "stop"
    assert resp.usage.total_tokens is not None and resp.usage.total_tokens > 0
    assert resp.diagnostics.ttft_ms is not None
    assert resp.diagnostics.total_duration_ms >= 0.0


@pytest.mark.asyncio
async def test_mock_llm_provider_stream():
    provider = MockLLMProvider(model_name="mock-model")
    req = GenerationRequest(
        query="Explain RRF.",
        context_text="[Doc-1] (doc_001): RRF is reciprocal rank fusion.",
        temperature=0.0,
        max_tokens=100,
    )

    chunks: list[StreamChunk] = []
    async for chunk in provider.stream(req):
        chunks.append(chunk)

    assert len(chunks) >= 2
    assert chunks[0].is_first is True
    assert chunks[-1].is_final is True
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].diagnostics is not None
    assert chunks[-1].diagnostics.ttft_ms is not None

    full_text = "".join(c.delta for c in chunks)
    assert "[Doc-1]" in full_text


@pytest.mark.asyncio
async def test_mock_llm_provider_fail_mode():
    provider = MockLLMProvider(simulate_error=ProviderError("Simulated error"))
    req = GenerationRequest(query="test", context_text="[Doc-1]: test")

    with pytest.raises(ProviderError):
        await provider.generate(req)

    with pytest.raises(ProviderError):
        async for _ in provider.stream(req):
            pass


@pytest.mark.asyncio
async def test_mock_llm_provider_empty_context():
    provider = MockLLMProvider()
    req = GenerationRequest(query="What is XYZ?", context_text="")
    resp = await provider.generate(req)

    assert resp.answer == "I do not have sufficient information in the provided context to answer this question."
    assert resp.context_insufficient is True


# -------------------------------------------------------------------------
# Test OpenAICompatibleProvider
# -------------------------------------------------------------------------

def test_openai_provider_missing_api_key():
    with patch.dict("os.environ", {}, clear=True):
        provider = OpenAICompatibleProvider(api_key="")
        with pytest.raises(ProviderAuthenticationError):
            provider._ensure_api_key()


def test_openai_provider_defaults():
    provider = OpenAICompatibleProvider(api_key="sk-test-secret-key-12345", model_name="gpt-4o-mini")
    assert provider._ensure_api_key() == "sk-test-secret-key-12345"
    assert provider.provider_name == "openai_compatible"
    assert provider.model_name == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_openai_provider_generate_success():
    provider = OpenAICompatibleProvider(api_key="sk-test-key", model_name="gpt-4o-mini")

    mock_json_response = {
        "id": "chatcmpl-123",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "message": {"content": "RRF combines rankings [Doc-1].", "role": "assistant"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "total_tokens": 60,
        },
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_json_response

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("app.generation.providers.httpx.AsyncClient", return_value=mock_client):
        req = GenerationRequest(query="Explain RRF", context_text="[Doc-1]: info")
        res = await provider.generate(req)

        assert res.answer == "RRF combines rankings [Doc-1]."
        assert res.finish_reason == "stop"
        assert res.usage.prompt_tokens == 50
        assert res.usage.completion_tokens == 10
        assert res.usage.is_estimated is False
        assert res.diagnostics.ttft_ms is not None


@pytest.mark.asyncio
async def test_openai_provider_generate_401():
    provider = OpenAICompatibleProvider(api_key="sk-bad-key", model_name="gpt-4o-mini")

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = '{"error": {"message": "Incorrect API key provided"}}'

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("app.generation.providers.httpx.AsyncClient", return_value=mock_client):
        req = GenerationRequest(query="test", context_text="test")
        with pytest.raises(ProviderAuthenticationError):
            await provider.generate(req)


@pytest.mark.asyncio
async def test_openai_provider_generate_timeout():
    provider = OpenAICompatibleProvider(api_key="sk-key", model_name="gpt-4o-mini")

    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.TimeoutException("Read timed out")
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("app.generation.providers.httpx.AsyncClient", return_value=mock_client):
        req = GenerationRequest(query="test", context_text="test")
        with pytest.raises(ProviderTimeoutError):
            await provider.generate(req)


@pytest.mark.asyncio
async def test_openai_provider_streaming_success():
    provider = OpenAICompatibleProvider(api_key="sk-key", model_name="gpt-4o-mini")

    sse_lines = [
        'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}',
        'data: {"choices": [{"delta": {"content": " world!"}, "finish_reason": "stop"}]}',
        'data: [DONE]',
    ]

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.aiter_lines = mock_aiter_lines

    # Create async context manager for stream()
    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("app.generation.providers.httpx.AsyncClient", return_value=mock_client):
        req = GenerationRequest(query="Greet me", context_text="context")
        chunks: list[StreamChunk] = []
        async for chunk in provider.stream(req):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[0].is_first is True
        assert chunks[1].delta == " world!"
        assert chunks[2].is_final is True
        assert chunks[2].finish_reason == "stop"


# -------------------------------------------------------------------------
# Test RAGGenerator Coordinator
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_generator_citation_resolution():
    provider = MockLLMProvider()
    generator = RAGGenerator(provider=provider)

    candidate1 = RetrievalResult(
        chunk_id="chunk_001_0",
        document_id="doc_001",
        rank=1,
        score=0.95,
        score_type="rrf",
        text="FastRAG uses reciprocal rank fusion for hybrid retrieval.",
        source="hybrid",
    )
    candidate2 = RetrievalResult(
        chunk_id="chunk_002_0",
        document_id="doc_002",
        rank=2,
        score=0.90,
        score_type="rrf",
        text="CrossEncoder reranking refines candidate scores.",
        source="hybrid",
    )

    assembler = ContextAssembler(max_tokens=800, max_chunks=4)
    context = assembler.assemble(candidates=[candidate1, candidate2])

    resp = await generator.generate_answer(
        query="What does FastRAG use?",
        context=context,
    )

    assert resp.answer != ""
    assert "doc_001" in resp.cited_doc_ids
    assert resp.context_insufficient is False
    assert resp.provider == "mock"


@pytest.mark.asyncio
async def test_rag_generator_empty_context():
    provider = MockLLMProvider()
    generator = RAGGenerator(provider=provider)

    assembler = ContextAssembler(max_tokens=800, max_chunks=4)
    context = assembler.assemble(candidates=[])

    resp = await generator.generate_answer(
        query="What is the capital of Mars?",
        context=context,
    )

    assert resp.context_insufficient is True
    assert len(resp.cited_doc_ids) == 0
    assert resp.answer == "I do not have sufficient information in the provided context to answer this question."


@pytest.mark.asyncio
async def test_rag_generator_abstention_does_not_call_provider():
    """Verify that when context is insufficient, generator returns abstention without invoking provider."""
    # A provider that throws an exception if generate() or stream() is called
    bad_provider = MockLLMProvider(simulate_error=RuntimeError("Provider MUST NOT be called!"))
    generator = RAGGenerator(provider=bad_provider)

    # Empty context
    assembler = ContextAssembler()
    empty_context = assembler.assemble([])
    assert empty_context.context_insufficient is True

    # Generation should succeed with deterministic abstention without raising RuntimeError
    resp = await generator.generate_answer(query="What is the speed of light in vacuum?", context=empty_context)
    assert resp.context_insufficient is True
    assert resp.answer == "I do not have sufficient information in the provided context to answer this question."

    # Streaming should also succeed with deterministic abstention without raising RuntimeError
    chunks: list[StreamChunk] = []
    async for chunk in generator.stream_answer(query="What is the speed of light in vacuum?", context=empty_context):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].delta == "I do not have sufficient information in the provided context to answer this question."
    assert chunks[1].is_final is True


@pytest.mark.asyncio
async def test_rag_generator_preserves_answerable_generation():
    """Verify that answerable queries with valid context continue normal generation and citation."""
    provider = MockLLMProvider()
    generator = RAGGenerator(provider=provider)

    candidate = RetrievalResult(
        chunk_id="chunk_valid",
        document_id="doc_valid",
        rank=1,
        score=8.5,
        score_type="cross_encoder",
        text="FastRAG achieves high retrieval recall via hybrid search.",
        source="doc_valid.txt",
    )
    assembler = ContextAssembler()
    context = assembler.assemble([candidate])
    assert context.context_insufficient is False

    resp = await generator.generate_answer(query="How does FastRAG achieve recall?", context=context)
    assert resp.context_insufficient is False
    assert resp.answer != "I do not have sufficient information in the provided context to answer this question."
    assert "doc_valid" in resp.cited_doc_ids
