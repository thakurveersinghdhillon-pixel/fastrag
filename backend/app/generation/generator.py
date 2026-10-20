"""FastRAG Grounded Answer Generator.

Coordinates prompt assembly, provider dispatch, citation verification, and real-time streaming
strictly grounded on Phase 6 AssembledContext.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, AsyncIterator

from app.core.config import get_settings
from app.core.exceptions import GenerationError
from app.core.logging_config import get_logger
from app.generation.models import (
    GenerationDiagnostics,
    GenerationRequest,
    GenerationResponse,
    StreamChunk,
    UsageStats,
)
from app.generation.providers import (
    LLMProvider,
    MockLLMProvider,
    OpenAICompatibleProvider,
)
from app.retrieval.context_assembler import AssembledContext

logger = get_logger("fastrag.generation.generator")

INSUFFICIENT_CONTEXT_ANSWER = (
    "I do not have sufficient information in the provided context to answer this question."
)


class RAGGenerator:
    """Coordinates grounded answer generation and real-time token streaming.

    Consumes Phase 6 AssembledContext, enforces system grounding and citation rules,
    and interfaces with pluggable LLM providers (Mock or OpenAI-compatible).
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        default_temperature: float | None = None,
        default_max_tokens: int | None = None,
        default_system_prompt: str | None = None,
    ) -> None:
        settings = get_settings()
        if provider is not None:
            self.provider = provider
        else:
            if settings.llm_provider.lower() == "mock":
                self.provider = MockLLMProvider(model_name=settings.llm_model)
            else:
                self.provider = OpenAICompatibleProvider(
                    api_key=settings.llm_api_key,
                    model_name=settings.llm_model,
                    base_url=settings.llm_base_url,
                    timeout_seconds=settings.llm_timeout_seconds,
                )

        self.default_temperature = (
            default_temperature if default_temperature is not None else settings.llm_temperature
        )
        self.default_max_tokens = (
            default_max_tokens if default_max_tokens is not None else settings.llm_max_tokens
        )
        self.default_system_prompt = default_system_prompt

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    def _prepare_request(
        self,
        query: str,
        context: AssembledContext,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> GenerationRequest:
        """Construct normalized GenerationRequest from user query and AssembledContext."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty")

        eff_temp = temperature if temperature is not None else self.default_temperature
        eff_max = max_tokens if max_tokens is not None else self.default_max_tokens
        eff_prompt = system_prompt or self.default_system_prompt

        return GenerationRequest(
            query=query.strip(),
            context_text=context.context_text,
            selected_chunk_ids=list(context.selected_chunk_ids),
            model=model or self.model_name,
            temperature=eff_temp,
            max_tokens=eff_max,
            system_prompt=eff_prompt,
            metadata={
                "total_chunks_selected": context.total_chunks_selected,
                "approximate_token_count": context.approximate_token_count,
                "budget_exceeded": context.budget_exceeded,
                "selected_document_ids": context.selected_document_ids,
                "context_insufficient": context.context_insufficient,
            },
        )

    async def generate_answer(
        self,
        query: str,
        context: AssembledContext,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> GenerationResponse:
        """Asynchronously generate a grounded answer from assembled context."""
        if context.context_insufficient or not context.context_text.strip():
            logger.info("Context insufficient or empty; returning deterministic abstention response.")
            return GenerationResponse(
                query=query.strip(),
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                model=model or self.model_name,
                provider=self.provider_name,
                finish_reason="stop",
                usage=UsageStats(
                    prompt_tokens=0,
                    completion_tokens=len(INSUFFICIENT_CONTEXT_ANSWER.split()),
                    total_tokens=len(INSUFFICIENT_CONTEXT_ANSWER.split()),
                    is_estimated=True,
                ),
                diagnostics=GenerationDiagnostics(
                    ttft_ms=0.0,
                    generation_ms=0.0,
                    total_duration_ms=0.0,
                    chunk_count=0,
                    model_name=model or self.model_name,
                    provider_name=self.provider_name,
                ),
                cited_doc_ids=[],
                context_insufficient=True,
            )

        request = self._prepare_request(
            query=query,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            model=model,
        )

        response = await self.provider.generate(request)

        # Cross-reference cited tags [Doc-N] against available citation markers
        if context.selected_chunks:
            valid_citations = {c.citation_id: c.document_id for c in context.selected_chunks}
            cited_tags = re.findall(r"\[Doc-\d+\]", response.answer)
            resolved_docs = [
                valid_citations[tag] for tag in cited_tags if tag in valid_citations
            ]
            response.cited_doc_ids = list(dict.fromkeys(resolved_docs))

        return response

    async def stream_answer(
        self,
        query: str,
        context: AssembledContext,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Asynchronously stream incremental answer tokens from assembled context."""
        if context.context_insufficient or not context.context_text.strip():
            logger.info("Context insufficient or empty; streaming deterministic abstention response.")
            yield StreamChunk(
                delta=INSUFFICIENT_CONTEXT_ANSWER,
                is_first=True,
                is_final=False,
            )
            yield StreamChunk(
                delta="",
                is_first=False,
                is_final=True,
                finish_reason="stop",
                usage=UsageStats(
                    prompt_tokens=0,
                    completion_tokens=len(INSUFFICIENT_CONTEXT_ANSWER.split()),
                    total_tokens=len(INSUFFICIENT_CONTEXT_ANSWER.split()),
                    is_estimated=True,
                ),
                diagnostics=GenerationDiagnostics(
                    ttft_ms=0.0,
                    generation_ms=0.0,
                    total_duration_ms=0.0,
                    chunk_count=1,
                    model_name=model or self.model_name,
                    provider_name=self.provider_name,
                ),
            )
            return

        request = self._prepare_request(
            query=query,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            model=model,
        )

        async for chunk in self.provider.stream(request):
            yield chunk

    def generate_sync(
        self,
        query: str,
        context: AssembledContext,
        **kwargs: Any,
    ) -> GenerationResponse:
        """Synchronous wrapper for generate_answer."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.generate_answer(query=query, context=context, **kwargs),
                )
                return future.result()
        else:
            return asyncio.run(self.generate_answer(query=query, context=context, **kwargs))

    async def aclose(self) -> None:
        """Gracefully close provider resources."""
        if hasattr(self.provider, "aclose"):
            await self.provider.aclose()

    async def __aenter__(self) -> RAGGenerator:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()
