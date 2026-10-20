"""FastRAG LLM Provider Abstraction.

Implements provider protocols, an async OpenAI-compatible client powered by HTTPX,
and a deterministic mock provider for zero-dependency offline testing and development.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable

import httpx

from app.core.config import get_settings
from app.core.exceptions import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderQuotaExhaustedError,
    ProviderTimeoutError,
    StreamInterruptedError,
)
from app.core.logging_config import get_logger
from app.generation.models import (
    GenerationDiagnostics,
    GenerationRequest,
    GenerationResponse,
    StreamChunk,
    UsageStats,
)
from app.generation.prompts import build_grounded_prompt

logger = get_logger("fastrag.generation.providers")


class LLMProvider(ABC):
    """Abstract interface for LLM text generation and real-time streaming."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Asynchronously generate a complete answer for a request."""
        pass

    @abstractmethod
    def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        """Asynchronously yield incremental stream chunks as they arrive."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Configured model name."""
        pass


class MockLLMProvider(LLMProvider):
    """Deterministic, zero-dependency LLM provider for unit tests and local development.

    Supports custom response mappings, simulated latency, streaming chunk emission,
    grounded citation generation, and controlled error injection.
    """

    def __init__(
        self,
        model_name: str = "mock-llm-v1",
        chunk_delay_s: float = 0.0,
        fixed_responses: dict[str, str] | None = None,
        custom_fn: Callable[[GenerationRequest], str] | None = None,
        simulate_error: Exception | None = None,
    ) -> None:
        self._model_name = model_name
        self.chunk_delay_s = chunk_delay_s
        self.fixed_responses = fixed_responses or {}
        self.custom_fn = custom_fn
        self.simulate_error = simulate_error

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _determine_answer(self, request: GenerationRequest) -> str:
        if self.simulate_error is not None:
            raise self.simulate_error

        # Check explicit query mapping
        if request.query in self.fixed_responses:
            return self.fixed_responses[request.query]

        if self.custom_fn is not None:
            return self.custom_fn(request)

        # Handle empty or insufficient context
        if request.metadata.get("context_insufficient") or not request.context_text.strip():
            return "I do not have sufficient information in the provided context to answer this question."

        # Extract citation IDs present in context (e.g. [Doc-1], [Doc-2])
        citations = re.findall(r"\[Doc-\d+\]", request.context_text)
        primary_cite = citations[0] if citations else "[Doc-1]"

        # Create realistic synthesized grounded response using context snippets
        lines = [line.strip() for line in request.context_text.split("\n") if line.strip() and not line.startswith("[Doc-")]
        snippet = lines[0] if lines else "the retrieved factual evidence"
        if len(snippet) > 80:
            snippet = snippet[:80] + "..."

        return f"Based on {primary_cite}, {snippet}. This directly addresses the query '{request.query}'."

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        start_t = time.perf_counter()
        answer = self._determine_answer(request)

        if self.chunk_delay_s > 0:
            await asyncio.sleep(self.chunk_delay_s)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        # Estimate tokens
        p_tokens = len(request.query.split()) + len(request.context_text.split())
        c_tokens = len(answer.split())

        usage = UsageStats(
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens,
            is_estimated=True,
        )
        diagnostics = GenerationDiagnostics(
            ttft_ms=elapsed_ms,
            generation_ms=elapsed_ms,
            total_duration_ms=elapsed_ms,
            chunk_count=1,
            model_name=self._model_name,
            provider_name="mock",
        )

        cited = list(set(re.findall(r"\[Doc-\d+\]", answer)))
        is_insufficient = bool(request.metadata.get("context_insufficient")) or not bool(request.context_text.strip())

        return GenerationResponse(
            query=request.query,
            answer=answer,
            model=self._model_name,
            provider="mock",
            finish_reason="stop",
            usage=usage,
            diagnostics=diagnostics,
            cited_doc_ids=cited,
            context_insufficient=is_insufficient,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        start_t = time.perf_counter()
        answer = self._determine_answer(request)

        words = answer.split(" ")
        first_chunk = True
        ttft_ms: float | None = None
        chunk_count = 0

        for idx, word in enumerate(words):
            if self.chunk_delay_s > 0:
                await asyncio.sleep(self.chunk_delay_s)

            delta = word if idx == len(words) - 1 else word + " "
            chunk_count += 1

            if first_chunk:
                ttft_ms = (time.perf_counter() - start_t) * 1000.0
                yield StreamChunk(delta=delta, is_first=True, is_final=False)
                first_chunk = False
            else:
                yield StreamChunk(delta=delta, is_first=False, is_final=False)

        total_elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        generation_ms = total_elapsed_ms - (ttft_ms or 0.0)

        p_tokens = len(request.query.split()) + len(request.context_text.split())
        c_tokens = len(answer.split())
        usage = UsageStats(
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens,
            is_estimated=True,
        )
        diagnostics = GenerationDiagnostics(
            ttft_ms=ttft_ms,
            generation_ms=generation_ms,
            total_duration_ms=total_elapsed_ms,
            chunk_count=chunk_count,
            model_name=self._model_name,
            provider_name="mock",
        )

        yield StreamChunk(
            delta="",
            is_first=False,
            is_final=True,
            finish_reason="stop",
            usage=usage,
            diagnostics=diagnostics,
        )


class OpenAICompatibleProvider(LLMProvider):
    """Asynchronous client for OpenAI-compatible chat endpoints using HTTPX.

    Works with OpenAI, LocalAI, vLLM, Ollama, Groq, Together, DeepSeek, or any
    standard OpenAI chat completions API (`/chat/completions`).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.llm_api_key
        self._model_name = model_name or settings.llm_model
        self._base_url = (base_url or settings.llm_base_url).rstrip("/")
        self._timeout_seconds = timeout_seconds or settings.llm_timeout_seconds
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        """Return persistent AsyncClient instance with connection pooling."""
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=60.0)
            self._client = httpx.AsyncClient(timeout=self._timeout_seconds, limits=limits)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Close persistent HTTP client session."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> OpenAICompatibleProvider:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_api_key(self) -> str:
        """Validate API key presence without logging its value."""
        if not self._api_key or not self._api_key.strip():
            raise ProviderAuthenticationError(
                "LLM API key is not configured. Provide it in constructor or set "
                "FASTRAG_LLM_API_KEY / OPENAI_API_KEY environment variable."
            )
        return self._api_key.strip()

    def _build_messages(self, request: GenerationRequest) -> list[dict[str, str]]:
        system_prompt, user_content = build_grounded_prompt(
            query=request.query,
            context_text=request.context_text,
            custom_system_prompt=request.system_prompt,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        api_key = self._ensure_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": request.model or self._model_name,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }

        start_t = time.perf_counter()
        try:
            client = self._get_client()
            response = await client.post(url, headers=headers, json=payload)

            if response.status_code in (401, 403) or (
                response.status_code == 400 and "api key" in response.text.lower()
            ):
                raise ProviderAuthenticationError(
                    f"Authentication failed with provider ({response.status_code}). Check API credentials."
                )
            if response.status_code == 429:
                err_text = response.text
                if any(
                    k in err_text.lower()
                    for k in ("quota", "resource_exhausted", "free_tier_requests", "billing", "insufficient_quota")
                ):
                    raise ProviderQuotaExhaustedError(
                        f"Provider quota exhausted ({response.status_code}): {err_text[:400]}"
                    )
                raise ProviderError(
                    f"Provider rate limit exceeded ({response.status_code}): {err_text[:400]}"
                )
            if response.status_code != 200:
                raise ProviderError(
                    f"Provider returned error status {response.status_code}: {response.text[:200]}"
                )

            data = response.json()
            total_elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            choice = data["choices"][0]
            answer = choice["message"].get("content") or ""
            finish_reason = choice.get("finish_reason", "stop")

            usage_data = data.get("usage", {})
            usage = UsageStats(
                prompt_tokens=usage_data.get("prompt_tokens"),
                completion_tokens=usage_data.get("completion_tokens"),
                total_tokens=usage_data.get("total_tokens"),
                is_estimated=False,
            )

            diagnostics = GenerationDiagnostics(
                ttft_ms=total_elapsed_ms,
                generation_ms=total_elapsed_ms,
                total_duration_ms=total_elapsed_ms,
                chunk_count=1,
                model_name=request.model or self._model_name,
                provider_name=self.provider_name,
            )

            cited = list(set(re.findall(r"\[Doc-\d+\]", answer)))

            return GenerationResponse(
                query=request.query,
                answer=answer,
                model=request.model or self._model_name,
                provider=self.provider_name,
                finish_reason=finish_reason,
                usage=usage,
                diagnostics=diagnostics,
                cited_doc_ids=cited,
                context_insufficient=not bool(request.context_text.strip()),
            )

        except httpx.TimeoutException as exc:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            raise ProviderTimeoutError(
                f"Provider request timed out after {self._timeout_seconds:.1f}s ({elapsed_ms:.1f}ms elapsed)"
            ) from exc
        except (ProviderAuthenticationError, ProviderError, ProviderTimeoutError):
            raise
        except Exception as exc:
            raise ProviderError(f"Unexpected provider network error: {exc}") from exc

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        api_key = self._ensure_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": request.model or self._model_name,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        start_t = time.perf_counter()
        first_token = True
        ttft_ms: float | None = None
        chunk_count = 0
        final_usage: UsageStats | None = None
        finish_reason: str = "stop"

        try:
            client = self._get_client()
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code in (401, 403):
                        raise ProviderAuthenticationError(
                            f"Authentication failed with provider ({response.status_code}). Check API credentials."
                        )
                    if response.status_code != 200:
                        body_err = await response.aread()
                        body_str = body_err.decode("utf-8", errors="ignore")
                        if response.status_code == 400 and "api key" in body_str.lower():
                            raise ProviderAuthenticationError(
                                f"Authentication failed with provider ({response.status_code}). Check API credentials."
                            )
                        raise ProviderError(
                            f"Provider stream failed with status {response.status_code}: {body_str[:200]}"
                        )

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue

                        raw_data = line[6:].strip()
                        if raw_data == "[DONE]":
                            break

                        try:
                            chunk_json = json.loads(raw_data)
                        except json.JSONDecodeError:
                            continue

                        # Extract usage if provided in stream_options
                        if "usage" in chunk_json and chunk_json["usage"]:
                            u = chunk_json["usage"]
                            final_usage = UsageStats(
                                prompt_tokens=u.get("prompt_tokens"),
                                completion_tokens=u.get("completion_tokens"),
                                total_tokens=u.get("total_tokens"),
                                is_estimated=False,
                            )

                        choices = chunk_json.get("choices", [])
                        if not choices:
                            continue

                        choice = choices[0]
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]

                        delta_content = choice.get("delta", {}).get("content", "")
                        if delta_content:
                            chunk_count += 1
                            if first_token:
                                ttft_ms = (time.perf_counter() - start_t) * 1000.0
                                yield StreamChunk(delta=delta_content, is_first=True, is_final=False)
                                first_token = False
                            else:
                                yield StreamChunk(delta=delta_content, is_first=False, is_final=False)

            total_elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            gen_ms = total_elapsed_ms - (ttft_ms or 0.0)

            diagnostics = GenerationDiagnostics(
                ttft_ms=ttft_ms,
                generation_ms=gen_ms,
                total_duration_ms=total_elapsed_ms,
                chunk_count=chunk_count,
                model_name=request.model or self._model_name,
                provider_name=self.provider_name,
            )

            # Emit final completion chunk
            yield StreamChunk(
                delta="",
                is_first=False,
                is_final=True,
                finish_reason=finish_reason,
                usage=final_usage or UsageStats(is_estimated=True),
                diagnostics=diagnostics,
            )

        except httpx.TimeoutException as exc:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            raise ProviderTimeoutError(
                f"Provider stream timed out after {self._timeout_seconds:.1f}s ({elapsed_ms:.1f}ms elapsed)"
            ) from exc
        except (ProviderAuthenticationError, ProviderError, ProviderTimeoutError):
            raise
        except asyncio.CancelledError:
            logger.info("Generation stream cancelled by client")
            raise
        except Exception as exc:
            raise StreamInterruptedError(f"Streaming failed unexpectedly: {exc}") from exc
