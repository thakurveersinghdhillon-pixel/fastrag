#!/usr/bin/env python3
"""FastRAG External Model Baseline Comparison Runner.

Implements Phase 9.3 comparison protocol (docs/PHASE_9_3_COMPARISON_PROTOCOL.md)
and Phase 9.3.2 controlled evaluation enhancements:
- Paired evaluation of FastRAG (RAG mode) vs Direct LLM baseline (closed-book parametric QA).
- Staged execution with conservative request pacing (inter-query delay) to respect rate limits.
- Bounded retries with exponential backoff, uniform jitter, and Retry-After header parsing.
- Explicit cache state tracking (uncached / bypassed) to prevent comparing warm cache to cold API.
- Comprehensive paired statistical analysis: paired mean differences, paired standard deviations,
  paired 95% confidence intervals, and paired two-tailed t-test p-values.
- Missing pair tracking and failure isolation (failed requests are never silently substituted with mock).
- Latency percentiles (median, p95, mean) and verified provider token/cost telemetry.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Sequence, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import httpx
import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

from enum import Enum

from app.core.config import get_settings
from app.core.exceptions import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderQuotaExhaustedError,
)
from app.core.logging_config import get_logger
from app.generation.generator import RAGGenerator
from app.generation.models import GenerationRequest, GenerationResponse, UsageStats
from app.generation.prompts import DEFAULT_SYSTEM_PROMPT
from app.generation.providers import LLMProvider, MockLLMProvider, OpenAICompatibleProvider
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.context_assembler import AssembledContext, ContextAssembler
from app.retrieval.dense import DenseRetriever, MockEmbeddingEncoder, SentenceTransformerEncoder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pipeline import PipelineResponse, RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker, SentenceTransformerCrossEncoder
from evaluation.generation_eval import (
    MetricSummaryStats,
    compute_citation_metrics,
    compute_evidence_completeness,
    compute_groundedness,
    compute_metric_stats,
    compute_token_f1,
    detect_abstention,
    extract_cited_document_ids,
)

logger = get_logger("fastrag.evaluation.comparison")

T = TypeVar("T")

PROMPT_VERSION = "v1.0.0"

# Direct closed-book LLM baseline system prompt (from docs/PHASE_9_3_COMPARISON_PROTOCOL.md)
DIRECT_SYSTEM_PROMPT = """You are a technical question-answering assistant.
Answer the user's question accurately and concisely based on your general technical knowledge.
If you do not know the answer or lack sufficient knowledge to answer accurately, explicitly state:
"I do not have sufficient information to answer this question." Do not fabricate facts."""


# ---------------------------------------------------------------------------
# Bounded Retries & Rate-Limit Handling
# ---------------------------------------------------------------------------

class RateLimitCategory(str, Enum):
    """Classification of LLM API rate-limit and service responses."""

    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    RATE_LIMIT_RPM = "RATE_LIMIT_RPM"
    RATE_LIMIT_TPM = "RATE_LIMIT_TPM"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    SERVER_ERROR = "SERVER_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    UNKNOWN = "UNKNOWN"


def sanitize_error_string(text: str) -> str:
    """Strip API keys, authorization tokens, and query credentials from error strings."""
    if not text:
        return ""
    # Strip Bearer tokens
    sanitized = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    # Strip API key query parameters
    sanitized = re.sub(r"([?&]key=)[^&\s]+", r"\1[REDACTED]", sanitized, flags=re.IGNORECASE)
    # Strip json or header key fields
    sanitized = re.sub(r"(api[_-]?key[\"':\s=]+)[A-Za-z0-9_\-\.]{8,}", r"\1[REDACTED]", sanitized, flags=re.IGNORECASE)
    return sanitized


def classify_rate_limit_error(
    exc: Exception | str,
    status_code: int | None = None,
    response_text: str | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Sanitize and classify an LLM API error into standardized categories.

    Categories:
        QUOTA_EXHAUSTED: Non-transient hard quota ceiling reached (daily limit, free tier request cap, billing requirement).
        RATE_LIMIT_RPM: Transient requests-per-minute rate limit.
        RATE_LIMIT_TPM: Transient tokens-per-minute rate limit.
        AUTHENTICATION_ERROR: Non-transient credential or permissions failure (HTTP 401/403).
        SERVER_ERROR: Transient upstream provider service outage (HTTP 500/502/503/504).
        NETWORK_ERROR: Transient client-side transport/DNS/connection error.
        UNKNOWN: Unclassified error.
    """
    raw_str = str(exc) if isinstance(exc, Exception) else exc
    resp_body = response_text or ""
    resp_headers = headers or {}
    sc = status_code

    if isinstance(exc, httpx.HTTPStatusError):
        if sc is None and exc.response is not None:
            sc = exc.response.status_code
        if not resp_body and exc.response is not None:
            resp_body = exc.response.text
        if not resp_headers and exc.response is not None:
            resp_headers = dict(exc.response.headers)

    combined = f"{raw_str} {resp_body}".strip()
    sanitized_text = sanitize_error_string(combined)
    lower_text = combined.lower()

    # Retry-After header parsing
    retry_after: float | None = None
    retry_after_hdr = resp_headers.get("Retry-After") or resp_headers.get("retry-after")
    if retry_after_hdr:
        try:
            retry_after = float(retry_after_hdr)
        except (ValueError, TypeError):
            pass

    # Extract metric if present
    metric_match = re.search(r"(?:Quota exceeded for metric:?|metric:?)\s*([a-zA-Z0-9_\-\./]+)", combined, re.IGNORECASE)
    metric = metric_match.group(1).rstrip(",;.") if metric_match else None

    # Extract limit if present
    limit_match = re.search(r"(?:limit:?)\s*(\d+)", combined, re.IGNORECASE)
    limit = int(limit_match.group(1)) if limit_match else None

    # 1. Quota Exhaustion (Hard limit)
    is_quota_exc = isinstance(exc, ProviderQuotaExhaustedError)
    quota_keywords = (
        "quota",
        "resource_exhausted",
        "free_tier_requests",
        "insufficient_quota",
        "check your plan and billing details",
        "billing",
        "daily limit",
        "project quota",
    )
    if is_quota_exc or ((sc == 429 or "429" in combined) and any(k in lower_text for k in quota_keywords)):
        summary = f"Hard quota exhausted ({metric or 'quota_limit'})"
        if limit is not None:
            summary += f" [limit: {limit}]"
        return {
            "category": RateLimitCategory.QUOTA_EXHAUSTED,
            "is_transient": False,
            "status_code": sc or 429,
            "metric": metric,
            "limit": limit,
            "retry_after_seconds": retry_after,
            "summary": summary,
            "raw_sanitized": sanitized_text[:300],
        }

    # 2. Authentication / Authorization Error (HTTP 401 / 403)
    if isinstance(exc, ProviderAuthenticationError) or sc in (401, 403) or "unauthorized" in lower_text or "forbidden" in lower_text:
        return {
            "category": RateLimitCategory.AUTHENTICATION_ERROR,
            "is_transient": False,
            "status_code": sc or (401 if "401" in combined else 403),
            "metric": None,
            "limit": None,
            "retry_after_seconds": None,
            "summary": f"Authentication/permission failed (HTTP {sc or '401/403'})",
            "raw_sanitized": sanitized_text[:300],
        }

    # 3. Rate Limit TPM (Tokens per minute)
    tpm_keywords = ("tpm", "tokens per minute", "tokens/minute", "token rate", "token limit")
    if (sc == 429 or "429" in combined) and any(k in lower_text for k in tpm_keywords):
        return {
            "category": RateLimitCategory.RATE_LIMIT_TPM,
            "is_transient": True,
            "status_code": sc or 429,
            "metric": metric,
            "limit": limit,
            "retry_after_seconds": retry_after,
            "summary": "Tokens-per-minute rate limit reached (HTTP 429)",
            "raw_sanitized": sanitized_text[:300],
        }

    # 4. Rate Limit RPM (Requests per minute)
    rpm_keywords = ("rpm", "requests per minute", "requests/minute", "too many requests", "rate limit")
    if sc == 429 or "429" in combined or any(k in lower_text for k in rpm_keywords):
        return {
            "category": RateLimitCategory.RATE_LIMIT_RPM,
            "is_transient": True,
            "status_code": sc or 429,
            "metric": metric,
            "limit": limit,
            "retry_after_seconds": retry_after,
            "summary": "Requests-per-minute rate limit reached (HTTP 429)",
            "raw_sanitized": sanitized_text[:300],
        }

    # 5. Upstream Server Error (HTTP 5xx)
    if sc in (500, 502, 503, 504) or any(s in lower_text for s in ("503", "502", "504", "internal server error", "service unavailable", "bad gateway")):
        return {
            "category": RateLimitCategory.SERVER_ERROR,
            "is_transient": True,
            "status_code": sc,
            "metric": None,
            "limit": None,
            "retry_after_seconds": retry_after,
            "summary": f"Upstream provider server error (HTTP {sc or '5xx'})",
            "raw_sanitized": sanitized_text[:300],
        }

    # 6. Network / Transport Error
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError, OSError)) or any(
        n in lower_text for n in ("connecterror", "connection refused", "timeout", "timed out", "remote protocol")
    ):
        return {
            "category": RateLimitCategory.NETWORK_ERROR,
            "is_transient": True,
            "status_code": None,
            "metric": None,
            "limit": None,
            "retry_after_seconds": None,
            "summary": f"Network transport error: {type(exc).__name__ if isinstance(exc, Exception) else 'TransportError'}",
            "raw_sanitized": sanitized_text[:300],
        }

    # 7. Unknown / Fallback
    return {
        "category": RateLimitCategory.UNKNOWN,
        "is_transient": False,
        "status_code": sc,
        "metric": None,
        "limit": None,
        "retry_after_seconds": None,
        "summary": f"Unclassified error: {sanitized_text[:100]}",
        "raw_sanitized": sanitized_text[:300],
    }


async def retry_with_backoff(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    timeout_seconds: float = 30.0,
    retry_on_exceptions: tuple[type[Exception], ...] = (
        httpx.HTTPStatusError,
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
        ProviderError,
        OSError,
    ),
) -> T:
    """Execute coroutine with bounded exponential backoff, jitter, Retry-After support, and quota fail-fast.

    Args:
        coro_factory: Callable returning a fresh coroutine for each attempt.
        max_retries: Maximum number of retry attempts.
        initial_delay: Base delay in seconds.
        backoff_factor: Multiplier for exponential growth.
        jitter: If True, add uniform random jitter in [0, 1) seconds.
        timeout_seconds: Timeout per individual attempt.
        retry_on_exceptions: Exceptions deemed transient.

    Returns:
        Result of successful coroutine invocation.
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await asyncio.wait_for(coro_factory(), timeout=timeout_seconds)
        except retry_on_exceptions as exc:
            last_exception = exc
            classification = classify_rate_limit_error(exc)

            # Non-transient errors (hard quota exhaustion, auth errors) fail fast immediately
            if not classification.get("is_transient", True):
                logger.error(
                    f"Non-transient error detected ({classification['category'].value}): "
                    f"{classification['summary']}. Aborting retries immediately."
                )
                raise exc

            # Client errors other than 429 are never retried
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code != 429 and exc.response.status_code < 500:
                raise exc

            if attempt >= max_retries:
                logger.error(f"Exceeded max retries ({max_retries}). Last error: {classification['summary']}")
                raise exc

            delay = initial_delay * (backoff_factor ** attempt)
            if jitter:
                delay += random.uniform(0.0, 1.0)

            # Honor Retry-After if present
            retry_after = classification.get("retry_after_seconds")
            if retry_after is not None:
                delay = max(delay, retry_after + 0.5)
            elif isinstance(exc, ProviderError) and "429" in str(exc):
                delay = max(delay, 5.0 * (attempt + 1))

            logger.warning(
                f"Transient error on attempt {attempt + 1}/{max_retries + 1} ({classification['category'].value}): "
                f"{classification['summary']}. Retrying in {delay:.2f}s..."
            )
            await asyncio.sleep(delay)

    raise last_exception or RuntimeError("Unexpected retry loop termination")


# ---------------------------------------------------------------------------
# Data Models for Controlled Paired Comparison
# ---------------------------------------------------------------------------

@dataclass
class QueryComparisonResult:
    """Individual query outcome under a specific evaluation mode."""

    query_id: str
    query: str
    category: str
    split: str
    provenance_status: str
    mode: str  # "rag" or "direct"
    system_name: str
    generated_answer: str
    reference_answer: str | None
    source_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    cited_doc_ids: list[str]
    is_unanswerable: bool
    correct_abstention: bool
    token_f1: float
    token_precision: float
    token_recall: float
    completeness: float
    groundedness: float
    citation_precision: float
    citation_recall: float
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cache_state: str = "uncached_bypassed"
    prompt_version: str = PROMPT_VERSION
    timestamp_utc: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryComparisonResult:
        """Construct QueryComparisonResult from dictionary, filtering unknown keys and supplying defaults."""
        valid_fields = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in valid_fields}
        defaults = {
            "query_id": "",
            "query": "",
            "category": "unknown",
            "split": "unknown",
            "provenance_status": "unknown",
            "mode": "rag",
            "system_name": "unknown",
            "generated_answer": "",
            "reference_answer": None,
            "source_doc_ids": [],
            "retrieved_doc_ids": [],
            "cited_doc_ids": [],
            "is_unanswerable": False,
            "correct_abstention": False,
            "token_f1": 0.0,
            "token_precision": 0.0,
            "token_recall": 0.0,
            "completeness": 0.0,
            "groundedness": 0.0,
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "latency_ms": 0.0,
        }
        for k, def_val in defaults.items():
            if k not in kwargs:
                kwargs[k] = def_val
        return cls(**kwargs)


@dataclass
class ModeEvaluationSummary:
    """Aggregated evaluation summary for one system mode."""

    mode: str
    system_name: str
    total_evaluated: int
    successful_count: int
    failed_count: int
    summary_metrics: dict[str, dict[str, Any]]
    category_metrics: dict[str, dict[str, Any]]
    unanswerable_metrics: dict[str, Any]
    per_query_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PairedMetricSummary:
    """Statistical summary of paired differences (RAG - Direct) on matched queries."""

    metric: str
    rag_mean: float
    direct_mean: float
    mean_delta: float
    std_delta: float
    ci95_margin: float
    ci95_range: list[float]
    p_value: float | None
    paired_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "rag_mean": round(self.rag_mean, 4),
            "direct_mean": round(self.direct_mean, 4),
            "mean_delta": round(self.mean_delta, 4),
            "std_delta": round(self.std_delta, 4),
            "ci95_margin": round(self.ci95_margin, 4),
            "ci95_range": [round(x, 4) for x in self.ci95_range],
            "p_value": round(self.p_value, 4) if self.p_value is not None else None,
            "paired_count": self.paired_count,
        }


@dataclass
class PairedAnalysisReport:
    """Complete paired comparative analysis between FastRAG and Direct LLM."""

    eligible_queries_count: int
    rag_attempted: int
    rag_successful: int
    rag_failed: int
    direct_attempted: int
    direct_successful: int
    direct_failed: int
    paired_successful_count: int
    missing_pairs: list[dict[str, Any]]
    paired_metrics: dict[str, dict[str, Any]]
    telemetry: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExternalComparisonReport:
    """Comprehensive comparison report contrasting FastRAG against baseline."""

    experiment_id: str
    timestamp_utc: str
    dataset_version: str
    split: str
    provenance_filter: str
    provider_type: str
    model_identifier: str
    eligible_queries_count: int
    cache_state: str
    pacing_delay_seconds: float
    modes: dict[str, ModeEvaluationSummary]
    paired_analysis: PairedAnalysisReport
    deltas: dict[str, float]  # RAG minus Direct
    environment_disclosure: dict[str, Any]
    output_path: str | None = None
    is_completed: bool = False
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "timestamp_utc": self.timestamp_utc,
            "dataset_version": self.dataset_version,
            "split": self.split,
            "provenance_filter": self.provenance_filter,
            "provider_type": self.provider_type,
            "model_identifier": self.model_identifier,
            "eligible_queries_count": self.eligible_queries_count,
            "cache_state": self.cache_state,
            "pacing_delay_seconds": self.pacing_delay_seconds,
            "is_completed": self.is_completed,
            "status": self.status,
            "modes": {k: v.to_dict() for k, v in self.modes.items()},
            "paired_analysis": self.paired_analysis.to_dict(),
            "deltas": self.deltas,
            "environment_disclosure": self.environment_disclosure,
            "output_path": self.output_path,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class PreflightSummary:
    """Preflight safety check and dry-run summary."""

    dataset_version: str
    split: str
    provenance_filter: str
    total_eligible: int
    completed_pairs_count: int
    incomplete_pairs_count: int
    unattempted_count: int
    rag_completed_count: int
    direct_completed_count: int
    estimated_rag_calls: int
    estimated_direct_calls: int
    estimated_total_calls: int
    provider_name: str
    model_name: str
    resumed_from: str | None
    api_key_configured: bool
    pacing_delay: float
    temperature: float
    max_tokens: int
    live_mode: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_preflight_summary(summary: PreflightSummary) -> str:
    """Format human-readable preflight safety check and dry-run summary."""
    lines = [
        "=" * 88,
        " FastRAG External Model Comparison: Preflight Safety Check & Dry Run",
        "=" * 88,
        f" Dataset:                   Dataset {summary.dataset_version} (datasets/{summary.dataset_version})",
        f" Split:                     {summary.split.upper()} (provenance: {summary.provenance_filter})",
        f" Eligible Query IDs:        {summary.total_eligible} queries",
        f" Resumed From:              {summary.resumed_from or 'None (starting fresh)'}",
        f" Already Completed Pairs:   {summary.completed_pairs_count} pairs",
        f" Incomplete Pairs:          {summary.incomplete_pairs_count} pairs (RAG valid: {summary.rag_completed_count}, Direct valid: {summary.direct_completed_count})",
        f" Unattempted Pairs:         {summary.unattempted_count} pairs",
        "-" * 88,
        " Estimated Provider Requests:",
        f"   - FastRAG (RAG Mode):    {summary.estimated_rag_calls} requests",
        f"   - Direct LLM Baseline:   {summary.estimated_direct_calls} requests",
        f"   - Total Remaining Calls: {summary.estimated_total_calls} requests",
        "-" * 88,
        " Evaluation Configuration:",
        f"   - Provider:              {summary.provider_name} ({summary.model_name})",
        f"   - API Key Configured:    {'YES (presence verified; active quota/billing UNVERIFIED)' if summary.api_key_configured else 'NO'}",
        f"   - Sampling Parameters:   temperature={summary.temperature}, max_tokens={summary.max_tokens}",
        f"   - Inter-Query Pacing:    {summary.pacing_delay:.1f}s delay",
        f"   - Execution Mode:        {'LIVE EXECUTION (--live)' if summary.live_mode else 'DRY RUN (zero API requests)'}",
        "=" * 88,
    ]
    if not summary.live_mode:
        lines.extend([
            " [!] SAFETY GUARD: Zero live requests made to external provider.",
            "     To execute real API calls, pass the explicit opt-in flag: --live",
            "=" * 88,
        ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Direct LLM Runner (Closed-Book Parametric QA)
# ---------------------------------------------------------------------------

class DirectLLMRunner:
    """Executes closed-book direct LLM query generation with zero retrieval context."""

    def __init__(
        self,
        provider: LLMProvider,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.provider = provider
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    async def generate_direct_answer(
        self,
        query: str,
        category: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[str, UsageStats | None, float, str | None]:
        """Generate closed-book answer directly via LLM provider.

        Returns:
            Tuple of (answer_text, usage_stats, latency_ms, error_message).
        """
        t0 = time.perf_counter()

        if isinstance(self.provider, MockLLMProvider):
            # Deterministic mock closed-book behavior
            if category == "unanswerable_negative":
                ans = "I do not have sufficient information to answer this question."
            elif query in self.provider.fixed_responses:
                ans = self.provider.fixed_responses[query]
            else:
                ans = f"Direct parametric response for technical query: {query.strip()}"

            latency_ms = (time.perf_counter() - t0) * 1000.0
            usage = UsageStats(
                prompt_tokens=len(query.split()) + 30,
                completion_tokens=len(ans.split()),
                total_tokens=len(query.split()) + len(ans.split()) + 30,
                is_estimated=True,
            )
            return ans, usage, latency_ms, None

        if isinstance(self.provider, OpenAICompatibleProvider):
            api_key = self.provider._ensure_api_key()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            url = f"{self.provider._base_url}/chat/completions"
            payload = {
                "model": self.provider._model_name,
                "messages": [
                    {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
                    {"role": "user", "content": query.strip()},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }

            async def _make_call() -> httpx.Response:
                client = self.provider._get_client()
                resp = await client.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
                if resp.status_code in (401, 403) or (
                    resp.status_code == 400 and "api key" in resp.text.lower()
                ):
                    raise ProviderAuthenticationError(
                        f"Authentication failed with provider ({resp.status_code}). Check API credentials."
                    )
                if resp.status_code == 429:
                    err_text = resp.text
                    if any(
                        k in err_text.lower()
                        for k in ("quota", "resource_exhausted", "free_tier_requests", "billing", "insufficient_quota")
                    ):
                        raise ProviderQuotaExhaustedError(
                            f"Provider quota exhausted ({resp.status_code}): {err_text[:400]}"
                        )
                    raise ProviderError(
                        f"Provider rate limit exceeded ({resp.status_code}): {err_text[:400]}"
                    )
                resp.raise_for_status()
                return resp

            try:
                response = await retry_with_backoff(
                    _make_call,
                    max_retries=self.max_retries,
                    timeout_seconds=self.timeout_seconds,
                )
                data = response.json()
                content = data["choices"][0]["message"].get("content") or ""
                u_data = data.get("usage", {})
                usage = UsageStats(
                    prompt_tokens=u_data.get("prompt_tokens", 0),
                    completion_tokens=u_data.get("completion_tokens", 0),
                    total_tokens=u_data.get("total_tokens", 0),
                    is_estimated=False,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return content.strip(), usage, latency_ms, None
            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return "", None, latency_ms, sanitize_error_string(str(exc))

        # Generic provider fallback
        req = GenerationRequest(
            query=query.strip(),
            context_text="",
            selected_chunk_ids=[],
            model=self.provider.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=DIRECT_SYSTEM_PROMPT,
        )
        try:
            gen_resp = await self.provider.generate(req)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return gen_resp.answer, gen_resp.usage, latency_ms, None
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return "", None, latency_ms, sanitize_error_string(str(exc))


# ---------------------------------------------------------------------------
# FastRAG Comparison Runner
# ---------------------------------------------------------------------------

class FastRAGComparisonRunner:
    """Orchestrates side-by-side evaluation of FastRAG (RAG) vs Direct LLM."""

    def __init__(
        self,
        pipeline: RAGPipeline | None = None,
        provider: LLMProvider | None = None,
        dataset_dir: Path | str | None = None,
        results_dir: Path | str | None = None,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
        pacing_delay_seconds: float = 0.0,
    ) -> None:
        self.pipeline = pipeline
        self.provider = provider
        self.direct_runner = DirectLLMRunner(
            provider=provider,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        ) if provider is not None else None
        self.dataset_dir = Path(dataset_dir) if dataset_dir else PROJECT_ROOT / "datasets" / "v2"
        self.results_dir = Path(results_dir) if results_dir else PROJECT_ROOT / "benchmarks" / "results"
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.pacing_delay_seconds = pacing_delay_seconds

    def load_eligible_queries(
        self,
        split: str = "dev",
        provenance_filter: str = "human_reviewed",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load and filter evaluation queries by split and provenance status."""
        split_file = f"queries_{split}.jsonl" if split in {"dev", "test"} else "queries.jsonl"
        q_path = self.dataset_dir / split_file
        if not q_path.is_file():
            q_path = self.dataset_dir / "queries.jsonl"

        with open(q_path, "r", encoding="utf-8") as f:
            all_queries = [json.loads(line) for line in f if line.strip()]

        eligible: list[dict[str, Any]] = []
        for q in all_queries:
            if split in {"dev", "test"} and q.get("split") != split:
                continue
            prov = q.get("annotation_provenance", {}).get("status", "unknown")
            if provenance_filter != "all" and prov != provenance_filter:
                continue
            eligible.append(q)

        if limit is not None and limit > 0:
            eligible = eligible[:limit]

        return eligible

    def load_previous_results(
        self,
        resume_source: Path | str | None,
        split: str,
        provenance_filter: str,
    ) -> tuple[dict[str, QueryComparisonResult], dict[str, QueryComparisonResult], Path | None]:
        """Load and validate completed query results from a previous evaluation file.

        Args:
            resume_source: Path to specific JSON results file, or 'latest' to find most recent.
            split: Target evaluation split ('dev' or 'test').
            provenance_filter: Target provenance filter ('human_reviewed' or 'all').

        Returns:
            Tuple of (completed_rag_dict, completed_direct_dict, file_path_used).
        """
        if not resume_source:
            return {}, {}, None

        target_path: Path | None = None
        if isinstance(resume_source, (str, Path)) and str(resume_source).lower() != "latest":
            target_path = Path(resume_source)
            if not target_path.is_file():
                raise FileNotFoundError(f"Resume results file does not exist: {target_path}")
        else:
            candidates = list(self.results_dir.glob(f"external_comparison_{split}_{provenance_filter}_*.json"))
            if not candidates:
                logger.warning(
                    f"No previous comparison file found in {self.results_dir} matching "
                    f"split '{split}' and provenance '{provenance_filter}'. Starting fresh."
                )
                return {}, {}, None

            expected_model = self.provider.model_name if self.provider else None
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for cand in candidates:
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if (not expected_model or meta.get("model_identifier") == expected_model) and meta.get("dataset_version") == "v2":
                        target_path = cand
                        break
                except Exception:
                    continue

            if not target_path:
                logger.warning(
                    f"No compatible previous comparison file matching model '{expected_model}' "
                    f"found in {self.results_dir}. Starting fresh."
                )
                return {}, {}, None

        logger.info(f"Loading previous results to resume from: {target_path}")
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_split = data.get("split")
        if file_split != split:
            raise ValueError(
                f"Cannot resume from {target_path}: split mismatch (file has '{file_split}', requested '{split}')"
            )

        file_prov = data.get("provenance_filter")
        if file_prov != provenance_filter:
            raise ValueError(
                f"Cannot resume from {target_path}: provenance mismatch (file has '{file_prov}', requested '{provenance_filter}')"
            )

        file_model = data.get("model_identifier")
        expected_model = self.provider.model_name if self.provider else None
        if expected_model and file_model != expected_model:
            raise ValueError(
                f"Cannot resume from {target_path}: model mismatch (file has '{file_model}', requested '{expected_model}')"
            )

        file_dataset_ver = data.get("dataset_version")
        if file_dataset_ver != "v2":
            raise ValueError(
                f"Cannot resume from {target_path}: dataset version mismatch (file has '{file_dataset_ver}', requested 'v2')"
            )

        completed_rag: dict[str, QueryComparisonResult] = {}
        completed_direct: dict[str, QueryComparisonResult] = {}

        modes = data.get("modes", {})
        rag_per_q = modes.get("rag", {}).get("per_query_results", [])
        for item in rag_per_q:
            if item.get("error") is None and item.get("generated_answer") is not None:
                if item.get("prompt_version", PROMPT_VERSION) != PROMPT_VERSION:
                    logger.warning(f"Skipping cached RAG result for {item.get('query_id')}: prompt version mismatch")
                    continue
                res = QueryComparisonResult.from_dict(item)
                completed_rag[res.query_id] = res

        direct_per_q = modes.get("direct", {}).get("per_query_results", [])
        for item in direct_per_q:
            if item.get("error") is None and item.get("generated_answer") is not None:
                if item.get("prompt_version", PROMPT_VERSION) != PROMPT_VERSION:
                    logger.warning(f"Skipping cached Direct result for {item.get('query_id')}: prompt version mismatch")
                    continue
                res = QueryComparisonResult.from_dict(item)
                completed_direct[res.query_id] = res

        logger.info(
            f"Successfully resumed: {len(completed_rag)} completed FastRAG results, "
            f"{len(completed_direct)} completed Direct LLM results."
        )
        return completed_rag, completed_direct, target_path

    def compute_preflight_summary(
        self,
        split: str = "dev",
        provenance_filter: str = "human_reviewed",
        modes_to_run: Sequence[str] = ("rag", "direct"),
        limit: int | None = None,
        resume_source: Path | str | None = None,
        live_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> PreflightSummary:
        """Compute dry-run safety summary before executing live queries."""
        eligible_queries = self.load_eligible_queries(
            split=split,
            provenance_filter=provenance_filter,
            limit=limit,
        )
        total_eligible = len(eligible_queries)

        completed_rag, completed_direct, resume_path = self.load_previous_results(
            resume_source=resume_source,
            split=split,
            provenance_filter=provenance_filter,
        )

        completed_pairs_count = 0
        incomplete_pairs_count = 0
        unattempted_count = 0
        estimated_rag_calls = 0
        estimated_direct_calls = 0

        for q in eligible_queries:
            qid = q["query_id"]
            r_done = qid in completed_rag
            d_done = qid in completed_direct

            if r_done and d_done:
                completed_pairs_count += 1
            elif r_done or d_done:
                incomplete_pairs_count += 1
            else:
                unattempted_count += 1

            if "rag" in modes_to_run and not r_done:
                estimated_rag_calls += 1
            if "direct" in modes_to_run and not d_done:
                estimated_direct_calls += 1

        estimated_total = estimated_rag_calls + estimated_direct_calls

        settings = get_settings()
        api_key_configured = bool(
            settings.llm_api_key if self.provider and self.provider.provider_name == "openai_compatible" else True
        )

        return PreflightSummary(
            dataset_version="v2",
            split=split,
            provenance_filter=provenance_filter,
            total_eligible=total_eligible,
            completed_pairs_count=completed_pairs_count,
            incomplete_pairs_count=incomplete_pairs_count,
            unattempted_count=unattempted_count,
            rag_completed_count=len(completed_rag),
            direct_completed_count=len(completed_direct),
            estimated_rag_calls=estimated_rag_calls,
            estimated_direct_calls=estimated_direct_calls,
            estimated_total_calls=estimated_total,
            provider_name=self.provider.provider_name if self.provider else "unconfigured",
            model_name=self.provider.model_name if self.provider else "unconfigured",
            resumed_from=str(resume_path) if resume_path else None,
            api_key_configured=api_key_configured,
            pacing_delay=self.pacing_delay_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
            live_mode=live_mode,
        )

    def _create_not_run_result(
        self,
        query_record: dict[str, Any],
        mode: str,
        reason: str = "not_run",
    ) -> QueryComparisonResult:
        """Create explicit placeholder result for un-run or interrupted query."""
        now_str = datetime.now(timezone.utc).isoformat()
        return QueryComparisonResult(
            query_id=query_record["query_id"],
            query=query_record["query"],
            category=query_record.get("category", "unknown"),
            split=query_record.get("split", "unknown"),
            provenance_status=query_record.get("annotation_provenance", {}).get("status", "unknown"),
            mode=mode,
            system_name=f"FastRAG ({self.provider.model_name})" if mode == "rag" else f"Direct LLM ({self.provider.model_name})",
            generated_answer="",
            reference_answer=query_record.get("reference_answer"),
            source_doc_ids=query_record.get("source_doc_ids", []),
            retrieved_doc_ids=[],
            cited_doc_ids=[],
            is_unanswerable=query_record.get("category") == "unanswerable_negative",
            correct_abstention=False,
            token_f1=0.0,
            token_precision=0.0,
            token_recall=0.0,
            completeness=0.0,
            groundedness=0.0,
            citation_precision=0.0,
            citation_recall=0.0,
            latency_ms=0.0,
            timestamp_utc=now_str,
            error=reason,
        )

    async def evaluate_query_rag(
        self,
        query_record: dict[str, Any],
        candidate_k: int = 20,
        rerank_top_k: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> QueryComparisonResult:
        """Run single query through full FastRAG pipeline (uncached)."""
        q_id = query_record["query_id"]
        q_text = query_record["query"]
        category = query_record.get("category", "unknown")
        split = query_record.get("split", "unknown")
        prov_status = query_record.get("annotation_provenance", {}).get("status", "unknown")
        ref_answer = query_record.get("reference_answer")
        source_doc_ids = query_record.get("source_doc_ids", [])
        expected_ev = query_record.get("expected_evidence", [])
        rel_judgments = query_record.get("relevance_judgments", {})
        relevant_doc_ids = {doc for doc, rel in rel_judgments.items() if rel > 0}
        is_unanswerable = category == "unanswerable_negative" or len(relevant_doc_ids) == 0

        t0 = time.perf_counter()
        raw_error: str | None = None
        pipe_resp: PipelineResponse | None = None

        async def _exec_pipeline() -> PipelineResponse:
            return await self.pipeline.execute_async(
                query=q_text,
                candidate_k=candidate_k,
                rerank_top_k=rerank_top_k,
                generate_answer=True,
                generation_temperature=temperature,
                generation_max_tokens=max_tokens,
            )

        try:
            pipe_resp = await retry_with_backoff(
                _exec_pipeline,
                max_retries=self.max_retries,
                timeout_seconds=self.timeout_seconds,
            )
        except Exception as exc:
            raw_error = str(exc)
            logger.warning(f"RAG execution error for {q_id}: {exc}")

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if pipe_resp is None or pipe_resp.generation is None:
            gen_answer = ""
            retrieved_doc_ids: list[str] = []
            cited_doc_ids: list[str] = []
            p_tokens: int | None = None
            c_tokens: int | None = None
            tot_tokens: int | None = None
            context_insufficient = False
        else:
            gen = pipe_resp.generation
            gen_answer = gen.answer
            retrieved_doc_ids = [c.document_id for c in pipe_resp.reranked_results]
            cited_doc_ids = extract_cited_document_ids(
                answer=gen.answer,
                context=pipe_resp.context,
                explicit_cited_docs=gen.cited_doc_ids,
            )
            p_tokens = gen.usage.prompt_tokens if gen.usage else None
            c_tokens = gen.usage.completion_tokens if gen.usage else None
            tot_tokens = gen.usage.total_tokens if gen.usage else None
            context_insufficient = gen.context_insufficient

        abstained = detect_abstention(gen_answer, context_insufficient)
        correct_abstention = (is_unanswerable and abstained) or (not is_unanswerable and not abstained)

        f1_dict = compute_token_f1(gen_answer, ref_answer) if not is_unanswerable else {
            "precision": 1.0 if abstained else 0.0,
            "recall": 1.0 if abstained else 0.0,
            "f1": 1.0 if abstained else 0.0,
        }
        completeness = compute_evidence_completeness(gen_answer, expected_ev) if not is_unanswerable else (
            1.0 if abstained else 0.0
        )
        context_str = pipe_resp.context.context_text if pipe_resp else ""
        groundedness = compute_groundedness(gen_answer, context_str)

        citation_scores = compute_citation_metrics(
            cited_doc_ids=cited_doc_ids,
            source_doc_ids=source_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
        )

        now_str = datetime.now(timezone.utc).isoformat()
        return QueryComparisonResult(
            query_id=q_id,
            query=q_text,
            category=category,
            split=split,
            provenance_status=prov_status,
            mode="rag",
            system_name=f"FastRAG ({self.provider.model_name})",
            generated_answer=gen_answer,
            reference_answer=ref_answer,
            source_doc_ids=source_doc_ids,
            retrieved_doc_ids=retrieved_doc_ids,
            cited_doc_ids=cited_doc_ids,
            is_unanswerable=is_unanswerable,
            correct_abstention=correct_abstention,
            token_f1=f1_dict["f1"],
            token_precision=f1_dict["precision"],
            token_recall=f1_dict["recall"],
            completeness=completeness,
            groundedness=groundedness,
            citation_precision=citation_scores["citation_precision"],
            citation_recall=citation_scores["citation_recall"],
            latency_ms=latency_ms,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=tot_tokens,
            cache_state="uncached_bypassed",
            prompt_version=PROMPT_VERSION,
            timestamp_utc=now_str,
            error=raw_error,
        )

    async def evaluate_query_direct(
        self,
        query_record: dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> QueryComparisonResult:
        """Run single query in Direct LLM mode (closed-book, zero retrieval context)."""
        q_id = query_record["query_id"]
        q_text = query_record["query"]
        category = query_record.get("category", "unknown")
        split = query_record.get("split", "unknown")
        prov_status = query_record.get("annotation_provenance", {}).get("status", "unknown")
        ref_answer = query_record.get("reference_answer")
        source_doc_ids = query_record.get("source_doc_ids", [])
        expected_ev = query_record.get("expected_evidence", [])
        rel_judgments = query_record.get("relevance_judgments", {})
        relevant_doc_ids = {doc for doc, rel in rel_judgments.items() if rel > 0}
        is_unanswerable = category == "unanswerable_negative" or len(relevant_doc_ids) == 0

        ans, usage, latency_ms, error = await self.direct_runner.generate_direct_answer(
            query=q_text,
            category=category,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        abstained = detect_abstention(ans)
        correct_abstention = (is_unanswerable and abstained) or (not is_unanswerable and not abstained)

        f1_dict = compute_token_f1(ans, ref_answer) if not is_unanswerable else {
            "precision": 1.0 if abstained else 0.0,
            "recall": 1.0 if abstained else 0.0,
            "f1": 1.0 if abstained else 0.0,
        }
        completeness = compute_evidence_completeness(ans, expected_ev) if not is_unanswerable else (
            1.0 if abstained else 0.0
        )
        # Direct generation has no context: groundedness is 0.0 by definition
        groundedness = 0.0

        # Direct generation produces no retrieval citation tags
        cited_doc_ids = extract_cited_document_ids(answer=ans, context=None)
        citation_scores = compute_citation_metrics(
            cited_doc_ids=cited_doc_ids,
            source_doc_ids=source_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
        )

        p_tokens = usage.prompt_tokens if usage else None
        c_tokens = usage.completion_tokens if usage else None
        tot_tokens = usage.total_tokens if usage else None

        now_str = datetime.now(timezone.utc).isoformat()
        return QueryComparisonResult(
            query_id=q_id,
            query=q_text,
            category=category,
            split=split,
            provenance_status=prov_status,
            mode="direct",
            system_name=f"Direct LLM ({self.provider.model_name})",
            generated_answer=ans,
            reference_answer=ref_answer,
            source_doc_ids=source_doc_ids,
            retrieved_doc_ids=[],
            cited_doc_ids=cited_doc_ids,
            is_unanswerable=is_unanswerable,
            correct_abstention=correct_abstention,
            token_f1=f1_dict["f1"],
            token_precision=f1_dict["precision"],
            token_recall=f1_dict["recall"],
            completeness=completeness,
            groundedness=groundedness,
            citation_precision=citation_scores["citation_precision"],
            citation_recall=citation_scores["citation_recall"],
            latency_ms=latency_ms,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=tot_tokens,
            cache_state="uncached_bypassed",
            prompt_version=PROMPT_VERSION,
            timestamp_utc=now_str,
            error=error,
        )

    def _aggregate_mode_results(
        self,
        mode: str,
        system_name: str,
        results: list[QueryComparisonResult],
    ) -> ModeEvaluationSummary:
        """Compute summary and category statistics for an evaluated mode."""
        successful = [r for r in results if r.error is None]
        total_eval = len(results)
        succ_count = len(successful)
        fail_count = total_eval - succ_count

        metric_keys = [
            "token_f1",
            "token_precision",
            "token_recall",
            "completeness",
            "groundedness",
            "citation_precision",
            "citation_recall",
            "latency_ms",
        ]

        # Use successful queries for valid metric reporting; fall back to all if none succeeded
        eval_pool = successful if successful else results

        summary_metrics: dict[str, dict[str, Any]] = {}
        for k in metric_keys:
            vals = [getattr(r, k) for r in eval_pool]
            summary_metrics[k] = compute_metric_stats(vals).to_dict()

        # Category breakdown
        by_cat: dict[str, list[QueryComparisonResult]] = {}
        for r in eval_pool:
            by_cat.setdefault(r.category, []).append(r)

        category_metrics: dict[str, dict[str, Any]] = {}
        for cat, cat_res in by_cat.items():
            cat_summary: dict[str, Any] = {"count": len(cat_res)}
            for k in metric_keys:
                cat_summary[k] = compute_metric_stats([getattr(r, k) for r in cat_res]).to_dict()
            category_metrics[cat] = cat_summary

        # Unanswerable queries breakdown
        unanswerable_res = [r for r in eval_pool if r.is_unanswerable]
        total_u = len(unanswerable_res)
        abstained_u = sum(1 for r in unanswerable_res if r.correct_abstention)
        unanswerable_metrics = {
            "total": total_u,
            "abstained_count": abstained_u,
            "abstention_accuracy": round(abstained_u / total_u, 4) if total_u > 0 else 1.0,
        }

        return ModeEvaluationSummary(
            mode=mode,
            system_name=system_name,
            total_evaluated=total_eval,
            successful_count=succ_count,
            failed_count=fail_count,
            summary_metrics=summary_metrics,
            category_metrics=category_metrics,
            unanswerable_metrics=unanswerable_metrics,
            per_query_results=[r.to_dict() for r in results],
        )

    def _compute_paired_analysis(
        self,
        eligible_queries: list[dict[str, Any]],
        rag_results: list[QueryComparisonResult],
        direct_results: list[QueryComparisonResult],
    ) -> PairedAnalysisReport:
        """Perform rigorous paired comparison analysis on mutually successful queries."""
        rag_by_id = {r.query_id: r for r in rag_results}
        dir_by_id = {r.query_id: r for r in direct_results}

        paired_successful: list[tuple[QueryComparisonResult, QueryComparisonResult]] = []
        missing_pairs: list[dict[str, Any]] = []

        for q in eligible_queries:
            qid = q["query_id"]
            r_res = rag_by_id.get(qid)
            d_res = dir_by_id.get(qid)
            r_ok = r_res is not None and r_res.error is None
            d_ok = d_res is not None and d_res.error is None

            if r_ok and d_ok:
                paired_successful.append((r_res, d_res))
            else:
                missing_pairs.append({
                    "query_id": qid,
                    "query": q.get("query", ""),
                    "category": q.get("category", "unknown"),
                    "rag_status": "success" if r_ok else (r_res.error if r_res else "not_run"),
                    "direct_status": "success" if d_ok else (d_res.error if d_res else "not_run"),
                })

        metric_keys = [
            "token_f1",
            "token_precision",
            "token_recall",
            "completeness",
            "groundedness",
            "citation_precision",
            "citation_recall",
            "latency_ms",
        ]

        paired_metrics: dict[str, dict[str, Any]] = {}
        n_paired = len(paired_successful)

        for k in metric_keys:
            if n_paired > 0:
                r_vals = [getattr(p[0], k) for p in paired_successful]
                d_vals = [getattr(p[1], k) for p in paired_successful]
                deltas = [r - d for r, d in zip(r_vals, d_vals)]

                r_mean = float(np.mean(r_vals))
                d_mean = float(np.mean(d_vals))
                m_delta = float(np.mean(deltas))
                s_delta = float(np.std(deltas, ddof=1)) if n_paired > 1 else 0.0
                se = s_delta / math.sqrt(n_paired) if n_paired > 0 else 0.0
                ci95 = 1.96 * se

                p_val: float | None = None
                if n_paired >= 3 and scipy_stats is not None:
                    try:
                        res_t = scipy_stats.ttest_rel(r_vals, d_vals)
                        p_val = float(res_t.pvalue) if not math.isnan(res_t.pvalue) else (0.0 if abs(m_delta) > 1e-6 else 1.0)
                    except Exception:
                        p_val = None

                stat = PairedMetricSummary(
                    metric=k,
                    rag_mean=r_mean,
                    direct_mean=d_mean,
                    mean_delta=m_delta,
                    std_delta=s_delta,
                    ci95_margin=ci95,
                    ci95_range=[m_delta - ci95, m_delta + ci95],
                    p_value=p_val,
                    paired_count=n_paired,
                )
                paired_metrics[k] = stat.to_dict()
            else:
                stat = PairedMetricSummary(
                    metric=k,
                    rag_mean=0.0,
                    direct_mean=0.0,
                    mean_delta=0.0,
                    std_delta=0.0,
                    ci95_margin=0.0,
                    ci95_range=[0.0, 0.0],
                    p_value=None,
                    paired_count=0,
                )
                paired_metrics[k] = stat.to_dict()

        # Telemetry aggregation
        def _calc_telemetry(items: list[QueryComparisonResult]) -> dict[str, Any]:
            succ = [x for x in items if x.error is None]
            lats = [x.latency_ms for x in succ] if succ else [0.0]
            pts = [x.prompt_tokens for x in succ if x.prompt_tokens is not None]
            cts = [x.completion_tokens for x in succ if x.completion_tokens is not None]
            tts = [x.total_tokens for x in succ if x.total_tokens is not None]

            # Google AI Studio Gemini Pay-as-you-go reference rate: $0.075 / 1M input, $0.30 / 1M output
            tot_p = sum(pts)
            tot_c = sum(cts)
            cost_usd = (tot_p * 0.075 + tot_c * 0.30) / 1_000_000.0

            return {
                "queries_successful": len(succ),
                "total_prompt_tokens": tot_p,
                "total_completion_tokens": tot_c,
                "total_tokens": sum(tts),
                "estimated_cost_usd_reference": round(cost_usd, 6),
                "free_tier_cost_usd": 0.0,
                "latency_median_ms": round(float(np.percentile(lats, 50)), 2),
                "latency_p95_ms": round(float(np.percentile(lats, 95)), 2),
                "latency_mean_ms": round(float(np.mean(lats)), 2),
                "latency_min_ms": round(float(np.min(lats)), 2),
                "latency_max_ms": round(float(np.max(lats)), 2),
            }

        telemetry = {
            "fastrag_telemetry": _calc_telemetry(rag_results),
            "direct_telemetry": _calc_telemetry(direct_results),
            "pricing_disclosure": "Estimated cost based on Google AI Studio pay-as-you-go ($0.075/1M input, $0.30/1M output). Telemetry measured from actual response usage.",
        }

        return PairedAnalysisReport(
            eligible_queries_count=len(eligible_queries),
            rag_attempted=len(rag_results),
            rag_successful=sum(1 for r in rag_results if r.error is None),
            rag_failed=sum(1 for r in rag_results if r.error is not None),
            direct_attempted=len(direct_results),
            direct_successful=sum(1 for r in direct_results if r.error is None),
            direct_failed=sum(1 for r in direct_results if r.error is not None),
            paired_successful_count=n_paired,
            missing_pairs=missing_pairs,
            paired_metrics=paired_metrics,
            telemetry=telemetry,
        )

    async def run_comparison(
        self,
        experiment_id: str,
        split: str = "dev",
        provenance_filter: str = "human_reviewed",
        modes_to_run: Sequence[str] = ("rag", "direct"),
        candidate_k: int = 20,
        rerank_top_k: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        limit: int | None = None,
        save_results: bool = True,
        paired_interleaved: bool = True,
        resume_source: Path | str | None = None,
        live_mode: bool = False,
    ) -> ExternalComparisonReport:
        """Run complete controlled comparison contrasting FastRAG vs Direct LLM."""
        if not live_mode and self.provider and self.provider.provider_name != "mock":
            raise RuntimeError(
                "Live execution blocked: live_mode is False. "
                "Real external provider evaluations require the explicit opt-in flag: --live"
            )

        eligible_queries = self.load_eligible_queries(
            split=split,
            provenance_filter=provenance_filter,
            limit=limit,
        )
        total_queries = len(eligible_queries)

        completed_rag, completed_direct, resume_path = self.load_previous_results(
            resume_source=resume_source,
            split=split,
            provenance_filter=provenance_filter,
        )

        rag_results: list[QueryComparisonResult] = []
        direct_results: list[QueryComparisonResult] = []

        print(f"[*] Starting Controlled Comparison Experiment: '{experiment_id}'")
        print(f"    - Split: {split.upper()} | Filter: {provenance_filter} | Eligible Queries: {total_queries}")
        print(f"    - Provider: {self.provider.provider_name if self.provider else 'none'} ({self.provider.model_name if self.provider else 'none'})")
        print(f"    - Pacing Delay: {self.pacing_delay_seconds}s | Paired Interleaved: {paired_interleaved}")
        print(f"    - Cache State: UNCACHED (Bypassed)")
        if resume_path:
            print(f"    - Resumed From: {resume_path.name}")
            print(f"      Previously Completed: RAG={len(completed_rag)}, Direct={len(completed_direct)}")
        else:
            print(f"    - Resumed From: None (fresh run)")
        print()

        quota_exhausted = False
        quota_metric: str | None = None
        quota_limit: int | None = None
        quota_summary: str | None = None

        if paired_interleaved and "rag" in modes_to_run and "direct" in modes_to_run:
            for idx, q in enumerate(eligible_queries, start=1):
                qid = q["query_id"]
                has_rag = qid in completed_rag
                has_direct = qid in completed_direct

                if has_rag and has_direct:
                    print(f"[{idx}/{total_queries}] Query {qid}: Both RAG and Direct already completed. Skipping API calls.")
                    rag_results.append(completed_rag[qid])
                    direct_results.append(completed_direct[qid])
                    continue

                if quota_exhausted:
                    rag_results.append(completed_rag.get(qid, self._create_not_run_result(q, "rag", "not_run: quota_exhausted")))
                    direct_results.append(completed_direct.get(qid, self._create_not_run_result(q, "direct", "not_run: quota_exhausted")))
                    continue

                print(f"[{idx}/{total_queries}] Evaluating paired query {qid}...")

                # 1. Evaluate FastRAG (RAG mode)
                if has_rag:
                    print(f"    * FastRAG: Reusing previously completed result.")
                    res_r = completed_rag[qid]
                else:
                    res_r = await self.evaluate_query_rag(
                        query_record=q,
                        candidate_k=candidate_k,
                        rerank_top_k=rerank_top_k,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if res_r.error:
                        print(f"    * FastRAG error: {res_r.error}")
                        clf_r = classify_rate_limit_error(res_r.error)
                        if clf_r["category"] == RateLimitCategory.QUOTA_EXHAUSTED:
                            quota_exhausted = True
                            quota_metric = clf_r.get("metric")
                            quota_limit = clf_r.get("limit")
                            quota_summary = clf_r.get("summary")
                            print(f"\n[!] Hard quota exhaustion detected: {quota_summary}")
                            print("    Stopping live requests immediately to protect API account.")
                rag_results.append(res_r)

                if not has_rag and self.pacing_delay_seconds > 0 and not quota_exhausted:
                    await asyncio.sleep(self.pacing_delay_seconds)

                # 2. Evaluate Direct LLM mode
                if quota_exhausted:
                    if has_direct:
                        res_d = completed_direct[qid]
                    else:
                        res_d = self._create_not_run_result(q, "direct", "not_run: quota_exhausted")
                else:
                    if has_direct:
                        print(f"    * Direct LLM: Reusing previously completed result.")
                        res_d = completed_direct[qid]
                    else:
                        res_d = await self.evaluate_query_direct(
                            query_record=q,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        if res_d.error:
                            print(f"    * Direct LLM error: {res_d.error}")
                            clf_d = classify_rate_limit_error(res_d.error)
                            if clf_d["category"] == RateLimitCategory.QUOTA_EXHAUSTED:
                                quota_exhausted = True
                                quota_metric = clf_d.get("metric")
                                quota_limit = clf_d.get("limit")
                                quota_summary = clf_d.get("summary")
                                print(f"\n[!] Hard quota exhaustion detected: {quota_summary}")
                                print("    Stopping live requests immediately to protect API account.")
                direct_results.append(res_d)

                if not has_direct and self.pacing_delay_seconds > 0 and idx < total_queries and not quota_exhausted:
                    await asyncio.sleep(self.pacing_delay_seconds)

        else:
            if "rag" in modes_to_run:
                print(f"[*] Executing FastRAG (RAG mode) on {total_queries} queries...")
                for idx, q in enumerate(eligible_queries, start=1):
                    qid = q["query_id"]
                    if qid in completed_rag:
                        print(f"[{idx}/{total_queries}] Query {qid}: FastRAG already completed. Skipping API call.")
                        rag_results.append(completed_rag[qid])
                        continue
                    if quota_exhausted:
                        rag_results.append(self._create_not_run_result(q, "rag", "not_run: quota_exhausted"))
                        continue

                    res_r = await self.evaluate_query_rag(
                        query_record=q,
                        candidate_k=candidate_k,
                        rerank_top_k=rerank_top_k,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    rag_results.append(res_r)
                    if res_r.error:
                        clf_r = classify_rate_limit_error(res_r.error)
                        if clf_r["category"] == RateLimitCategory.QUOTA_EXHAUSTED:
                            quota_exhausted = True
                            quota_metric = clf_r.get("metric")
                            quota_limit = clf_r.get("limit")
                            quota_summary = clf_r.get("summary")
                            print(f"\n[!] Hard quota exhaustion detected: {quota_summary}")
                            print("    Stopping live requests immediately to protect API account.")
                    if self.pacing_delay_seconds > 0 and idx < total_queries and not quota_exhausted:
                        await asyncio.sleep(self.pacing_delay_seconds)

            if "direct" in modes_to_run:
                print(f"[*] Executing Direct LLM baseline on {total_queries} queries...")
                for idx, q in enumerate(eligible_queries, start=1):
                    qid = q["query_id"]
                    if qid in completed_direct:
                        print(f"[{idx}/{total_queries}] Query {qid}: Direct LLM already completed. Skipping API call.")
                        direct_results.append(completed_direct[qid])
                        continue
                    if quota_exhausted:
                        direct_results.append(self._create_not_run_result(q, "direct", "not_run: quota_exhausted"))
                        continue

                    res_d = await self.evaluate_query_direct(
                        query_record=q,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    direct_results.append(res_d)
                    if res_d.error:
                        clf_d = classify_rate_limit_error(res_d.error)
                        if clf_d["category"] == RateLimitCategory.QUOTA_EXHAUSTED:
                            quota_exhausted = True
                            quota_metric = clf_d.get("metric")
                            quota_limit = clf_d.get("limit")
                            quota_summary = clf_d.get("summary")
                            print(f"\n[!] Hard quota exhaustion detected: {quota_summary}")
                            print("    Stopping live requests immediately to protect API account.")
                    if self.pacing_delay_seconds > 0 and idx < total_queries and not quota_exhausted:
                        await asyncio.sleep(self.pacing_delay_seconds)

        modes_summary: dict[str, ModeEvaluationSummary] = {}
        if rag_results:
            modes_summary["rag"] = self._aggregate_mode_results(
                mode="rag",
                system_name=f"FastRAG ({self.provider.model_name})" if self.provider else "FastRAG",
                results=rag_results,
            )
        if direct_results:
            modes_summary["direct"] = self._aggregate_mode_results(
                mode="direct",
                system_name=f"Direct LLM ({self.provider.model_name})" if self.provider else "Direct LLM",
                results=direct_results,
            )

        paired_analysis = self._compute_paired_analysis(
            eligible_queries=eligible_queries,
            rag_results=rag_results,
            direct_results=direct_results,
        )

        deltas: dict[str, float] = {}
        for m_key, p_stat in paired_analysis.paired_metrics.items():
            deltas[m_key] = p_stat["mean_delta"]

        total_eval = len(eligible_queries)
        rag_succ = sum(1 for r in rag_results if r.error is None)
        dir_succ = sum(1 for r in direct_results if r.error is None)
        paired_succ = paired_analysis.paired_successful_count

        if paired_interleaved and "rag" in modes_to_run and "direct" in modes_to_run:
            is_completed = (paired_succ == total_eval) and not quota_exhausted
        else:
            is_completed = (
                (rag_succ == total_eval if "rag" in modes_to_run else True)
                and (dir_succ == total_eval if "direct" in modes_to_run else True)
                and not quota_exhausted
            )

        if is_completed:
            status = "completed"
        elif quota_exhausted:
            status = "partial_quota_exhausted"
        else:
            status = "partial_incomplete"

        now_utc = datetime.now(timezone.utc).isoformat()
        env_disclosure = {
            "provider_type": self.provider.provider_name if self.provider else "unconfigured",
            "model_identifier": self.provider.model_name if self.provider else "unconfigured",
            "is_mock": isinstance(self.provider, MockLLMProvider) if self.provider else False,
            "chatgpt_api_available": bool(os.environ.get("OPENAI_API_KEY")),
            "gemini_api_key_configured": bool(get_settings().llm_api_key),
            "cache_state": "uncached_bypassed",
            "pacing_delay_seconds": self.pacing_delay_seconds,
            "paired_interleaved": paired_interleaved,
            "live_mode": live_mode,
            "resumed_from": str(resume_path) if resume_path else None,
            "consumer_chatgpt_web_ui_note": (
                "Consumer ChatGPT Web UI is excluded per Protocol Section 6; "
                "reproducible baseline requires official OpenAI API endpoint."
            ),
            "quota_exhausted": quota_exhausted,
            "quota_metric": quota_metric,
            "quota_limit": quota_limit,
            "quota_summary": quota_summary,
        }

        report = ExternalComparisonReport(
            experiment_id=experiment_id,
            timestamp_utc=now_utc,
            dataset_version="v2",
            split=split,
            provenance_filter=provenance_filter,
            provider_type=self.provider.provider_name if self.provider else "unconfigured",
            model_identifier=self.provider.model_name if self.provider else "unconfigured",
            eligible_queries_count=total_queries,
            cache_state="uncached_bypassed",
            pacing_delay_seconds=self.pacing_delay_seconds,
            modes=modes_summary,
            paired_analysis=paired_analysis,
            deltas=deltas,
            environment_disclosure=env_disclosure,
            is_completed=is_completed,
            status=status,
        )

        if save_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            ts_slug = now_utc.replace(":", "-").replace(".", "_")
            prov_label = self.provider.provider_name if self.provider else "unconfigured"
            out_file = self.results_dir / f"external_comparison_{split}_{provenance_filter}_{prov_label}_{ts_slug}.json"
            report.output_path = str(out_file)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(report.to_json(indent=2))
            print(f"\n[+] Results successfully saved to: {out_file}")

        return report


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def format_comparison_table(report: ExternalComparisonReport) -> str:
    """Format comparative side-by-side results table for console and docs."""
    p_analysis = report.paired_analysis
    status_label = "COMPLETED BENCHMARK" if report.is_completed else f"PARTIAL EVALUATION ({report.status.upper()})"
    lines = [
        "=" * 96,
        f" FastRAG vs External Model Controlled Comparison: {report.experiment_id}",
        f" Status:    [{status_label}]",
        f" Timestamp: {report.timestamp_utc} | Split: {report.split.upper()} | Filter: {report.provenance_filter}",
        f" Provider:  {report.provider_type} ({report.model_identifier}) | Cache: {report.cache_state}",
        f" Evaluated Queries: {report.eligible_queries_count} | Paired Complete: {p_analysis.paired_successful_count} | Missing Pairs: {len(p_analysis.missing_pairs)}",
    ]
    resumed_from = report.environment_disclosure.get("resumed_from")
    if resumed_from:
        lines.append(f" Resumed From:      {resumed_from}")
    lines.append("=" * 96)

    if report.environment_disclosure.get("quota_exhausted"):
        lines.extend([
            f" [!] HARD QUOTA EXHAUSTION: {report.environment_disclosure.get('quota_summary')}",
            "     Requests halted early to protect provider account. Incomplete pairs logged below.",
            "=" * 96,
        ])

    lines.extend([
        f" {'Metric':<22} | {'FastRAG (RAG)':<14} | {'Direct LLM':<14} | {'Paired Delta':<14} | {'95% CI':<14} | {'p-val':<7}",
        "-" * 96,
    ])

    metric_labels = [
        ("token_f1", "Token F1"),
        ("token_precision", "Token Precision"),
        ("token_recall", "Token Recall"),
        ("completeness", "Completeness"),
        ("groundedness", "Groundedness"),
        ("citation_precision", "Citation Precision"),
        ("citation_recall", "Citation Recall"),
        ("latency_ms", "Latency (ms)"),
    ]

    for key, label in metric_labels:
        p_stat = p_analysis.paired_metrics.get(key, {})
        rag_val = f"{p_stat.get('rag_mean', 0.0):.4f}" if p_stat else "N/A"
        dir_val = f"{p_stat.get('direct_mean', 0.0):.4f}" if p_stat else "N/A"
        delta_val = f"{p_stat.get('mean_delta', 0.0):+.4f}" if p_stat else "N/A"
        ci_str = f"±{p_stat.get('ci95_margin', 0.0):.4f}" if p_stat else "N/A"
        p_str = f"{p_stat.get('p_value'):.4f}" if p_stat.get('p_value') is not None else "N/A"
        lines.append(f" {label:<22} | {rag_val:<14} | {dir_val:<14} | {delta_val:<14} | {ci_str:<14} | {p_str:<7}")

    lines.append("=" * 96)

    # Missing pairs breakdown
    if p_analysis.missing_pairs:
        lines.append(f" Missing Pairs Breakdown ({len(p_analysis.missing_pairs)} incomplete):")
        for mp in p_analysis.missing_pairs:
            lines.append(f"   * {mp['query_id']} ({mp['category']}) -> RAG: {mp['rag_status'][:25]} | Direct: {mp['direct_status'][:25]}")
        lines.append("-" * 96)

    # Telemetry and latency percentiles
    telem = p_analysis.telemetry
    rag_t = telem.get("fastrag_telemetry", {})
    dir_t = telem.get("direct_telemetry", {})
    lines.append(" Latency & Token Telemetry:")
    lines.append(
        f"   * FastRAG Latency: median={rag_t.get('latency_median_ms')}ms | p95={rag_t.get('latency_p95_ms')}ms | mean={rag_t.get('latency_mean_ms')}ms"
    )
    lines.append(
        f"   * Direct  Latency: median={dir_t.get('latency_median_ms')}ms | p95={dir_t.get('latency_p95_ms')}ms | mean={dir_t.get('latency_mean_ms')}ms"
    )
    lines.append(
        f"   * Token Usage: FastRAG={rag_t.get('total_tokens')} total ({rag_t.get('total_prompt_tokens')} prompt, {rag_t.get('total_completion_tokens')} comp) | "
        f"Direct={dir_t.get('total_tokens')} total ({dir_t.get('total_prompt_tokens')} prompt, {dir_t.get('total_completion_tokens')} comp)"
    )
    lines.append(
        f"   * Cost Disclosure: Free Tier = $0.00 | Pay-as-you-go Ref = ${rag_t.get('estimated_cost_usd_reference', 0.0):.6f} (RAG) vs ${dir_t.get('estimated_cost_usd_reference', 0.0):.6f} (Direct)"
    )
    lines.append("=" * 96)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FastRAG vs External Model Baseline Comparison (Phase 9.3)."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "v2",
        help="Path to dataset directory (default: datasets/v2)",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Evaluation split (default: dev). NOTE: test split is quarantined until protocol freeze.",
    )
    parser.add_argument(
        "--provenance",
        choices=["human_reviewed", "all"],
        default="human_reviewed",
        help="Filter by annotation provenance (default: human_reviewed)",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "rag", "direct"],
        default="all",
        help="Which systems to evaluate (default: all -> both RAG and Direct baseline)",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "gemini", "openai"],
        default="mock",
        help="LLM provider (default: mock)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier override (e.g., gemini-3.8-flash, gpt-4o, mock-llm-v1)",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Candidates to retrieve in RAG mode (default: 20)",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=10,
        help="Candidates after CrossEncoder reranking (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max generation tokens (default: 1024)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for transient API errors (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--pacing-delay",
        type=float,
        default=None,
        help="Inter-query pacing delay in seconds (default: 4.0 for real APIs, 0.0 for mock)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries evaluated (for fast dry runs and staged pilots)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output JSON file to benchmarks/results/",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Custom experiment identifier",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Explicit opt-in to execute live external API calls. If omitted for real providers, defaults to safe dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Force safe dry-run mode (zero API requests, displays preflight summary and exits).",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from previous comparison results. Pass file path or omit argument to auto-resume from 'latest'.",
    )
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()

    if args.split == "test":
        print(
            "[!] WARNING: The held-out test split is strictly quarantined per protocol. "
            "Proceeding only for final evaluation.",
            file=sys.stderr,
        )

    # 1. Inspect provider & credentials
    settings = get_settings()
    if args.provider == "mock":
        model_name = args.model or "mock-llm-v1"
        provider: LLMProvider = MockLLMProvider(model_name=model_name)
        pacing_delay = args.pacing_delay if args.pacing_delay is not None else 0.0
    elif args.provider in ("gemini", "openai"):
        model_name = args.model or (settings.llm_model if args.provider == "gemini" else "gpt-4o")
        api_key = settings.llm_api_key if args.provider == "gemini" else os.environ.get("OPENAI_API_KEY", "")
        base_url = settings.llm_base_url if args.provider == "gemini" else "https://api.openai.com/v1"
        provider = OpenAICompatibleProvider(
            api_key=api_key or "dry_run_placeholder",
            model_name=model_name,
            base_url=base_url,
            timeout_seconds=args.timeout,
        )
        pacing_delay = args.pacing_delay if args.pacing_delay is not None else 4.0
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    modes_to_run = ["rag", "direct"] if args.mode == "all" else [args.mode]

    # 2. Preflight Safety Check & Dry-Run Evaluation
    # If dry-run requested OR live mode not opted in for real provider:
    if args.dry_run or (not args.live and args.provider != "mock"):
        preflight_runner = FastRAGComparisonRunner(
            pipeline=None,
            provider=provider,
            dataset_dir=args.dataset_dir,
            results_dir=PROJECT_ROOT / "benchmarks" / "results",
            max_retries=args.max_retries,
            timeout_seconds=args.timeout,
            pacing_delay_seconds=pacing_delay,
        )
        summary = preflight_runner.compute_preflight_summary(
            split=args.split,
            provenance_filter=args.provenance,
            modes_to_run=modes_to_run,
            limit=args.limit,
            resume_source=args.resume,
            live_mode=False,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(format_preflight_summary(summary))
        return 0

    # 3. For live runs, verify API key
    if args.provider in ("gemini", "openai"):
        api_key = settings.llm_api_key if args.provider == "gemini" else os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print(
                f"Error: API key for provider '{args.provider}' is not configured.",
                file=sys.stderr,
            )
            return 1

    # 4. Initialize pipeline components
    indexes_root = PROJECT_ROOT / "datasets" / "indexes"
    bm25_dir = indexes_root / "bm25"
    dense_dir = indexes_root / "dense"

    bm25 = BM25Retriever()
    bm25.load_index(bm25_dir)

    encoder = SentenceTransformerEncoder()
    dense = DenseRetriever(encoder=encoder)
    dense.load_index(dense_dir)

    base_retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        dense_candidate_count=args.candidate_k,
        bm25_candidate_count=args.candidate_k,
        top_k=args.candidate_k,
    )

    reranker = CrossEncoderReranker(
        scorer=SentenceTransformerCrossEncoder(),
        default_top_k=args.rerank_top_k,
    )

    generator = RAGGenerator(provider=provider)
    assembler = ContextAssembler()

    pipeline = RAGPipeline(
        retriever=base_retriever,
        reranker=reranker,
        context_assembler=assembler,
        generator=generator,
    )
    pipeline.warmup()

    runner = FastRAGComparisonRunner(
        pipeline=pipeline,
        provider=provider,
        dataset_dir=args.dataset_dir,
        results_dir=PROJECT_ROOT / "benchmarks" / "results",
        max_retries=args.max_retries,
        timeout_seconds=args.timeout,
        pacing_delay_seconds=pacing_delay,
    )

    summary = runner.compute_preflight_summary(
        split=args.split,
        provenance_filter=args.provenance,
        modes_to_run=modes_to_run,
        limit=args.limit,
        resume_source=args.resume,
        live_mode=True,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(format_preflight_summary(summary))
    print()

    exp_id = args.experiment_id or f"comparison_{args.split}_{args.provenance}_{args.provider}"

    try:
        report = await runner.run_comparison(
            experiment_id=exp_id,
            split=args.split,
            provenance_filter=args.provenance,
            modes_to_run=modes_to_run,
            candidate_k=args.candidate_k,
            rerank_top_k=args.rerank_top_k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            limit=args.limit,
            save_results=not args.no_save,
            paired_interleaved=True,
            resume_source=args.resume,
            live_mode=True,
        )
        print("\n" + format_comparison_table(report))
        return 0
    except Exception as exc:
        print(f"\nComparison run failed with error: {exc}", file=sys.stderr)
        return 1
    finally:
        await pipeline.aclose()


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
