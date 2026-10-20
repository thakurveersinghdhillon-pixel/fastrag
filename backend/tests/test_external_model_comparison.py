"""Unit tests for Phase 9.3 External Model Baseline Comparison.

Verifies:
- Exponential backoff and retry behavior on transient errors
- Immediate failure without retry on non-retryable errors
- Retry-After header parsing on HTTP 429 rate limits
- Direct LLM runner prompt construction and execution
- Human-reviewed provenance filtering on dev split
- Paired comparison analysis, missing pair detection, and paired statistical testing
- Latency percentiles and token telemetry computation
- End-to-end comparison execution (RAG vs Direct) and JSON report generation
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import json
from unittest.mock import AsyncMock, MagicMock
import pytest
import httpx

from app.core.exceptions import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderQuotaExhaustedError,
)
from app.generation.generator import RAGGenerator
from app.generation.providers import MockLLMProvider, OpenAICompatibleProvider
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.context_assembler import ContextAssembler
from app.retrieval.dense import DenseRetriever, MockEmbeddingEncoder
from app.retrieval.pipeline import RAGPipeline
from scripts.run_external_model_comparison import (
    DIRECT_SYSTEM_PROMPT,
    DirectLLMRunner,
    ExternalComparisonReport,
    FastRAGComparisonRunner,
    PreflightSummary,
    PROMPT_VERSION,
    QueryComparisonResult,
    RateLimitCategory,
    classify_rate_limit_error,
    format_comparison_table,
    format_preflight_summary,
    retry_with_backoff,
    sanitize_error_string,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_V2_DIR = PROJECT_ROOT / "datasets" / "v2"


@pytest.mark.asyncio
async def test_retry_with_backoff_success() -> None:
    """Verify retry helper succeeds on first attempt."""
    calls = 0

    async def _op() -> str:
        nonlocal calls
        calls += 1
        return "success"

    res = await retry_with_backoff(_op, max_retries=3, initial_delay=0.01)
    assert res == "success"
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_with_backoff_transient_recovery() -> None:
    """Verify retry recovers after transient 503 error."""
    calls = 0

    async def _op() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            req = httpx.Request("POST", "http://test")
            resp = httpx.Response(503, request=req)
            raise httpx.HTTPStatusError("Service Unavailable", request=req, response=resp)
        return "recovered"

    res = await retry_with_backoff(_op, max_retries=2, initial_delay=0.01, jitter=False)
    assert res == "recovered"
    assert calls == 2


@pytest.mark.asyncio
async def test_retry_with_backoff_retry_after_header() -> None:
    """Verify retry parses and honors Retry-After header on 429."""
    calls = 0

    async def _op() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            req = httpx.Request("POST", "http://test")
            resp = httpx.Response(429, headers={"Retry-After": "0.05"}, request=req)
            raise httpx.HTTPStatusError("Too Many Requests", request=req, response=resp)
        return "recovered_after_429"

    t0 = asyncio.get_event_loop().time()
    res = await retry_with_backoff(_op, max_retries=2, initial_delay=0.01, jitter=False)
    elapsed = asyncio.get_event_loop().time() - t0

    assert res == "recovered_after_429"
    assert calls == 2
    assert elapsed >= 0.05  # Honored the Retry-After delay


@pytest.mark.asyncio
async def test_retry_with_backoff_non_retryable_error() -> None:
    """Verify non-retryable error (e.g. 401 Unauthorized) raises immediately."""
    calls = 0

    async def _op() -> str:
        nonlocal calls
        calls += 1
        req = httpx.Request("POST", "http://test")
        resp = httpx.Response(401, request=req)
        raise httpx.HTTPStatusError("Unauthorized", request=req, response=resp)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await retry_with_backoff(_op, max_retries=3, initial_delay=0.01)

    assert exc_info.value.response.status_code == 401
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_with_backoff_exhaustion() -> None:
    """Verify retry loop exhausts and raises last exception."""
    calls = 0

    async def _op() -> str:
        nonlocal calls
        calls += 1
        raise httpx.TimeoutException("Connection timed out")

    with pytest.raises(httpx.TimeoutException):
        await retry_with_backoff(_op, max_retries=2, initial_delay=0.01, jitter=False)

    assert calls == 3


@pytest.mark.asyncio
async def test_direct_llm_runner_mock() -> None:
    """Verify DirectLLMRunner handles answerable and unanswerable queries."""
    provider = MockLLMProvider(model_name="mock-test")
    runner = DirectLLMRunner(provider=provider)

    # 1. Answerable query
    ans, usage, lat, err = await runner.generate_direct_answer(
        query="What is BM25?",
        category="factual_direct",
    )
    assert "Direct parametric response" in ans
    assert err is None
    assert lat > 0.0
    assert usage is not None

    # 2. Unanswerable query
    ans_u, usage_u, lat_u, err_u = await runner.generate_direct_answer(
        query="What is the quantum flux capacitor in doc_999?",
        category="unanswerable_negative",
    )
    assert ans_u == "I do not have sufficient information to answer this question."
    assert err_u is None


def test_comparison_runner_query_loading() -> None:
    """Verify loading and provenance filtering of dev queries."""
    bm25 = BM25Retriever()
    dense = DenseRetriever(encoder=MockEmbeddingEncoder())
    generator = RAGGenerator(provider=MockLLMProvider())
    assembler = ContextAssembler()
    pipeline = RAGPipeline(retriever=bm25, context_assembler=assembler, generator=generator)

    runner = FastRAGComparisonRunner(
        pipeline=pipeline,
        provider=generator.provider,
        dataset_dir=DATASETS_V2_DIR,
    )

    # Exactly 35 human-reviewed dev queries
    dev_human = runner.load_eligible_queries(split="dev", provenance_filter="human_reviewed")
    assert len(dev_human) == 35

    # All dev queries (60 total)
    dev_all = runner.load_eligible_queries(split="dev", provenance_filter="all")
    assert len(dev_all) == 60


def test_paired_analysis_with_missing_pairs() -> None:
    """Verify paired analysis tracks completed pairs and isolates missing pairs."""
    bm25 = BM25Retriever()
    dense = DenseRetriever(encoder=MockEmbeddingEncoder())
    generator = RAGGenerator(provider=MockLLMProvider())
    assembler = ContextAssembler()
    pipeline = RAGPipeline(retriever=bm25, context_assembler=assembler, generator=generator)

    runner = FastRAGComparisonRunner(
        pipeline=pipeline,
        provider=generator.provider,
        dataset_dir=DATASETS_V2_DIR,
    )

    eligible = [
        {"query_id": "q1", "query": "query 1", "category": "factual_direct"},
        {"query_id": "q2", "query": "query 2", "category": "factual_direct"},
    ]

    # q1 succeeds in both; q2 fails in direct
    r1 = QueryComparisonResult(
        query_id="q1", query="query 1", category="factual_direct", split="dev",
        provenance_status="human_reviewed", mode="rag", system_name="RAG",
        generated_answer="ans1", reference_answer="ref1", source_doc_ids=["doc_001"],
        retrieved_doc_ids=["doc_001"], cited_doc_ids=["doc_001"], is_unanswerable=False,
        correct_abstention=True, token_f1=0.8, token_precision=0.8, token_recall=0.8,
        completeness=1.0, groundedness=1.0, citation_precision=1.0, citation_recall=1.0,
        latency_ms=100.0, prompt_tokens=500, completion_tokens=30, total_tokens=530,
        error=None,
    )
    r2 = QueryComparisonResult(
        query_id="q2", query="query 2", category="factual_direct", split="dev",
        provenance_status="human_reviewed", mode="rag", system_name="RAG",
        generated_answer="ans2", reference_answer="ref2", source_doc_ids=["doc_002"],
        retrieved_doc_ids=["doc_002"], cited_doc_ids=["doc_002"], is_unanswerable=False,
        correct_abstention=True, token_f1=0.6, token_precision=0.6, token_recall=0.6,
        completeness=1.0, groundedness=1.0, citation_precision=1.0, citation_recall=1.0,
        latency_ms=120.0, prompt_tokens=500, completion_tokens=30, total_tokens=530,
        error=None,
    )

    d1 = QueryComparisonResult(
        query_id="q1", query="query 1", category="factual_direct", split="dev",
        provenance_status="human_reviewed", mode="direct", system_name="Direct",
        generated_answer="ans1_d", reference_answer="ref1", source_doc_ids=["doc_001"],
        retrieved_doc_ids=[], cited_doc_ids=[], is_unanswerable=False,
        correct_abstention=True, token_f1=0.4, token_precision=0.4, token_recall=0.4,
        completeness=0.5, groundedness=0.0, citation_precision=0.0, citation_recall=0.0,
        latency_ms=50.0, prompt_tokens=50, completion_tokens=20, total_tokens=70,
        error=None,
    )
    d2 = QueryComparisonResult(
        query_id="q2", query="query 2", category="factual_direct", split="dev",
        provenance_status="human_reviewed", mode="direct", system_name="Direct",
        generated_answer="", reference_answer="ref2", source_doc_ids=["doc_002"],
        retrieved_doc_ids=[], cited_doc_ids=[], is_unanswerable=False,
        correct_abstention=False, token_f1=0.0, token_precision=0.0, token_recall=0.0,
        completeness=0.0, groundedness=0.0, citation_precision=0.0, citation_recall=0.0,
        latency_ms=5000.0, prompt_tokens=None, completion_tokens=None, total_tokens=None,
        error="Client error '429 Too Many Requests'",
    )

    p_analysis = runner._compute_paired_analysis(
        eligible_queries=eligible,
        rag_results=[r1, r2],
        direct_results=[d1, d2],
    )

    assert p_analysis.eligible_queries_count == 2
    assert p_analysis.rag_successful == 2
    assert p_analysis.direct_successful == 1
    assert p_analysis.direct_failed == 1
    assert p_analysis.paired_successful_count == 1
    assert len(p_analysis.missing_pairs) == 1
    assert p_analysis.missing_pairs[0]["query_id"] == "q2"
    assert "429" in p_analysis.missing_pairs[0]["direct_status"]

    # Check paired difference on valid pair (q1 only)
    f1_stat = p_analysis.paired_metrics["token_f1"]
    assert f1_stat["rag_mean"] == 0.8
    assert f1_stat["direct_mean"] == 0.4
    assert f1_stat["mean_delta"] == 0.4


@pytest.mark.asyncio
async def test_comparison_runner_end_to_end(tmp_path: Path) -> None:
    """Verify end-to-end comparison execution of FastRAG vs Direct LLM."""
    bm25 = BM25Retriever()
    bm25.load_index(PROJECT_ROOT / "datasets" / "indexes" / "bm25")

    dense = DenseRetriever(encoder=MockEmbeddingEncoder())
    dense.load_index(PROJECT_ROOT / "datasets" / "indexes" / "dense")

    provider = MockLLMProvider(model_name="mock-eval")
    generator = RAGGenerator(provider=provider)
    assembler = ContextAssembler()

    pipeline = RAGPipeline(retriever=bm25, context_assembler=assembler, generator=generator)

    runner = FastRAGComparisonRunner(
        pipeline=pipeline,
        provider=provider,
        dataset_dir=DATASETS_V2_DIR,
        results_dir=tmp_path,
        pacing_delay_seconds=0.0,
    )

    report = await runner.run_comparison(
        experiment_id="test_controlled_comparison",
        split="dev",
        provenance_filter="human_reviewed",
        modes_to_run=["rag", "direct"],
        candidate_k=5,
        limit=3,
        save_results=True,
        paired_interleaved=True,
    )

    assert isinstance(report, ExternalComparisonReport)
    assert report.eligible_queries_count == 3
    assert report.cache_state == "uncached_bypassed"
    assert "rag" in report.modes
    assert "direct" in report.modes

    # FastRAG should have groundedness and citations; Direct should not
    rag_mode = report.modes["rag"]
    direct_mode = report.modes["direct"]

    assert rag_mode.summary_metrics["groundedness"]["mean"] > 0.0
    assert direct_mode.summary_metrics["groundedness"]["mean"] == 0.0
    assert rag_mode.summary_metrics["citation_recall"]["mean"] > 0.0
    assert direct_mode.summary_metrics["citation_recall"]["mean"] == 0.0

    # Paired analysis checks
    p_report = report.paired_analysis
    assert p_report.paired_successful_count == 3
    assert len(p_report.missing_pairs) == 0
    assert "token_f1" in p_report.paired_metrics
    assert p_report.telemetry["fastrag_telemetry"]["total_tokens"] > 0

    # Table format
    table = format_comparison_table(report)
    assert "FastRAG (RAG)" in table
    assert "Direct LLM" in table
    assert "Paired Delta" in table
    assert "Latency & Token Telemetry" in table

    # Verify JSON file exists
    assert report.output_path is not None
    assert Path(report.output_path).is_file()


def test_classify_rate_limit_error_google_quota_exhausted() -> None:
    """Verify classification of exact Google Generative Language quota exhaustion error."""
    google_err = (
        'Provider returned error status 429: {\n  "error": {\n    "code": 429,\n'
        '    "message": "You exceeded your current quota, please check your plan and billing details. '
        "For more information on this error, head here: https://ai.dev/rate-limit. "
        "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
        'limit: 20",\n    "status": "RESOURCE_EXHAUSTED"\n  }\n}'
    )
    clf = classify_rate_limit_error(google_err)

    assert clf["category"] == RateLimitCategory.QUOTA_EXHAUSTED
    assert clf["is_transient"] is False
    assert clf["status_code"] == 429
    assert clf["metric"] == "generativelanguage.googleapis.com/generate_content_free_tier_requests"
    assert clf["limit"] == 20
    assert "Hard quota exhausted" in clf["summary"]


def test_classify_rate_limit_error_rpm_and_tpm() -> None:
    """Verify differentiation between transient RPM and TPM limits."""
    # RPM limit
    rpm_err = "Client error '429 Too Many Requests': Rate limit reached for requests per minute (RPM)."
    clf_rpm = classify_rate_limit_error(rpm_err)
    assert clf_rpm["category"] == RateLimitCategory.RATE_LIMIT_RPM
    assert clf_rpm["is_transient"] is True

    # TPM limit
    tpm_err = "429 Too Many Requests: Tokens per minute (TPM) limit exceeded. Limit: 100000"
    clf_tpm = classify_rate_limit_error(tpm_err)
    assert clf_tpm["category"] == RateLimitCategory.RATE_LIMIT_TPM
    assert clf_tpm["is_transient"] is True


def test_classify_rate_limit_error_auth_and_server() -> None:
    """Verify classification of authentication failures and 5xx server errors."""
    auth_err = ProviderAuthenticationError("Authentication failed with provider (401). Check API credentials.")
    clf_auth = classify_rate_limit_error(auth_err)
    assert clf_auth["category"] == RateLimitCategory.AUTHENTICATION_ERROR
    assert clf_auth["is_transient"] is False

    req = httpx.Request("POST", "http://test")
    resp_503 = httpx.Response(503, request=req, text="Service Unavailable: Model overloaded")
    server_err = httpx.HTTPStatusError("503 Service Unavailable", request=req, response=resp_503)
    clf_server = classify_rate_limit_error(server_err)
    assert clf_server["category"] == RateLimitCategory.SERVER_ERROR
    assert clf_server["is_transient"] is True


def test_sanitize_error_string() -> None:
    """Verify secrets and API keys are redacted from error strings."""
    secret_key = "AIzaSyD-SecretKey1234567890abcdef"
    raw_str = f"Error calling url 'https://generativelanguage.googleapis.com/v1beta?key={secret_key}' with Bearer {secret_key}"
    sanitized = sanitize_error_string(raw_str)

    assert secret_key not in sanitized
    assert "[REDACTED]" in sanitized


@pytest.mark.asyncio
async def test_retry_with_backoff_quota_exhaustion_immediate_abort() -> None:
    """Verify retry loop aborts immediately on ProviderQuotaExhaustedError without retries."""
    calls = 0

    async def _quota_exhausted_call() -> str:
        nonlocal calls
        calls += 1
        raise ProviderQuotaExhaustedError(
            "Provider quota exhausted (429): Quota exceeded for metric: "
            "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20"
        )

    with pytest.raises(ProviderQuotaExhaustedError):
        await retry_with_backoff(_quota_exhausted_call, max_retries=3, initial_delay=0.01)

    assert calls == 1  # Aborted immediately on first attempt without retrying!


@pytest.mark.asyncio
async def test_retry_with_backoff_http429_quota_immediate_abort() -> None:
    """Verify retry loop aborts immediately on HTTP 429 containing quota exhaustion."""
    calls = 0

    async def _http_quota_call() -> str:
        nonlocal calls
        calls += 1
        req = httpx.Request("POST", "http://test")
        resp = httpx.Response(
            429,
            request=req,
            text='{"error": {"code": 429, "message": "You exceeded your current quota. Quota exceeded for metric: requests, limit: 20"}}',
        )
        raise httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp)

    with pytest.raises(httpx.HTTPStatusError):
        await retry_with_backoff(_http_quota_call, max_retries=3, initial_delay=0.01)

    assert calls == 1  # Aborted immediately on attempt 1!


@pytest.mark.asyncio
async def test_runner_quota_exhaustion_stops_and_records_missing_pairs(tmp_path: Path) -> None:
    """Verify runner stops cleanly on quota exhaustion, preserves completed pairs, and tracks un-run pairs."""
    bm25 = BM25Retriever()
    bm25.load_index(PROJECT_ROOT / "datasets" / "indexes" / "bm25")

    dense = DenseRetriever(encoder=MockEmbeddingEncoder())
    dense.load_index(PROJECT_ROOT / "datasets" / "indexes" / "dense")

    # Mock provider that fails with quota exhausted on second call
    provider = MockLLMProvider(model_name="mock-quota-test")
    generator = RAGGenerator(provider=provider)
    assembler = ContextAssembler()

    pipeline = RAGPipeline(retriever=bm25, context_assembler=assembler, generator=generator)

    runner = FastRAGComparisonRunner(
        pipeline=pipeline,
        provider=provider,
        dataset_dir=DATASETS_V2_DIR,
        results_dir=tmp_path,
        pacing_delay_seconds=0.0,
    )

    call_count = 0
    orig_evaluate_rag = runner.evaluate_query_rag

    async def _mock_evaluate_rag(query_record: dict, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            now_str = datetime.now(timezone.utc).isoformat()
            return QueryComparisonResult(
                query_id=query_record["query_id"],
                query=query_record["query"],
                category=query_record.get("category", "unknown"),
                split="dev",
                provenance_status="human_reviewed",
                mode="rag",
                system_name="FastRAG (mock)",
                generated_answer="",
                reference_answer=None,
                source_doc_ids=[],
                retrieved_doc_ids=[],
                cited_doc_ids=[],
                is_unanswerable=False,
                correct_abstention=False,
                token_f1=0.0,
                token_precision=0.0,
                token_recall=0.0,
                completeness=0.0,
                groundedness=0.0,
                citation_precision=0.0,
                citation_recall=0.0,
                latency_ms=10.0,
                timestamp_utc=now_str,
                error="Provider quota exhausted (429): Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20",
            )
        return await orig_evaluate_rag(query_record, **kwargs)

    runner.evaluate_query_rag = _mock_evaluate_rag

    report = await runner.run_comparison(
        experiment_id="test_quota_stop",
        split="dev",
        provenance_filter="human_reviewed",
        modes_to_run=["rag", "direct"],
        candidate_k=5,
        limit=4,
        save_results=True,
        paired_interleaved=True,
    )

    # Total queries requested was 4, but quota exhausted on query 2
    assert report.environment_disclosure["quota_exhausted"] is True
    assert "generativelanguage.googleapis.com" in (report.environment_disclosure["quota_metric"] or "")
    # Query 1 succeeded as a complete pair
    assert report.paired_analysis.paired_successful_count == 1
    # Remaining 3 queries are in missing_pairs (q2 failed, q3 and q4 not run)
    # Check that q3 and q4 are marked with not_run
    statuses = [mp["rag_status"] for mp in report.paired_analysis.missing_pairs]
    assert any("not_run" in s for s in statuses)
    assert report.is_completed is False
    assert report.status == "partial_quota_exhausted"


@pytest.mark.asyncio
async def test_dry_run_sends_zero_api_requests(tmp_path: Path) -> None:
    """Verify dry-run mode and preflight summary generate accurate plans with zero API calls."""
    provider = MockLLMProvider(model_name="mock-preflight-test")
    runner = FastRAGComparisonRunner(
        pipeline=None,
        provider=provider,
        dataset_dir=DATASETS_V2_DIR,
        results_dir=tmp_path,
        pacing_delay_seconds=0.0,
    )

    summary = runner.compute_preflight_summary(
        split="dev",
        provenance_filter="human_reviewed",
        modes_to_run=["rag", "direct"],
        limit=5,
        live_mode=False,
    )

    assert summary.total_eligible == 5
    assert summary.completed_pairs_count == 0
    assert summary.incomplete_pairs_count == 0
    assert summary.unattempted_count == 5
    assert summary.estimated_rag_calls == 5
    assert summary.estimated_direct_calls == 5
    assert summary.estimated_total_calls == 10
    assert summary.live_mode is False

    text_summary = format_preflight_summary(summary)
    assert "DRY RUN (zero API requests)" in text_summary
    assert "Eligible Query IDs:        5 queries" in text_summary
    assert "Total Remaining Calls: 10 requests" in text_summary

    # Safety guard: run_comparison without live_mode=True for real provider must raise RuntimeError
    real_style_provider = OpenAICompatibleProvider(
        api_key="test_key",
        model_name="gpt-4o",
        base_url="https://api.openai.com/v1",
    )
    guard_runner = FastRAGComparisonRunner(
        pipeline=None,
        provider=real_style_provider,
        dataset_dir=DATASETS_V2_DIR,
        results_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="Live execution blocked: live_mode is False"):
        await guard_runner.run_comparison(
            experiment_id="test_guard",
            split="dev",
            provenance_filter="human_reviewed",
            live_mode=False,
        )


@pytest.mark.asyncio
async def test_resume_skips_completed_pairs(tmp_path: Path) -> None:
    """Verify resuming skips queries where both RAG and Direct have valid completed results."""
    # Create previous results file with Q1 and Q2 complete
    prev_file = tmp_path / "external_comparison_dev_human_reviewed_mock-test_prev.json"
    prev_data = {
        "experiment_id": "prev_pilot",
        "timestamp_utc": "2026-10-17T00:00:00+00:00",
        "dataset_version": "v2",
        "split": "dev",
        "provenance_filter": "human_reviewed",
        "provider_type": "mock",
        "model_identifier": "mock-test",
        "modes": {
            "rag": {
                "per_query_results": [
                    {
                        "query_id": "q2_dev_001",
                        "query": "Test Q1",
                        "category": "factual_single_hop",
                        "split": "dev",
                        "provenance_status": "human_reviewed",
                        "mode": "rag",
                        "system_name": "FastRAG (mock-test)",
                        "generated_answer": "Valid RAG Answer 1",
                        "prompt_version": PROMPT_VERSION,
                        "error": None,
                        "token_f1": 0.85,
                        "token_precision": 0.85,
                        "token_recall": 0.85,
                        "completeness": 0.9,
                        "groundedness": 0.9,
                        "citation_precision": 1.0,
                        "citation_recall": 1.0,
                        "latency_ms": 100.0,
                    },
                    {
                        "query_id": "q2_dev_002",
                        "query": "Test Q2",
                        "category": "multi_hop_synthesis",
                        "split": "dev",
                        "provenance_status": "human_reviewed",
                        "mode": "rag",
                        "system_name": "FastRAG (mock-test)",
                        "generated_answer": "Valid RAG Answer 2",
                        "prompt_version": PROMPT_VERSION,
                        "error": None,
                        "token_f1": 0.75,
                        "token_precision": 0.75,
                        "token_recall": 0.75,
                        "completeness": 0.8,
                        "groundedness": 0.8,
                        "citation_precision": 1.0,
                        "citation_recall": 1.0,
                        "latency_ms": 120.0,
                    },
                ]
            },
            "direct": {
                "per_query_results": [
                    {
                        "query_id": "q2_dev_001",
                        "query": "Test Q1",
                        "category": "factual_single_hop",
                        "split": "dev",
                        "provenance_status": "human_reviewed",
                        "mode": "direct",
                        "system_name": "Direct LLM (mock-test)",
                        "generated_answer": "Valid Direct Answer 1",
                        "prompt_version": PROMPT_VERSION,
                        "error": None,
                        "token_f1": 0.60,
                        "token_precision": 0.60,
                        "token_recall": 0.60,
                        "completeness": 0.5,
                        "groundedness": 0.0,
                        "citation_precision": 0.0,
                        "citation_recall": 0.0,
                        "latency_ms": 50.0,
                    },
                    {
                        "query_id": "q2_dev_002",
                        "query": "Test Q2",
                        "category": "multi_hop_synthesis",
                        "split": "dev",
                        "provenance_status": "human_reviewed",
                        "mode": "direct",
                        "system_name": "Direct LLM (mock-test)",
                        "generated_answer": "Valid Direct Answer 2",
                        "prompt_version": PROMPT_VERSION,
                        "error": None,
                        "token_f1": 0.50,
                        "token_precision": 0.50,
                        "token_recall": 0.50,
                        "completeness": 0.4,
                        "groundedness": 0.0,
                        "citation_precision": 0.0,
                        "citation_recall": 0.0,
                        "latency_ms": 55.0,
                    },
                ]
            },
        },
    }
    with open(prev_file, "w", encoding="utf-8") as f:
        json.dump(prev_data, f)

    provider = MockLLMProvider(model_name="mock-test")
    runner = FastRAGComparisonRunner(
        pipeline=None,
        provider=provider,
        dataset_dir=DATASETS_V2_DIR,
        results_dir=tmp_path,
        pacing_delay_seconds=0.0,
    )

    rag_eval_calls: list[str] = []
    direct_eval_calls: list[str] = []

    async def _mock_eval_rag(query_record: dict, **kwargs):
        qid = query_record["query_id"]
        rag_eval_calls.append(qid)
        now_str = datetime.now(timezone.utc).isoformat()
        return QueryComparisonResult(
            query_id=qid,
            query=query_record["query"],
            category=query_record.get("category", "unknown"),
            split="dev",
            provenance_status="human_reviewed",
            mode="rag",
            system_name="FastRAG (mock-test)",
            generated_answer="New RAG Answer",
            reference_answer=query_record.get("reference_answer"),
            source_doc_ids=[],
            retrieved_doc_ids=[],
            cited_doc_ids=[],
            is_unanswerable=False,
            correct_abstention=False,
            token_f1=0.8,
            token_precision=0.8,
            token_recall=0.8,
            completeness=0.8,
            groundedness=0.8,
            citation_precision=1.0,
            citation_recall=1.0,
            latency_ms=80.0,
            timestamp_utc=now_str,
            error=None,
        )

    async def _mock_eval_direct(query_record: dict, **kwargs):
        qid = query_record["query_id"]
        direct_eval_calls.append(qid)
        now_str = datetime.now(timezone.utc).isoformat()
        return QueryComparisonResult(
            query_id=qid,
            query=query_record["query"],
            category=query_record.get("category", "unknown"),
            split="dev",
            provenance_status="human_reviewed",
            mode="direct",
            system_name="Direct LLM (mock-test)",
            generated_answer="New Direct Answer",
            reference_answer=query_record.get("reference_answer"),
            source_doc_ids=[],
            retrieved_doc_ids=[],
            cited_doc_ids=[],
            is_unanswerable=False,
            correct_abstention=False,
            token_f1=0.6,
            token_precision=0.6,
            token_recall=0.6,
            completeness=0.6,
            groundedness=0.0,
            citation_precision=0.0,
            citation_recall=0.0,
            latency_ms=40.0,
            timestamp_utc=now_str,
            error=None,
        )

    runner.evaluate_query_rag = _mock_eval_rag
    runner.evaluate_query_direct = _mock_eval_direct

    # Run on first 3 queries. Q1 and Q2 should be skipped; only Q3 should execute.
    report = await runner.run_comparison(
        experiment_id="test_resumed_run",
        split="dev",
        provenance_filter="human_reviewed",
        limit=3,
        resume_source=prev_file,
        live_mode=True,
    )

    assert rag_eval_calls == ["q2_dev_003"]
    assert direct_eval_calls == ["q2_dev_003"]
    assert report.is_completed is True
    assert report.status == "completed"
    assert report.paired_analysis.paired_successful_count == 3
    # Check that Q1 and Q2 answers match the resumed file
    rag_q1 = next(r for r in report.modes["rag"].per_query_results if r["query_id"] == "q2_dev_001")
    assert rag_q1["generated_answer"] == "Valid RAG Answer 1"


@pytest.mark.asyncio
async def test_resume_incomplete_pairs_preserves_valid_output(tmp_path: Path) -> None:
    """Verify resuming an incomplete pair reuses the completed condition without re-evaluating it."""
    # Previous file: Q1 succeeded in RAG mode, but Direct mode failed
    prev_file = tmp_path / "external_comparison_incomplete_pair.json"
    prev_data = {
        "experiment_id": "prev_partial",
        "timestamp_utc": "2026-10-17T00:00:00+00:00",
        "dataset_version": "v2",
        "split": "dev",
        "provenance_filter": "human_reviewed",
        "provider_type": "mock",
        "model_identifier": "mock-test",
        "modes": {
            "rag": {
                "per_query_results": [
                    {
                        "query_id": "q2_dev_001",
                        "query": "Test Q1",
                        "category": "factual_single_hop",
                        "split": "dev",
                        "provenance_status": "human_reviewed",
                        "mode": "rag",
                        "system_name": "FastRAG (mock-test)",
                        "generated_answer": "Preserved RAG Answer",
                        "prompt_version": PROMPT_VERSION,
                        "error": None,
                        "token_f1": 0.9,
                        "token_precision": 0.9,
                        "token_recall": 0.9,
                        "completeness": 0.9,
                        "groundedness": 0.9,
                        "citation_precision": 1.0,
                        "citation_recall": 1.0,
                        "latency_ms": 95.0,
                    }
                ]
            },
            "direct": {
                "per_query_results": [
                    {
                        "query_id": "q2_dev_001",
                        "query": "Test Q1",
                        "mode": "direct",
                        "error": "HTTP 429: Rate limit exceeded",
                    }
                ]
            },
        },
    }
    with open(prev_file, "w", encoding="utf-8") as f:
        json.dump(prev_data, f)

    provider = MockLLMProvider(model_name="mock-test")
    runner = FastRAGComparisonRunner(
        pipeline=None,
        provider=provider,
        dataset_dir=DATASETS_V2_DIR,
        results_dir=tmp_path,
        pacing_delay_seconds=0.0,
    )

    rag_calls = 0
    direct_calls = 0

    async def _mock_eval_rag(query_record: dict, **kwargs):
        nonlocal rag_calls
        rag_calls += 1
        raise AssertionError("evaluate_query_rag should NOT be called for already completed RAG query!")

    async def _mock_eval_direct(query_record: dict, **kwargs):
        nonlocal direct_calls
        direct_calls += 1
        now_str = datetime.now(timezone.utc).isoformat()
        return QueryComparisonResult(
            query_id=query_record["query_id"],
            query=query_record["query"],
            category=query_record.get("category", "unknown"),
            split="dev",
            provenance_status="human_reviewed",
            mode="direct",
            system_name="Direct LLM (mock-test)",
            generated_answer="Newly Recovered Direct Answer",
            reference_answer=query_record.get("reference_answer"),
            source_doc_ids=[],
            retrieved_doc_ids=[],
            cited_doc_ids=[],
            is_unanswerable=False,
            correct_abstention=False,
            token_f1=0.7,
            token_precision=0.7,
            token_recall=0.7,
            completeness=0.7,
            groundedness=0.0,
            citation_precision=0.0,
            citation_recall=0.0,
            latency_ms=45.0,
            timestamp_utc=now_str,
            error=None,
        )

    runner.evaluate_query_rag = _mock_eval_rag
    runner.evaluate_query_direct = _mock_eval_direct

    report = await runner.run_comparison(
        experiment_id="test_recover_incomplete",
        split="dev",
        provenance_filter="human_reviewed",
        limit=1,
        resume_source=prev_file,
        live_mode=True,
    )

    assert rag_calls == 0
    assert direct_calls == 1
    assert report.paired_analysis.paired_successful_count == 1
    rag_res = report.modes["rag"].per_query_results[0]
    direct_res = report.modes["direct"].per_query_results[0]
    assert rag_res["generated_answer"] == "Preserved RAG Answer"
    assert direct_res["generated_answer"] == "Newly Recovered Direct Answer"


@pytest.mark.asyncio
async def test_resume_incompatible_configuration_rejection(tmp_path: Path) -> None:
    """Verify runner rejects resuming from results with mismatched split, provenance, or model."""
    runner = FastRAGComparisonRunner(
        pipeline=None,
        provider=MockLLMProvider(model_name="target-model"),
        dataset_dir=DATASETS_V2_DIR,
        results_dir=tmp_path,
    )

    base_meta = {
        "dataset_version": "v2",
        "split": "dev",
        "provenance_filter": "human_reviewed",
        "model_identifier": "target-model",
        "modes": {},
    }

    # 1. Mismatched split
    p_split = tmp_path / "bad_split.json"
    with open(p_split, "w") as f:
        json.dump({**base_meta, "split": "test"}, f)
    with pytest.raises(ValueError, match="split mismatch"):
        runner.load_previous_results(resume_source=p_split, split="dev", provenance_filter="human_reviewed")

    # 2. Mismatched model
    p_model = tmp_path / "bad_model.json"
    with open(p_model, "w") as f:
        json.dump({**base_meta, "model_identifier": "different-model"}, f)
    with pytest.raises(ValueError, match="model mismatch"):
        runner.load_previous_results(resume_source=p_model, split="dev", provenance_filter="human_reviewed")

    # 3. Mismatched provenance
    p_prov = tmp_path / "bad_prov.json"
    with open(p_prov, "w") as f:
        json.dump({**base_meta, "provenance_filter": "all"}, f)
    with pytest.raises(ValueError, match="provenance mismatch"):
        runner.load_previous_results(resume_source=p_prov, split="dev", provenance_filter="human_reviewed")

    # 4. Mismatched dataset version
    p_ver = tmp_path / "bad_ver.json"
    with open(p_ver, "w") as f:
        json.dump({**base_meta, "dataset_version": "v1"}, f)
    with pytest.raises(ValueError, match="dataset version mismatch"):
        runner.load_previous_results(resume_source=p_ver, split="dev", provenance_filter="human_reviewed")


@pytest.mark.asyncio
async def test_missing_results_are_not_scored_as_zero(tmp_path: Path) -> None:
    """Verify incomplete or un-run queries are excluded from aggregate metric scores."""
    runner = FastRAGComparisonRunner(
        pipeline=None,
        provider=MockLLMProvider(model_name="mock-score-test"),
        dataset_dir=DATASETS_V2_DIR,
        results_dir=tmp_path,
    )

    res_success = QueryComparisonResult(
        query_id="q1",
        query="Q1",
        category="factual_single_hop",
        split="dev",
        provenance_status="human_reviewed",
        mode="rag",
        system_name="FastRAG",
        generated_answer="Answer 1",
        reference_answer="Answer 1",
        source_doc_ids=[],
        retrieved_doc_ids=[],
        cited_doc_ids=[],
        is_unanswerable=False,
        correct_abstention=False,
        token_f1=0.80,
        token_precision=0.80,
        token_recall=0.80,
        completeness=0.80,
        groundedness=0.80,
        citation_precision=1.0,
        citation_recall=1.0,
        latency_ms=100.0,
        timestamp_utc="2026-10-17T00:00:00Z",
        error=None,
    )

    res_not_run = QueryComparisonResult(
        query_id="q2",
        query="Q2",
        category="factual_single_hop",
        split="dev",
        provenance_status="human_reviewed",
        mode="rag",
        system_name="FastRAG",
        generated_answer="",
        reference_answer="Answer 2",
        source_doc_ids=[],
        retrieved_doc_ids=[],
        cited_doc_ids=[],
        is_unanswerable=False,
        correct_abstention=False,
        token_f1=0.0,
        token_precision=0.0,
        token_recall=0.0,
        completeness=0.0,
        groundedness=0.0,
        citation_precision=0.0,
        citation_recall=0.0,
        latency_ms=0.0,
        timestamp_utc="2026-10-17T00:00:00Z",
        error="not_run: quota_exhausted",
    )

    # Mode aggregation must score ONLY successful results
    summary = runner._aggregate_mode_results(
        mode="rag",
        system_name="FastRAG",
        results=[res_success, res_not_run],
    )
    assert summary.total_evaluated == 2
    assert summary.successful_count == 1
    assert summary.failed_count == 1
    # Average F1 must be 0.80 (from q1), NOT diluted to 0.40 by q2
    assert summary.summary_metrics["token_f1"]["mean"] == 0.80

    # Paired analysis must only compute deltas over mutually successful pairs
    paired_analysis = runner._compute_paired_analysis(
        eligible_queries=[{"query_id": "q1", "category": "factual_single_hop"}, {"query_id": "q2", "category": "factual_single_hop"}],
        rag_results=[res_success, res_not_run],
        direct_results=[res_success, res_not_run],
    )
    assert paired_analysis.paired_successful_count == 1
    assert len(paired_analysis.missing_pairs) == 1
    assert paired_analysis.paired_metrics["token_f1"]["rag_mean"] == 0.80
    assert paired_analysis.paired_metrics["token_f1"]["direct_mean"] == 0.80


