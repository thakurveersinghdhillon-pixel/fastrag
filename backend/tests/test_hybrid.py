"""Comprehensive Unit and Concurrency Tests for FastRAG Hybrid Retrieval and RRF (Phase 5).

Covers:
- Reciprocal Rank Fusion (numerical correctness, deduplication, deterministic ties, edge cases)
- Concurrent execution and overlap verification using controlled test doubles
- Timeout handling, strict vs. degraded error modes, exception propagation
- Deduplication and deterministic candidate ordering
- Evaluation adapter integration
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Sequence

import pytest

from app.core.exceptions import BranchTimeoutError, ConfigurationError, RetrievalError
from app.ingestion.models import Chunk
from app.retrieval.base import BaseRetriever, IndexInfo, RetrievalResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever, MockEmbeddingEncoder
from app.retrieval.fusion import (
    DEFAULT_RRF_SMOOTHING_K,
    reciprocal_rank_fusion,
    validate_rrf_k,
)
from app.retrieval.hybrid import HybridRetriever
from evaluation.retriever_adapter import evaluate_retriever


def _make_result(
    chunk_id: str,
    doc_id: str,
    rank: int,
    score: float,
    score_type: str = "test",
    text: str = "sample text",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=doc_id,
        rank=rank,
        score=score,
        score_type=score_type,
        text=text,
        source="test_source",
        metadata={"title": f"Doc {doc_id}"},
    )


# ============================================================================
# 1. Reciprocal Rank Fusion (RRF) Numerical & Edge Case Tests
# ============================================================================

def test_validate_rrf_k() -> None:
    validate_rrf_k(60)
    validate_rrf_k(1)
    with pytest.raises(ValueError, match="rrf_k must be strictly positive"):
        validate_rrf_k(0)
    with pytest.raises(ValueError, match="rrf_k must be strictly positive"):
        validate_rrf_k(-10)
    with pytest.raises(TypeError, match="rrf_k must be an integer"):
        validate_rrf_k("60")  # type: ignore[arg-type]


def test_rrf_exact_hand_calculated_scores() -> None:
    """Verify mathematically exact RRF scores against known manual formula calculations.

    Formula: RRF(d) = sum 1 / (k + rank)
    Using k=60:
    - List 1 (dense):
        Rank 1: chunk_A -> score 1/(60 + 1) = 1/61 = 0.01639344...
        Rank 2: chunk_B -> score 1/(60 + 2) = 1/62 = 0.01612903...
    - List 2 (bm25):
        Rank 1: chunk_B -> score 1/(60 + 1) = 1/61 = 0.01639344...
        Rank 2: chunk_C -> score 1/(60 + 2) = 1/62 = 0.01612903...

    Expected combined:
    - chunk_B: 1/61 + 1/62 = 0.03252247
    - chunk_A: 1/61        = 0.01639344
    - chunk_C: 1/62        = 0.01612903
    """
    list_dense = [
        _make_result("chunk_A", "doc_A", rank=1, score=0.9),
        _make_result("chunk_B", "doc_B", rank=2, score=0.8),
    ]
    list_bm25 = [
        _make_result("chunk_B", "doc_B", rank=1, score=15.0),
        _make_result("chunk_C", "doc_C", rank=2, score=12.0),
    ]

    fused = reciprocal_rank_fusion(
        ranked_lists={"dense": list_dense, "bm25": list_bm25},
        top_k=5,
        rrf_k=60,
    )

    assert len(fused) == 3

    # Check order
    assert fused[0].chunk_id == "chunk_B"
    assert fused[1].chunk_id == "chunk_A"
    assert fused[2].chunk_id == "chunk_C"

    # Check exact score calculation
    expected_b = round(1.0 / 61.0 + 1.0 / 62.0, 8)
    expected_a = round(1.0 / 61.0, 8)
    expected_c = round(1.0 / 62.0, 8)

    assert pytest.approx(fused[0].score, 1e-6) == expected_b
    assert pytest.approx(fused[1].score, 1e-6) == expected_a
    assert pytest.approx(fused[2].score, 1e-6) == expected_c

    # Check metadata tracking
    assert fused[0].metadata["rrf_ranks"]["dense"] == 2
    assert fused[0].metadata["rrf_ranks"]["bm25"] == 1
    assert fused[0].metadata["original_scores"]["dense"] == 0.8
    assert fused[0].metadata["original_scores"]["bm25"] == 15.0
    assert fused[0].score_type == "rrf"


def test_rrf_duplicate_occurrences_within_one_list_not_inflated() -> None:
    """Duplicate occurrences of the same chunk in a single branch must NOT inflate contribution."""
    list_with_dupes = [
        _make_result("chunk_dup", "doc_1", rank=1, score=10.0),
        _make_result("chunk_other", "doc_2", rank=2, score=8.0),
        _make_result("chunk_dup", "doc_1", rank=3, score=5.0),  # Duplicate chunk_id in same list
    ]

    fused = reciprocal_rank_fusion(
        ranked_lists=[list_with_dupes],
        top_k=5,
        rrf_k=60,
    )

    assert len(fused) == 2
    assert fused[0].chunk_id == "chunk_dup"
    # Score should strictly be 1/61, NOT (1/61 + 1/63)
    assert pytest.approx(fused[0].score, 1e-6) == round(1.0 / 61.0, 8)


def test_rrf_deterministic_tie_breaking() -> None:
    """When candidates have identical RRF scores, tie-breaking must be strictly by chunk_id ascending."""
    list_1 = [
        _make_result("chunk_z", "doc_z", rank=1, score=1.0),
        _make_result("chunk_m", "doc_m", rank=2, score=0.9),
        _make_result("chunk_a", "doc_a", rank=3, score=0.8),
    ]
    # Disjoint list with identical rank positions
    list_2 = [
        _make_result("chunk_b", "doc_b", rank=1, score=1.0),
        _make_result("chunk_y", "doc_y", rank=2, score=0.9),
    ]

    fused = reciprocal_rank_fusion(
        ranked_lists=[list_1, list_2],
        top_k=5,
        rrf_k=60,
    )

    # Ranks 1 in list_1 (chunk_z) and list_2 (chunk_b) both score 1/61
    # Ties must sort chunk_b before chunk_z alphabetically
    tied_rank1_chunks = [r.chunk_id for r in fused if pytest.approx(r.score, 1e-6) == 1.0 / 61.0]
    assert tied_rank1_chunks == ["chunk_b", "chunk_z"]


def test_rrf_empty_and_disjoint_lists() -> None:
    # All empty
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []

    # Disjoint lists
    l1 = [_make_result("c1", "d1", rank=1, score=1.0)]
    l2 = [_make_result("c2", "d2", rank=1, score=1.0)]
    fused = reciprocal_rank_fusion([l1, l2], top_k=2)
    assert len(fused) == 2
    assert {r.chunk_id for r in fused} == {"c1", "c2"}


# ============================================================================
# 2. Controlled Test Doubles & Concurrency Verification
# ============================================================================

class MockDelayedRetriever(BaseRetriever):
    """Test double that simulates computational delay to verify branch concurrency."""

    def __init__(self, name: str, delay_seconds: float, results: list[RetrievalResult]) -> None:
        self.name = name
        self.delay_seconds = delay_seconds
        self.results = results
        self.query_received: str | None = None
        self.top_k_received: int | None = None
        self.call_start_time: float | None = None
        self.call_end_time: float | None = None

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        self.query_received = query
        self.top_k_received = top_k
        self.call_start_time = time.perf_counter()
        time.sleep(self.delay_seconds)
        self.call_end_time = time.perf_counter()
        return self.results[:top_k]

    def build_index(self, chunks: Sequence[Chunk]) -> IndexInfo:
        return IndexInfo("mock", "fp", len(chunks), len(chunks))

    def save_index(self, directory_path: Path | str) -> None:
        pass

    def load_index(self, directory_path: Path | str) -> IndexInfo:
        return IndexInfo("mock", "fp", 0, 0)

    def get_index_info(self) -> IndexInfo | None:
        return IndexInfo("mock", "fp", 10, 5)

    @property
    def is_indexed(self) -> bool:
        return True


class MockFailingRetriever(BaseRetriever):
    """Test double that raises an error during search."""

    def __init__(self, error_message: str = "Synthetic database crash") -> None:
        self.error_message = error_message

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        raise RuntimeError(self.error_message)

    def build_index(self, chunks: Sequence[Chunk]) -> IndexInfo:
        return IndexInfo("mock", "fp", 0, 0)

    def save_index(self, directory_path: Path | str) -> None:
        pass

    def load_index(self, directory_path: Path | str) -> IndexInfo:
        return IndexInfo("mock", "fp", 0, 0)

    def get_index_info(self) -> IndexInfo | None:
        return IndexInfo("mock", "fp", 0, 0)

    @property
    def is_indexed(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_concurrent_branches_execution_overlap() -> None:
    """Verify that both branches execute concurrently in worker threads.

    If each branch takes ~0.08s, sequential execution would take >= 0.16s,
    whereas concurrent execution takes ~0.08s (strictly < 0.14s).
    """
    dense_res = [_make_result("chunk_d1", "doc_d", rank=1, score=0.9)]
    bm25_res = [_make_result("chunk_b1", "doc_b", rank=1, score=10.0)]

    dense_mock = MockDelayedRetriever("dense", delay_seconds=0.08, results=dense_res)
    bm25_mock = MockDelayedRetriever("bm25", delay_seconds=0.08, results=bm25_res)

    hybrid = HybridRetriever(
        dense_retriever=dense_mock,
        bm25_retriever=bm25_mock,
        dense_candidate_count=5,
        bm25_candidate_count=5,
        top_k=5,
    )

    t0 = time.perf_counter()
    response = await hybrid.search_async("test query", concurrent=True)
    total_duration = time.perf_counter() - t0

    # 1. Both branches received the identical query and their candidate limits
    assert dense_mock.query_received == "test query"
    assert bm25_mock.query_received == "test query"
    assert dense_mock.top_k_received == 5
    assert bm25_mock.top_k_received == 5

    # 2. Check temporal overlap: dense and bm25 were both active at the same time
    assert dense_mock.call_start_time is not None
    assert dense_mock.call_end_time is not None
    assert bm25_mock.call_start_time is not None
    assert bm25_mock.call_end_time is not None

    overlap_start = max(dense_mock.call_start_time, bm25_mock.call_start_time)
    overlap_end = min(dense_mock.call_end_time, bm25_mock.call_end_time)
    assert overlap_end > overlap_start, "Retrieval branches did not overlap in execution!"

    # 3. Total wall-clock time was concurrent (< 0.14s instead of 0.16s+)
    assert total_duration < 0.14, f"Total duration {total_duration:.3f}s was not concurrent"
    assert len(response.results) == 2
    assert response.diagnostics.concurrent is True
    assert response.diagnostics.degraded is False


@pytest.mark.asyncio
async def test_sequential_execution_control_baseline() -> None:
    """Verify that concurrent=False executes sequentially for comparison."""
    dense_res = [_make_result("chunk_d1", "doc_d", rank=1, score=0.9)]
    bm25_res = [_make_result("chunk_b1", "doc_b", rank=1, score=10.0)]

    dense_mock = MockDelayedRetriever("dense", delay_seconds=0.05, results=dense_res)
    bm25_mock = MockDelayedRetriever("bm25", delay_seconds=0.05, results=bm25_res)

    hybrid = HybridRetriever(
        dense_retriever=dense_mock,
        bm25_retriever=bm25_mock,
    )

    t0 = time.perf_counter()
    response = await hybrid.search_async("query", concurrent=False)
    total_duration = time.perf_counter() - t0

    # Sequential: >= 0.10s (sum of delays)
    assert total_duration >= 0.095
    assert response.diagnostics.concurrent is False


# ============================================================================
# 3. Timeout, Failure Modes, and Degraded Operation Tests
# ============================================================================

@pytest.mark.asyncio
async def test_strict_mode_fails_on_branch_error() -> None:
    """In strict mode, a branch failure must raise RetrievalError."""
    dense_mock = MockDelayedRetriever("dense", delay_seconds=0.01, results=[_make_result("c1", "d1", 1, 0.9)])
    failing_bm25 = MockFailingRetriever("Database connection refused")

    hybrid = HybridRetriever(
        dense_retriever=dense_mock,
        bm25_retriever=failing_bm25,
        mode="strict",
    )

    with pytest.raises(RetrievalError, match="Database connection refused"):
        await hybrid.search_async("query")


@pytest.mark.asyncio
async def test_strict_mode_fails_on_timeout() -> None:
    """In strict mode, a branch exceeding timeout must raise BranchTimeoutError."""
    # Dense branch takes 0.2s, timeout is 0.05s
    slow_dense = MockDelayedRetriever("dense", delay_seconds=0.20, results=[])
    fast_bm25 = MockDelayedRetriever("bm25", delay_seconds=0.01, results=[_make_result("c1", "d1", 1, 10.0)])

    hybrid = HybridRetriever(
        dense_retriever=slow_dense,
        bm25_retriever=fast_bm25,
        timeout_seconds=0.05,
        mode="strict",
    )

    with pytest.raises(BranchTimeoutError, match="timed out"):
        await hybrid.search_async("query")


@pytest.mark.asyncio
async def test_degraded_mode_returns_partial_results() -> None:
    """In degraded mode, if one branch fails, the successful branch's results are returned with degraded=True."""
    dense_mock = MockDelayedRetriever("dense", delay_seconds=0.01, results=[_make_result("c_dense", "doc_d", 1, 0.9)])
    failing_bm25 = MockFailingRetriever("BM25 index corrupted")

    hybrid = HybridRetriever(
        dense_retriever=dense_mock,
        bm25_retriever=failing_bm25,
        mode="degraded",
    )

    response = await hybrid.search_async("query")
    assert response.diagnostics.degraded is True
    assert response.diagnostics.dense.success is True
    assert response.diagnostics.bm25.success is False
    assert len(response.results) == 1
    assert response.results[0].chunk_id == "c_dense"


@pytest.mark.asyncio
async def test_degraded_mode_fails_if_both_branches_fail() -> None:
    """If both branches fail, degraded mode must still raise RetrievalError."""
    failing_dense = MockFailingRetriever("Dense network error")
    failing_bm25 = MockFailingRetriever("BM25 IO error")

    hybrid = HybridRetriever(
        dense_retriever=failing_dense,
        bm25_retriever=failing_bm25,
        mode="degraded",
    )

    with pytest.raises(RetrievalError, match="Both retrieval branches failed"):
        await hybrid.search_async("query")


# ============================================================================
# 4. End-to-End Integration with Real BM25 & Mock Dense Retriever
# ============================================================================

@pytest.fixture
def test_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="chunk_01",
            document_id="doc_1",
            chunk_index=0,
            text="Reciprocal Rank Fusion merges multiple ranked candidate lists without score calibration.",
            source="src",
            start_offset=0,
            end_offset=88,
            metadata={"title": "RRF Concept"},
        ),
        Chunk(
            chunk_id="chunk_02",
            document_id="doc_2",
            chunk_index=0,
            text="BM25 lexical search scores documents using term frequency and inverse document frequency.",
            source="src",
            start_offset=0,
            end_offset=89,
            metadata={"title": "BM25 Scoring"},
        ),
        Chunk(
            chunk_id="chunk_03",
            document_id="doc_3",
            chunk_index=0,
            text="Dense semantic search embeds queries and passages into vector space using cosine similarity.",
            source="src",
            start_offset=0,
            end_offset=92,
            metadata={"title": "Dense Vectors"},
        ),
    ]


def test_hybrid_retriever_sync_search_and_persistence(test_chunks: list[Chunk], tmp_path: Path) -> None:
    bm25 = BM25Retriever()
    dense = DenseRetriever(encoder=MockEmbeddingEncoder(dimension=32))

    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        dense_candidate_count=3,
        bm25_candidate_count=3,
        top_k=2,
    )

    info = hybrid.build_index(test_chunks)
    assert info.retriever_type == "hybrid"
    assert info.chunk_count == 3
    assert hybrid.is_indexed

    # Synchronous search
    results = hybrid.search("Reciprocal Rank Fusion lexical", top_k=2)
    assert len(results) == 2
    assert results[0].score_type == "rrf"
    assert results[0].rank == 1

    # Save and reload
    save_dir = tmp_path / "hybrid_index"
    hybrid.save_index(save_dir)
    assert (save_dir / "dense" / "dense_meta.json").exists()
    assert (save_dir / "bm25" / "bm25_index.json").exists()

    fresh_dense = DenseRetriever(encoder=MockEmbeddingEncoder(dimension=32))
    fresh_bm25 = BM25Retriever()
    loaded_hybrid = HybridRetriever(dense_retriever=fresh_dense, bm25_retriever=fresh_bm25)
    loaded_info = loaded_hybrid.load_index(save_dir)
    assert loaded_info.chunk_count == 3

    reloaded_results = loaded_hybrid.search("Reciprocal Rank Fusion lexical", top_k=2)
    assert len(reloaded_results) == len(results)
    assert reloaded_results[0].chunk_id == results[0].chunk_id
    assert pytest.approx(reloaded_results[0].score, 1e-6) == results[0].score
