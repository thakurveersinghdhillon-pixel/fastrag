"""FastRAG Concurrent Hybrid Retrieval with Reciprocal Rank Fusion (RRF).

Combines dense semantic vector retrieval and BM25 lexical retrieval concurrently
using asyncio, fuses candidate pools via Reciprocal Rank Fusion, and produces
deterministic, deduplicated ranked results with microsecond-level telemetry.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from app.core.exceptions import BranchTimeoutError, ConfigurationError, RetrievalError
from app.core.logging_config import get_logger
from app.ingestion.models import Chunk
from app.retrieval.base import BaseRetriever, IndexInfo, RetrievalResult, validate_top_k
from app.retrieval.fusion import DEFAULT_RRF_SMOOTHING_K, reciprocal_rank_fusion, validate_rrf_k

logger = get_logger("fastrag.retrieval.hybrid")


@dataclass
class BranchDiagnostics:
    """Latency and execution telemetry for an individual retrieval branch."""

    branch_name: str
    candidate_count: int
    duration_ms: float
    success: bool
    error_message: str | None = None
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HybridRetrievalDiagnostics:
    """Aggregated latency, concurrency, and failure telemetry for hybrid retrieval."""

    dense: BranchDiagnostics
    bm25: BranchDiagnostics
    fusion_duration_ms: float
    total_duration_ms: float
    degraded: bool
    mode: str
    concurrent: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "dense": self.dense.to_dict(),
            "bm25": self.bm25.to_dict(),
            "fusion_duration_ms": round(self.fusion_duration_ms, 3),
            "total_duration_ms": round(self.total_duration_ms, 3),
            "degraded": self.degraded,
            "mode": self.mode,
            "concurrent": self.concurrent,
        }


@dataclass
class HybridRetrievalResponse:
    """Full hybrid retrieval response containing ranked candidates and diagnostics."""

    results: list[RetrievalResult]
    diagnostics: HybridRetrievalDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "diagnostics": self.diagnostics.to_dict(),
        }


class HybridRetriever(BaseRetriever):
    """Orchestrates concurrent Dense semantic and BM25 lexical retrieval with RRF.

    Executes both retrieval branches concurrently in worker threads, applies
    Reciprocal Rank Fusion, and supports strict vs. degraded error handling.
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        dense_candidate_count: int = 20,
        bm25_candidate_count: int = 20,
        top_k: int = 10,
        rrf_k: int = DEFAULT_RRF_SMOOTHING_K,
        timeout_seconds: float | None = None,
        mode: Literal["strict", "degraded"] = "strict",
    ) -> None:
        if not isinstance(dense_retriever, BaseRetriever):
            raise TypeError("dense_retriever must implement BaseRetriever")
        if not isinstance(bm25_retriever, BaseRetriever):
            raise TypeError("bm25_retriever must implement BaseRetriever")

        validate_top_k(top_k)
        validate_top_k(dense_candidate_count)
        validate_top_k(bm25_candidate_count)
        validate_rrf_k(rrf_k)

        if mode not in ("strict", "degraded"):
            raise ConfigurationError(f"Invalid mode '{mode}': must be 'strict' or 'degraded'")

        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_candidate_count = dense_candidate_count
        self.bm25_candidate_count = bm25_candidate_count
        self.default_top_k = top_k
        self.rrf_k = rrf_k
        self.timeout_seconds = timeout_seconds
        self.mode: Literal["strict", "degraded"] = mode

        self.last_diagnostics: HybridRetrievalDiagnostics | None = None

    @property
    def is_indexed(self) -> bool:
        """True if both underlying retrievers have built or loaded indexes."""
        return self.dense_retriever.is_indexed and self.bm25_retriever.is_indexed

    def warmup(self) -> None:
        """Pre-warm underlying dense and BM25 retrievers."""
        if hasattr(self.dense_retriever, "warmup"):
            self.dense_retriever.warmup()
        if hasattr(self.bm25_retriever, "warmup"):
            self.bm25_retriever.warmup()

    def build_index(self, chunks: Sequence[Chunk]) -> IndexInfo:
        """Build indexes for both Dense and BM25 retrievers."""
        dense_info = self.dense_retriever.build_index(chunks)
        bm25_info = self.bm25_retriever.build_index(chunks)

        return IndexInfo(
            retriever_type="hybrid",
            corpus_fingerprint=dense_info.corpus_fingerprint,
            chunk_count=dense_info.chunk_count,
            document_count=dense_info.document_count,
            hyperparameters={
                "dense": dense_info.hyperparameters,
                "bm25": bm25_info.hyperparameters,
                "dense_candidates": self.dense_candidate_count,
                "bm25_candidates": self.bm25_candidate_count,
                "rrf_k": self.rrf_k,
                "mode": self.mode,
            },
        )

    def save_index(self, directory_path: Path | str) -> None:
        """Save both Dense and BM25 indexes under subdirectories."""
        base_dir = Path(directory_path)
        self.dense_retriever.save_index(base_dir / "dense")
        self.bm25_retriever.save_index(base_dir / "bm25")

    def load_index(self, directory_path: Path | str) -> IndexInfo:
        """Load both Dense and BM25 indexes from subdirectories."""
        base_dir = Path(directory_path)
        dense_info = self.dense_retriever.load_index(base_dir / "dense")
        bm25_info = self.bm25_retriever.load_index(base_dir / "bm25")

        return IndexInfo(
            retriever_type="hybrid",
            corpus_fingerprint=dense_info.corpus_fingerprint,
            chunk_count=dense_info.chunk_count,
            document_count=dense_info.document_count,
            hyperparameters={
                "dense": dense_info.hyperparameters,
                "bm25": bm25_info.hyperparameters,
                "dense_candidates": self.dense_candidate_count,
                "bm25_candidates": self.bm25_candidate_count,
                "rrf_k": self.rrf_k,
                "mode": self.mode,
            },
        )

    def get_index_info(self) -> IndexInfo | None:
        """Return combined index info if both indexes are loaded."""
        dense_info = self.dense_retriever.get_index_info()
        bm25_info = self.bm25_retriever.get_index_info()
        if dense_info is None or bm25_info is None:
            return None

        return IndexInfo(
            retriever_type="hybrid",
            corpus_fingerprint=dense_info.corpus_fingerprint,
            chunk_count=dense_info.chunk_count,
            document_count=dense_info.document_count,
            hyperparameters={
                "dense": dense_info.hyperparameters,
                "bm25": bm25_info.hyperparameters,
                "dense_candidates": self.dense_candidate_count,
                "bm25_candidates": self.bm25_candidate_count,
                "rrf_k": self.rrf_k,
                "mode": self.mode,
            },
        )

    async def _execute_branch(
        self,
        retriever: BaseRetriever,
        branch_name: str,
        query: str,
        candidate_count: int,
        timeout: float | None,
    ) -> tuple[list[RetrievalResult], BranchDiagnostics]:
        """Execute a single retrieval branch in a worker thread with timing and error isolation."""
        start = time.perf_counter()
        diag = BranchDiagnostics(
            branch_name=branch_name,
            candidate_count=0,
            duration_ms=0.0,
            success=False,
        )

        try:
            # Dispatch synchronous CPU/model search to worker thread
            task = asyncio.to_thread(retriever.search, query, candidate_count)
            if timeout is not None:
                results = await asyncio.wait_for(task, timeout=timeout)
            else:
                results = await task

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            diag.candidate_count = len(results)
            diag.duration_ms = elapsed_ms
            diag.success = True
            return results, diag

        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            diag.duration_ms = elapsed_ms
            diag.timed_out = True
            diag.error_message = f"Branch '{branch_name}' timed out after {timeout:.3f}s"
            logger.warning(diag.error_message)
            return [], diag

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            diag.duration_ms = elapsed_ms
            diag.error_message = f"Branch '{branch_name}' failed: {str(exc)}"
            logger.error(diag.error_message, exc_info=True)
            return [], diag

    async def search_async(
        self,
        query: str,
        top_k: int | None = None,
        dense_candidates: int | None = None,
        bm25_candidates: int | None = None,
        rrf_k: int | None = None,
        timeout_seconds: float | None = None,
        mode: Literal["strict", "degraded"] | None = None,
        concurrent: bool = True,
    ) -> HybridRetrievalResponse:
        """Asynchronously execute hybrid retrieval and Reciprocal Rank Fusion.

        Args:
            query: Query string.
            top_k: Final number of ranked results to return (defaults to self.default_top_k).
            dense_candidates: Candidate limit for dense retrieval branch.
            bm25_candidates: Candidate limit for BM25 retrieval branch.
            rrf_k: RRF smoothing constant (defaults to self.rrf_k).
            timeout_seconds: Per-branch timeout in seconds.
            mode: Error mode ('strict' or 'degraded').
            concurrent: If True, execute branches concurrently; if False, execute sequentially (for baseline benchmarking).

        Returns:
            HybridRetrievalResponse containing fused candidates and detailed branch diagnostics.
        """
        effective_top_k = top_k if top_k is not None else self.default_top_k
        validate_top_k(effective_top_k)

        effective_dense_k = dense_candidates if dense_candidates is not None else self.dense_candidate_count
        effective_bm25_k = bm25_candidates if bm25_candidates is not None else self.bm25_candidate_count
        effective_rrf_k = rrf_k if rrf_k is not None else self.rrf_k
        effective_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        effective_mode = mode if mode is not None else self.mode

        validate_top_k(effective_dense_k)
        validate_top_k(effective_bm25_k)
        validate_rrf_k(effective_rrf_k)

        # Empty or whitespace queries short-circuit immediately
        if not query or not query.strip():
            empty_diag = HybridRetrievalDiagnostics(
                dense=BranchDiagnostics("dense", 0, 0.0, True),
                bm25=BranchDiagnostics("bm25", 0, 0.0, True),
                fusion_duration_ms=0.0,
                total_duration_ms=0.0,
                degraded=False,
                mode=effective_mode,
                concurrent=concurrent,
            )
            self.last_diagnostics = empty_diag
            return HybridRetrievalResponse(results=[], diagnostics=empty_diag)

        overall_start = time.perf_counter()

        if concurrent:
            # Concurrent execution via asyncio.gather
            dense_coro = self._execute_branch(
                self.dense_retriever, "dense", query, effective_dense_k, effective_timeout
            )
            bm25_coro = self._execute_branch(
                self.bm25_retriever, "bm25", query, effective_bm25_k, effective_timeout
            )
            (dense_results, dense_diag), (bm25_results, bm25_diag) = await asyncio.gather(
                dense_coro, bm25_coro
            )
        else:
            # Sequential execution (control baseline for latency comparison)
            dense_results, dense_diag = await self._execute_branch(
                self.dense_retriever, "dense", query, effective_dense_k, effective_timeout
            )
            bm25_results, bm25_diag = await self._execute_branch(
                self.bm25_retriever, "bm25", query, effective_bm25_k, effective_timeout
            )

        # Failure / Timeout evaluation based on configured mode
        both_failed = (not dense_diag.success) and (not bm25_diag.success)
        one_failed = (not dense_diag.success) or (not bm25_diag.success)

        if both_failed:
            err_msg = (
                f"Both retrieval branches failed: dense='{dense_diag.error_message}', "
                f"bm25='{bm25_diag.error_message}'"
            )
            logger.error(err_msg)
            raise RetrievalError(err_msg)

        if one_failed:
            if effective_mode == "strict":
                failed_branch = "dense" if not dense_diag.success else "bm25"
                failed_diag = dense_diag if not dense_diag.success else bm25_diag
                if failed_diag.timed_out:
                    raise BranchTimeoutError(failed_diag.error_message or f"{failed_branch} timed out")
                raise RetrievalError(failed_diag.error_message or f"{failed_branch} retrieval failed")

            # Degraded mode: log explicitly and proceed with available candidates
            logger.warning(
                "Operating in degraded hybrid retrieval mode: returning partial results. "
                f"Dense success={dense_diag.success}, BM25 success={bm25_diag.success}"
            )

        # Reciprocal Rank Fusion
        fusion_start = time.perf_counter()
        fused = reciprocal_rank_fusion(
            ranked_lists={"dense": dense_results, "bm25": bm25_results},
            top_k=effective_top_k,
            rrf_k=effective_rrf_k,
        )
        fusion_elapsed_ms = (time.perf_counter() - fusion_start) * 1000.0
        total_elapsed_ms = (time.perf_counter() - overall_start) * 1000.0

        diagnostics = HybridRetrievalDiagnostics(
            dense=dense_diag,
            bm25=bm25_diag,
            fusion_duration_ms=fusion_elapsed_ms,
            total_duration_ms=total_elapsed_ms,
            degraded=one_failed,
            mode=effective_mode,
            concurrent=concurrent,
        )
        self.last_diagnostics = diagnostics

        logger.info(
            f"Hybrid retrieval finished in {total_elapsed_ms:.2f}ms "
            f"(dense: {dense_diag.duration_ms:.2f}ms, bm25: {bm25_diag.duration_ms:.2f}ms, "
            f"fusion: {fusion_elapsed_ms:.2f}ms, degraded={diagnostics.degraded})",
            extra={"operation": "hybrid_search", "latency_ms": total_elapsed_ms},
        )

        return HybridRetrievalResponse(results=fused, diagnostics=diagnostics)

    def search(self, query: str, top_k: int = 10, **kwargs: Any) -> list[RetrievalResult]:
        """Synchronous wrapper for search_async conforming to BaseRetriever interface.

        Can be called directly from synchronous CLI scripts or evaluation adapters.
        """
        try:
            # Check if there is an active event loop in the current thread
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # If already inside an asynchronous loop (e.g., inside FastAPI handler or pytest-asyncio)
            # Use nest_asyncio or worker thread to avoid blocking loop re-entrancy
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.search_async(query, top_k=top_k, **kwargs))
                response = future.result()
                return response.results
        else:
            response = asyncio.run(self.search_async(query, top_k=top_k, **kwargs))
            return response.results
