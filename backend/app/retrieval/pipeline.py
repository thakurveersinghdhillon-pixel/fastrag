"""FastRAG End-to-End Retrieval Pipeline Orchestration.

Coordinates Candidate Retrieval (BM25, Dense, or Hybrid) -> CrossEncoder Reranking
-> Context Assembly with token budgeting and structured citation tagging.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Sequence

from app.core.logging_config import get_logger
from app.generation.models import GenerationResponse, StreamChunk, StreamEvent
from app.ingestion.models import Chunk
from app.retrieval.base import BaseRetriever, IndexInfo, RetrievalResult
from app.retrieval.context_assembler import AssembledContext, ContextAssembler
from app.retrieval.hybrid import HybridRetrievalResponse, HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker, RerankDiagnostics

if TYPE_CHECKING:
    from app.generation.generator import RAGGenerator

logger = get_logger("fastrag.retrieval.pipeline")


@dataclass
class PipelineDiagnostics:
    """Execution telemetry across all stages of the retrieval and generation pipeline."""

    retrieval_ms: float
    rerank_ms: float
    assembly_ms: float
    total_ms: float
    candidate_count: int
    reranked_count: int
    selected_context_count: int
    generation_ms: float = 0.0
    ttft_ms: float | None = None
    hybrid_breakdown: dict[str, Any] | None = None
    rerank_diagnostics: dict[str, Any] | None = None
    generation_diagnostics: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "retrieval_ms": round(self.retrieval_ms, 3),
            "rerank_ms": round(self.rerank_ms, 3),
            "assembly_ms": round(self.assembly_ms, 3),
            "generation_ms": round(self.generation_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "candidate_count": self.candidate_count,
            "reranked_count": self.reranked_count,
            "selected_context_count": self.selected_context_count,
            "ttft_ms": round(self.ttft_ms, 3) if self.ttft_ms is not None else None,
            "hybrid_breakdown": self.hybrid_breakdown,
            "rerank_diagnostics": self.rerank_diagnostics,
            "generation_diagnostics": self.generation_diagnostics,
            "usage": self.usage,
        }
        return data


@dataclass
class PipelineResponse:
    """Structured response containing all pipeline artifacts and telemetry."""

    query: str
    candidates: list[RetrievalResult]
    reranked_results: list[RetrievalResult]
    context: AssembledContext
    diagnostics: PipelineDiagnostics
    generation: GenerationResponse | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "query": self.query,
            "candidates": [c.to_dict() for c in self.candidates],
            "reranked_results": [r.to_dict() for r in self.reranked_results],
            "context": self.context.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
        }
        if self.generation is not None:
            data["generation"] = self.generation.to_dict()
        return data


class RAGPipeline:
    """Orchestrates candidate retrieval, neural reranking, and context assembly.

    Maintains component decoupling:
    - Retriever: Any BaseRetriever (HybridRetriever, DenseRetriever, or BM25Retriever).
    - Reranker: CrossEncoderReranker (optional; can be bypassed if None or via flag).
    - ContextAssembler: Token budgeter and citation formatter.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: CrossEncoderReranker | None = None,
        context_assembler: ContextAssembler | None = None,
        generator: RAGGenerator | None = None,
        query_cache: Any | None = None,
        rerank_blending_alpha: float = 0.0,
    ) -> None:
        if not isinstance(retriever, BaseRetriever):
            raise TypeError("retriever must implement BaseRetriever")
        self.retriever = retriever
        self.reranker = reranker
        self.context_assembler = context_assembler or ContextAssembler()
        self.generator = generator
        self.query_cache = query_cache
        if rerank_blending_alpha < 0.0:
            raise ValueError(
                f"rerank_blending_alpha must be non-negative (>= 0.0), got {rerank_blending_alpha}"
            )
        self.rerank_blending_alpha = float(rerank_blending_alpha)

    def warmup(self) -> None:
        """Pre-warm underlying models (dense encoder and CrossEncoder reranker)."""
        if hasattr(self.retriever, "warmup"):
            self.retriever.warmup()
        if self.reranker is not None and hasattr(self.reranker, "warmup"):
            self.reranker.warmup()

    async def aclose(self) -> None:
        """Gracefully close resources such as persistent HTTP connection pools."""
        if self.generator is not None and hasattr(self.generator, "aclose"):
            await self.generator.aclose()

    async def execute_async(
        self,
        query: str,
        candidate_k: int = 20,
        rerank_top_k: int = 10,
        context_max_tokens: int | None = None,
        context_max_chunks: int | None = None,
        bypass_reranking: bool = False,
        generate_answer: bool = False,
        generation_temperature: float | None = None,
        generation_max_tokens: int | None = None,
        generation_system_prompt: str | None = None,
        generation_model: str | None = None,
        min_relevance_score: float | None = None,
        ordering_strategy: str | None = None,
        rerank_blending_alpha: float | None = None,
        **retriever_kwargs: Any,
    ) -> PipelineResponse:
        """Asynchronously execute retrieval, reranking, context assembly, and optional answer generation.

        Args:
            query: User search query.
            candidate_k: Number of candidate chunks to retrieve from base retriever.
            rerank_top_k: Number of candidates to retain after reranking.
            context_max_tokens: Override maximum tokens in assembled context.
            context_max_chunks: Override maximum chunks in assembled context.
            bypass_reranking: If True, skips neural reranker stage.
            generate_answer: If True and generator is available, invokes LLM answer generation.
            generation_temperature: Sampling temperature override for LLM generator.
            generation_max_tokens: Maximum completion tokens override for LLM generator.
            generation_system_prompt: Custom system prompt override for LLM generator.
            generation_model: Model name override for LLM generator.
            min_relevance_score: Optional relevance score threshold to filter candidate chunks.
            ordering_strategy: Passage layout ordering ('descending' or 'boundary').
            rerank_blending_alpha: Optional rank blending weight between CrossEncoder and base ranks.
            **retriever_kwargs: Passed to hybrid/retriever search.

        Returns:
            PipelineResponse containing candidates, reranked results, context, generation, and diagnostics.
        """
        overall_start = time.perf_counter()

        # Step 0: Check Tier 1 Query Response Cache (if answer generation is enabled)
        eff_temp = generation_temperature if generation_temperature is not None else (
            self.generator.default_temperature if self.generator else 0.0
        )
        eff_model = generation_model or (self.generator.model_name if self.generator else "default")
        eff_prompt = generation_system_prompt or (self.generator.default_system_prompt if self.generator else None)
        eff_max_tokens = context_max_tokens if context_max_tokens is not None else self.context_assembler.max_tokens
        eff_max_chunks = context_max_chunks if context_max_chunks is not None else self.context_assembler.max_chunks
        eff_min_score = min_relevance_score if min_relevance_score is not None else self.context_assembler.min_relevance_score
        eff_ordering = ordering_strategy if ordering_strategy is not None else self.context_assembler.ordering_strategy
        eff_blending_alpha = (
            float(rerank_blending_alpha)
            if rerank_blending_alpha is not None
            else self.rerank_blending_alpha
        )
        if eff_blending_alpha < 0.0:
            raise ValueError(
                f"rerank_blending_alpha must be non-negative (>= 0.0), got {eff_blending_alpha}"
            )
        tenant_id = retriever_kwargs.pop("tenant_id", None)

        index_info = self.get_index_info()
        corpus_fp = index_info.corpus_fingerprint if index_info else ""

        if (
            generate_answer
            and self.query_cache is not None
            and getattr(self.query_cache, "enabled", False)
        ):
            try:
                cached_resp = self.query_cache.get(
                    query=query,
                    candidate_k=candidate_k,
                    rerank_top_k=rerank_top_k,
                    context_max_tokens=eff_max_tokens,
                    context_max_chunks=eff_max_chunks,
                    bypass_reranking=bypass_reranking,
                    model=eff_model,
                    temperature=eff_temp,
                    system_prompt=eff_prompt,
                    corpus_fingerprint=corpus_fp,
                    tenant_id=tenant_id,
                    max_tokens=generation_max_tokens,
                    min_relevance_score=eff_min_score,
                    ordering_strategy=eff_ordering,
                    rerank_blending_alpha=eff_blending_alpha,
                )
                if cached_resp is not None:
                    total_cache_ms = (time.perf_counter() - overall_start) * 1000.0
                    cached_resp.diagnostics.total_ms = round(total_cache_ms, 3)
                    if cached_resp.diagnostics.generation_diagnostics is None:
                        cached_resp.diagnostics.generation_diagnostics = {}
                    cached_resp.diagnostics.generation_diagnostics["cache_hit"] = True
                    self.last_diagnostics = cached_resp.diagnostics
                    return cached_resp
            except Exception as exc:
                logger.warning(f"Query cache read failed safely: {exc}", exc_info=True)

        # Step 1: Candidate Retrieval
        retrieval_start = time.perf_counter()
        hybrid_diag: dict[str, Any] | None = None

        if isinstance(self.retriever, HybridRetriever):
            hybrid_resp: HybridRetrievalResponse = await self.retriever.search_async(
                query=query,
                top_k=candidate_k,
                **retriever_kwargs,
            )
            candidates = hybrid_resp.results
            hybrid_diag = hybrid_resp.diagnostics.to_dict()
        else:
            candidates = await asyncio.to_thread(self.retriever.search, query, candidate_k)

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # Step 2: CrossEncoder Reranking
        rerank_start = time.perf_counter()
        rerank_diag_dict: dict[str, Any] | None = None

        if self.reranker is not None and not bypass_reranking:
            # Run neural reranking (CPU/GPU bound) in thread with optional reciprocal rank blending
            reranked_results = await asyncio.to_thread(
                self.reranker.rerank,
                query,
                candidates,
                rerank_top_k,
                eff_blending_alpha,
            )
            if self.reranker.last_diagnostics:
                rerank_diag_dict = self.reranker.last_diagnostics.to_dict()
        else:
            # Bypass reranker: take top rerank_top_k from candidates directly
            reranked_results = list(candidates[:rerank_top_k])

        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0

        # Step 3: Context Assembly
        assembly_start = time.perf_counter()
        assembled_context = self.context_assembler.assemble(
            candidates=reranked_results,
            max_tokens=context_max_tokens,
            max_chunks=context_max_chunks,
            min_relevance_score=min_relevance_score,
            ordering_strategy=ordering_strategy,
        )
        assembly_ms = (time.perf_counter() - assembly_start) * 1000.0

        # Step 4: Grounded LLM Generation (Optional)
        gen_response: GenerationResponse | None = None
        gen_ms: float = 0.0
        ttft_ms: float | None = None
        gen_diag_dict: dict[str, Any] | None = None
        gen_usage_dict: dict[str, Any] | None = None

        if generate_answer and self.generator is not None:
            gen_start = time.perf_counter()
            gen_response = await self.generator.generate_answer(
                query=query,
                context=assembled_context,
                temperature=generation_temperature,
                max_tokens=generation_max_tokens,
                system_prompt=generation_system_prompt,
                model=generation_model,
            )
            gen_ms = (time.perf_counter() - gen_start) * 1000.0
            ttft_ms = gen_response.diagnostics.ttft_ms
            gen_diag_dict = gen_response.diagnostics.to_dict()
            gen_usage_dict = gen_response.usage.to_dict()

        total_ms = (time.perf_counter() - overall_start) * 1000.0

        diagnostics = PipelineDiagnostics(
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            assembly_ms=assembly_ms,
            generation_ms=gen_ms,
            total_ms=total_ms,
            candidate_count=len(candidates),
            reranked_count=len(reranked_results),
            selected_context_count=assembled_context.total_chunks_selected,
            ttft_ms=ttft_ms,
            hybrid_breakdown=hybrid_diag,
            rerank_diagnostics=rerank_diag_dict,
            generation_diagnostics=gen_diag_dict,
            usage=gen_usage_dict,
        )

        self.last_diagnostics = diagnostics
        response = PipelineResponse(
            query=query,
            candidates=candidates,
            reranked_results=reranked_results,
            context=assembled_context,
            diagnostics=diagnostics,
            generation=gen_response,
        )

        # Populate Tier 1 Query Response Cache on successful generation
        if (
            generate_answer
            and gen_response is not None
            and self.query_cache is not None
            and getattr(self.query_cache, "enabled", False)
        ):
            try:
                self.query_cache.set(
                    response=response,
                    candidate_k=candidate_k,
                    rerank_top_k=rerank_top_k,
                    context_max_tokens=eff_max_tokens,
                    context_max_chunks=eff_max_chunks,
                    bypass_reranking=bypass_reranking,
                    model=eff_model,
                    temperature=eff_temp,
                    system_prompt=eff_prompt,
                    corpus_fingerprint=corpus_fp,
                    tenant_id=tenant_id,
                    max_tokens=generation_max_tokens,
                    min_relevance_score=eff_min_score,
                    ordering_strategy=eff_ordering,
                    rerank_blending_alpha=eff_blending_alpha,
                )
            except Exception as exc:
                logger.warning(f"Query cache write failed safely: {exc}", exc_info=True)

        return response

    async def execute_stream_async(
        self,
        query: str,
        candidate_k: int = 20,
        rerank_top_k: int = 10,
        context_max_tokens: int | None = None,
        context_max_chunks: int | None = None,
        bypass_reranking: bool = False,
        generation_temperature: float | None = None,
        generation_max_tokens: int | None = None,
        generation_system_prompt: str | None = None,
        generation_model: str | None = None,
        min_relevance_score: float | None = None,
        ordering_strategy: str | None = None,
        rerank_blending_alpha: float | None = None,
        **retriever_kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Asynchronously stream RAG pipeline execution via Server-Sent Events (SSE).

        Yields:
            1. 'context' event containing assembled context and retrieval telemetry.
            2. 'token' events containing incremental text deltas as emitted by LLM provider.
            3. 'done' event with final usage statistics and generation diagnostics.
        """
        overall_start = time.perf_counter()

        # Step 0: Check Tier 1 Query Response Cache
        eff_temp = generation_temperature if generation_temperature is not None else (
            self.generator.default_temperature if self.generator else 0.0
        )
        eff_model = generation_model or (self.generator.model_name if self.generator else "default")
        eff_prompt = generation_system_prompt or (self.generator.default_system_prompt if self.generator else None)
        eff_max_tokens = context_max_tokens if context_max_tokens is not None else self.context_assembler.max_tokens
        eff_max_chunks = context_max_chunks if context_max_chunks is not None else self.context_assembler.max_chunks
        eff_min_score = min_relevance_score if min_relevance_score is not None else self.context_assembler.min_relevance_score
        eff_ordering = ordering_strategy if ordering_strategy is not None else self.context_assembler.ordering_strategy
        eff_blending_alpha = (
            float(rerank_blending_alpha)
            if rerank_blending_alpha is not None
            else self.rerank_blending_alpha
        )
        if eff_blending_alpha < 0.0:
            raise ValueError(
                f"rerank_blending_alpha must be non-negative (>= 0.0), got {eff_blending_alpha}"
            )
        tenant_id = retriever_kwargs.pop("tenant_id", None)

        index_info = self.get_index_info()
        corpus_fp = index_info.corpus_fingerprint if index_info else ""

        if self.query_cache is not None and getattr(self.query_cache, "enabled", False):
            cached_resp = None
            try:
                cached_resp = self.query_cache.get(
                    query=query,
                    candidate_k=candidate_k,
                    rerank_top_k=rerank_top_k,
                    context_max_tokens=eff_max_tokens,
                    context_max_chunks=eff_max_chunks,
                    bypass_reranking=bypass_reranking,
                    model=eff_model,
                    temperature=eff_temp,
                    system_prompt=eff_prompt,
                    corpus_fingerprint=corpus_fp,
                    tenant_id=tenant_id,
                    max_tokens=generation_max_tokens,
                    min_relevance_score=eff_min_score,
                    ordering_strategy=eff_ordering,
                    rerank_blending_alpha=eff_blending_alpha,
                )
            except Exception as exc:
                logger.warning(f"Query cache read failed safely during stream: {exc}", exc_info=True)

            if cached_resp is not None and cached_resp.generation is not None:
                # Replay cached response over SSE with zero retrieval/generation wait
                yield StreamEvent(
                    event="context",
                    data={
                        "query": query,
                        "retrieval_ms": 0.0,
                        "rerank_ms": 0.0,
                        "assembly_ms": 0.0,
                        "selected_chunks": [c.to_dict() for c in cached_resp.context.selected_chunks],
                        "approximate_tokens": cached_resp.context.approximate_token_count,
                        "cache_hit": True,
                    },
                )
                words = cached_resp.generation.answer.split(" ")
                for i, word in enumerate(words):
                    delta = word if i == 0 else " " + word
                    yield StreamEvent(
                        event="token",
                        data={"delta": delta, "is_first": (i == 0)},
                    )
                total_ms = (time.perf_counter() - overall_start) * 1000.0
                yield StreamEvent(
                    event="done",
                    data={
                        "finish_reason": cached_resp.generation.finish_reason or "stop",
                        "usage": cached_resp.generation.usage.to_dict() if cached_resp.generation.usage else None,
                        "diagnostics": {"cache_hit": True, "total_ms": round(total_ms, 2)},
                        "total_pipeline_ms": round(total_ms, 2),
                    },
                )
                return

        # Step 1: Candidate Retrieval
        retrieval_start = time.perf_counter()
        if isinstance(self.retriever, HybridRetriever):
            hybrid_resp = await self.retriever.search_async(
                query=query, top_k=candidate_k, **retriever_kwargs
            )
            candidates = hybrid_resp.results
            hybrid_diag = hybrid_resp.diagnostics.to_dict()
        else:
            candidates = await asyncio.to_thread(self.retriever.search, query, candidate_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # Step 2: CrossEncoder Reranking
        rerank_start = time.perf_counter()
        if self.reranker is not None and not bypass_reranking:
            reranked_results = await asyncio.to_thread(
                self.reranker.rerank, query, candidates, rerank_top_k, eff_blending_alpha
            )
        else:
            reranked_results = list(candidates[:rerank_top_k])
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0

        # Step 3: Context Assembly
        assembly_start = time.perf_counter()
        assembled_context = self.context_assembler.assemble(
            candidates=reranked_results,
            max_tokens=context_max_tokens,
            max_chunks=context_max_chunks,
            min_relevance_score=min_relevance_score,
            ordering_strategy=ordering_strategy,
        )
        assembly_ms = (time.perf_counter() - assembly_start) * 1000.0

        # Emit context event
        yield StreamEvent(
            event="context",
            data={
                "query": query,
                "retrieval_ms": round(retrieval_ms, 2),
                "rerank_ms": round(rerank_ms, 2),
                "assembly_ms": round(assembly_ms, 2),
                "selected_chunks": [c.to_dict() for c in assembled_context.selected_chunks],
                "approximate_tokens": assembled_context.approximate_token_count,
            },
        )

        # Step 4: Stream Generation
        if self.generator is not None:
            async for chunk in self.generator.stream_answer(
                query=query,
                context=assembled_context,
                temperature=generation_temperature,
                max_tokens=generation_max_tokens,
                system_prompt=generation_system_prompt,
                model=generation_model,
            ):
                if chunk.is_final:
                    total_ms = (time.perf_counter() - overall_start) * 1000.0
                    yield StreamEvent(
                        event="done",
                        data={
                            "finish_reason": chunk.finish_reason or "stop",
                            "usage": chunk.usage.to_dict() if chunk.usage else None,
                            "diagnostics": chunk.diagnostics.to_dict() if chunk.diagnostics else None,
                            "total_pipeline_ms": round(total_ms, 2),
                        },
                    )
                else:
                    yield StreamEvent(
                        event="token",
                        data={
                            "delta": chunk.delta,
                            "is_first": chunk.is_first,
                        },
                    )

    def execute(
        self,
        query: str,
        candidate_k: int = 20,
        rerank_top_k: int = 10,
        context_max_tokens: int | None = None,
        context_max_chunks: int | None = None,
        bypass_reranking: bool = False,
        generate_answer: bool = False,
        rerank_blending_alpha: float | None = None,
        **kwargs: Any,
    ) -> PipelineResponse:
        """Synchronous wrapper for execute_async."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.execute_async(
                        query=query,
                        candidate_k=candidate_k,
                        rerank_top_k=rerank_top_k,
                        context_max_tokens=context_max_tokens,
                        context_max_chunks=context_max_chunks,
                        bypass_reranking=bypass_reranking,
                        generate_answer=generate_answer,
                        rerank_blending_alpha=rerank_blending_alpha,
                        **kwargs,
                    ),
                )
                return future.result()
        else:
            return asyncio.run(
                self.execute_async(
                    query=query,
                    candidate_k=candidate_k,
                    rerank_top_k=rerank_top_k,
                    context_max_tokens=context_max_tokens,
                    context_max_chunks=context_max_chunks,
                    bypass_reranking=bypass_reranking,
                    generate_answer=generate_answer,
                    rerank_blending_alpha=rerank_blending_alpha,
                    **kwargs,
                )
            )

    # BaseRetriever Interface Compliance
    def search(self, query: str, top_k: int = 10, **kwargs: Any) -> list[RetrievalResult]:
        """Execute search conforming to BaseRetriever interface."""
        candidate_k = kwargs.pop("candidate_k", max(top_k * 2, 20))
        rerank_blending_alpha = kwargs.pop("rerank_blending_alpha", None)
        resp = self.execute(
            query=query,
            candidate_k=candidate_k,
            rerank_top_k=top_k,
            rerank_blending_alpha=rerank_blending_alpha,
            **kwargs,
        )
        return resp.reranked_results

    @property
    def is_indexed(self) -> bool:
        """Return True if base retriever is indexed."""
        return self.retriever.is_indexed

    def build_index(self, chunks: Sequence[Chunk]) -> IndexInfo:
        """Delegate build_index to base retriever."""
        return self.retriever.build_index(chunks)

    def save_index(self, directory_path: Path | str) -> None:
        """Delegate save_index to base retriever."""
        self.retriever.save_index(directory_path)

    def load_index(self, directory_path: Path | str) -> IndexInfo:
        """Delegate load_index to base retriever."""
        return self.retriever.load_index(directory_path)

    def get_index_info(self) -> IndexInfo | None:
        """Return index info augmented with reranker metadata."""
        base_info = self.retriever.get_index_info()
        if base_info is None:
            return None
        return IndexInfo(
            retriever_type=f"{base_info.retriever_type}+rerank" if self.reranker else base_info.retriever_type,
            corpus_fingerprint=base_info.corpus_fingerprint,
            chunk_count=base_info.chunk_count,
            document_count=base_info.document_count,
            hyperparameters={
                "base_index": base_info.hyperparameters,
                "reranker_model": self.reranker.model_name if self.reranker else None,
            },
        )


RetrievalPipeline = RAGPipeline

