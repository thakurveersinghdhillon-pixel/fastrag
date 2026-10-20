"""FastRAG v1 API Routes.

Exposes REST endpoints for non-streaming grounded answer generation and real-time SSE streaming.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_pipeline
from app.core.exceptions import (
    GenerationError,
    PipelineError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    RerankingError,
    RetrievalError,
)
from app.core.logging_config import get_logger
from app.generation.models import StreamEvent
from app.retrieval.pipeline import RAGPipeline
from app.schemas.query import QueryRequest, QueryResponse

logger = get_logger("fastrag.api.v1.routes")

router = APIRouter(tags=["RAG Generation"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Execute grounded RAG query",
    description="Executes candidate retrieval, CrossEncoder reranking, context assembly, and LLM answer generation.",
)
async def query_endpoint(
    payload: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """Execute end-to-end RAG query and return structured JSON answer."""
    try:
        response = await pipeline.execute_async(
            query=payload.query,
            candidate_k=payload.candidate_k,
            rerank_top_k=payload.rerank_top_k,
            context_max_tokens=payload.context_max_tokens,
            context_max_chunks=payload.context_max_chunks,
            bypass_reranking=payload.bypass_reranking,
            generate_answer=True,
            generation_temperature=payload.temperature,
            generation_max_tokens=payload.max_tokens,
            generation_system_prompt=payload.system_prompt,
            generation_model=payload.model,
            min_relevance_score=payload.min_relevance_score,
            ordering_strategy=payload.ordering_strategy,
        )
    except ProviderAuthenticationError as exc:
        logger.error(f"Provider authentication error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"LLM provider authentication failed: {exc}",
        ) from exc
    except ProviderTimeoutError as exc:
        logger.error(f"Provider timeout error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"LLM provider timeout: {exc}",
        ) from exc
    except ProviderError as exc:
        logger.error(f"Provider error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM provider error: {exc}",
        ) from exc
    except (GenerationError, RetrievalError, RerankingError, PipelineError) as exc:
        logger.error(f"Pipeline execution error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG execution error: {exc}",
        ) from exc

    gen = response.generation
    if gen is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pipeline completed without generation response",
        )

    citation_map = {c.citation_id: c.document_id for c in response.context.selected_chunks}
    return QueryResponse(
        query=response.query,
        answer=gen.answer,
        model=gen.model,
        provider=gen.provider,
        finish_reason=gen.finish_reason,
        cited_doc_ids=gen.cited_doc_ids,
        context_insufficient=gen.context_insufficient,
        context={
            "total_chunks_selected": response.context.total_chunks_selected,
            "approximate_tokens": response.context.approximate_token_count,
            "citations": citation_map,
        },
        diagnostics=response.diagnostics.to_dict(),
        usage=gen.usage.to_dict(),
    )


@router.post(
    "/query/stream",
    summary="Execute grounded RAG query with real-time SSE streaming",
    description="Streams context assembly, incremental tokens, and completion metadata via Server-Sent Events (SSE).",
)
async def query_stream_endpoint(
    payload: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> StreamingResponse:
    """Stream RAG response tokens incrementally via text/event-stream."""

    async def sse_event_generator() -> AsyncIterator[str]:
        try:
            async for event in pipeline.execute_stream_async(
                query=payload.query,
                candidate_k=payload.candidate_k,
                rerank_top_k=payload.rerank_top_k,
                context_max_tokens=payload.context_max_tokens,
                context_max_chunks=payload.context_max_chunks,
                bypass_reranking=payload.bypass_reranking,
                generation_temperature=payload.temperature,
                generation_max_tokens=payload.max_tokens,
                generation_system_prompt=payload.system_prompt,
                generation_model=payload.model,
            ):
                yield event.to_sse()
        except asyncio.CancelledError:
            logger.info("Client disconnected from SSE stream", extra={"query": payload.query})
            raise
        except Exception as exc:
            logger.error(f"Exception during SSE stream generation: {exc}", exc_info=True)
            err_event = StreamEvent(
                event="error",
                data={"error": str(exc), "type": type(exc).__name__},
            )
            yield err_event.to_sse()

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
