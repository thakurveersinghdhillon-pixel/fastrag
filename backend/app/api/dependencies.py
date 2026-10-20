"""FastAPI dependencies for FastRAG API endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from fastapi import Request

from app.caching.embedding_cache import QueryEmbeddingCache
from app.caching.query_cache import QueryResponseCache
from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger
from app.generation.generator import RAGGenerator
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.context_assembler import ContextAssembler
from app.retrieval.dense import DenseRetriever, MockEmbeddingEncoder, SentenceTransformerEncoder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker, MockCrossEncoderScorer

logger = get_logger("fastrag.api.dependencies")

_pipeline_instance: RAGPipeline | None = None
_embedding_cache_instance: QueryEmbeddingCache | None = None
_query_cache_instance: QueryResponseCache | None = None


def get_embedding_cache(settings: Settings | None = None) -> QueryEmbeddingCache | None:
    """Return shared QueryEmbeddingCache singleton based on settings."""
    global _embedding_cache_instance
    cfg = settings or get_settings()
    if not cfg.enable_embedding_cache:
        return None
    if _embedding_cache_instance is None:
        _embedding_cache_instance = QueryEmbeddingCache(
            max_size=cfg.embedding_cache_size,
            enabled=cfg.enable_embedding_cache,
        )
    return _embedding_cache_instance


def get_query_cache(settings: Settings | None = None) -> QueryResponseCache | None:
    """Return shared QueryResponseCache singleton based on settings."""
    global _query_cache_instance
    cfg = settings or get_settings()
    if not cfg.enable_query_cache:
        return None
    if _query_cache_instance is None:
        _query_cache_instance = QueryResponseCache(
            max_size=cfg.query_cache_size,
            ttl_seconds=cfg.query_cache_ttl_seconds,
            enabled=cfg.enable_query_cache,
        )
    return _query_cache_instance


def create_pipeline(
    settings: Settings | None = None,
    mock_encoders: bool = False,
    mock_reranker: bool = False,
    warmup: bool = False,
) -> RAGPipeline:
    """Build a complete RAGPipeline configured from application settings."""
    cfg = settings or get_settings()

    # Determine project root and index directories
    project_root = Path(__file__).resolve().parents[3]
    indexes_root = project_root / "datasets" / "indexes"
    bm25_dir = indexes_root / "bm25"
    dense_dir = indexes_root / "dense"

    # Setup retriever
    if bm25_dir.exists() and dense_dir.exists():
        bm25 = BM25Retriever()
        bm25.load_index(bm25_dir)

        if mock_encoders or cfg.environment == "test":
            encoder = MockEmbeddingEncoder()
        else:
            encoder = SentenceTransformerEncoder()

        emb_cache = get_embedding_cache(cfg)
        dense = DenseRetriever(encoder=encoder, embedding_cache=emb_cache)
        dense.load_index(dense_dir)

        retriever = HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            dense_candidate_count=cfg.default_top_k * 2,
            bm25_candidate_count=cfg.default_top_k * 2,
            top_k=cfg.default_top_k,
        )
    else:
        logger.warning(
            "Index files not found; falling back to standalone BM25 retriever",
            extra={"bm25_dir": str(bm25_dir), "dense_dir": str(dense_dir)},
        )
        retriever = BM25Retriever()

    # Setup reranker
    if mock_reranker or cfg.environment == "test":
        reranker = CrossEncoderReranker(
            scorer=MockCrossEncoderScorer(),
            default_top_k=cfg.default_top_k,
        )
    else:
        reranker = CrossEncoderReranker(
            model_name=cfg.reranker_model_name,
            batch_size=cfg.reranker_batch_size,
            device=cfg.reranker_device,
            default_top_k=cfg.default_top_k,
        )

    # Setup context assembler
    assembler = ContextAssembler(
        max_tokens=cfg.context_max_tokens,
        max_chunks=cfg.context_max_chunks,
    )

    # Setup generator
    generator = RAGGenerator()
    q_cache = get_query_cache(cfg)

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        context_assembler=assembler,
        generator=generator,
        query_cache=q_cache,
    )
    if warmup:
        pipeline.warmup()
    return pipeline


def get_pipeline(request: Request) -> RAGPipeline:
    """FastAPI dependency to retrieve the shared RAGPipeline instance."""
    global _pipeline_instance

    if hasattr(request.app.state, "pipeline") and request.app.state.pipeline is not None:
        return request.app.state.pipeline

    if _pipeline_instance is None:
        _pipeline_instance = create_pipeline()

    return _pipeline_instance
