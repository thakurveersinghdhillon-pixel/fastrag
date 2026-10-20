"""Integration tests for the FastRAG Retrieval-Reranking-Assembly Pipeline."""

import pytest
from app.core.exceptions import RetrievalError
from app.ingestion.models import Chunk
from app.retrieval.base import BaseRetriever, IndexInfo, RetrievalResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.context_assembler import ContextAssembler
from app.retrieval.dense import DenseRetriever, MockEmbeddingEncoder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker, MockCrossEncoderScorer


def _build_test_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="c_hnsw",
            document_id="doc_vector",
            chunk_index=0,
            text="HNSW is a graph-based approximate nearest neighbor vector indexing structure.",
            source="test_corpus.jsonl",
            start_offset=0,
            end_offset=80,
            metadata={"title": "Vector Indexing with HNSW"},
        ),
        Chunk(
            chunk_id="c_bm25",
            document_id="doc_lexical",
            chunk_index=0,
            text="Okapi BM25 is a probabilistic term matching retrieval algorithm.",
            source="test_corpus.jsonl",
            start_offset=0,
            end_offset=67,
            metadata={"title": "BM25 Scoring"},
        ),
        Chunk(
            chunk_id="c_rrf",
            document_id="doc_fusion",
            chunk_index=0,
            text="Reciprocal Rank Fusion merges multiple ranked lists without score calibration.",
            source="test_corpus.jsonl",
            start_offset=0,
            end_offset=78,
            metadata={"title": "RRF Fusion"},
        ),
    ]


@pytest.fixture
def mock_retrievers():
    chunks = _build_test_chunks()
    bm25 = BM25Retriever()
    bm25.build_index(chunks)

    dense = DenseRetriever(encoder=MockEmbeddingEncoder(dimension=32))
    dense.build_index(chunks)

    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25, top_k=5)
    return bm25, dense, hybrid


def test_pipeline_end_to_end_execution(mock_retrievers):
    """Verify full query -> Hybrid -> Rerank -> Assemble pipeline execution."""
    bm25, dense, hybrid = mock_retrievers

    fixed_scores = {
        "Reciprocal Rank Fusion merges multiple ranked lists without score calibration.": 9.9,
        "HNSW is a graph-based approximate nearest neighbor vector indexing structure.": 4.5,
    }
    reranker = CrossEncoderReranker(scorer=MockCrossEncoderScorer(fixed_scores=fixed_scores))
    assembler = ContextAssembler(max_tokens=500, max_chunks=3)

    pipeline = RAGPipeline(
        retriever=hybrid,
        reranker=reranker,
        context_assembler=assembler,
    )

    response = pipeline.execute("What is Reciprocal Rank Fusion?", candidate_k=3, rerank_top_k=2)

    # 1. Candidate retrieval
    assert len(response.candidates) > 0
    # 2. Reranking
    assert len(response.reranked_results) == 2
    assert response.reranked_results[0].chunk_id == "c_rrf"
    assert response.reranked_results[0].score == 9.9
    assert response.reranked_results[0].score_type == "cross_encoder"
    # 3. Context assembly
    assert response.context.total_chunks_selected == 2
    assert response.context.selected_chunk_ids[0] == "c_rrf"
    assert "[Doc-1] Title: RRF Fusion" in response.context.context_text
    # 4. Telemetry diagnostics
    assert response.diagnostics.candidate_count > 0
    assert response.diagnostics.reranked_count == 2
    assert response.diagnostics.selected_context_count == 2
    assert response.diagnostics.total_ms > 0


def test_pipeline_bypass_reranking(mock_retrievers):
    """Verify that bypass_reranking skips the neural reranker without error."""
    bm25, dense, hybrid = mock_retrievers

    pipeline = RAGPipeline(retriever=hybrid, reranker=None)
    response = pipeline.execute("vector indexing", bypass_reranking=True, candidate_k=2, rerank_top_k=2)

    assert len(response.reranked_results) == 2
    assert response.diagnostics.rerank_ms >= 0.0
    # Result scores should remain original retrieval scores (RRF)
    assert response.reranked_results[0].score_type == "rrf"


def test_pipeline_bm25_only_workflow_without_cross_encoder_loading(mock_retrievers):
    """Verify that BM25-only workflow does not require CrossEncoder loading."""
    bm25, _, _ = mock_retrievers

    # Pipeline with no reranker attached
    pipeline = RAGPipeline(retriever=bm25, reranker=None)
    response = pipeline.execute("Okapi BM25 algorithm", candidate_k=2)

    assert len(response.candidates) > 0
    assert response.candidates[0].score_type == "bm25"
    assert response.context.total_chunks_selected > 0


def test_pipeline_dense_only_workflow_without_cross_encoder_loading(mock_retrievers):
    """Verify that Dense-only workflow functions without CrossEncoder loading."""
    _, dense, _ = mock_retrievers

    pipeline = RAGPipeline(retriever=dense, reranker=None)
    response = pipeline.execute("HNSW graph", candidate_k=2)

    assert len(response.candidates) > 0
    assert response.candidates[0].score_type == "cosine_similarity"
    assert response.context.total_chunks_selected > 0


@pytest.mark.asyncio
async def test_pipeline_async_execution(mock_retrievers):
    """Verify execute_async method operates correctly with asyncio."""
    _, _, hybrid = mock_retrievers
    reranker = CrossEncoderReranker(scorer=MockCrossEncoderScorer())

    pipeline = RAGPipeline(retriever=hybrid, reranker=reranker)
    response = await pipeline.execute_async("test query", candidate_k=2, rerank_top_k=2)

    assert response.query == "test query"
    assert response.diagnostics.total_ms > 0


def test_pipeline_rerank_blending_alpha_execution(mock_retrievers):
    """Verify that rerank_blending_alpha applies blending and preserves telemetry."""
    _, _, hybrid = mock_retrievers
    scorer = MockCrossEncoderScorer()
    reranker = CrossEncoderReranker(scorer=scorer)

    pipeline = RAGPipeline(retriever=hybrid, reranker=reranker, rerank_blending_alpha=0.5)
    response = pipeline.execute("vector indexing", candidate_k=3, rerank_top_k=2)

    assert len(response.reranked_results) == 2
    assert response.reranked_results[0].score_type == "blended_rrf"
    assert "rerank_score" in response.reranked_results[0].metadata
    assert response.reranked_results[0].metadata["blended_alpha"] == 0.5


def test_pipeline_blending_with_min_relevance_score_thresholding(mock_retrievers):
    """Verify Phase 9.4 min_relevance_score filters based on CrossEncoder logit when blending."""
    _, _, hybrid = mock_retrievers

    fixed_scores = {
        "HNSW is a graph-based approximate nearest neighbor vector indexing structure.": 8.0,
        "Okapi BM25 is a probabilistic term matching retrieval algorithm.": -8.0,  # Below threshold
        "Reciprocal Rank Fusion merges multiple ranked lists without score calibration.": -9.0,  # Below threshold
    }
    scorer = MockCrossEncoderScorer(fixed_scores=fixed_scores)
    reranker = CrossEncoderReranker(scorer=scorer)
    assembler = ContextAssembler(min_relevance_score=0.0)

    pipeline = RAGPipeline(
        retriever=hybrid,
        reranker=reranker,
        context_assembler=assembler,
        rerank_blending_alpha=1.0,
    )

    response = pipeline.execute("indexing", candidate_k=3, rerank_top_k=3)

    # 3 reranked results returned by reranker with blending
    assert len(response.reranked_results) == 3
    # But only 1 candidate had CrossEncoder score >= 0.0, so context assembler selects only that 1!
    assert response.context.total_chunks_selected == 1
    assert response.context.selected_chunk_ids == ["c_hnsw"]
    assert response.context.selected_chunks[0].rerank_score == 8.0


def test_pipeline_blending_with_boundary_ordering(mock_retrievers):
    """Verify Phase 9.4 boundary ordering correctly arranges blended candidates."""
    _, _, hybrid = mock_retrievers

    fixed_scores = {
        "HNSW is a graph-based approximate nearest neighbor vector indexing structure.": 10.0,
        "Okapi BM25 is a probabilistic term matching retrieval algorithm.": 5.0,
        "Reciprocal Rank Fusion merges multiple ranked lists without score calibration.": 2.0,
    }
    scorer = MockCrossEncoderScorer(fixed_scores=fixed_scores)
    reranker = CrossEncoderReranker(scorer=scorer)
    assembler = ContextAssembler(ordering_strategy="boundary")

    pipeline = RAGPipeline(
        retriever=hybrid,
        reranker=reranker,
        context_assembler=assembler,
        rerank_blending_alpha=0.5,
    )

    response = pipeline.execute("search", candidate_k=3, rerank_top_k=3)

    # 3 candidates: rank 1 first, rank 2 last, rank 3 middle
    assert response.context.total_chunks_selected == 3
    assert response.context.selected_chunks[0].rank == 1
    assert response.context.selected_chunks[1].rank == 3
    assert response.context.selected_chunks[2].rank == 2


def test_pipeline_negative_alpha_raises_value_error(mock_retrievers):
    """Verify negative rerank_blending_alpha raises ValueError."""
    _, _, hybrid = mock_retrievers

    with pytest.raises(ValueError, match="rerank_blending_alpha"):
        RAGPipeline(retriever=hybrid, rerank_blending_alpha=-0.2)

    pipeline = RAGPipeline(retriever=hybrid)
    with pytest.raises(ValueError, match="rerank_blending_alpha"):
        pipeline.execute("query", rerank_blending_alpha=-1.0)

