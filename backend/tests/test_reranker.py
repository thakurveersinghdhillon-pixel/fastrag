"""Tests for CrossEncoder Reranking Module."""

import pytest
from app.core.exceptions import RetrievalError
from app.retrieval.base import RetrievalResult
from app.retrieval.reranker import (
    CrossEncoderReranker,
    CrossEncoderScorer,
    MockCrossEncoderScorer,
    RerankDiagnostics,
    SentenceTransformerCrossEncoder,
)


def _make_candidate(
    chunk_id: str,
    document_id: str,
    rank: int,
    score: float,
    score_type: str = "rrf",
    text: str = "Sample chunk text",
    metadata: dict | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        rank=rank,
        score=score,
        score_type=score_type,
        text=text,
        source="test_corpus.jsonl",
        metadata=metadata or {"original_info": "preserved"},
    )


def test_reranker_ordering_with_mock_scores():
    """Verify that candidates are ordered by descending CrossEncoder scores."""
    c1 = _make_candidate("chunk_1", "doc_A", 1, 0.032, text="Passage about vector indexing")
    c2 = _make_candidate("chunk_2", "doc_B", 2, 0.031, text="Passage about BM25 ranking")
    c3 = _make_candidate("chunk_3", "doc_C", 3, 0.030, text="Passage about reciprocal rank fusion")

    # Give chunk_3 highest score, chunk_1 middle, chunk_2 lowest
    fixed_scores = {
        "Passage about reciprocal rank fusion": 9.5,
        "Passage about vector indexing": 5.0,
        "Passage about BM25 ranking": 1.2,
    }
    scorer = MockCrossEncoderScorer(fixed_scores=fixed_scores)
    reranker = CrossEncoderReranker(scorer=scorer)

    results = reranker.rerank("What is RRF?", [c1, c2, c3], top_k=3)

    assert len(results) == 3
    assert [r.chunk_id for r in results] == ["chunk_3", "chunk_1", "chunk_2"]
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].score == 9.5
    assert results[0].score_type == "cross_encoder"


def test_reranker_preserves_original_scores_and_metadata():
    """Verify that original retrieval scores, ranks, and metadata are preserved."""
    c1 = _make_candidate(
        "chunk_1",
        "doc_A",
        1,
        0.75,
        score_type="cosine_similarity",
        text="Text A",
        metadata={"custom_key": "val123", "rrf_ranks": {"dense": 1, "bm25": 4}},
    )
    scorer = MockCrossEncoderScorer(fixed_scores={"Text A": 8.0})
    reranker = CrossEncoderReranker(scorer=scorer)

    results = reranker.rerank("query", [c1], top_k=1)
    assert len(results) == 1
    r = results[0]

    assert r.score == 8.0
    assert r.score_type == "cross_encoder"
    assert r.metadata["original_score"] == 0.75
    assert r.metadata["original_score_type"] == "cosine_similarity"
    assert r.metadata["original_rank"] == 1
    assert r.metadata["rerank_score"] == 8.0
    assert r.metadata["custom_key"] == "val123"
    assert r.metadata["rrf_ranks"] == {"dense": 1, "bm25": 4}
    assert r.metadata["rerank_model"] == "mock-cross-encoder-v1"


def test_reranker_deterministic_tie_breaking():
    """Verify that ties on identical score are broken deterministically by chunk_id ascending."""
    # Three candidates with identical scores
    c_b = _make_candidate("chunk_B", "doc_B", 1, 0.5, text="Identical score text B")
    c_c = _make_candidate("chunk_C", "doc_C", 2, 0.5, text="Identical score text C")
    c_a = _make_candidate("chunk_A", "doc_A", 3, 0.5, text="Identical score text A")

    # Scorer assigns exact same score to all
    scorer = MockCrossEncoderScorer(score_fn=lambda q, p: 4.0)
    reranker = CrossEncoderReranker(scorer=scorer)

    # Pass in shuffled order
    results = reranker.rerank("query", [c_c, c_b, c_a], top_k=3)
    assert [r.chunk_id for r in results] == ["chunk_A", "chunk_B", "chunk_C"]


def test_reranker_empty_and_whitespace_inputs():
    """Verify safe handling of empty candidates list and empty query strings."""
    reranker = CrossEncoderReranker(scorer=MockCrossEncoderScorer())

    # Empty candidate list
    res_empty_cands = reranker.rerank("query", [])
    assert res_empty_cands == []
    assert reranker.last_diagnostics is not None
    assert reranker.last_diagnostics.candidate_count == 0

    # Empty query
    c1 = _make_candidate("c1", "d1", 1, 0.5, text="Valid text")
    res_empty_q = reranker.rerank("", [c1])
    assert res_empty_q == []
    res_ws_q = reranker.rerank("   \t  \n ", [c1])
    assert res_ws_q == []


def test_reranker_single_candidate():
    """Verify behavior on a single candidate."""
    c1 = _make_candidate("c1", "d1", 5, 0.2, text="Single passage")
    scorer = MockCrossEncoderScorer(fixed_scores={"Single passage": 3.14})
    reranker = CrossEncoderReranker(scorer=scorer)

    results = reranker.rerank("query", [c1], top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].rank == 1
    assert results[0].score == 3.14


def test_reranker_duplicate_chunk_ids_handled_safely():
    """Verify that duplicate candidate chunk_ids are not scored multiple times."""
    scored_calls = 0

    def mock_counting_fn(q: str, p: str) -> float:
        nonlocal scored_calls
        scored_calls += 1
        return 1.0

    scorer = MockCrossEncoderScorer(score_fn=mock_counting_fn)
    reranker = CrossEncoderReranker(scorer=scorer)

    c1 = _make_candidate("c1", "d1", 1, 0.9, text="Text 1")
    c1_dup = _make_candidate("c1", "d1", 2, 0.8, text="Text 1 duplicate")
    c2 = _make_candidate("c2", "d2", 3, 0.7, text="Text 2")

    results = reranker.rerank("query", [c1, c1_dup, c2], top_k=10)

    # Only 2 unique chunk IDs should have been scored
    assert scored_calls == 2
    assert len(results) == 2
    assert [r.chunk_id for r in results] == ["c1", "c2"]


def test_reranker_invalid_top_k():
    """Verify validation of top_k argument."""
    reranker = CrossEncoderReranker(scorer=MockCrossEncoderScorer())
    c1 = _make_candidate("c1", "d1", 1, 0.9, text="Text")

    with pytest.raises(ValueError, match="top_k must be strictly positive"):
        reranker.rerank("query", [c1], top_k=0)

    with pytest.raises(ValueError, match="top_k must be strictly positive"):
        reranker.rerank("query", [c1], top_k=-5)

    with pytest.raises(TypeError, match="top_k must be an integer"):
        reranker.rerank("query", [c1], top_k="10")  # type: ignore[arg-type]


def test_reranker_scorer_failure_raises_retrieval_error():
    """Verify that scorer exceptions are wrapped cleanly in RetrievalError."""
    class FailingScorer(CrossEncoderScorer):
        @property
        def model_name(self) -> str:
            return "failing-model"

        def score_pairs(self, pairs):
            raise RuntimeError("CUDA Out of Memory in model layer")

    reranker = CrossEncoderReranker(scorer=FailingScorer())
    c1 = _make_candidate("c1", "d1", 1, 0.9, text="Text")

    with pytest.raises(RetrievalError, match="CUDA Out of Memory in model layer"):
        reranker.rerank("query", [c1])


def test_reranker_lazy_loading_and_instance_reuse():
    """Verify that the model is loaded on demand and reused without reloading."""
    load_count = 0

    class TrackingEncoder(SentenceTransformerCrossEncoder):
        def _ensure_model(self):
            nonlocal load_count
            load_count += 1
            # Return dummy object simulating predict method
            class DummyModel:
                def predict(self, pairs, **kwargs):
                    return [5.0] * len(pairs)
            if self._model is None:
                self._model = DummyModel()
            return self._model

    encoder = TrackingEncoder(model_name="mock-lazy-model")
    reranker = CrossEncoderReranker(scorer=encoder)

    # Before rerank, model has not been loaded
    assert load_count == 0

    c1 = _make_candidate("c1", "d1", 1, 0.5, text="Text 1")
    c2 = _make_candidate("c2", "d2", 2, 0.4, text="Text 2")

    # First call loads model
    res1 = reranker.rerank("query 1", [c1])
    assert len(res1) == 1
    assert load_count == 1

    # Second call reuses existing loaded model
    res2 = reranker.rerank("query 2", [c2])
    assert len(res2) == 1
    assert load_count == 2  # _ensure_model called but returns cached self._model
    assert encoder._model is not None


def test_reranker_candidate_limit_respects_top_k():
    """Verify that top_k strictly limits the output candidate count."""
    candidates = [
        _make_candidate(f"c_{i}", f"d_{i}", i + 1, 1.0 / (i + 1), text=f"Text {i}")
        for i in range(15)
    ]
    scorer = MockCrossEncoderScorer()
    reranker = CrossEncoderReranker(scorer=scorer)

    results = reranker.rerank("query", candidates, top_k=4)
    assert len(results) == 4
    assert [r.rank for r in results] == [1, 2, 3, 4]


# Phase 9.5 Reciprocal Rank Blending Tests

def test_reranker_default_alpha_preserves_pure_cross_encoder():
    """Verify default blending_alpha=0.0 reproduces existing pure CrossEncoder behavior."""
    c1 = _make_candidate("chunk_1", "doc_A", 1, 0.9, text="Text 1")
    c2 = _make_candidate("chunk_2", "doc_B", 2, 0.8, text="Text 2")
    c3 = _make_candidate("chunk_3", "doc_C", 3, 0.7, text="Text 3")

    # CrossEncoder inverts the ranking
    fixed_scores = {"Text 1": -5.0, "Text 2": 2.0, "Text 3": 10.0}
    scorer = MockCrossEncoderScorer(fixed_scores=fixed_scores)
    reranker = CrossEncoderReranker(scorer=scorer, default_blending_alpha=0.0)

    # Calling with default alpha (None or 0.0)
    results = reranker.rerank("query", [c1, c2, c3], top_k=3)

    assert len(results) == 3
    assert [r.chunk_id for r in results] == ["chunk_3", "chunk_2", "chunk_1"]
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].score == 10.0
    assert results[0].score_type == "cross_encoder"
    assert results[0].metadata["rerank_score"] == 10.0
    assert "blended_score" not in results[0].metadata


def test_reranker_blending_promotes_suppressed_candidate():
    """Verify that blending_alpha > 0.0 promotes candidates suppressed by CrossEncoder."""
    # c1 is base rank 1, but CrossEncoder scores it low (-10.0)
    # c2 is base rank 2, CrossEncoder scores it highest (+5.0)
    # c3 is base rank 3, CrossEncoder scores it medium (-9.0)
    c1 = _make_candidate("chunk_1", "doc_A", 1, 0.95, text="Base Rank 1")
    c2 = _make_candidate("chunk_2", "doc_B", 2, 0.80, text="Base Rank 2")
    c3 = _make_candidate("chunk_3", "doc_C", 3, 0.70, text="Base Rank 3")

    fixed_scores = {
        "Base Rank 1": -10.0,
        "Base Rank 2": 5.0,
        "Base Rank 3": -9.0,
    }
    scorer = MockCrossEncoderScorer(fixed_scores=fixed_scores)
    reranker = CrossEncoderReranker(scorer=scorer)

    # With alpha=0.0: CE order is c2 (rank 1), c3 (rank 2), c1 (rank 3)
    res_ce = reranker.rerank("query", [c1, c2, c3], top_k=3, blending_alpha=0.0)
    assert [r.chunk_id for r in res_ce] == ["chunk_2", "chunk_3", "chunk_1"]

    # With alpha=1.0:
    # c2: CE rank 1, base rank 2 -> 1/(60+1) + 1.0/(60+2) = 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252
    # c1: CE rank 3, base rank 1 -> 1/(60+3) + 1.0/(60+1) = 1/63 + 1/61 = 0.01587 + 0.01639 = 0.03227
    # c3: CE rank 2, base rank 3 -> 1/(60+2) + 1.0/(60+3) = 1/62 + 1/63 = 0.01613 + 0.01587 = 0.03200
    # Thus c1 is promoted ahead of c3!
    res_blended = reranker.rerank("query", [c1, c2, c3], top_k=3, blending_alpha=1.0)
    assert [r.chunk_id for r in res_blended] == ["chunk_2", "chunk_1", "chunk_3"]
    assert res_blended[1].chunk_id == "chunk_1"
    assert res_blended[1].score_type == "blended_rrf"
    assert res_blended[1].metadata["rerank_score"] == -10.0  # CE logit preserved!
    assert res_blended[1].metadata["cross_encoder_rank"] == 3
    assert res_blended[1].metadata["original_rank"] == 1
    assert res_blended[1].metadata["blended_alpha"] == 1.0


def test_reranker_blending_deterministic_tie_breaking():
    """Verify that ties on identical blended score break deterministically by chunk_id ascending."""
    c_b = _make_candidate("chunk_B", "doc_B", 1, 0.5, text="Passage B")
    c_a = _make_candidate("chunk_A", "doc_A", 1, 0.5, text="Passage A")

    # Identical CE score and identical base rank -> identical blended score
    scorer = MockCrossEncoderScorer(score_fn=lambda q, p: 3.0)
    reranker = CrossEncoderReranker(scorer=scorer)

    # Pass in [c_b, c_a]
    results = reranker.rerank("query", [c_b, c_a], top_k=2, blending_alpha=0.5)
    assert len(results) == 2
    assert results[0].chunk_id == "chunk_A"
    assert results[1].chunk_id == "chunk_B"


def test_reranker_negative_alpha_raises_value_error():
    """Verify that negative blending alpha raises ValueError."""
    c1 = _make_candidate("chunk_1", "doc_A", 1, 0.5, text="Text")
    scorer = MockCrossEncoderScorer()

    with pytest.raises(ValueError, match="default_blending_alpha"):
        CrossEncoderReranker(scorer=scorer, default_blending_alpha=-0.5)

    reranker = CrossEncoderReranker(scorer=scorer)
    with pytest.raises(ValueError, match="blending_alpha"):
        reranker.rerank("query", [c1], blending_alpha=-1.0)

