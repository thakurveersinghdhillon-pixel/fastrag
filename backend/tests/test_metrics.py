"""Tests for FastRAG Information Retrieval Metrics (Recall@K, MRR@K, nDCG@K)."""

import math
import pytest

from evaluation.metrics import (
    dcg_at_k,
    evaluate_query_metrics,
    idcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_recall_at_k_known_ranking() -> None:
    """Verify Recall@K calculation on known deterministic rankings."""
    qrels = {"d1": 2, "d2": 1, "d3": 0, "d4": 2}  # 3 relevant documents: d1, d2, d4
    retrieved = ["d3", "d1", "d5", "d2"]

    assert recall_at_k(retrieved, qrels, k=1) == 0.0  # d3 is irrelevant
    assert recall_at_k(retrieved, qrels, k=2) == pytest.approx(1 / 3)  # d1 retrieved
    assert recall_at_k(retrieved, qrels, k=3) == pytest.approx(1 / 3)  # d5 irrelevant
    assert recall_at_k(retrieved, qrels, k=4) == pytest.approx(2 / 3)  # d1 and d2 retrieved
    assert recall_at_k(retrieved, qrels, k=10) == pytest.approx(2 / 3)  # k > len(retrieved)


def test_reciprocal_rank_at_k_known_ranking() -> None:
    """Verify Reciprocal Rank / MRR@K on known deterministic rankings."""
    qrels = {"d1": 2, "d2": 1, "d3": 0}

    # First relevant at rank 1
    assert reciprocal_rank_at_k(["d1", "d2"], qrels, k=5) == 1.0

    # First relevant at rank 2
    assert reciprocal_rank_at_k(["d3", "d2"], qrels, k=5) == 0.5

    # First relevant at rank 3
    assert reciprocal_rank_at_k(["d3", "d4", "d1"], qrels, k=5) == pytest.approx(1 / 3)

    # Cutoff K drops the relevant document
    assert reciprocal_rank_at_k(["d3", "d4", "d1"], qrels, k=2) == 0.0


def test_mean_reciprocal_rank() -> None:
    """Verify MRR calculation across multiple query reciprocal ranks."""
    rrs = [1.0, 0.5, 0.25, 0.0]
    assert mean_reciprocal_rank(rrs) == pytest.approx(1.75 / 4)
    assert mean_reciprocal_rank([]) == 0.0


def test_ndcg_at_k_graded_relevance_hand_calculated() -> None:
    """Verify nDCG@K matches theoretical closed-form values."""
    qrels = {"d1": 2, "d2": 1, "d3": 0}
    # Ideal order: d1 (rel=2), d2 (rel=1)
    # IDCG@2 = (2^2 - 1)/log2(2) + (2^1 - 1)/log2(3) = 3 + 1/log2(3)
    idcg_2 = 3.0 + 1.0 / math.log2(3.0)

    # Perfect ranking: ["d1", "d2"]
    assert ndcg_at_k(["d1", "d2"], qrels, k=2) == 1.0

    # Suboptimal ranking: ["d2", "d1"]
    # DCG@2 = (2^1 - 1)/log2(2) + (2^2 - 1)/log2(3) = 1 + 3/log2(3)
    dcg_2 = 1.0 + 3.0 / math.log2(3.0)
    expected_ndcg = dcg_2 / idcg_2
    assert ndcg_at_k(["d2", "d1"], qrels, k=2) == pytest.approx(expected_ndcg)

    # Completely irrelevant ranking: ["d3", "d4"]
    assert ndcg_at_k(["d3", "d4"], qrels, k=2) == 0.0


def test_empty_retrieval_results() -> None:
    """Verify all metrics handle empty retrieval lists safely."""
    qrels = {"d1": 2}
    assert recall_at_k([], qrels, k=5) == 0.0
    assert reciprocal_rank_at_k([], qrels, k=5) == 0.0
    assert ndcg_at_k([], qrels, k=5) == 0.0
    assert dcg_at_k([], qrels, k=5) == 0.0


def test_no_relevant_documents_in_qrels() -> None:
    """Verify all metrics return 0.0 when no relevant documents exist for query."""
    qrels_empty: dict[str, int] = {}
    qrels_zero: dict[str, int] = {"d1": 0, "d2": 0}

    for qrels in [qrels_empty, qrels_zero]:
        assert recall_at_k(["d1", "d2"], qrels, k=5) == 0.0
        assert reciprocal_rank_at_k(["d1", "d2"], qrels, k=5) == 0.0
        assert ndcg_at_k(["d1", "d2"], qrels, k=5) == 0.0
        assert idcg_at_k(qrels, k=5) == 0.0


def test_k_equal_to_one() -> None:
    """Verify boundary condition K=1."""
    qrels = {"d1": 2, "d2": 2}
    assert recall_at_k(["d1", "d2"], qrels, k=1) == 0.5
    assert reciprocal_rank_at_k(["d1", "d2"], qrels, k=1) == 1.0
    assert ndcg_at_k(["d1", "d2"], qrels, k=1) == 1.0


def test_k_larger_than_retrieved() -> None:
    """Verify K larger than length of retrieved list."""
    qrels = {"d1": 2, "d2": 1}
    retrieved = ["d1"]
    assert recall_at_k(retrieved, qrels, k=100) == 0.5
    assert reciprocal_rank_at_k(retrieved, qrels, k=100) == 1.0
    assert ndcg_at_k(retrieved, qrels, k=100) == pytest.approx(3.0 / (3.0 + 1.0 / math.log2(3.0)))


def test_invalid_k_values() -> None:
    """Verify ValueError is raised on non-positive or non-integer K."""
    qrels = {"d1": 1}
    for invalid_k in [0, -1, -10]:
        with pytest.raises(ValueError):
            recall_at_k(["d1"], qrels, k=invalid_k)
        with pytest.raises(ValueError):
            reciprocal_rank_at_k(["d1"], qrels, k=invalid_k)
        with pytest.raises(ValueError):
            ndcg_at_k(["d1"], qrels, k=invalid_k)

    for bad_type_k in [1.5, "10", None, True]:
        with pytest.raises(TypeError):
            recall_at_k(["d1"], qrels, k=bad_type_k)  # type: ignore[arg-type]


def test_evaluate_query_metrics_batch_helper() -> None:
    """Verify evaluate_query_metrics parses metric specifications properly."""
    qrels = {"d1": 2, "d2": 1}
    retrieved = ["d1", "d2"]
    metrics = evaluate_query_metrics(retrieved, qrels, ["recall@1", "recall@2", "mrr@5", "ndcg@2"])

    assert metrics["recall@1"] == 0.5
    assert metrics["recall@2"] == 1.0
    assert metrics["mrr@5"] == 1.0
    assert metrics["ndcg@2"] == 1.0


def test_invalid_metric_specifications() -> None:
    """Verify error reporting on malformed metric strings."""
    with pytest.raises(ValueError, match="Invalid metric specification"):
        evaluate_query_metrics(["d1"], {"d1": 1}, ["invalid_metric"])

    with pytest.raises(ValueError, match="Invalid cutoff K"):
        evaluate_query_metrics(["d1"], {"d1": 1}, ["recall@abc"])

    with pytest.raises(ValueError, match="Unsupported metric type"):
        evaluate_query_metrics(["d1"], {"d1": 1}, ["f1@5"])


def test_document_level_deduplication_recall_at_k() -> None:
    """Verify duplicate chunks of the same document do not shrink the evaluated document pool."""
    qrels = {"doc_relevant": 1}
    # doc_irrelevant has 2 chunks retrieved ahead of doc_relevant
    retrieved = ["doc_irrelevant", "doc_irrelevant", "doc_relevant", "doc_relevant"]

    # At K=2, if not deduplicated, top-2 is ["doc_irrelevant", "doc_irrelevant"] (0 hits)
    # With document-level deduplication, top-2 unique documents are ["doc_irrelevant", "doc_relevant"] (1 hit)
    assert recall_at_k(retrieved, qrels, k=2) == 1.0


def test_document_level_deduplication_mrr_at_k() -> None:
    """Verify duplicate chunks do not artificially demote the first relevant document rank."""
    qrels = {"doc_relevant": 1}
    # doc_irrelevant has 2 chunks before doc_relevant
    retrieved = ["doc_irrelevant", "doc_irrelevant", "doc_relevant"]

    # Without deduplication, doc_relevant appears at position 3 -> RR = 1/3
    # With document deduplication, doc_relevant is at unique rank 2 -> RR = 1/2 = 0.5
    assert reciprocal_rank_at_k(retrieved, qrels, k=5) == 0.5


def test_document_level_deduplication_ndcg_at_k_prevents_multiple_gain() -> None:
    """Verify duplicate chunks of the same relevant document do not receive multiple gain discounts."""
    qrels = {"doc_A": 2, "doc_B": 1}
    # doc_A appears twice followed by doc_B
    retrieved = ["doc_A", "doc_A", "doc_B"]

    # Ideal ranking: doc_A (rel=2) at rank 1, doc_B (rel=1) at rank 2
    # If doc_A were counted twice, DCG would artificially exceed or distort IDCG
    score = ndcg_at_k(retrieved, qrels, k=2)
    assert score == pytest.approx(1.0)


def test_multi_chunk_corpus_relevant_chunk_not_first() -> None:
    """Verify pipeline preserves relevance when the relevant chunk is not the first retrieved chunk.

    Scenario:
    - Corpus contains doc_target (2 chunks) and doc_other (1 chunk).
    - Base retrieval returns:
      Rank 1: doc_other (chunk_other)
      Rank 2: doc_target (chunk_intro, low relevance)
      Rank 3: doc_target (chunk_content, high relevance)
    - CrossEncoder reranks all candidates:
      chunk_content gets score 0.95 -> Rank 1
      chunk_other gets score 0.60 -> Rank 2
      chunk_intro gets score 0.10 -> Rank 3
    - Deduplicating reranked chunks yields unique documents: [doc_target, doc_other].
    - Verifies that candidate pools preserve all chunks into reranking before deduplication.
    """
    class Candidate:
        def __init__(self, doc_id: str, chunk_id: str, score: float):
            self.document_id = doc_id
            self.chunk_id = chunk_id
            self.score = score

    # Base retrieval results (chunk_intro before chunk_content)
    base_candidates = [
        Candidate("doc_other", "doc_other_c0", 0.85),
        Candidate("doc_target", "doc_target_c0_intro", 0.75),
        Candidate("doc_target", "doc_target_c1_content", 0.65),
    ]

    # Reranker scores: chunk_content is the true relevant chunk
    rerank_scores = {
        "doc_target_c1_content": 0.95,
        "doc_other_c0": 0.60,
        "doc_target_c0_intro": 0.10,
    }

    # Simulate reranking sorted by rerank_scores
    reranked = sorted(base_candidates, key=lambda c: rerank_scores[c.chunk_id], reverse=True)
    assert [c.chunk_id for c in reranked] == ["doc_target_c1_content", "doc_other_c0", "doc_target_c0_intro"]

    # Deduplicate reranked chunks to unique document IDs
    unique_doc_ids = list(dict.fromkeys(c.document_id for c in reranked))
    assert unique_doc_ids == ["doc_target", "doc_other"]

    # Evaluate metrics against ground truth
    qrels = {"doc_target": 2, "doc_other": 1}
    metrics = evaluate_query_metrics(unique_doc_ids, qrels, ["recall@1", "mrr@1", "ndcg@1"])

    # doc_target is top-1 document thanks to chunk_content promotion
    assert metrics["recall@1"] == 0.5  # 1 out of 2 relevant docs
    assert metrics["mrr@1"] == 1.0     # first relevant doc at rank 1
    assert metrics["ndcg@1"] == 1.0    # ideal doc at rank 1

