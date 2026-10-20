"""FastRAG Information Retrieval Evaluation Metrics.

Implements standard deterministic IR evaluation metrics:
- Recall@K (binary relevance: judgment > 0)
- Reciprocal Rank / MRR@K (position of first relevant item within top-K)
- nDCG@K (Normalized Discounted Cumulative Gain with graded relevance 0, 1, 2)
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence, Set


def _validate_k(k: int) -> None:
    """Validate that K is a strictly positive integer."""
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"Cutoff parameter K must be an integer, got {type(k).__name__}")
    if k <= 0:
        raise ValueError(f"Cutoff parameter K must be strictly positive (>= 1), got {k}")


def _extract_relevant_set(relevant_docs: Mapping[str, int] | Set[str]) -> set[str]:
    """Extract set of relevant document IDs where relevance > 0."""
    if isinstance(relevant_docs, set):
        return relevant_docs
    if isinstance(relevant_docs, (dict, Mapping)):
        return {doc_id for doc_id, rel in relevant_docs.items() if rel > 0}
    raise TypeError(
        f"relevant_docs must be a set or mapping of doc_id to integer relevance, "
        f"got {type(relevant_docs).__name__}"
    )


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_docs: Mapping[str, int] | Set[str],
    k: int,
) -> float:
    """Calculate the fraction of relevant documents retrieved within the top-K results.

    Conventions:
    - Binary relevance: any document with relevance judgment > 0 is treated as relevant.
    - If there are zero relevant documents for the query in the ground truth, Recall@K is defined as 0.0.
    - If retrieved_ids is empty, Recall@K is 0.0.
    - If K exceeds len(retrieved_ids), all retrieved documents are evaluated.

    Args:
        retrieved_ids: Ranked list of retrieved document identifiers.
        relevant_docs: Mapping of doc_id to relevance score, or set of relevant doc_ids.
        k: Cutoff rank (must be >= 1).

    Returns:
        Float in range [0.0, 1.0].
    """
    _validate_k(k)
    target_relevant = _extract_relevant_set(relevant_docs)
    if not target_relevant:
        return 0.0

    if not retrieved_ids:
        return 0.0

    # Consider top-k unique retrieved documents in rank order
    unique_retrieved = list(dict.fromkeys(retrieved_ids))
    top_k_candidates = set(unique_retrieved[:k])
    hits = len(top_k_candidates & target_relevant)
    return hits / len(target_relevant)


def reciprocal_rank_at_k(
    retrieved_ids: Sequence[str],
    relevant_docs: Mapping[str, int] | Set[str],
    k: int,
) -> float:
    """Calculate the reciprocal rank of the first relevant document retrieved within top-K.

    Conventions:
    - Rank is 1-indexed (first item has rank 1, reciprocal rank 1.0).
    - If no relevant document appears within the top-K cutoff, reciprocal rank is 0.0.
    - If there are zero relevant documents in the ground truth, reciprocal rank is 0.0.

    Args:
        retrieved_ids: Ranked list of retrieved document identifiers.
        relevant_docs: Mapping of doc_id to relevance score, or set of relevant doc_ids.
        k: Cutoff rank (must be >= 1).

    Returns:
        Float in range [0.0, 1.0].
    """
    _validate_k(k)
    target_relevant = _extract_relevant_set(relevant_docs)
    if not target_relevant or not retrieved_ids:
        return 0.0

    unique_retrieved = list(dict.fromkeys(retrieved_ids))
    for rank_idx, doc_id in enumerate(unique_retrieved[:k], start=1):
        if doc_id in target_relevant:
            return 1.0 / rank_idx

    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: Sequence[float]) -> float:
    """Calculate the arithmetic mean of reciprocal ranks across a set of queries.

    Args:
        reciprocal_ranks: Sequence of individual query reciprocal rank values.

    Returns:
        Float in range [0.0, 1.0]. Returns 0.0 if input sequence is empty.
    """
    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def dcg_at_k(
    retrieved_ids: Sequence[str],
    relevance_map: Mapping[str, int],
    k: int,
) -> float:
    """Calculate Discounted Cumulative Gain at cutoff K using conventional exponential gain.

    Formula:
        DCG@K = sum_{i=1}^{min(K, |retrieved|)} (2^{rel_i} - 1) / log2(i + 1)

    Args:
        retrieved_ids: Ranked list of retrieved document identifiers.
        relevance_map: Mapping of doc_id to integer graded relevance level.
        k: Cutoff rank (must be >= 1).

    Returns:
        Non-negative float DCG score.
    """
    _validate_k(k)
    if not retrieved_ids or not relevance_map:
        return 0.0

    unique_retrieved = list(dict.fromkeys(retrieved_ids))
    dcg = 0.0
    for rank_idx, doc_id in enumerate(unique_retrieved[:k], start=1):
        rel = relevance_map.get(doc_id, 0)
        if rel > 0:
            gain = (2.0 ** rel) - 1.0
            discount = math.log2(rank_idx + 1.0)
            dcg += gain / discount

    return dcg


def idcg_at_k(
    relevance_map: Mapping[str, int],
    k: int,
) -> float:
    """Calculate Ideal Discounted Cumulative Gain at cutoff K.

    Sorts all positive relevance judgments in descending order and calculates DCG@K.

    Args:
        relevance_map: Mapping of doc_id to integer graded relevance level.
        k: Cutoff rank (must be >= 1).

    Returns:
        Non-negative float IDCG score. Returns 0.0 if no positive relevance exists.
    """
    _validate_k(k)
    if not relevance_map:
        return 0.0

    sorted_rels = sorted((r for r in relevance_map.values() if r > 0), reverse=True)
    if not sorted_rels:
        return 0.0

    idcg = 0.0
    for rank_idx, rel in enumerate(sorted_rels[:k], start=1):
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(rank_idx + 1.0)
        idcg += gain / discount

    return idcg


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance_map: Mapping[str, int],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at cutoff K.

    Formula:
        nDCG@K = DCG@K / IDCG@K

    Conventions:
    - If IDCG@K is 0.0 (no relevant documents exist for the query), nDCG@K is defined as 0.0.
    - Graded relevance scores: 0 (not relevant), 1 (partially relevant), 2 (highly relevant).
    - Result is bounded within [0.0, 1.0].

    Args:
        retrieved_ids: Ranked list of retrieved document identifiers.
        relevance_map: Mapping of doc_id to integer graded relevance level.
        k: Cutoff rank (must be >= 1).

    Returns:
        Float in range [0.0, 1.0].
    """
    _validate_k(k)
    ideal_dcg = idcg_at_k(relevance_map, k)
    if ideal_dcg == 0.0:
        return 0.0

    actual_dcg = dcg_at_k(retrieved_ids, relevance_map, k)
    score = actual_dcg / ideal_dcg
    return min(max(score, 0.0), 1.0)


def evaluate_query_metrics(
    retrieved_ids: Sequence[str],
    relevance_map: Mapping[str, int],
    metric_names: Sequence[str],
) -> dict[str, float]:
    """Calculate a dictionary of configured metrics for a single query's ranked retrieval results.

    Supported metric name formats (case-insensitive):
    - recall@K (e.g. recall@5, recall@10)
    - mrr@K or rr@K (e.g. mrr@10, rr@10)
    - ndcg@K (e.g. ndcg@5, ndcg@10)

    Args:
        retrieved_ids: Ranked list of retrieved document identifiers.
        relevance_map: Ground-truth mapping of doc_id to integer relevance level.
        metric_names: List of metric specifiers to compute.

    Returns:
        Mapping of canonical metric name to computed float score.
    """
    results: dict[str, float] = {}

    for metric_spec in metric_names:
        clean_spec = metric_spec.strip().lower()
        if "@" not in clean_spec:
            raise ValueError(
                f"Invalid metric specification '{metric_spec}'. "
                f"Expected format '<metric>@<K>', e.g. 'recall@10', 'ndcg@5', 'mrr@10'"
            )

        metric_type, k_str = clean_spec.split("@", 1)
        try:
            k_val = int(k_str)
        except ValueError:
            raise ValueError(f"Invalid cutoff K in metric '{metric_spec}': '{k_str}' is not an integer")

        if metric_type == "recall":
            results[clean_spec] = recall_at_k(retrieved_ids, relevance_map, k_val)
        elif metric_type in ("mrr", "rr", "reciprocal_rank"):
            results[clean_spec] = reciprocal_rank_at_k(retrieved_ids, relevance_map, k_val)
        elif metric_type == "ndcg":
            results[clean_spec] = ndcg_at_k(retrieved_ids, relevance_map, k_val)
        else:
            raise ValueError(
                f"Unsupported metric type '{metric_type}' in '{metric_spec}'. "
                f"Supported types: recall, mrr, ndcg."
            )

    return results
