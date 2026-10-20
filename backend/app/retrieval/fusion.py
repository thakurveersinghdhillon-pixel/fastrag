"""Reciprocal Rank Fusion (RRF) for FastRAG Hybrid Retrieval.

Implements standard Reciprocal Rank Fusion (RRF) to combine multiple ranked candidate
lists without requiring floating-point score calibration between heterogeneous retrievers.

Formula:
    RRF(d) = sum_{r in R} (1 / (k + rank_r(d)))
where:
    - R is the collection of ranked candidate lists
    - k is the smoothing constant (default: 60)
    - rank_r(d) is the 1-based rank position of candidate d in list r
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.core.logging_config import get_logger
from app.retrieval.base import RetrievalResult, validate_top_k

logger = get_logger("fastrag.retrieval.fusion")

DEFAULT_RRF_SMOOTHING_K = 60


def validate_rrf_k(rrf_k: int) -> None:
    """Validate that RRF smoothing constant is a positive integer."""
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool):
        raise TypeError(f"rrf_k must be an integer, got {type(rrf_k).__name__}")
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be strictly positive (> 0), got {rrf_k}")


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievalResult]] | Mapping[str, Sequence[RetrievalResult]],
    top_k: int = 10,
    rrf_k: int = DEFAULT_RRF_SMOOTHING_K,
) -> list[RetrievalResult]:
    """Fuse multiple ranked candidate lists using standard Reciprocal Rank Fusion.

    Args:
        ranked_lists: Sequence or Mapping of ranked lists of RetrievalResult.
                      If a Mapping is passed, the keys represent branch names (e.g., 'dense', 'bm25').
        top_k: Maximum number of fused candidates to return.
        rrf_k: Smoothing constant k added to rank denominator (default: 60).

    Returns:
        Deduplicated, re-ranked list of RetrievalResult with score_type='rrf',
        ordered by descending RRF score, with ties broken deterministically by chunk_id.
    """
    validate_top_k(top_k)
    validate_rrf_k(rrf_k)

    # Normalize input into a sequence of (branch_name, results_list)
    normalized_lists: list[tuple[str, Sequence[RetrievalResult]]]
    if isinstance(ranked_lists, Mapping):
        normalized_lists = [(str(name), res_list) for name, res_list in ranked_lists.items()]
    else:
        normalized_lists = [(f"branch_{idx}", res_list) for idx, res_list in enumerate(ranked_lists)]

    if not normalized_lists:
        return []

    # Map chunk_id to candidate details and aggregated score
    # Structure: chunk_id -> {"score": float, "chunk": RetrievalResult, "ranks": dict, "raw_scores": dict}
    candidates: dict[str, dict[str, Any]] = {}

    for branch_name, r_list in normalized_lists:
        seen_in_branch: set[str] = set()

        for rank_idx, result in enumerate(r_list, start=1):
            cid = result.chunk_id

            # Guard: do not allow duplicate occurrences within the same list to inflate contribution
            if cid in seen_in_branch:
                continue
            seen_in_branch.add(cid)

            rrf_contribution = 1.0 / (rrf_k + rank_idx)

            if cid not in candidates:
                candidates[cid] = {
                    "score": rrf_contribution,
                    "chunk": result,
                    "branch_ranks": {branch_name: rank_idx},
                    "branch_scores": {branch_name: result.score},
                }
            else:
                candidates[cid]["score"] += rrf_contribution
                candidates[cid]["branch_ranks"][branch_name] = rank_idx
                candidates[cid]["branch_scores"][branch_name] = result.score

    if not candidates:
        return []

    # Sort candidates by descending RRF score; break ties deterministically by chunk_id ascending
    sorted_candidates = sorted(
        candidates.values(),
        key=lambda entry: (-entry["score"], entry["chunk"].chunk_id),
    )

    fused_results: list[RetrievalResult] = []
    for rank, entry in enumerate(sorted_candidates[:top_k], start=1):
        original = entry["chunk"]
        merged_metadata = dict(original.metadata) if original.metadata else {}
        merged_metadata["rrf_ranks"] = entry["branch_ranks"]
        merged_metadata["original_scores"] = entry["branch_scores"]

        fused_results.append(
            RetrievalResult(
                chunk_id=original.chunk_id,
                document_id=original.document_id,
                rank=rank,
                score=round(entry["score"], 8),
                score_type="rrf",
                text=original.text,
                source=original.source,
                metadata=merged_metadata,
            )
        )

    return fused_results
