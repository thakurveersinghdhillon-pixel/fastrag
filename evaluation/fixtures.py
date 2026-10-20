"""FastRAG Synthetic Ranking Fixtures for Unit Testing and Evaluation Verification.

DISCLAIMER:
The fixtures in this file are PRE-DETERMINED TEST FIXTURES designed solely to verify
the correctness of metric calculations, edge-case behavior, and the evaluation runner.
THEY DO NOT REPRESENT ACTUAL PERFORMANCE BENCHMARKS OR RETRIEVAL OUTPUTS FROM
FASTRAG RETRIEVAL ALGORITHMS. FastRAG retrieval engines have not yet been implemented.
"""

from __future__ import annotations

from typing import Mapping


def get_synthetic_perfect_rankings(
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, list[str]]:
    """Generate a synthetic test fixture where relevant documents are placed at the top ranks.

    Intended solely for unit testing to verify that metrics achieve 1.0 under ideal ranking.
    """
    rankings: dict[str, list[str]] = {}
    for query_id, doc_judgments in qrels.items():
        # Sort documents by relevance descending (2 first, then 1, then 0)
        sorted_docs = sorted(
            doc_judgments.keys(),
            key=lambda d: doc_judgments[d],
            reverse=True,
        )
        rankings[query_id] = sorted_docs
    return rankings


def get_synthetic_inverse_rankings(
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, list[str]]:
    """Generate a synthetic test fixture where irrelevant documents are placed at the top.

    Intended solely for unit testing to verify behavior under worst-case rankings.
    """
    rankings: dict[str, list[str]] = {}
    for query_id, doc_judgments in qrels.items():
        # Sort documents by relevance ascending (0 first, then 1, then 2)
        sorted_docs = sorted(
            doc_judgments.keys(),
            key=lambda d: doc_judgments[d],
        )
        rankings[query_id] = sorted_docs
    return rankings


def get_synthetic_partial_rankings(
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, list[str]]:
    """Generate a synthetic test fixture simulating partial retrieval hits.

    Picks the first relevant document at rank 2 or 3, leaving rank 1 as irrelevant.
    """
    rankings: dict[str, list[str]] = {}
    for query_id, doc_judgments in qrels.items():
        relevant = [d for d, r in doc_judgments.items() if r > 0]
        irrelevant = [d for d, r in doc_judgments.items() if r == 0]

        # Place one irrelevant document first if available, then relevant docs
        mixed: list[str] = []
        if irrelevant:
            mixed.append(irrelevant[0])
        mixed.extend(relevant)
        if len(irrelevant) > 1:
            mixed.extend(irrelevant[1:])

        rankings[query_id] = mixed
    return rankings


def get_synthetic_empty_rankings(
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, list[str]]:
    """Generate an empty ranking fixture to verify 0.0 metric calculations on zero results."""
    return {query_id: [] for query_id in qrels.keys()}
