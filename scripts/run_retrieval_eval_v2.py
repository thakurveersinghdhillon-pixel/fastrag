#!/usr/bin/env python3
"""FastRAG Retrieval Quality Evaluation CLI on Dataset v2.

Evaluates BM25, Dense, Hybrid RRF, and Hybrid+Rerank configurations across
Dataset v2, supporting provenance filtering (human_reviewed), category breakdowns,
and separate unanswerable/abstention analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker, SentenceTransformerCrossEncoder
from evaluation.metrics import evaluate_query_metrics
from evaluation.runner import load_qrels


def compute_file_sha256(path: Path) -> str | None:
    """Compute SHA256 hexadecimal digest of a file, returning None if missing."""
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dense_index_fingerprint(dense_dir: Path) -> dict[str, str | None]:
    """Compute SHA256 cryptographic fingerprints for all artifacts comprising the dense index."""
    return {
        "dense_vectors_sha256": compute_file_sha256(dense_dir / "dense_vectors.npz"),
        "dense_meta_sha256": compute_file_sha256(dense_dir / "dense_meta.json"),
    }


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FastRAG Retrieval Quality Benchmarks across 4 configurations on Dataset v2."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "v2",
        help="Path to Dataset v2 directory (default: datasets/v2)",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Split to evaluate (default: dev). Held-out test split must not be used for tuning.",
    )
    parser.add_argument(
        "--provenance",
        choices=["human_reviewed", "heuristic_candidate", "pending_review", "all"],
        default="human_reviewed",
        help="Annotation provenance filter (default: human_reviewed)",
    )
    parser.add_argument(
        "--query-ids",
        type=str,
        default=None,
        help="Comma-separated query IDs to evaluate/diagnose (e.g. q2_dev_003,q2_dev_010)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K candidates to retrieve (default: 10)",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Initial candidates for hybrid and reranker (default: 20)",
    )
    parser.add_argument(
        "--blending-alpha",
        type=float,
        default=0.0,
        help="Reciprocal rank blending weight (alpha >= 0.0) between CrossEncoder rank and base candidate rank (default: 0.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "results" / "retrieval_v2_benchmarks.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        default=None,
        help="Optional path to output JSON file for detailed diagnostic rankings",
    )
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queries_file = args.dataset_dir / f"queries_{args.split}.jsonl"
    qrels_file = args.dataset_dir / f"qrels_{args.split}.tsv"

    if not queries_file.is_file() or not qrels_file.is_file():
        print(f"Error: Dataset v2 files missing for split '{args.split}'", file=sys.stderr)
        return 1

    # Load queries and filter by provenance
    with open(queries_file, "r", encoding="utf-8") as f:
        all_queries = [json.loads(line) for line in f if line.strip()]

    eligible_queries = [
        q for q in all_queries
        if args.provenance == "all" or q.get("annotation_provenance", {}).get("status") == args.provenance
    ]

    qrels = load_qrels(qrels_file)

    # Optional query filter
    requested_query_ids: list[str] | None = None
    if args.query_ids:
        requested_query_ids = [qid.strip() for qid in args.query_ids.split(",") if qid.strip()]
        requested_set = set(requested_query_ids)
        all_ids_in_split = {q["query_id"] for q in all_queries}
        missing_ids = [qid for qid in requested_query_ids if qid not in all_ids_in_split]
        if missing_ids:
            print(f" Warning: Requested query IDs missing from split '{args.split}': {missing_ids}")

        excluded_by_prov = [
            q["query_id"] for q in all_queries
            if q["query_id"] in requested_set
            and args.provenance != "all"
            and q.get("annotation_provenance", {}).get("status") != args.provenance
        ]
        if excluded_by_prov:
            print(f" Notice: Requested query IDs excluded by provenance filter '{args.provenance}': {excluded_by_prov}")

        query_map = {q["query_id"]: q for q in eligible_queries if q["query_id"] in requested_set}
        eligible_queries = [query_map[qid] for qid in requested_query_ids if qid in query_map]

    print("=" * 75)
    print(" FastRAG Retrieval Quality Evaluation on Dataset v2")
    print(f" Split: {args.split.upper()} | Provenance Filter: {args.provenance}")
    print(f" Blending Alpha: {args.blending_alpha}")
    print(f" Total Queries: {len(all_queries)} | Eligible Queries: {len(eligible_queries)}")
    if requested_query_ids is not None:
        selected_ids = [q["query_id"] for q in eligible_queries]
        print(f" Selected Target Query IDs ({len(selected_ids)}): {', '.join(selected_ids)}")
    print("=" * 75)

    if not eligible_queries:
        print("Error: No eligible queries found matching specified criteria.", file=sys.stderr)
        return 1

    # Separate answerable vs unanswerable queries
    answerable_queries = []
    unanswerable_queries = []
    for q in eligible_queries:
        q_id = q["query_id"]
        rel_map = qrels.get(q_id, {})
        has_positive_rel = any(rel > 0 for rel in rel_map.values())
        if has_positive_rel:
            answerable_queries.append(q)
        else:
            unanswerable_queries.append(q)

    print(f" Answerable Queries: {len(answerable_queries)}")
    print(f" Unanswerable Queries: {len(unanswerable_queries)}")
    print("=" * 75)

    # Initialize retrieval components
    indexes_root = PROJECT_ROOT / "datasets" / "indexes"
    candidate_pool = max(args.candidate_k, args.top_k * 2)

    bm25 = BM25Retriever()
    bm25.load_index(indexes_root / "bm25")

    encoder = SentenceTransformerEncoder()
    dense = DenseRetriever(encoder=encoder)
    dense.load_index(indexes_root / "dense")

    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        dense_candidate_count=candidate_pool,
        bm25_candidate_count=candidate_pool,
        top_k=candidate_pool,
    )

    reranker = CrossEncoderReranker(
        scorer=SentenceTransformerCrossEncoder(),
        default_top_k=args.top_k,
        default_blending_alpha=args.blending_alpha,
    )

    # Configurations to evaluate
    configurations = [
        ("bm25", bm25, False),
        ("dense", dense, False),
        ("hybrid", hybrid, False),
        ("hybrid_rerank", hybrid, True),
    ]

    metrics_list = ["recall@5", "recall@10", "mrr@10", "ndcg@5", "ndcg@10"]
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    all_results: dict[str, Any] = {
        "dataset_version": "2.0.0",
        "timestamp_utc": timestamp_utc,
        "split": args.split,
        "provenance_filter": args.provenance,
        "blending_alpha": args.blending_alpha,
        "eligible_query_count": len(eligible_queries),
        "answerable_count": len(answerable_queries),
        "unanswerable_count": len(unanswerable_queries),
        "benchmark_metadata": {
            "candidate_k": args.candidate_k,
            "candidate_pool_size": candidate_pool,
            "top_k": args.top_k,
            "blending_alpha": args.blending_alpha,
            "models": {
                "dense_encoder": "sentence-transformers/all-MiniLM-L6-v2",
                "reranker_cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
            "fingerprints": {
                "corpus_sha256": compute_file_sha256(args.dataset_dir / "corpus.jsonl"),
                "queries_sha256": compute_file_sha256(queries_file),
                "qrels_sha256": compute_file_sha256(qrels_file),
                "bm25_index_sha256": compute_file_sha256(indexes_root / "bm25" / "bm25_index.json"),
                "dense_index_sha256": compute_file_sha256(indexes_root / "dense" / "dense_vectors.npz"),
                "dense_meta_sha256": compute_file_sha256(indexes_root / "dense" / "dense_meta.json"),
            },
        },
        "configurations": {},
    }

    # Diagnostic data tracking
    diagnostic_data: dict[str, dict[str, Any]] = {}
    if args.diagnostic_output or requested_query_ids:
        for q in answerable_queries:
            q_id = q["query_id"]
            q_text = q["query"]
            cat = q.get("category", "unknown")
            prov = q.get("annotation_provenance", {}).get("status", "unknown")
            query_qrels = qrels.get(q_id, {})
            rel_docs = [
                doc for doc, rel in sorted(query_qrels.items(), key=lambda item: item[1], reverse=True)
                if rel > 0
            ]
            diagnostic_data[q_id] = {
                "query_id": q_id,
                "query": q_text,
                "category": cat,
                "provenance_status": prov,
                "relevant_document_ids": rel_docs,
                "hybrid_candidate_document_ids": [],
                "reranked_document_ids": [],
                "per_query_metrics": {},
                "metrics": {},
            }

    for config_name, retriever_obj, use_reranker in configurations:
        t0 = time.perf_counter()

        per_query_scores: dict[str, dict[str, float]] = {}
        category_scores: dict[str, dict[str, list[float]]] = {}
        provenance_scores: dict[str, dict[str, list[float]]] = {}
        metric_accum: dict[str, list[float]] = {m: [] for m in metrics_list}

        # Evaluate Answerable Queries
        for q in answerable_queries:
            q_id = q["query_id"]
            q_text = q["query"]
            cat = q.get("category", "unknown")
            prov = q.get("annotation_provenance", {}).get("status", "unknown")
            query_qrels = qrels.get(q_id, {})

            if use_reranker:
                candidates = retriever_obj.search(q_text, top_k=candidate_pool)
                reranked = reranker.rerank(
                    q_text, candidates, top_k=len(candidates), blending_alpha=args.blending_alpha
                )
                retrieved_ids = list(dict.fromkeys(r.document_id for r in reranked))[:args.top_k]
            else:
                candidates = retriever_obj.search(q_text, top_k=candidate_pool)
                # Deduplicate chunks to unique doc_ids in rank order before top_k cutoff
                retrieved_ids = list(dict.fromkeys(r.document_id for r in candidates))[:args.top_k]

            q_metrics = evaluate_query_metrics(retrieved_ids, query_qrels, metrics_list)
            per_query_scores[q_id] = q_metrics

            if q_id in diagnostic_data:
                if config_name == "hybrid":
                    diagnostic_data[q_id]["hybrid_candidate_document_ids"] = list(retrieved_ids)
                elif config_name == "hybrid_rerank":
                    diagnostic_data[q_id]["reranked_document_ids"] = list(retrieved_ids)
                diagnostic_data[q_id]["per_query_metrics"][config_name] = q_metrics
                diagnostic_data[q_id]["metrics"][config_name] = q_metrics

            for m_name, val in q_metrics.items():
                metric_accum[m_name].append(val)

            if cat not in category_scores:
                category_scores[cat] = {m: [] for m in metrics_list}
            for m_name, val in q_metrics.items():
                category_scores[cat][m_name].append(val)

            if prov not in provenance_scores:
                provenance_scores[prov] = {m: [] for m in metrics_list}
            for m_name, val in q_metrics.items():
                provenance_scores[prov][m_name].append(val)

        # Evaluate Unanswerable Queries (Noise / False Candidate Analysis)
        unans_candidate_counts: list[int] = []
        for q in unanswerable_queries:
            q_text = q["query"]
            if use_reranker:
                candidates = retriever_obj.search(q_text, top_k=candidate_pool)
                reranked = reranker.rerank(
                    q_text, candidates, top_k=len(candidates), blending_alpha=args.blending_alpha
                )
                retrieved_ids = list(dict.fromkeys(r.document_id for r in reranked))[:args.top_k]
            else:
                candidates = retriever_obj.search(q_text, top_k=candidate_pool)
                retrieved_ids = list(dict.fromkeys(r.document_id for r in candidates))[:args.top_k]
            unans_candidate_counts.append(len(retrieved_ids))

        elapsed_s = time.perf_counter() - t0

        # Compute aggregate means
        agg_metrics = {
            m: round(sum(vals) / len(vals), 4) if vals else 0.0
            for m, vals in metric_accum.items()
        }

        cat_summary = {}
        for cat, m_dict in category_scores.items():
            cat_summary[cat] = {
                m: round(sum(vals) / len(vals), 4) if vals else 0.0
                for m, vals in m_dict.items()
            }

        prov_summary = {}
        for prov, m_dict in provenance_scores.items():
            prov_ans_count = len(m_dict[metrics_list[0]]) if metrics_list[0] in m_dict else 0
            prov_summary[prov] = {
                "answerable_count": prov_ans_count,
                "aggregate_metrics": {
                    m: round(sum(vals) / len(vals), 4) if vals else 0.0
                    for m, vals in m_dict.items()
                },
            }

        unans_summary = {
            "unanswerable_queries_count": len(unanswerable_queries),
            "unconstrained_retrieval_noise_rate": 1.0 if unanswerable_queries else 0.0,
            "avg_candidates_returned": round(sum(unans_candidate_counts) / len(unans_candidate_counts), 2) if unans_candidate_counts else 0.0,
            "interpretation": (
                "Unconstrained top-k retrievers return candidates even for unanswerable queries. "
                "Downstream generation or score thresholding is responsible for abstention."
            ),
        }

        all_results["configurations"][config_name] = {
            "aggregate_metrics": agg_metrics,
            "category_metrics": cat_summary,
            "provenance_metrics": prov_summary,
            "per_query_metrics": per_query_scores,
            "unanswerable_metrics": unans_summary,
            "wall_clock_seconds": round(elapsed_s, 3),
        }

    # Print comparative ASCII table
    print(f" {'Configuration':<16} | {'Recall@5':<9} | {'Recall@10':<9} | {'MRR@10':<8} | {'nDCG@5':<8} | {'nDCG@10':<8} | {'Wall (s)':<8}")
    print("-" * 75)
    for cfg, data in all_results["configurations"].items():
        m = data["aggregate_metrics"]
        print(
            f" {cfg:<16} | {m['recall@5']:<9.4f} | {m['recall@10']:<9.4f} | "
            f"{m['mrr@10']:<8.4f} | {m['ndcg@5']:<8.4f} | {m['ndcg@10']:<8.4f} | {data['wall_clock_seconds']:<8.2f}"
        )
    print("=" * 75)

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results written to: {args.output}")

    # Save diagnostic output if requested
    if args.diagnostic_output:
        diagnostic_payload = {
            "dataset_version": "2.0.0",
            "timestamp_utc": timestamp_utc,
            "benchmark_configuration": {
                "split": args.split,
                "provenance_filter": args.provenance,
                "top_k": args.top_k,
                "candidate_k": args.candidate_k,
                "blending_alpha": args.blending_alpha,
                "candidate_pool_size": candidate_pool,
                "models": all_results["benchmark_metadata"]["models"],
                "fingerprints": all_results["benchmark_metadata"]["fingerprints"],
            },
            "total_queries_diagnosed": len(diagnostic_data),
            "queries": list(diagnostic_data.values()),
        }
        args.diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.diagnostic_output, "w", encoding="utf-8") as f:
            json.dump(diagnostic_payload, f, indent=2)
        print(f"Diagnostic rankings written to: {args.diagnostic_output}")

    # If query IDs were requested, print detailed rankings to terminal
    if requested_query_ids:
        print("\n" + "=" * 75)
        print(" Targeted Diagnostic Rankings")
        print("=" * 75)
        for q_id, q_diag in diagnostic_data.items():
            print(f" Query ID: {q_id}")
            print(f" Query: {q_diag['query']}")
            print(f" Category: {q_diag['category']} | Provenance: {q_diag['provenance_status']}")
            print(f" Relevant Doc IDs: {q_diag['relevant_document_ids']}")
            print(f" Hybrid RRF Doc IDs (Top {args.top_k}): {q_diag['hybrid_candidate_document_ids']}")
            print(f" Hybrid + CrossEncoder Doc IDs (Top {args.top_k}): {q_diag['reranked_document_ids']}")
            h_m = q_diag['per_query_metrics'].get('hybrid', {})
            hr_m = q_diag['per_query_metrics'].get('hybrid_rerank', {})
            print(
                f" Metrics (Hybrid):        Recall@5={h_m.get('recall@5', 0):.4f} | "
                f"Recall@10={h_m.get('recall@10', 0):.4f} | MRR@10={h_m.get('mrr@10', 0):.4f} | "
                f"nDCG@5={h_m.get('ndcg@5', 0):.4f} | nDCG@10={h_m.get('ndcg@10', 0):.4f}"
            )
            print(
                f" Metrics (Hybrid+Rerank): Recall@5={hr_m.get('recall@5', 0):.4f} | "
                f"Recall@10={hr_m.get('recall@10', 0):.4f} | MRR@10={hr_m.get('mrr@10', 0):.4f} | "
                f"nDCG@5={hr_m.get('ndcg@5', 0):.4f} | nDCG@10={hr_m.get('ndcg@10', 0):.4f}"
            )
            print("-" * 75)
        if not args.diagnostic_output:
            print(" (Tip: Specify --diagnostic-output <path> to write these diagnostic rankings to a separate JSON file)")
        print("=" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
