#!/usr/bin/env python3
"""FastRAG Evaluation Runner CLI.

Evaluates retrievers (BM25, Dense), pre-computed rankings, or synthetic fixtures
against ground-truth qrels to compute Recall@K, MRR@K, and nDCG@K metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.fixtures import (
    get_synthetic_empty_rankings,
    get_synthetic_inverse_rankings,
    get_synthetic_partial_rankings,
    get_synthetic_perfect_rankings,
)
from evaluation.retriever_adapter import evaluate_retriever
from evaluation.runner import EvaluationRunner, load_qrels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastRAG Evaluation Runner: calculate retrieval metrics (Recall@K, MRR@K, nDCG@K)."
    )
    parser.add_argument(
        "--retriever",
        choices=["bm25", "dense", "hybrid", "hybrid-rerank"],
        default=None,
        help="Evaluate an actual retriever (bm25, dense, hybrid, or hybrid-rerank) against evaluation queries.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply CrossEncoder reranker to retrieved candidates before evaluation.",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=20,
        help="Number of candidates to retrieve and pass to the CrossEncoder reranker (default: 20).",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help="CrossEncoder model override (default: cross-encoder/ms-marco-MiniLM-L-6-v2).",
    )
    parser.add_argument(
        "--mock-reranker",
        action="store_true",
        help="Use deterministic mock CrossEncoder scorer for offline/fast evaluation without neural weights.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "queries.jsonl",
        help="Path to queries.jsonl (default: datasets/queries.jsonl)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Path to index directory for the retriever (default: datasets/indexes/<retriever>)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K results to retrieve per query during retriever evaluation (default: 10)",
    )
    parser.add_argument(
        "--dense-candidates",
        type=int,
        default=20,
        help="For hybrid retriever: dense branch candidate limit (default: 20)",
    )
    parser.add_argument(
        "--bm25-candidates",
        type=int,
        default=20,
        help="For hybrid retriever: BM25 branch candidate limit (default: 20)",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="For hybrid retriever: RRF smoothing constant (default: 60)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="For hybrid retriever: run branches sequentially instead of concurrently (for comparison).",
    )
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use deterministic mock embeddings for dense retriever instead of downloading neural model.",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "qrels.tsv",
        help="Path to ground truth qrels.tsv file (default: datasets/qrels.tsv)",
    )
    parser.add_argument(
        "--rankings",
        type=Path,
        default=None,
        help="Path to JSON file containing {query_id: [doc_id, ...]} rankings.",
    )
    parser.add_argument(
        "--fixture",
        choices=["perfect", "partial", "inverse", "empty"],
        default=None,
        help="Use a pre-determined synthetic ranking fixture for testing (e.g. perfect, partial, inverse, empty).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Unique identifier for this evaluation experiment.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Narrative description of the evaluation run.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["recall@5", "recall@10", "mrr@10", "ndcg@5", "ndcg@10"],
        help="List of metrics to compute (e.g. recall@5 mrr@10 ndcg@5).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write results to benchmarks/results/ directory.",
    )
    return parser.parse_args()


def run_retriever_eval(args: argparse.Namespace) -> int:
    from app.retrieval.bm25 import BM25Retriever
    from app.retrieval.dense import DenseRetriever, MockEmbeddingEncoder, SentenceTransformerEncoder
    from app.retrieval.hybrid import HybridRetriever

    search_kwargs: dict[str, Any] = {}

    is_hybrid = args.retriever in ("hybrid", "hybrid-rerank")
    should_rerank = args.rerank or (args.retriever == "hybrid-rerank")

    if is_hybrid:
        indexes_root = PROJECT_ROOT / "datasets" / "indexes"
        bm25_dir = indexes_root / "bm25"
        dense_dir = indexes_root / "dense"

        if not bm25_dir.exists() or not dense_dir.exists():
            print("Error: Indexes missing. Run 'python3 scripts/build_indexes.py --type all'", file=sys.stderr)
            return 1

        bm25 = BM25Retriever()
        bm25.load_index(bm25_dir)

        encoder = MockEmbeddingEncoder() if args.mock_embeddings else SentenceTransformerEncoder()
        dense = DenseRetriever(encoder=encoder)
        dense.load_index(dense_dir)

        base_retriever = HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            dense_candidate_count=args.dense_candidates,
            bm25_candidate_count=args.bm25_candidates,
            top_k=args.top_k,
            rrf_k=args.rrf_k,
        )
        search_kwargs["concurrent"] = not args.sequential
        mode_label = "sequential" if args.sequential else "concurrent"
        exp_id = args.experiment_id or (
            f"hybrid_reranked_eval" if should_rerank else f"hybrid_{mode_label}_baseline"
        )
        desc = args.description or (
            f"FastRAG Phase 6 Hybrid + CrossEncoder reranking evaluation"
            if should_rerank
            else f"FastRAG Phase 5 Hybrid RRF evaluation ({mode_label})"
        )
        retriever = base_retriever
    else:
        index_dir = args.index_dir or (PROJECT_ROOT / "datasets" / "indexes" / args.retriever)
        if not index_dir.exists():
            print(f"Error: Index directory does not exist: {index_dir}", file=sys.stderr)
            print("Please build index first using 'python3 scripts/build_indexes.py --type all'", file=sys.stderr)
            return 1

        if args.retriever == "bm25":
            base_retriever = BM25Retriever()
            base_retriever.load_index(index_dir)
        else:
            encoder = MockEmbeddingEncoder() if args.mock_embeddings else SentenceTransformerEncoder()
            base_retriever = DenseRetriever(encoder=encoder)
            base_retriever.load_index(index_dir)

        exp_id = args.experiment_id or f"{args.retriever}_retrieval_eval"
        desc = args.description or f"FastRAG Phase 4 {args.retriever.upper()} retrieval evaluation"
        retriever = base_retriever

    if should_rerank:
        from app.retrieval.pipeline import RAGPipeline
        from app.retrieval.reranker import (
            CrossEncoderReranker,
            MockCrossEncoderScorer,
            SentenceTransformerCrossEncoder,
        )

        if args.mock_reranker:
            scorer = MockCrossEncoderScorer()
        else:
            scorer = SentenceTransformerCrossEncoder(model_name=args.reranker_model)

        reranker = CrossEncoderReranker(scorer=scorer, default_top_k=args.top_k)
        retriever = RAGPipeline(retriever=base_retriever, reranker=reranker)
        search_kwargs["candidate_k"] = args.rerank_candidates

    report, latency_stats = evaluate_retriever(
        retriever=retriever,
        queries_path=args.queries,
        qrels_path=args.qrels,
        experiment_id=exp_id,
        top_k=args.top_k,
        metrics=args.metrics,
        description=desc,
        save_results=not args.no_save,
        results_dir=PROJECT_ROOT / "benchmarks" / "results",
        search_kwargs=search_kwargs,
    )

    print("\n" + report.summary())
    print("\n" + "=" * 55)
    print("Latency Performance Statistics")
    print("=" * 55)
    print(f"Cold-start latency:  {latency_stats.cold_start_ms:8.2f} ms")
    print(f"Warm mean latency:    {latency_stats.warm_mean_ms:8.2f} ms")
    print(f"Warm p50 latency:     {latency_stats.warm_p50_ms:8.2f} ms")
    print(f"Warm p95 latency:     {latency_stats.warm_p95_ms:8.2f} ms")
    print(f"Warm p99 latency:     {latency_stats.warm_p99_ms:8.2f} ms")
    print(f"Warm queries count:   {latency_stats.warm_query_count}")
    print(f"Total time elapsed:   {latency_stats.total_duration_seconds:8.4f} s")

    # If hybrid breakdown stats were collected, display them
    retriever_cfg = report.experiment_metadata.get("retrieval_configuration", {})
    breakdown = retriever_cfg.get("hybrid_breakdown")
    if breakdown:
        print("-" * 55)
        print("Hybrid Subsystem Breakdown:")
        print(f"  Dense branch (median):  {breakdown.get('dense_p50_ms', 0):8.2f} ms (mean: {breakdown.get('dense_mean_ms', 0):.2f} ms)")
        print(f"  BM25 branch (median):   {breakdown.get('bm25_p50_ms', 0):8.2f} ms (mean: {breakdown.get('bm25_mean_ms', 0):.2f} ms)")
        print(f"  RRF fusion (mean):      {breakdown.get('fusion_mean_ms', 0):8.2f} ms")
        print(f"  Degraded query count:   {breakdown.get('degraded_queries', 0)}")

    rerank_breakdown = retriever_cfg.get("reranker_breakdown")
    if rerank_breakdown:
        print("-" * 55)
        print("CrossEncoder Reranker Breakdown:")
        print(f"  Rerank latency (p50):   {rerank_breakdown.get('rerank_p50_ms', 0):8.2f} ms (mean: {rerank_breakdown.get('rerank_mean_ms', 0):.2f} ms)")
        print(f"  Rerank query count:     {rerank_breakdown.get('rerank_query_count', 0)}")
    print("=" * 55)
    return 0


def main() -> int:
    args = parse_args()

    if not args.qrels.is_file():
        print(f"Error: qrels file not found at {args.qrels}", file=sys.stderr)
        return 1

    # Check if evaluating an actual retriever
    if args.retriever:
        return run_retriever_eval(args)

    qrels = load_qrels(args.qrels)

    # Determine rankings source
    rankings: dict[str, list[str]] = {}
    if args.rankings:
        if not args.rankings.is_file():
            print(f"Error: rankings file not found at {args.rankings}", file=sys.stderr)
            return 1
        with open(args.rankings, "r", encoding="utf-8") as f:
            rankings = json.load(f)
    elif args.fixture:
        print(f"[NOTE] Using synthetic '{args.fixture}' ranking fixture for verification.")
        if args.fixture == "perfect":
            rankings = get_synthetic_perfect_rankings(qrels)
        elif args.fixture == "partial":
            rankings = get_synthetic_partial_rankings(qrels)
        elif args.fixture == "inverse":
            rankings = get_synthetic_inverse_rankings(qrels)
        elif args.fixture == "empty":
            rankings = get_synthetic_empty_rankings(qrels)
    else:
        print(
            "Notice: Neither --retriever, --rankings nor --fixture was specified. Defaulting to '--fixture partial' for demonstration.",
            file=sys.stderr,
        )
        rankings = get_synthetic_partial_rankings(qrels)

    runner = EvaluationRunner(
        qrels=qrels,
        metrics=args.metrics,
        datasets_dir=PROJECT_ROOT / "datasets",
        results_dir=PROJECT_ROOT / "benchmarks" / "results",
    )

    exp_id = args.experiment_id or "baseline_eval"
    desc = args.description or "FastRAG Phase 2 baseline evaluation"

    report = runner.run(
        rankings=rankings,
        experiment_id=exp_id,
        description=desc,
        retrieval_configuration={"fixture_mode": args.fixture or "custom_file"},
        save_results=not args.no_save,
    )

    print("\n" + report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
