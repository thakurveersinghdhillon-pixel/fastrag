#!/usr/bin/env python3
"""FastRAG Hybrid Query CLI.

Executes concurrent Dense and BM25 retrieval combined with Reciprocal Rank Fusion (RRF).
Displays detailed branch timings, fusion duration, and degraded mode diagnostics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever, MockEmbeddingEncoder, SentenceTransformerEncoder
from app.retrieval.hybrid import HybridRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastRAG Hybrid Query CLI: concurrent Dense + BM25 search with Reciprocal Rank Fusion."
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Search query string.",
    )
    parser.add_argument(
        "--dense-index-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "indexes" / "dense",
        help="Path to Dense index directory (default: datasets/indexes/dense)",
    )
    parser.add_argument(
        "--bm25-index-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "indexes" / "bm25",
        help="Path to BM25 index directory (default: datasets/indexes/bm25)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final fused candidates to return (default: 5).",
    )
    parser.add_argument(
        "--dense-candidates",
        type=int,
        default=20,
        help="Number of candidates to retrieve from the dense branch (default: 20).",
    )
    parser.add_argument(
        "--bm25-candidates",
        type=int,
        default=20,
        help="Number of candidates to retrieve from the BM25 branch (default: 20).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal Rank Fusion smoothing constant k (default: 60).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-branch retrieval timeout in seconds (default: None).",
    )
    parser.add_argument(
        "--mode",
        choices=["strict", "degraded"],
        default="strict",
        help="Failure handling mode: 'strict' raises errors; 'degraded' returns partial results (default: strict).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Execute retrieval branches sequentially instead of concurrently (control baseline).",
    )
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use deterministic mock embeddings for dense retriever instead of neural model.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON response with diagnostics.",
    )
    return parser.parse_args()


async def run_hybrid_search(args: argparse.Namespace) -> int:
    if not args.query:
        print("Error: query string must be provided.", file=sys.stderr)
        return 1

    if not args.bm25_index_dir.exists():
        print(f"Error: BM25 index not found at: {args.bm25_index_dir}", file=sys.stderr)
        print("Hint: Build indexes first using 'python3 scripts/build_indexes.py --type all'", file=sys.stderr)
        return 1

    if not args.dense_index_dir.exists():
        print(f"Error: Dense index not found at: {args.dense_index_dir}", file=sys.stderr)
        print("Hint: Build indexes first using 'python3 scripts/build_indexes.py --type all'", file=sys.stderr)
        return 1

    # Initialize BM25 retriever
    bm25 = BM25Retriever()
    bm25.load_index(args.bm25_index_dir)

    # Initialize Dense retriever
    encoder = MockEmbeddingEncoder() if args.mock_embeddings else SentenceTransformerEncoder()
    dense = DenseRetriever(encoder=encoder)
    dense.load_index(args.dense_index_dir)

    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        dense_candidate_count=args.dense_candidates,
        bm25_candidate_count=args.bm25_candidates,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        timeout_seconds=args.timeout,
        mode=args.mode,
    )

    try:
        response = await hybrid.search_async(
            query=args.query,
            top_k=args.top_k,
            dense_candidates=args.dense_candidates,
            bm25_candidates=args.bm25_candidates,
            rrf_k=args.rrf_k,
            timeout_seconds=args.timeout,
            mode=args.mode,
            concurrent=not args.sequential,
        )
    except Exception as exc:
        print(f"Hybrid retrieval error: {exc}", file=sys.stderr)
        return 1

    diag = response.diagnostics

    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
        return 0

    print("=" * 75)
    exec_type = "CONCURRENT" if diag.concurrent else "SEQUENTIAL"
    print(f"FastRAG Hybrid Retrieval ({exec_type} + RRF k={args.rrf_k})")
    print("=" * 75)
    print(f"Query:               '{args.query}'")
    print(f"Total Latency:        {diag.total_duration_ms:8.2f} ms")
    print(f"  - Dense Branch:     {diag.dense.duration_ms:8.2f} ms (hits: {diag.dense.candidate_count})")
    print(f"  - BM25 Branch:      {diag.bm25.duration_ms:8.2f} ms (hits: {diag.bm25.candidate_count})")
    print(f"  - RRF Fusion:       {diag.fusion_duration_ms:8.2f} ms")
    print(f"Execution Mode:       {diag.mode.upper()}")
    if diag.degraded:
        print("WARNING: RETRIEVAL OPERATING IN DEGRADED MODE (One branch failed/timed out)")
    print(f"Top-K Requested:      {args.top_k}")
    print(f"Fused Hits Returned:  {len(response.results)}")
    print("-" * 75)

    if not response.results:
        print("No matching chunks retrieved.")
        return 0

    for res in response.results:
        meta = res.metadata or {}
        orig_ranks = meta.get("rrf_ranks", {})
        orig_scores = meta.get("original_scores", {})
        title = meta.get("title", "Untitled")

        dense_info = f"rank={orig_ranks.get('dense', '-')}, score={orig_scores.get('dense', 0.0):.4f}" if "dense" in orig_ranks else "absent"
        bm25_info = f"rank={orig_ranks.get('bm25', '-')}, score={orig_scores.get('bm25', 0.0):.4f}" if "bm25" in orig_ranks else "absent"

        snippet = res.text.replace("\n", " ")[:110] + "..." if len(res.text) > 110 else res.text.replace("\n", " ")

        print(f"Rank {res.rank:<2} | RRF Score: {res.score:8.6f} | Chunk: {res.chunk_id} | Doc: {res.document_id}")
        print(f"        Title:       {title}")
        print(f"        Dense hit:   [{dense_info}]")
        print(f"        BM25 hit:    [{bm25_info}]")
        print(f"        Snippet:     {snippet}")
        print("-" * 75)

    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_hybrid_search(args))


if __name__ == "__main__":
    sys.exit(main())
