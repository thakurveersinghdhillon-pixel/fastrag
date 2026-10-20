#!/usr/bin/env python3
"""FastRAG Query Index CLI.

Executes queries directly against saved BM25 or Dense indexes.
Useful for interactive testing, debugging, and verification of retrieval rankings.
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastRAG Query Index CLI: test retrieval against persisted BM25 or Dense indexes."
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Search query string.",
    )
    parser.add_argument(
        "--type",
        choices=["bm25", "dense"],
        default="bm25",
        help="Retriever index type to query (default: bm25).",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Custom index directory path (defaults to datasets/indexes/<type>).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidates to retrieve (default: 5).",
    )
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="For dense retriever: use deterministic hash-based mock encoder instead of neural model.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON array of retrieval results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.query:
        print("Error: query string must be provided.", file=sys.stderr)
        return 1

    indexes_root = PROJECT_ROOT / "datasets" / "indexes"
    index_path = args.index_dir or (indexes_root / args.type)

    if not index_path.exists():
        print(f"Error: Index path does not exist: {index_path}", file=sys.stderr)
        print("Hint: Build the index first using 'python3 scripts/build_indexes.py --type all'", file=sys.stderr)
        return 1

    # Initialize retriever
    start_load = time.perf_counter()
    if args.type == "bm25":
        retriever = BM25Retriever()
        try:
            retriever.load_index(index_path)
        except Exception as e:
            print(f"Error loading BM25 index: {e}", file=sys.stderr)
            return 1
    else:
        encoder = MockEmbeddingEncoder() if args.mock_embeddings else SentenceTransformerEncoder()
        retriever = DenseRetriever(encoder=encoder)
        try:
            retriever.load_index(index_path)
        except Exception as e:
            print(f"Error loading Dense index: {e}", file=sys.stderr)
            return 1
    load_time_ms = (time.perf_counter() - start_load) * 1000.0

    # Execute search
    start_search = time.perf_counter()
    results = retriever.search(args.query, top_k=args.top_k)
    search_time_ms = (time.perf_counter() - start_search) * 1000.0

    if args.json:
        output_data = {
            "query": args.query,
            "retriever_type": args.type,
            "top_k": args.top_k,
            "load_time_ms": round(load_time_ms, 2),
            "search_time_ms": round(search_time_ms, 2),
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(output_data, indent=2))
        return 0

    print("=" * 70)
    print(f"FastRAG Query Execution ({args.type.upper()})")
    print("=" * 70)
    print(f"Query:        '{args.query}'")
    print(f"Index:        {index_path}")
    print(f"Search Time:  {search_time_ms:.2f} ms (Index load: {load_time_ms:.2f} ms)")
    print(f"Top-K:        {args.top_k}")
    print(f"Hits:         {len(results)}")
    print("-" * 70)

    if not results:
        print("No matching chunks retrieved.")
        return 0

    for res in results:
        preview = res.text.replace("\n", " ")[:100] + "..." if len(res.text) > 100 else res.text.replace("\n", " ")
        print(f"Rank {res.rank:<2} | Score: {res.score:8.4f} | Chunk: {res.chunk_id:<20} | Doc: {res.document_id}")
        print(f"        Snippet: {preview}")
        print("-" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
