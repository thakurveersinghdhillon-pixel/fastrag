#!/usr/bin/env python3
"""FastRAG Index Construction CLI.

Builds Dense and BM25 search indexes from processed chunk JSONL files and
saves them to datasets/indexes/ with cryptographic corpus verification.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root and backend to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.retrieval.dense import MockEmbeddingEncoder, SentenceTransformerEncoder
from app.retrieval.indexer import (
    build_and_save_bm25_index,
    build_and_save_dense_index,
    load_chunks_for_indexing,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="FastRAG Index Builder CLI.")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "chunks.jsonl",
        help="Path to processed chunks JSONL file (default: datasets/processed/chunks.jsonl)",
    )
    parser.add_argument(
        "--type",
        choices=["all", "dense", "bm25"],
        default="all",
        help="Type of index to construct (default: all)",
    )
    parser.add_argument(
        "--dense-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "indexes" / "dense",
        help="Destination directory for dense index",
    )
    parser.add_argument(
        "--bm25-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "indexes" / "bm25",
        help="Destination directory for BM25 index",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name for dense index",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 term saturation parameter k1 (default: 1.5)",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 document length normalization parameter b (default: 0.75)",
    )
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use deterministic mock embeddings instead of downloading neural models",
    )

    args = parser.parse_args()

    if not args.chunks.is_file():
        print(f"Error: Chunks file not found at '{args.chunks}'.", file=sys.stderr)
        print("Run 'python3 scripts/process_documents.py' first to generate chunks.", file=sys.stderr)
        return 1

    print(f"Loading chunks from: {args.chunks}")
    chunks = load_chunks_for_indexing(args.chunks)
    print(f"Loaded {len(chunks)} chunks.")

    # 1. Build BM25 Index
    if args.type in ("all", "bm25"):
        print(f"\nBuilding BM25 Index (k1={args.k1}, b={args.b})...")
        t0 = time.perf_counter()
        _, bm25_info = build_and_save_bm25_index(
            chunks=chunks,
            output_dir=args.bm25_dir,
            k1=args.k1,
            b=args.b,
        )
        t_bm25 = time.perf_counter() - t0
        print(f"BM25 index built in {t_bm25*1000:.2f}ms.")
        print(f"Saved to: {args.bm25_dir} (vocab={bm25_info.hyperparameters.get('vocab_size')})")

    # 2. Build Dense Index
    if args.type in ("all", "dense"):
        encoder = None
        if args.mock_embeddings:
            print("\n[NOTE] Using deterministic MockEmbeddingEncoder (--mock-embeddings).")
            encoder = MockEmbeddingEncoder(dimension=64)
        else:
            print(f"\nInitializing Dense Encoder ({args.model_name})...")
            encoder = SentenceTransformerEncoder(model_name=args.model_name)

        print(f"Building Dense Index (batch_size={args.batch_size})...")
        t0 = time.perf_counter()
        try:
            _, dense_info = build_and_save_dense_index(
                chunks=chunks,
                output_dir=args.dense_dir,
                model_name=args.model_name,
                batch_size=args.batch_size,
                encoder=encoder,
            )
            t_dense = time.perf_counter() - t0
            print(f"Dense index built in {t_dense:.2f}s.")
            print(f"Saved to: {args.dense_dir} (dim={dense_info.hyperparameters.get('dimension')})")
        except Exception as exc:
            print(f"Error building dense index: {exc}", file=sys.stderr)
            return 1

    print("\nIndexing complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
