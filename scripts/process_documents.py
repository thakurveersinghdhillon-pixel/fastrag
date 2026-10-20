#!/usr/bin/env python3
"""FastRAG Document Processing and Ingestion CLI.

Loads documents from JSONL or TXT files, applies conservative normalization,
chunks text into configurable deterministic windows, validates chunk records,
and writes processed chunks to JSONL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root and backend to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.pipeline import IngestionPipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FastRAG Document Ingestion CLI: normalize, chunk, and export documents."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "corpus.jsonl",
        help="Input document file (.jsonl, .txt) or directory (default: datasets/corpus.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "chunks.jsonl",
        help="Destination JSONL file for processed chunks (default: datasets/processed/chunks.jsonl)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk character length (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap character length (default: 150)",
    )
    parser.add_argument(
        "--no-preserve-paragraphs",
        action="store_true",
        help="Disable paragraph boundary preference when splitting chunks",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Halt processing immediately on the first invalid document record",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input path '{args.input}' does not exist.", file=sys.stderr)
        return 1

    if args.chunk_size <= 0:
        print(f"Error: --chunk-size must be strictly positive (got {args.chunk_size})", file=sys.stderr)
        return 1

    if args.chunk_overlap < 0:
        print(f"Error: --chunk-overlap must be non-negative (got {args.chunk_overlap})", file=sys.stderr)
        return 1

    if args.chunk_overlap >= args.chunk_size:
        print(
            f"Error: --chunk-overlap ({args.chunk_overlap}) must be strictly less than --chunk-size ({args.chunk_size})",
            file=sys.stderr,
        )
        return 1

    print(f"Starting FastRAG document ingestion...")
    print(f"  Input:               {args.input}")
    print(f"  Output:              {args.output}")
    print(f"  Chunk Size:          {args.chunk_size}")
    print(f"  Chunk Overlap:       {args.chunk_overlap}")
    print(f"  Preserve Paragraphs: {not args.no_preserve_paragraphs}")
    print(f"  Error Policy:        {'fail_fast' if args.fail_fast else 'collect'}")

    pipeline = IngestionPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        preserve_paragraphs=not args.no_preserve_paragraphs,
        on_error="fail_fast" if args.fail_fast else "collect",
    )

    try:
        chunks, summary = pipeline.process_file(args.input, output_path=args.output)
    except Exception as exc:
        print(f"\nIngestion failed: {exc}", file=sys.stderr)
        return 1

    print("\n" + summary.summary())

    if summary.documents_rejected > 0 and args.fail_fast:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
