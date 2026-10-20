#!/usr/bin/env python3
"""FastRAG Reranked Query Execution CLI.

Executes candidate retrieval (Hybrid, Dense, or BM25) followed by CrossEncoder reranking
and token-budgeted context assembly, displaying rank scores, source provenance, and
component-level latency diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.core.exceptions import RetrievalError
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.context_assembler import ContextAssembler
from app.retrieval.dense import DenseRetriever, MockEmbeddingEncoder, SentenceTransformerEncoder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pipeline import RAGPipeline
from app.retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    MockCrossEncoderScorer,
    SentenceTransformerCrossEncoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastRAG Reranked Query CLI: Candidate Retrieval -> CrossEncoder -> Context Assembly"
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default="What is Reciprocal Rank Fusion RRF?",
        help="Query text string to search for (default: 'What is Reciprocal Rank Fusion RRF?')",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "dense", "bm25"],
        default="hybrid",
        help="Candidate retrieval engine (default: hybrid)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=20,
        help="Number of initial candidates to retrieve per branch (default: 20)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top reranked chunks to retain (default: 5)",
    )
    parser.add_argument(
        "--context-max-tokens",
        type=int,
        default=800,
        help="Maximum approximate tokens in assembled context (default: 800)",
    )
    parser.add_argument(
        "--context-max-chunks",
        type=int,
        default=4,
        help="Maximum chunks in assembled context (default: 4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"CrossEncoder model name override (default from config or {DEFAULT_RERANKER_MODEL})",
    )
    parser.add_argument(
        "--mock-reranker",
        action="store_true",
        help="Use deterministic mock CrossEncoder scorer for offline/fast testing without neural weights.",
    )
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use deterministic mock embeddings for dense retrieval instead of downloading neural model.",
    )
    parser.add_argument(
        "--bypass-reranking",
        action="store_true",
        help="Bypass the neural reranking stage and pass raw candidate rankings directly to context assembler.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON response including candidates, reranked results, context, and diagnostics.",
    )
    return parser.parse_args()


def load_retriever(args: argparse.Namespace) -> Any:
    indexes_root = PROJECT_ROOT / "datasets" / "indexes"

    if args.mode == "bm25":
        bm25_dir = indexes_root / "bm25"
        if not bm25_dir.exists():
            raise FileNotFoundError(f"BM25 index not found at {bm25_dir}. Run build_indexes.py first.")
        bm25 = BM25Retriever()
        bm25.load_index(bm25_dir)
        return bm25

    elif args.mode == "dense":
        dense_dir = indexes_root / "dense"
        if not dense_dir.exists():
            raise FileNotFoundError(f"Dense index not found at {dense_dir}. Run build_indexes.py first.")
        encoder = MockEmbeddingEncoder() if args.mock_embeddings else SentenceTransformerEncoder()
        dense = DenseRetriever(encoder=encoder)
        dense.load_index(dense_dir)
        return dense

    else:  # hybrid
        bm25_dir = indexes_root / "bm25"
        dense_dir = indexes_root / "dense"
        if not bm25_dir.exists() or not dense_dir.exists():
            raise FileNotFoundError("Indexes missing. Run 'python3 scripts/build_indexes.py --type all'")

        bm25 = BM25Retriever()
        bm25.load_index(bm25_dir)

        encoder = MockEmbeddingEncoder() if args.mock_embeddings else SentenceTransformerEncoder()
        dense = DenseRetriever(encoder=encoder)
        dense.load_index(dense_dir)

        return HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            dense_candidate_count=args.candidates,
            bm25_candidate_count=args.candidates,
            top_k=args.candidates,
        )


def main() -> int:
    args = parse_args()

    try:
        retriever = load_retriever(args)
    except Exception as exc:
        print(f"Error loading retriever: {exc}", file=sys.stderr)
        return 1

    # Configure Scorer / Reranker
    if args.mock_reranker:
        scorer = MockCrossEncoderScorer()
        reranker = CrossEncoderReranker(scorer=scorer, default_top_k=args.top_k)
    else:
        reranker = CrossEncoderReranker(
            model_name=args.model,
            default_top_k=args.top_k,
        )

    # Configure Assembler
    assembler = ContextAssembler(
        max_tokens=args.context_max_tokens,
        max_chunks=args.context_max_chunks,
    )

    # Build Pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        context_assembler=assembler,
    )

    try:
        response = pipeline.execute(
            query=args.query,
            candidate_k=args.candidates,
            rerank_top_k=args.top_k,
            context_max_tokens=args.context_max_tokens,
            context_max_chunks=args.context_max_chunks,
            bypass_reranking=args.bypass_reranking,
        )
    except RetrievalError as exc:
        print(f"Retrieval / Reranking Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected Pipeline Failure: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
        return 0

    # Format Human-Readable Report
    d = response.diagnostics
    ctx = response.context
    scorer_type = "MOCK" if args.mock_reranker else ("BYPASSED" if args.bypass_reranking else reranker.model_name)

    print("=" * 76)
    print("FastRAG Retrieval & Reranking Pipeline Execution")
    print("=" * 76)
    print(f"Query:              '{response.query}'")
    print(f"Mode:               {args.mode.upper()} -> Reranker ({scorer_type}) -> ContextAssembler")
    print(f"Total Pipeline:     {d.total_ms:8.2f} ms")
    print(f"  - Candidate Search:{d.retrieval_ms:8.2f} ms (candidates: {d.candidate_count})")
    if d.hybrid_breakdown:
        h = d.hybrid_breakdown
        dense_ms = h.get("dense", {}).get("duration_ms", 0.0)
        bm25_ms = h.get("bm25", {}).get("duration_ms", 0.0)
        fusion_ms = h.get("fusion_duration_ms", 0.0)
        print(f"    * Dense branch:  {dense_ms:8.2f} ms")
        print(f"    * BM25 branch:   {bm25_ms:8.2f} ms")
        print(f"    * RRF fusion:    {fusion_ms:8.2f} ms")
    print(f"  - Reranking:      {d.rerank_ms:8.2f} ms (reranked: {d.reranked_count})")
    print(f"  - Context Assembly:{d.assembly_ms:8.2f} ms (selected chunks: {d.selected_context_count})")
    print("-" * 76)
    print("Reranked Top Candidates:")
    print("-" * 76)

    for r in response.reranked_results:
        meta = r.metadata or {}
        orig_sc = meta.get("original_score", "N/A")
        orig_rank = meta.get("original_rank", "N/A")
        title = meta.get("title") or "Untitled"
        print(f"Rank {r.rank:<2} | Rerank Score: {r.score:8.4f} | Orig Rank: {orig_rank} (Score: {orig_sc}) | Chunk: {r.chunk_id}")
        print(f"        Doc: {r.document_id} | Title: {title}")
        snippet = r.text.replace("\n", " ")
        if len(snippet) > 110:
            snippet = snippet[:110] + "..."
        print(f"        Snippet: {snippet}")
        print("-" * 76)

    print("Assembled Context Package:")
    print("-" * 76)
    print(f"Total Selected Chunks: {ctx.total_chunks_selected} / budget {ctx.max_chunks_budget}")
    print(f"Approximate Tokens:    {ctx.approximate_token_count} / budget {ctx.token_budget}")
    print(f"Budget Exceeded:       {ctx.budget_exceeded}")
    print(f"Truncation Applied:    {ctx.truncation_applied}")
    if ctx.truncation_reasons:
        for reason in ctx.truncation_reasons:
            print(f"  Notice: {reason}")
    print("\nFormatted Context Text:")
    print("----------------------------------------------------------------------------")
    print(ctx.context_text)
    print("=" * 76)

    return 0


if __name__ == "__main__":
    sys.exit(main())
