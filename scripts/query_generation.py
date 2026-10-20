#!/usr/bin/env python3
"""FastRAG End-to-End Generation CLI.

Executes complete RAG pipeline:
Candidate Retrieval (Hybrid, Dense, or BM25)
  -> CrossEncoder Reranking
  -> Token-Budgeted Context Assembly
  -> Grounded LLM Generation (Non-Streaming or Live Incremental Streaming)

Displays grounded answer, cited sources, token consumption, and component-level latency.
"""

from __future__ import annotations

import argparse
import asyncio
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
from app.core.exceptions import FastRAGError, GenerationError, RetrievalError
from app.generation.generator import RAGGenerator
from app.generation.providers import MockLLMProvider, OpenAICompatibleProvider
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
        description="FastRAG Generation CLI: Candidate Search -> CrossEncoder -> Context Assembly -> LLM Synthesis"
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default="What is Reciprocal Rank Fusion RRF?",
        help="Query text to search and answer (default: 'What is Reciprocal Rank Fusion RRF?')",
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
        help="Number of top reranked chunks for context assembly (default: 5)",
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
        "--bypass-reranking",
        action="store_true",
        help="Bypass neural reranking and pass raw retriever rankings to context assembler.",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help=f"CrossEncoder model name override (default: {DEFAULT_RERANKER_MODEL})",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "mock"],
        default=None,
        help="LLM provider override ('openai' or 'mock')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM generation model override (e.g. 'gpt-4o-mini', 'mock-llm-v1')",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key override for hosted LLM provider",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL override for OpenAI-compatible LLM endpoint",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for LLM generation (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable live real-time token streaming to stdout.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Shortcut to run entirely in mock mode (mock embeddings, mock reranker, mock LLM provider).",
    )
    parser.add_argument(
        "--mock-reranker",
        action="store_true",
        help="Use deterministic mock CrossEncoder scorer without downloading neural weights.",
    )
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use deterministic mock embeddings for dense retrieval.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON response with diagnostics, citations, and token usage.",
    )
    return parser.parse_args()


def load_retriever(args: argparse.Namespace) -> Any:
    indexes_root = PROJECT_ROOT / "datasets" / "indexes"
    use_mock_embeddings = args.mock or args.mock_embeddings

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
        encoder = MockEmbeddingEncoder(dimension=384) if use_mock_embeddings else SentenceTransformerEncoder()
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

        encoder = MockEmbeddingEncoder(dimension=384) if use_mock_embeddings else SentenceTransformerEncoder()
        dense = DenseRetriever(encoder=encoder)
        dense.load_index(dense_dir)

        return HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            dense_candidate_count=args.candidates,
            bm25_candidate_count=args.candidates,
            top_k=args.candidates,
        )


async def run_streaming(pipeline: RAGPipeline, args: argparse.Namespace) -> int:
    """Run pipeline with live streaming output to terminal."""
    accumulated_tokens: list[str] = []
    final_data: dict[str, Any] = {}
    context_data: dict[str, Any] = {}

    print(f"\n[Query]: {args.query}\n")
    print("Retrieving candidates, reranking, and assembling context...", end="", flush=True)

    try:
        async for event in pipeline.execute_stream_async(
            query=args.query,
            candidate_k=args.candidates,
            rerank_top_k=args.top_k,
            context_max_tokens=args.context_max_tokens,
            context_max_chunks=args.context_max_chunks,
            bypass_reranking=args.bypass_reranking,
            generation_temperature=args.temperature,
            generation_max_tokens=args.max_tokens,
            generation_model=args.model,
        ):
            if event.event == "context":
                context_data = event.data
                print(" Done!\n")
                chunks_count = len(context_data.get("selected_chunks", []))
                approx_tok = context_data.get("approximate_tokens", 0)
                print(f"[Context]: Selected {chunks_count} passages (~{approx_tok} tokens).")
                print("-" * 76)
                print("[Streaming Answer]:\n")

            elif event.event == "token":
                delta = event.data.get("delta", "")
                accumulated_tokens.append(delta)
                sys.stdout.write(delta)
                sys.stdout.flush()

            elif event.event == "done":
                final_data = event.data

            elif event.event == "error":
                print(f"\n[Stream Error]: {event.data.get('error')}", file=sys.stderr)
                return 1

    except Exception as exc:
        print(f"\n[Execution Error]: {exc}", file=sys.stderr)
        return 1

    print("\n\n" + "-" * 76)
    diag = final_data.get("diagnostics") or {}
    usage = final_data.get("usage") or {}
    print(f"[Streaming Telemetry]:")
    print(f"  Time to First Token (TTFT): {diag.get('ttft_ms', 'N/A')} ms")
    print(f"  Total Generation Latency:   {diag.get('total_duration_ms', 'N/A')} ms")
    print(f"  Total Pipeline Latency:     {final_data.get('total_pipeline_ms', 'N/A')} ms")
    print(f"  Token Usage: Prompt={usage.get('prompt_tokens', 'N/A')} | Completion={usage.get('completion_tokens', 'N/A')} | Total={usage.get('total_tokens', 'N/A')}")
    print("=" * 76 + "\n")
    return 0


async def run_non_streaming(pipeline: RAGPipeline, args: argparse.Namespace) -> int:
    """Run pipeline and display structured response."""
    try:
        response = await pipeline.execute_async(
            query=args.query,
            candidate_k=args.candidates,
            rerank_top_k=args.top_k,
            context_max_tokens=args.context_max_tokens,
            context_max_chunks=args.context_max_chunks,
            bypass_reranking=args.bypass_reranking,
            generate_answer=True,
            generation_temperature=args.temperature,
            generation_max_tokens=args.max_tokens,
            generation_model=args.model,
        )
    except FastRAGError as exc:
        print(f"[FastRAG Error]: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[Unexpected Pipeline Failure]: {exc}", file=sys.stderr)
        return 1

    gen = response.generation
    if gen is None:
        print("[Error]: Generation response is None", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "query": response.query,
            "answer": gen.answer,
            "model": gen.model,
            "provider": gen.provider,
            "finish_reason": gen.finish_reason,
            "cited_doc_ids": gen.cited_doc_ids,
            "context_insufficient": gen.context_insufficient,
            "diagnostics": response.diagnostics.to_dict(),
            "usage": gen.usage.to_dict(),
            "context": {
                "total_chunks_selected": response.context.total_chunks_selected,
                "approximate_token_count": response.context.approximate_token_count,
                "citations": {c.citation_id: c.document_id for c in response.context.selected_chunks},
            },
        }
        print(json.dumps(payload, indent=2))
        return 0

    d = response.diagnostics
    ctx = response.context

    print("=" * 76)
    print("FastRAG Grounded Retrieval & Generation Execution")
    print("=" * 76)
    print(f"Query:              '{response.query}'")
    print(f"Retrieval Engine:   {args.mode.upper()}")
    print(f"LLM Provider:       {gen.provider} ({gen.model})")
    print(f"Total Pipeline:     {d.total_ms:8.2f} ms")
    print(f"  - Candidate Search:{d.retrieval_ms:8.2f} ms (candidates: {d.candidate_count})")
    print(f"  - Reranking:       {d.rerank_ms:8.2f} ms (reranked: {d.reranked_count})")
    print(f"  - Context Assembly:{d.assembly_ms:8.2f} ms (passages: {ctx.total_chunks_selected}, ~{ctx.approximate_token_count} tokens)")
    print(f"  - LLM Generation:  {d.generation_ms:8.2f} ms")
    if d.ttft_ms is not None:
        print(f"  - Time to 1st Token:{d.ttft_ms:8.2f} ms")
    print("-" * 76)
    print(f"Token Consumption:  Prompt={gen.usage.prompt_tokens} | Completion={gen.usage.completion_tokens} | Total={gen.usage.total_tokens} (estimated: {gen.usage.is_estimated})")
    print(f"Cited Documents:    {', '.join(gen.cited_doc_ids) if gen.cited_doc_ids else 'None'}")
    print(f"Context Status:     {'INSUFFICIENT' if gen.context_insufficient else 'GROUNDED'}")
    print("-" * 76)
    print("Grounded Answer:\n")
    print(gen.answer)
    print("\n" + "=" * 76)
    return 0


def main() -> int:
    args = parse_args()
    settings = get_settings()

    # Load candidate retriever
    try:
        retriever = load_retriever(args)
    except Exception as exc:
        print(f"Error loading retriever: {exc}", file=sys.stderr)
        return 1

    # Configure Scorer / Reranker
    use_mock_reranker = args.mock or args.mock_reranker
    if use_mock_reranker:
        scorer = MockCrossEncoderScorer()
        reranker = CrossEncoderReranker(scorer=scorer, default_top_k=args.top_k)
    else:
        reranker = CrossEncoderReranker(
            model_name=args.reranker_model,
            default_top_k=args.top_k,
        )

    # Configure Context Assembler
    assembler = ContextAssembler(
        max_tokens=args.context_max_tokens,
        max_chunks=args.context_max_chunks,
    )

    # Configure LLM Provider
    use_mock_llm = args.mock or (args.provider == "mock") or (args.provider is None and settings.llm_provider == "mock")
    if use_mock_llm:
        provider = MockLLMProvider(model_name=args.model or "mock-llm-v1")
    else:
        provider = OpenAICompatibleProvider(
            api_key=args.api_key or settings.llm_api_key,
            model_name=args.model or settings.llm_model,
            base_url=args.base_url or settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    generator = RAGGenerator(
        provider=provider,
        default_temperature=args.temperature,
        default_max_tokens=args.max_tokens,
    )

    # Assemble End-to-End Pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        context_assembler=assembler,
        generator=generator,
    )

    if args.stream:
        return asyncio.run(run_streaming(pipeline, args))
    else:
        return asyncio.run(run_non_streaming(pipeline, args))


if __name__ == "__main__":
    sys.exit(main())
