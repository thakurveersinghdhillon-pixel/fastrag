#!/usr/bin/env python3
"""CLI script to run FastRAG Generation and Retrieval Quality Evaluation Harness.

Evaluates answer correctness, completeness, context groundedness, citation
precision/recall, and abstention on eligible dataset records.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.generation.generator import RAGGenerator
from app.generation.providers import MockLLMProvider, OpenAICompatibleProvider
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.context_assembler import ContextAssembler
from app.retrieval.dense import DenseRetriever, MockEmbeddingEncoder, SentenceTransformerEncoder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker, SentenceTransformerCrossEncoder
from evaluation.generation_eval import GenerationEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FastRAG Generation Quality Evaluation on Dataset v2."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "v2",
        help="Path to dataset directory (default: datasets/v2)",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Evaluation split to evaluate (default: dev). Never tune on test.",
    )
    parser.add_argument(
        "--provenance",
        choices=["human_reviewed", "heuristic_candidate", "pending_review", "all"],
        default="human_reviewed",
        help="Filter queries by annotation provenance (default: human_reviewed)",
    )
    parser.add_argument(
        "--category",
        choices=[
            "factual_direct",
            "multi_hop",
            "technical_identifiers",
            "unanswerable_negative",
            "ambiguous_underspecified",
            "citation_grounding",
            "all",
        ],
        default="all",
        help="Filter queries by evaluation category (default: all)",
    )
    parser.add_argument(
        "--retriever",
        choices=["hybrid-rerank", "hybrid", "dense", "bm25"],
        default="hybrid-rerank",
        help="Retriever configuration to evaluate (default: hybrid-rerank)",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "gemini", "openai"],
        default="mock",
        help="LLM generation provider (default: mock)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model override (e.g. gemini-3.8-flash, mock-llm-v1)",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Custom experiment identifier",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Number of initial candidates to retrieve (default: 20)",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=10,
        help="Number of candidates to retain after reranking (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output report to benchmarks/results/",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of evaluation queries (for fast dry runs)",
    )
    parser.add_argument(
        "--min-relevance-score",
        type=float,
        default=None,
        help="Optional minimum relevance score threshold for candidate context filtering",
    )
    parser.add_argument(
        "--ordering-strategy",
        choices=["descending", "boundary"],
        default="descending",
        help="Context chunk ordering strategy (default: descending)",
    )
    parser.add_argument(
        "--blending-alpha",
        type=float,
        default=0.0,
        help="Reciprocal rank blending weight (alpha >= 0.0) between CrossEncoder rank and base candidate rank (default: 0.0)",
    )
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    indexes_root = PROJECT_ROOT / "datasets" / "indexes"
    bm25_dir = indexes_root / "bm25"
    dense_dir = indexes_root / "dense"

    print("=" * 70)
    print(" FastRAG Generation and Retrieval Quality Evaluation Runner")
    print(f" Dataset: {args.dataset_dir} | Split: {args.split} | Provenance: {args.provenance}")
    print(f" Retriever: {args.retriever} | Provider: {args.provider}")
    print(f" Blending Alpha: {args.blending_alpha}")
    print("=" * 70)

    # 1. Initialize retriever components
    bm25 = BM25Retriever()
    bm25.load_index(bm25_dir)

    encoder = SentenceTransformerEncoder()
    dense = DenseRetriever(encoder=encoder)
    dense.load_index(dense_dir)

    if args.retriever in ("hybrid", "hybrid-rerank"):
        base_retriever = HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            dense_candidate_count=args.candidate_k,
            bm25_candidate_count=args.candidate_k,
            top_k=args.candidate_k,
        )
    elif args.retriever == "dense":
        base_retriever = dense
    elif args.retriever == "bm25":
        base_retriever = bm25
    else:
        raise ValueError(f"Unknown retriever: {args.retriever}")

    # 2. Initialize reranker if configured
    reranker: CrossEncoderReranker | None = None
    if args.retriever == "hybrid-rerank":
        reranker = CrossEncoderReranker(
            scorer=SentenceTransformerCrossEncoder(),
            default_top_k=args.rerank_top_k,
            default_blending_alpha=args.blending_alpha,
        )

    # 3. Initialize generator provider
    settings = get_settings()
    if args.provider == "mock":
        model_name = args.model or "mock-llm-v1"
        provider = MockLLMProvider(model_name=model_name)
    elif args.provider in ("gemini", "openai"):
        model_name = args.model or settings.llm_model
        api_key = settings.llm_api_key
        base_url = settings.llm_base_url
        if not api_key:
            print("Error: FASTRAG_LLM_API_KEY environment variable is required for real provider.", file=sys.stderr)
            return 1
        provider = OpenAICompatibleProvider(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    generator = RAGGenerator(provider=provider)
    assembler = ContextAssembler(
        min_relevance_score=args.min_relevance_score,
        ordering_strategy=args.ordering_strategy,
    )

    pipeline = RAGPipeline(
        retriever=base_retriever,
        reranker=reranker,
        context_assembler=assembler,
        generator=generator,
        rerank_blending_alpha=args.blending_alpha,
    )
    pipeline.warmup()

    # 4. Instantiate evaluator
    evaluator = GenerationEvaluator(
        pipeline=pipeline,
        dataset_dir=args.dataset_dir,
        results_dir=PROJECT_ROOT / "benchmarks" / "results",
    )

    suffix_parts = []
    if args.category != "all":
        suffix_parts.append(args.category)
    if args.blending_alpha > 0.0:
        suffix_parts.append(f"alpha_{args.blending_alpha}")
    if args.min_relevance_score is not None:
        suffix_parts.append(f"thresh_{args.min_relevance_score}")
    if args.ordering_strategy != "descending":
        suffix_parts.append(args.ordering_strategy)
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

    exp_id = args.experiment_id or f"generation_phase9_5_{args.split}_{args.provenance}_{args.retriever}_{args.provider}{suffix}"

    try:
        report = await evaluator.run_evaluation(
            experiment_id=exp_id,
            split=args.split,
            provenance_filter=args.provenance,
            candidate_k=args.candidate_k,
            rerank_top_k=args.rerank_top_k,
            save_results=not args.no_save,
            temperature=args.temperature,
            limit=args.limit,
            category_filter=None if args.category == "all" else args.category,
        )
        print("\n" + report.summary_table())
        return 0
    except Exception as exc:
        print(f"\nEvaluation failed with error: {exc}", file=sys.stderr)
        return 1
    finally:
        await pipeline.aclose()


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
