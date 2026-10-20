"""FastRAG Phase 8.2 Baseline & Optimization Benchmark.

Reproducibly measures latency across individual pipeline stages and end-to-end requests:
- BM25 Retrieval
- Dense Retrieval (Uncached vs. Cached Embedding)
- Hybrid Retrieval & RRF (Concurrent vs. Sequential)
- CrossEncoder Reranking
- Context Assembly
- LLM Generation (MockLLM baseline)
- End-to-End Pipeline (Uncached Retrieval-Only, Uncached Complete RAG, Warm Cached RAG)
Separates cold-start from warm runs with detailed statistical breakdowns.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.caching.embedding_cache import QueryEmbeddingCache
from app.caching.query_cache import QueryResponseCache
from app.generation.generator import RAGGenerator
from app.generation.providers import MockLLMProvider
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.context_assembler import ContextAssembler
from app.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker


def calc_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    s = sorted(latencies_ms)
    n = len(s)
    return {
        "mean": round(statistics.mean(s), 3),
        "median": round(statistics.median(s), 3),
        "p95": round(s[min(int(n * 0.95), n - 1)], 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
        "stdev": round(statistics.stdev(s) if n > 1 else 0.0, 3),
    }


async def run_stage_benchmarks():
    indexes_root = PROJECT_ROOT / "datasets" / "indexes"
    queries_file = PROJECT_ROOT / "datasets" / "queries.jsonl"

    with open(queries_file, encoding="utf-8") as f:
        queries = [json.loads(line)["query"] for line in f if line.strip()]

    print(f"Loaded {len(queries)} evaluation queries.")

    # -------------------------------------------------------------------------
    # 1. Cold Start Benchmarking (First load of models + first query execution)
    # -------------------------------------------------------------------------
    print("\n[1] Measuring Cold-Start Latencies...")
    
    # BM25 Cold load & query
    t0 = time.perf_counter()
    bm25 = BM25Retriever()
    bm25.load_index(indexes_root / "bm25")
    bm25_load_ms = (time.perf_counter() - t0) * 1000.0
    
    t0 = time.perf_counter()
    bm25_first_results = bm25.search(queries[0], top_k=20)
    bm25_cold_query_ms = (time.perf_counter() - t0) * 1000.0

    # Dense Cold load & query
    t0 = time.perf_counter()
    dense_encoder = SentenceTransformerEncoder()
    dense_retriever = DenseRetriever(encoder=dense_encoder)
    dense_retriever.load_index(indexes_root / "dense")
    dense_load_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    dense_first_results = dense_retriever.search(queries[0], top_k=20)
    dense_cold_query_ms = (time.perf_counter() - t0) * 1000.0

    # CrossEncoder Cold load & query
    t0 = time.perf_counter()
    reranker = CrossEncoderReranker()
    reranker_load_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    reranker_first_results = reranker.rerank(queries[0], dense_first_results, top_k=10)
    reranker_cold_query_ms = (time.perf_counter() - t0) * 1000.0

    # Context Assembler Cold
    assembler = ContextAssembler(max_tokens=1000, max_chunks=5)
    t0 = time.perf_counter()
    context_first = assembler.assemble(reranker_first_results)
    assembler_cold_ms = (time.perf_counter() - t0) * 1000.0

    # Generator Cold
    generator = RAGGenerator(provider=MockLLMProvider())
    t0 = time.perf_counter()
    gen_first = await generator.generate_answer(queries[0], context_first)
    gen_cold_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  BM25:         Load={bm25_load_ms:.2f}ms, First Query={bm25_cold_query_ms:.2f}ms")
    print(f"  Dense:        Load={dense_load_ms:.2f}ms, First Query={dense_cold_query_ms:.2f}ms (incl. model load)")
    print(f"  CrossEncoder: Load={reranker_load_ms:.2f}ms, First Query={reranker_cold_query_ms:.2f}ms (incl. model load)")
    print(f"  Assembler:    First={assembler_cold_ms:.2f}ms")
    print(f"  Generator:    First={gen_cold_ms:.2f}ms")

    # -------------------------------------------------------------------------
    # 2. Warm Stage Benchmarks (5 repetitions across 24 queries = 120 calls each)
    # -------------------------------------------------------------------------
    num_reps = 5
    print(f"\n[2] Measuring Warm Stage Latencies ({num_reps} repetitions x {len(queries)} queries = {num_reps * len(queries)} iterations)...")

    # A. BM25 Retrieval
    bm25_lats = []
    for _ in range(num_reps):
        for q in queries:
            t0 = time.perf_counter()
            _ = bm25.search(q, top_k=20)
            bm25_lats.append((time.perf_counter() - t0) * 1000.0)

    # B. Dense Retrieval (Uncached - raw transformer encoder + dot product)
    dense_uncached_lats = []
    dense_no_cache = DenseRetriever(encoder=dense_encoder, embedding_cache=None)
    dense_no_cache.load_index(indexes_root / "dense")
    for _ in range(num_reps):
        for q in queries:
            t0 = time.perf_counter()
            _ = dense_no_cache.search(q, top_k=20)
            dense_uncached_lats.append((time.perf_counter() - t0) * 1000.0)

    # C. Dense Retrieval (Cached Embedding - dot product only)
    emb_cache = QueryEmbeddingCache(max_size=1000, enabled=True)
    dense_cached = DenseRetriever(encoder=dense_encoder, embedding_cache=emb_cache)
    dense_cached.load_index(indexes_root / "dense")
    # Pre-populate embedding cache
    for q in queries:
        dense_cached.search(q, top_k=20)
    dense_cached_lats = []
    for _ in range(num_reps):
        for q in queries:
            t0 = time.perf_counter()
            _ = dense_cached.search(q, top_k=20)
            dense_cached_lats.append((time.perf_counter() - t0) * 1000.0)

    # D. Hybrid Retrieval (Concurrent vs Sequential)
    hybrid_concurrent = HybridRetriever(dense_retriever=dense_no_cache, bm25_retriever=bm25)
    hybrid_conc_lats = []
    for _ in range(num_reps):
        for q in queries:
            t0 = time.perf_counter()
            _ = await hybrid_concurrent.search_async(q, top_k=20, concurrent=True)
            hybrid_conc_lats.append((time.perf_counter() - t0) * 1000.0)

    hybrid_seq_lats = []
    for _ in range(num_reps):
        for q in queries:
            t0 = time.perf_counter()
            _ = await hybrid_concurrent.search_async(q, top_k=20, concurrent=False)
            hybrid_seq_lats.append((time.perf_counter() - t0) * 1000.0)

    # E. CrossEncoder Reranking
    # Use real candidate lists from hybrid retrieval for each query
    candidates_per_query = {}
    for q in queries:
        resp = await hybrid_concurrent.search_async(q, top_k=20)
        candidates_per_query[q] = resp.results

    reranker_lats = []
    for _ in range(num_reps):
        for q in queries:
            cands = candidates_per_query[q]
            t0 = time.perf_counter()
            _ = reranker.rerank(q, cands, top_k=10)
            reranker_lats.append((time.perf_counter() - t0) * 1000.0)

    # F. Context Assembly
    reranked_per_query = {}
    for q in queries:
        reranked_per_query[q] = reranker.rerank(q, candidates_per_query[q], top_k=10)

    assembler_lats = []
    for _ in range(num_reps):
        for q in queries:
            r_cands = reranked_per_query[q]
            t0 = time.perf_counter()
            _ = assembler.assemble(r_cands, max_tokens=1000, max_chunks=5)
            assembler_lats.append((time.perf_counter() - t0) * 1000.0)

    # G. LLM Generation (MockLLMProvider)
    contexts_per_query = {q: assembler.assemble(reranked_per_query[q]) for q in queries}
    gen_lats = []
    for _ in range(num_reps):
        for q in queries:
            ctx = contexts_per_query[q]
            t0 = time.perf_counter()
            _ = await generator.generate_answer(q, ctx)
            gen_lats.append((time.perf_counter() - t0) * 1000.0)

    # -------------------------------------------------------------------------
    # 3. End-to-End Pipeline Benchmarks
    # -------------------------------------------------------------------------
    print("\n[3] Measuring End-to-End Pipeline Latencies...")

    # A. Uncached Pipeline (Retrieval + Rerank + Assembly, generate_answer=False)
    p_uncached_search = RAGPipeline(
        retriever=hybrid_concurrent,
        reranker=reranker,
        context_assembler=assembler,
        generator=generator,
        query_cache=None,
    )
    e2e_search_lats = []
    for _ in range(num_reps):
        for q in queries:
            t0 = time.perf_counter()
            _ = await p_uncached_search.execute_async(q, generate_answer=False)
            e2e_search_lats.append((time.perf_counter() - t0) * 1000.0)

    # B. Uncached Pipeline Complete RAG (Retrieval + Rerank + Assembly + Generation)
    e2e_rag_lats = []
    for _ in range(num_reps):
        for q in queries:
            t0 = time.perf_counter()
            _ = await p_uncached_search.execute_async(q, generate_answer=True)
            e2e_rag_lats.append((time.perf_counter() - t0) * 1000.0)

    # C. Warm Cached Pipeline Complete RAG (Query Cache Enabled & Pre-warmed)
    q_cache = QueryResponseCache(max_size=1000, ttl_seconds=3600.0, enabled=True)
    p_cached_rag = RAGPipeline(
        retriever=hybrid_concurrent,
        reranker=reranker,
        context_assembler=assembler,
        generator=generator,
        query_cache=q_cache,
    )
    # Warm up cache
    for q in queries:
        await p_cached_rag.execute_async(q, generate_answer=True)
    
    e2e_cached_lats = []
    for _ in range(num_reps):
        for q in queries:
            t0 = time.perf_counter()
            _ = await p_cached_rag.execute_async(q, generate_answer=True)
            e2e_cached_lats.append((time.perf_counter() - t0) * 1000.0)

    # -------------------------------------------------------------------------
    # 4. Measure First-Query Latency WITH Warmup Pre-Executed
    # -------------------------------------------------------------------------
    print("\n[4] Measuring First-Query Latency WITH Warmup Pre-Executed...")
    p_prewarmed = RAGPipeline(
        retriever=HybridRetriever(
            dense_retriever=DenseRetriever(encoder=SentenceTransformerEncoder()),
            bm25_retriever=bm25,
        ),
        reranker=CrossEncoderReranker(),
        context_assembler=assembler,
        generator=generator,
        query_cache=None,
    )
    t_warmup_start = time.perf_counter()
    p_prewarmed.warmup()
    warmup_duration_ms = (time.perf_counter() - t_warmup_start) * 1000.0

    t_first_query = time.perf_counter()
    _ = await p_prewarmed.execute_async(queries[0], generate_answer=False)
    first_query_after_warmup_ms = (time.perf_counter() - t_first_query) * 1000.0

    print(f"  Warmup execution time:             {warmup_duration_ms:.2f} ms")
    print(f"  First user query AFTER warmup:     {first_query_after_warmup_ms:.2f} ms (vs ~6,095 ms cold start)")

    # Print summary table
    print("\n" + "=" * 90)
    print("FastRAG Phase 8.2 Optimized Latency Profile (All times in milliseconds)")
    print("=" * 90)
    print(f"{'Component / Scenario':<40} | {'Mean':>8} | {'Median':>8} | {'P95':>8} | {'Min':>8} | {'Max':>8} | {'Throughput':>10}")
    print("-" * 90)

    results = {
        "BM25 Retrieval": calc_stats(bm25_lats),
        "Dense Retrieval (Uncached Enc)": calc_stats(dense_uncached_lats),
        "Dense Retrieval (Cached Enc)": calc_stats(dense_cached_lats),
        "Hybrid RRF (Concurrent)": calc_stats(hybrid_conc_lats),
        "Hybrid RRF (Sequential)": calc_stats(hybrid_seq_lats),
        "CrossEncoder Reranker": calc_stats(reranker_lats),
        "Context Assembly": calc_stats(assembler_lats),
        "LLM Generation (Mock)": calc_stats(gen_lats),
        "E2E Uncached Search (No Gen)": calc_stats(e2e_search_lats),
        "E2E Uncached Complete RAG": calc_stats(e2e_rag_lats),
        "E2E Warm Cache Hit RAG": calc_stats(e2e_cached_lats),
    }

    for name, s in results.items():
        qps = 1000.0 / s["mean"] if s["mean"] > 0 else 0
        print(f"{name:<40} | {s['mean']:>7.3f} | {s['median']:>7.3f} | {s['p95']:>7.3f} | {s['min']:>7.3f} | {s['max']:>7.3f} | {qps:>8.1f} q/s")
    print("=" * 90)

    # Save results to benchmarks/results/phase8_2_optimized.json
    out_path = PROJECT_ROOT / "benchmarks" / "results" / "phase8_2_optimized.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cold_start": {
                    "bm25_load_ms": bm25_load_ms,
                    "bm25_first_query_ms": bm25_cold_query_ms,
                    "dense_load_ms": dense_load_ms,
                    "dense_first_query_ms": dense_cold_query_ms,
                    "reranker_load_ms": reranker_load_ms,
                    "reranker_first_query_ms": reranker_cold_query_ms,
                    "assembler_cold_ms": assembler_cold_ms,
                    "gen_cold_ms": gen_cold_ms,
                    "warmup_duration_ms": warmup_duration_ms,
                    "first_query_after_warmup_ms": first_query_after_warmup_ms,
                },
                "warm_profile": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved optimized benchmark data to {out_path}")

    # Load baseline if available and print comparison
    baseline_path = PROJECT_ROOT / "benchmarks" / "results" / "phase8_2_baseline.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)["warm_profile"]
        print("\n" + "=" * 90)
        print("FastRAG Phase 8.2 Optimization Comparison (Baseline vs Optimized Median Latency)")
        print("=" * 90)
        print(f"{'Component / Scenario':<40} | {'Baseline Median':>15} | {'Optimized Median':>16} | {'Speedup':>10}")
        print("-" * 90)
        for key in results:
            if key in base_data:
                b_med = base_data[key]["median"]
                o_med = results[key]["median"]
                ratio = b_med / o_med if o_med > 0 else 1.0
                print(f"{key:<40} | {b_med:>13.3f}ms | {o_med:>14.3f}ms | {ratio:>9.2f}x")
        print("=" * 90)


if __name__ == "__main__":
    asyncio.run(run_stage_benchmarks())
