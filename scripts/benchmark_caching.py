"""FastRAG Caching Performance Benchmark (Phase 8.1).

Measures repeated-query latency and throughput:
1. Caching Disabled (baseline control)
2. Caching Enabled: Cold / Cache-Miss Pass
3. Caching Enabled: Warm / Cache-Hit Pass
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.caching.embedding_cache import QueryEmbeddingCache
from app.caching.query_cache import QueryResponseCache
from app.core.config import get_settings
from app.generation.generator import RAGGenerator
from app.generation.providers import MockLLMProvider
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.context_assembler import ContextAssembler
from app.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker


def build_pipeline(enable_caches: bool = True):
    indexes_root = PROJECT_ROOT / "datasets" / "indexes"
    bm25 = BM25Retriever()
    bm25.load_index(indexes_root / "bm25")

    emb_cache = QueryEmbeddingCache(max_size=1000, enabled=enable_caches)
    encoder = SentenceTransformerEncoder()
    dense = DenseRetriever(encoder=encoder, embedding_cache=emb_cache)
    dense.load_index(indexes_root / "dense")

    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)
    reranker = CrossEncoderReranker()
    assembler = ContextAssembler()
    generator = RAGGenerator(provider=MockLLMProvider())

    query_cache = QueryResponseCache(
        max_size=1000,
        ttl_seconds=3600.0,
        enabled=enable_caches,
    )

    pipeline = RAGPipeline(
        retriever=hybrid,
        reranker=reranker,
        context_assembler=assembler,
        generator=generator,
        query_cache=query_cache,
    )
    return pipeline, emb_cache, query_cache


async def run_benchmark():
    queries_file = PROJECT_ROOT / "datasets" / "queries.jsonl"
    with open(queries_file, encoding="utf-8") as f:
        queries = [json.loads(line)["query"] for line in f if line.strip()]

    print("=" * 80)
    print(f"FastRAG Phase 8.1 Caching Benchmark ({len(queries)} queries)")
    print("=" * 80)

    # 1. Warm up model weights
    p_warmup, _, _ = build_pipeline(enable_caches=False)
    print("Warming up models...")
    _ = await p_warmup.execute_async("warmup query", generate_answer=True)

    # -------------------------------------------------------------------------
    # Experiment A: Caching Disabled
    # -------------------------------------------------------------------------
    p_disabled, _, _ = build_pipeline(enable_caches=False)
    disabled_latencies = []
    for q in queries:
        t0 = time.perf_counter()
        resp = await p_disabled.execute_async(q, generate_answer=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        disabled_latencies.append(elapsed_ms)

    # -------------------------------------------------------------------------
    # Experiment B: Caching Enabled — Cold Pass (Cache Misses)
    # -------------------------------------------------------------------------
    p_enabled, emb_cache, q_cache = build_pipeline(enable_caches=True)
    cold_latencies = []
    for q in queries:
        t0 = time.perf_counter()
        resp = await p_enabled.execute_async(q, generate_answer=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        cold_latencies.append(elapsed_ms)

    miss_stats = q_cache.stats()

    # -------------------------------------------------------------------------
    # Experiment C: Caching Enabled — Warm Pass (Cache Hits)
    # -------------------------------------------------------------------------
    warm_latencies = []
    for q in queries:
        t0 = time.perf_counter()
        resp = await p_enabled.execute_async(q, generate_answer=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        warm_latencies.append(elapsed_ms)

    hit_stats = q_cache.stats()

    def summarize(lats: list[float]) -> dict[str, float]:
        s = sorted(lats)
        n = len(s)
        return {
            "mean": round(statistics.mean(s), 3),
            "p50": round(s[int(n * 0.50)], 3),
            "p95": round(s[min(int(n * 0.95), n - 1)], 3),
            "p99": round(s[min(int(n * 0.99), n - 1)], 3),
            "min": round(s[0], 3),
            "max": round(s[-1], 3),
        }

    sum_disabled = summarize(disabled_latencies)
    sum_cold = summarize(cold_latencies)
    sum_warm = summarize(warm_latencies)

    print("\nBenchmark Results (Wall-Clock Latencies in ms):")
    print(f"{'Mode':<30} | {'Mean':>9} | {'P50':>9} | {'P95':>9} | {'Min':>9} | {'Max':>9}")
    print("-" * 80)
    print(f"{'Caching Disabled (Baseline)':<30} | {sum_disabled['mean']:>8.3f}ms | {sum_disabled['p50']:>8.3f}ms | {sum_disabled['p95']:>8.3f}ms | {sum_disabled['min']:>8.3f}ms | {sum_disabled['max']:>8.3f}ms")
    print(f"{'Caching Enabled: Cache Miss':<30} | {sum_cold['mean']:>8.3f}ms | {sum_cold['p50']:>8.3f}ms | {sum_cold['p95']:>8.3f}ms | {sum_cold['min']:>8.3f}ms | {sum_cold['max']:>8.3f}ms")
    print(f"{'Caching Enabled: Cache Hit':<30} | {sum_warm['mean']:>8.3f}ms | {sum_warm['p50']:>8.3f}ms | {sum_warm['p95']:>8.3f}ms | {sum_warm['min']:>8.3f}ms | {sum_warm['max']:>8.3f}ms")
    print("-" * 80)

    speedup_mean = sum_disabled["mean"] / sum_warm["mean"] if sum_warm["mean"] > 0 else 0
    speedup_p50 = sum_disabled["p50"] / sum_warm["p50"] if sum_warm["p50"] > 0 else 0
    print(f"\nMeasured Speedup on Cache Hit:")
    print(f"  Mean speedup: {speedup_mean:.1f}x ({sum_disabled['mean']:.2f}ms -> {sum_warm['mean']:.3f}ms)")
    print(f"  P50 speedup:  {speedup_p50:.1f}x ({sum_disabled['p50']:.2f}ms -> {sum_warm['p50']:.3f}ms)")
    print(f"  Tier 1 Response Cache Hits: {hit_stats.hits}/{hit_stats.total_lookups} (Hit Rate: {hit_stats.hit_rate * 100:.1f}%)")
    print(f"  Tier 2 Embedding Cache Stats: Hits={emb_cache.stats().hits}, Misses={emb_cache.stats().misses}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
