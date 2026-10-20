#!/usr/bin/env python3
"""FastRAG Throughput and Concurrency Benchmark.

Measures retrieval throughput (QPS) and latency distributions across:
- BM25 alone
- Dense alone
- Hybrid Sequential (control baseline)
- Hybrid Concurrent (asyncio branch dispatch)
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
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
from evaluation.retriever_adapter import load_evaluation_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastRAG Concurrency and Throughput Benchmark")
    parser.add_argument(
        "--queries",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "queries.jsonl",
        help="Path to queries.jsonl",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Number of passes through the query dataset (default: 2)",
    )
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use mock embeddings instead of neural encoder for quick verification",
    )
    return parser.parse_args()


async def benchmark_hybrid(
    hybrid: HybridRetriever,
    queries: list[str],
    concurrent: bool,
    label: str,
) -> dict[str, float]:
    latencies: list[float] = []

    # Warmup
    await hybrid.search_async("warmup query", top_k=5, concurrent=concurrent)

    start_total = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        await hybrid.search_async(q, top_k=5, concurrent=concurrent)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    total_time_s = time.perf_counter() - start_total
    qps = len(queries) / total_time_s if total_time_s > 0 else 0.0

    sorted_lats = sorted(latencies)
    n = len(sorted_lats)
    p50 = sorted_lats[int(n * 0.50)]
    p95 = sorted_lats[min(int(n * 0.95), n - 1)]
    p99 = sorted_lats[min(int(n * 0.99), n - 1)]
    mean_lat = statistics.mean(latencies)

    print(f"{label:<30} | {qps:6.2f} QPS | mean: {mean_lat:6.2f}ms | p50: {p50:6.2f}ms | p95: {p95:6.2f}ms")

    return {
        "qps": round(qps, 2),
        "mean_ms": round(mean_lat, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "total_time_s": round(total_time_s, 4),
    }


async def main() -> int:
    args = parse_args()

    indexes_root = PROJECT_ROOT / "datasets" / "indexes"
    bm25_dir = indexes_root / "bm25"
    dense_dir = indexes_root / "dense"

    if not bm25_dir.exists() or not dense_dir.exists():
        print("Indexes not found. Run 'python3 scripts/build_indexes.py --type all' first.", file=sys.stderr)
        return 1

    query_records = load_evaluation_queries(args.queries)
    base_queries = [q["query"] for q in query_records]
    queries = base_queries * args.iterations

    bm25 = BM25Retriever()
    bm25.load_index(bm25_dir)

    encoder = MockEmbeddingEncoder() if args.mock_embeddings else SentenceTransformerEncoder()
    dense = DenseRetriever(encoder=encoder)
    dense.load_index(dense_dir)

    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    print("=" * 80)
    print(f"FastRAG Concurrency & Throughput Benchmark ({len(queries)} queries, iterations={args.iterations})")
    print("=" * 80)

    # 1. Hybrid Sequential
    await benchmark_hybrid(hybrid, queries, concurrent=False, label="Hybrid Sequential (Control)")

    # 2. Hybrid Concurrent
    await benchmark_hybrid(hybrid, queries, concurrent=True, label="Hybrid Concurrent (asyncio)")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
