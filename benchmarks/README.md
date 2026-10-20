# FastRAG Performance Benchmarking

This directory contains benchmarking suites, load testing scripts, profiling tools, and benchmark results for measuring FastRAG throughput, latency distributions, and system resource efficiency.

## Benchmarking Objectives

The primary goal of FastRAG is to minimize end-to-end retrieval and generation latency while maximizing retrieval quality and answer accuracy. The benchmark suite systematically characterizes:

1. **End-to-End Latency Profiles**: Wall-clock time from incoming query to first generated token (TTFT - Time To First Token) and total response completion (TLT - Total Latency Time).
2. **Subsystem Latency Distributions**:
   - Query embedding inference latency.
   - Dense vector index lookup latency (HNSW/IVF).
   - Lexical inverted index retrieval latency (BM25).
   - Reciprocal Rank Fusion (RRF) computational overhead.
   - CrossEncoder reranking latency vs. batch size.
   - Context serialization and prompt assembly overhead.
3. **Throughput & Concurrency**: Sustained queries per second (QPS), saturation points, and error rates under varying concurrent client loads.
4. **Comparative Architectural Baselines**:
   - Lexical Only (BM25)
   - Dense Only (Vector Embedding)
   - Sequential Hybrid (Dense followed by Lexical)
   - Concurrent Hybrid (Asynchronous Dense + Lexical with RRF)
   - Hybrid + CrossEncoder RERANK

## Planned Structure (Phase 9)

```
benchmarks/
├── load_tests/           # Locust or custom async HTTP load testing scripts
├── microbenchmarks/      # Targeted microbenchmarks for RRF algorithms and vector lookups
├── profiles/             # CPU, memory, and async task execution profiles
└── results/              # Historical benchmark logs and comparative charts
```
