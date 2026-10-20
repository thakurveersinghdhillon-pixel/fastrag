# FastRAG Phase 8.2: Performance Optimization Audit and Plan

**Date**: October 15, 2026  
**Status**: Completed Baseline Audit & Optimization Plan  
**Target Project**: `/Users/thakurveersingh/Documents/fastrag`  
**Test Suite Baseline**: 178/178 Passing (100% Pass Rate in ~0.77s)  
**Retrieval Metric Baseline**: Recall@5 = 0.7299, Recall@10 = 0.8507, MRR@10 = 1.0000, nDCG@5 = 0.8731, nDCG@10 = 0.9043  

---

## 1. Executive Summary

Phase 8.1 established FastRAG's caching architecture, delivering sub-millisecond responses (<0.3 ms) on repeated queries. However, **uncached requests**—representing new queries, updated knowledge, or stochastic generations—must execute the full physical pipeline (lexical search, semantic embedding, RRF fusion, CrossEncoder reranking, context assembly, and LLM generation).

This audit establishes a reproducible latency baseline across all pipeline stages, isolates confirmed bottlenecks on uncached execution paths, and outlines a prioritized optimization plan designed to maximize uncached throughput and reduce p95/cold-start latency without altering retrieval quality or system architecture.

---

## 2. Empirical Baseline Measurements

Measurements were collected on the production evaluation dataset (`datasets/queries.jsonl`, 24 queries) using `scripts/benchmark_phase8_2.py` running in `.venv/` on macOS (Apple Silicon, 4 PyTorch CPU worker threads). Stage latencies were captured across 5 repetitions (120 individual iterations per stage) following model loading.

### 2.1 Cold-Start Latency Breakdown

| Subsystem / Operation | Initial Load / Query Latency | Primary Driver |
| :--- | :---: | :--- |
| **BM25 Inverted Index** | **1.07 ms** (load) / **0.04 ms** (first query) | JSON parsing of inverted tables |
| **Dense Semantic Index** | **1.50 ms** (load) / **5,312.83 ms** (first query) | Lazy loading of `sentence-transformers/all-MiniLM-L6-v2` PyTorch weights on first search |
| **CrossEncoder Reranker** | **4.58 ms** (load) / **781.91 ms** (first query) | Lazy loading of `cross-encoder/ms-marco-MiniLM-L-6-v2` weights on first rerank |
| **Context Assembler** | **0.07 ms** | In-memory token budgeting |
| **Mock LLM Generator** | **0.10 ms** | Local mock response mapping |
| **Total Uncached Cold Start** | **~6,095 ms** | First incoming query suffers complete weight load penalty |

### 2.2 Warm Stage Latency Profile (Uncached Execution)

| Pipeline Stage | Mean | Median (P50) | P95 | Min | Max | Stage Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 Retrieval** | 0.026 ms | 0.026 ms | 0.038 ms | 0.011 ms | 0.055 ms | 38,461.5 q/s |
| **Dense Retrieval (Uncached Enc)** | 23.378 ms | 4.712 ms | 191.994 ms | 3.564 ms | 731.130 ms | 42.8 q/s |
| **Dense Retrieval (Cached Enc)** | 0.024 ms | 0.024 ms | 0.027 ms | 0.019 ms | 0.036 ms | 41,666.7 q/s |
| **Hybrid RRF (Concurrent)** | 4.621 ms | 4.535 ms | 5.611 ms | 3.973 ms | 6.191 ms | 216.4 q/s |
| **Hybrid RRF (Sequential)** | 7.346 ms | 6.413 ms | 13.893 ms | 3.678 ms | 15.421 ms | 136.1 q/s |
| **CrossEncoder Reranker** | 43.814 ms | 35.989 ms | 50.527 ms | 31.916 ms | 238.360 ms | 22.8 q/s |
| **Context Assembly** | 0.023 ms | 0.021 ms | 0.033 ms | 0.019 ms | 0.070 ms | 43,478.3 q/s |
| **LLM Generation (Mock)** | 0.014 ms | 0.013 ms | 0.018 ms | 0.012 ms | 0.051 ms | 71,428.6 q/s |
| **E2E Uncached Search (No Gen)** | 41.565 ms | 41.390 ms | 46.236 ms | 37.240 ms | 50.456 ms | 24.1 q/s |
| **E2E Uncached Complete RAG** | 41.405 ms | 41.290 ms | 46.240 ms | 37.198 ms | 49.970 ms | 24.2 q/s |
| **E2E Warm Cache Hit RAG** | **0.265 ms** | **0.262 ms** | **0.292 ms** | **0.254 ms** | **0.313 ms** | **3,773.6 q/s** |

---

## 3. Confirmed Bottlenecks and Audit Analysis

### 3.1 Bottleneck 1: Uncached First-Request Cold-Start Stall (~6,095 ms)
- **Finding**: Both `SentenceTransformerEncoder` and `SentenceTransformerCrossEncoder` initialize model instances lazily via `_ensure_model()` inside their respective `encode()` and `score_pairs()` call paths.
- **Root Cause**: The FastAPI server boot path and dependency injector (`dependencies.py`) instantiate encoder classes without pre-loading model weights. Consequently, the first real user query absorbs the full weight loading penalty (5.31s for dense encoder + 0.78s for cross-encoder).
- **Evidence**: `dense_first_query_ms = 5312.83 ms`, `reranker_first_query_ms = 781.91 ms`.

### 3.2 Bottleneck 2: CrossEncoder Forward Pass Dominance (87.2% of Uncached Latency)
- **Finding**: In uncached execution, CrossEncoder reranking takes **35.989 ms median** out of **41.290 ms total end-to-end pipeline latency** (87.2% of total runtime).
- **Root Cause**:
  1. The reranker scores 20 (query, candidate) text pairs sequentially per request.
  2. PyTorch execution currently operates under default `torch.no_grad()` rather than `torch.inference_mode()`, generating unnecessary view tracking and version counter bookkeeping.
  3. Redundant Python imports (`import numpy as np`) inside tight loops and scoring methods.

### 3.3 Bottleneck 3: Per-Request HTTP Connection Overhead in `OpenAICompatibleProvider`
- **Finding**: In `backend/app/generation/providers.py`, both `generate()` and `stream()` construct a brand-new `httpx.AsyncClient(timeout=...)` inside an `async with` block on every single request.
- **Root Cause**: Creating a new `httpx.AsyncClient` per request forces:
  - Fresh DNS resolution on every query.
  - New TCP 3-way handshake on every query (~30–80 ms).
  - New TLS 1.3 handshake on every query (~50–120 ms).
  Total per-request network penalty: **80–200+ ms** per LLM call.
- **Solution**: Persistent connection pool client with HTTP keep-alive and graceful teardown (`aclose()`).

### 3.4 Concurrency & Correctness Verifications
- **Concurrent Retrieval**: Confirmed working. `HybridRetriever.search_async()` runs dense and BM25 branches in parallel via `asyncio.gather()` and `asyncio.to_thread()`, achieving **4.535 ms** median vs **6.413 ms** sequential (~29.3% reduction in candidate retrieval latency).
- **Context Assembly**: Negligible latency (0.021 ms). No bottlenecks observed.
- **Cache Keys & Invalidation**: Fully intact. 12 answer-affecting parameters, corpus fingerprint binding, and tenant isolation operate with zero regressions.

---

## 4. Prioritized Optimization Opportunities

| Rank | Optimization | Target Component | Expected Impact | Implementation Risk | Complexity |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1** | **Explicit Pipeline Pre-Warmup Hook** | `dense.py`, `reranker.py`, `pipeline.py`, `dependencies.py` | Eliminates **~6,095 ms** first-request cold stall. Guarantees predictable p99 on server start. | Low (No search logic changes) | Low |
| **2** | **Persistent HTTP Client Pooling** | `generation/providers.py` | Eliminates **80–200 ms** connection overhead per uncached live LLM call. | Low (Standard HTTP connection pool) | Low |
| **3** | **PyTorch `inference_mode` & Import Optimization** | `dense.py`, `reranker.py` | Reduces CPU tensor overhead and allocator churn across both transformer encoders. | Low (Exact mathematical parity) | Low |
| **4** | **Inverted Index Postings for BM25** | `bm25.py` | Replaces full document scan ($O(N)$) with postings list lookup ($O(\text{terms})$). | Low (Zero schema changes) | Low |
| **5** | **Vectorized Top-K Selection** | `dense.py` | Eliminates Python tuple construction and sorting for large vector sets. | Low (Deterministic tie-breaking preserved) | Low |

---

## 5. Expected Trade-offs and Invariants

1. **Retrieval Quality Invariance**:
   - Zero changes to candidate limits, RRF smoothing ($k=60$), CrossEncoder scoring equations, or rank ordering.
   - Ground truth metrics (Recall@5 = 0.7299, Recall@10 = 0.8507, MRR@10 = 1.0000, nDCG@10 = 0.9043) must remain identical.
2. **Memory Footprint**:
   - Pre-warming models loads ~500 MB of PyTorch weights during startup rather than deferring to the first user request. This is standard production best practice.
3. **HTTP Client Lifecycle**:
   - Reusing `httpx.AsyncClient` requires an explicit cleanup hook (`aclose()`) to prevent resource warnings on application exit.

---

## 6. Minimal Implementation Plan

### Step 1: Implement Model Pre-Warmup
- Add `warmup()` to `SentenceTransformerEncoder`, `SentenceTransformerCrossEncoder`, `DenseRetriever`, `CrossEncoderReranker`, and `RAGPipeline`.
- Invoke `warmup()` during `create_pipeline()` or application lifespan when running in production.

### Step 2: Implement Persistent HTTP Client Pooling in `OpenAICompatibleProvider`
- Allow `OpenAICompatibleProvider` to maintain an internal `httpx.AsyncClient` with keep-alive connection pooling.
- Provide `aclose()` for clean async teardown.

### Step 3: Implement PyTorch `inference_mode` in Encoder and CrossEncoder
- Guard `model.encode()` and `model.predict()` with `torch.inference_mode()` when torch is available.
- Clean up redundant `import numpy as np` from inner search loops.

### Step 4: Verification & Regression Testing
- Add regression tests in `backend/tests/` for warmup hooks, HTTP client pooling, and inference mode invariance.
- Verify full test suite (178+ tests passing).
- Run dataset validation and retrieval quality benchmark to prove 100% metric invariance.
- Re-run `scripts/benchmark_phase8_2.py` to record after-optimization latency metrics.
