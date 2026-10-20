# FastRAG Phase 4: Dense Retrieval and BM25 Retrieval — Report

## 1. Executive Summary

Phase 4 establishes independent dense semantic vector retrieval and BM25 lexical retrieval engines for FastRAG. Both retrievers operate on the chunked corpus produced in Phase 3 (`datasets/processed/chunks.jsonl`), share a typed retrieval interface, implement deterministic tie-breaking, persist indexes safely to local disk, and bridge directly into the Phase 2 evaluation harness.

In strict accordance with project phase boundaries:
- **No hybrid fusion (RRF)** has been implemented.
- **No CrossEncoder reranker** has been implemented.
- **No LLM generation or streaming** has been implemented.
- **No query caching or async orchestration** has been implemented.

---

## 2. Retrieval Subsystem Architecture

The retrieval subsystem resides under `backend/app/retrieval/`:

```
backend/app/retrieval/
├── __init__.py           # Package exports (BaseRetriever, BM25Retriever, DenseRetriever, etc.)
├── base.py               # Shared abstractions, schemas (RetrievalResult, IndexInfo), and utilities
├── bm25.py               # BM25Retriever with Okapi Robertson-Spärck Jones scoring & technical tokenizer
├── dense.py              # DenseRetriever with exact cosine similarity & lazy SentenceTransformer loading
└── indexer.py            # Index generation and atomic local persistence pipeline
```

Supporting evaluation and operations components:
```
evaluation/
└── retriever_adapter.py  # Bridges retrievers to Phase 2 EvaluationRunner, measuring cold/warm latencies
scripts/
├── build_indexes.py      # CLI runner to construct and persist BM25 and Dense indexes
├── query_index.py        # CLI interactive search tool for querying persisted indexes
└── run_eval.py           # Updated CLI supporting evaluation of BM25 and Dense retrievers
```

---

## 3. Retriever Implementation Details

### A. Shared Retrieval Interface (`backend/app/retrieval/base.py`)

Both retrievers inherit from `BaseRetriever` and implement the following contract:

- `build_index(chunks: Sequence[Chunk]) -> IndexInfo`
- `search(query: str, top_k: int = 10) -> list[RetrievalResult]`
- `save_index(directory_path: Path | str) -> None`
- `load_index(directory_path: Path | str) -> IndexInfo`
- `get_index_info() -> IndexInfo | None`
- `is_indexed -> bool`

#### Result Normalization & Deterministic Tie-Breaking
Every retrieval hit is returned as a typed `RetrievalResult`:
- `chunk_id: str`
- `document_id: str`
- `rank: int` (1-indexed)
- `score: float`
- `score_type: str` (`"bm25"` or `"cosine_similarity"`)
- `text: str`
- `source: str`
- `metadata: dict`

When candidate scores tie, ordering is broken deterministically by sorting on `chunk_id` ascending. Raw scores are never mixed between engines; score types are explicitly tagged.

#### Corpus Fingerprinting
Both indexes compute a deterministic SHA-256 fingerprint across all input chunks:
$$\text{fingerprint} = \text{SHA256}\left(\sum_{i} \text{chunk\_id}_i + \text{text}_i\right)$$
This ensures index integrity and alerts callers if chunk content has drifted.

---

### B. Lexical Search: BM25 Retriever (`backend/app/retrieval/bm25.py`)

- **Algorithm**: Standard Okapi BM25 formulation with Robertson-Spärck Jones IDF:
  $$\text{IDF}(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)$$
  $$\text{score}(D, Q) = \sum_{q_i \in Q} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
- **Hyperparameters**: $k_1 = 1.5, b = 0.75$.
- **Tokenizer**: Custom regular expression `_TOKEN_PATTERN` preserving technical dotted identifiers (e.g. `asyncio.gather`), hyphenated words (`bi-encoder`), underscores, and versions (`v1.2.3`).
- **Persistence**: Atomic write to `datasets/indexes/bm25/bm25_index.json` containing vocabulary, document frequencies, term frequencies, document lengths, and chunk payloads.

---

### C. Dense Semantic Search: Dense Retriever (`backend/app/retrieval/dense.py`)

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` producing normalized 384-dimensional dense vectors.
- **Similarity Metric**: Exact cosine dot product computed as matrix-vector multiplication in NumPy:
  $$\text{scores} = V \cdot \mathbf{q}$$
  where $V \in \mathbb{R}^{N \times D}$ and $\mathbf{q} \in \mathbb{R}^D$ are $L_2$-normalized float32 vectors.
- **Lazy Initialization & Offline Fallback**: The neural model is initialized lazily on the first encoding call. It checks `local_files_only=True` first to load from the local cache without network timeouts or remote Hub requests.
- **Testing Interface**: `MockEmbeddingEncoder` produces reproducible 64-dim float32 vectors derived from text hashes, allowing fast, offline unit testing without downloading neural weights.
- **Persistence**: Embeddings stored in compressed NumPy format (`datasets/indexes/dense/dense_vectors.npz`) and metadata stored in JSON (`datasets/indexes/dense/dense_meta.json`).

---

## 4. Empirical Evaluation Results

Both retrieval engines were evaluated against the 24 evaluation queries and 88 ground-truth relevance judgments in `datasets/` using `scripts/run_eval.py`.

### A. Metric Comparison

| Metric | BM25 Baseline | Dense Vector Baseline | Absolute Difference |
| :--- | :---: | :---: | :---: |
| **Recall@5** | 0.6465 | **0.7528** | +0.1063 |
| **Recall@10** | 0.7229 | **0.8326** | +0.1097 |
| **MRR@10** | 0.9375 | **1.0000** | +0.0625 |
| **nDCG@5** | 0.7808 | **0.8866** | +0.1058 |
| **nDCG@10** | 0.8152 | **0.9030** | +0.0878 |

*Artifact references: `benchmarks/results/bm25_baseline_*.json` and `benchmarks/results/dense_baseline_*.json`.*

### B. Latency Profiling (Cold Start vs. Warm Query Execution)

Latencies were measured per query using high-resolution monotonic timers (`time.perf_counter()`):

| Metric | BM25 Lexical Retriever | Dense Vector Retriever | Notes |
| :--- | :---: | :---: | :--- |
| **Cold-Start Latency** | 0.038 ms | 2,877.89 ms | Dense cold start includes PyTorch/Transformer model weight loading into memory. |
| **Warm Mean Latency** | 0.021 ms | 99.10 ms | Average query latency across 23 warm queries. |
| **Warm P50 (Median)** | **0.021 ms** | **10.54 ms** | Median online query latency. |
| **Warm P95** | 0.030 ms | 192.35 ms | Tail latency under CPU encoding. |
| **Warm P99** | 0.032 ms | 745.36 ms | Max tail latency spike during CPU inference. |
| **Total Query Batch (24 queries)** | 0.0006 s | 5.1573 s | Total wall-clock time to evaluate all 24 dataset queries. |

---

## 5. Key Engineering Insights

1. **Semantic Coverage vs. Lexical Precision**:
   - Dense retrieval outperforms BM25 across all recall and ranking metrics on the evaluation dataset (Recall@10: 0.8326 vs 0.7229; MRR@10: 1.0000 vs 0.9375). Dense captures synonyms, architectural paraphrasing, and semantic queries that lack exact keyword matches.
   - However, BM25 executes in **0.02 ms** (over 500x faster than warm dense query embedding generation on CPU), making it an ideal ultra-low latency filter.

2. **Cold-Start Isolation**:
   - Dense retrieval exhibits a substantial cold-start penalty (~2.88 seconds) during initial PyTorch model instantiation. In production, this initialization must occur during FastAPI lifespan startup (`lifespan`), completely shielding client requests from cold-start latency.

3. **Incommensurability of Raw Scores**:
   - BM25 scores are unbounded positive numbers ($[0, \infty)$) that scale with document length and term frequency.
   - Cosine similarity scores fall strictly within $[-1.0, 1.0]$.
   - Naive score addition or linear combination without rank normalization is mathematically unsound. This validates the design decision to defer candidate merging to Phase 5's **Reciprocal Rank Fusion (RRF)**.

4. **Persistence Safety**:
   - Dense indexes use `np.savez_compressed` for vector matrices and JSON for metadata; BM25 uses pure JSON.
   - Atomic replacement (`os.replace` from temporary staging files) guarantees that corrupted index files cannot be written if a crash occurs mid-save.

---

## 6. Verification and Test Suite Status

The test suite was expanded with 15 retrieval-specific unit and integration tests in `backend/tests/test_retrieval.py`:
- `test_validate_top_k`: Type and positivity validation for top-$k$.
- `test_corpus_fingerprint`: Determinism of chunk SHA-256 corpus hashing.
- `test_bm25_tokenizer`: Verification of technical identifiers and dotted notation preservation.
- `test_bm25_search_without_index` & `test_dense_search_without_index`: Safe handling of unindexed searches.
- `test_bm25_build_and_search`: Exact keyword match and metadata propagation.
- `test_bm25_empty_query` & `test_dense_empty_query`: Safe handling of blank queries.
- `test_bm25_tie_breaking` & `test_dense_tie_breaking`: Deterministic sorting by `chunk_id` ascending on tied scores.
- `test_bm25_persistence_roundtrip` & `test_dense_persistence_roundtrip`: Save/load parity verifying identical rankings and scores.
- `test_dense_mock_build_and_search`: Verification of deterministic `MockEmbeddingEncoder`.
- `test_dense_cosine_exact_match`: Unit test with orthogonal and unit vectors verifying mathematical dot-product correctness.
- `test_evaluate_retriever_adapter`: End-to-end integration with the evaluation runner and latency statistics.

### Test Execution Summary
- Total Tests: **84 passing** (0 failures, 0 skipped, 0 errors).
- Execution Time: **0.24 seconds**.
- Dataset Validation: **PASSED** (0 errors, 0 warnings, all 36 documents, 24 queries, 88 qrels verified).

---

## 7. Roadmap to Phase 5

Phase 5 will implement **Concurrent Hybrid Retrieval and Reciprocal Rank Fusion (RRF)**:
1. **Asynchronous Dispatch**: Dispatch Dense and BM25 queries concurrently via `asyncio.gather`, bounding total latency to $\approx \max(\Delta t_{\text{dense}}, \Delta t_{\text{bm25}})$.
2. **RRF Implementation**: Merge ranked candidate lists using rank position formula:
   $$\text{RRF}(d) = \sum_{r \in \mathcal{R}} \frac{1}{k_{\text{rrf}} + \text{rank}_r(d)}$$
3. **Candidate Deduplication**: Map chunk hits back to parent documents with rank weighting to eliminate duplicate document passages before downstream generation.
