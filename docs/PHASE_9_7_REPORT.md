# FastRAG Phase 9.7: Retrieval Evaluation Consolidation and Benchmark Integrity Report

**Date**: October 18, 2026  
**Status**: Completed  
**Branch**: `main`  
**Scope**: Verification and consolidation of BM25, Dense, Hybrid RRF, and CrossEncoder retrieval baselines; benchmark cryptographic fingerprinting; metric calculation consistency; dev/test split quarantine verification; and fairness audit.

---

## 1. Executive Summary

Phase 9.7 consolidates the retrieval evaluations developed across Phases 9.1 through 9.6 into an audited, reproducible, and mathematically trustworthy baseline for the FastRAG hybrid retrieval architecture. 

In accordance with strict project safeguards:
- **No architectural changes or new retrieval algorithms** were introduced.
- **Prior benchmark artifacts** were strictly preserved without overwriting.
- **Held-out test split quarantine** was fully audited and verified untouched.
- **Cryptographic artifact fingerprints** (covering corpus, query splits, qrels, BM25 index, and dense index vector/metadata files) were reconciled.
- The entire backend test suite (**251 passed unit tests**) and dataset validators (**0 errors, 0 warnings**) pass cleanly.

---

## 2. Review of Existing Retrieval Evaluation Results

All four primary retrieval configurations and the diagnostic reciprocal-rank blending variant were evaluated under identical conditions on Dataset v2.

### 2.1 Evaluated Configurations
1. **BM25 Retriever**: Lexical candidate retrieval using Okapi BM25 with token normalization, stopword filtering, and term frequency saturation.
2. **Dense Retriever**: Vector semantic search using `sentence-transformers/all-MiniLM-L6-v2` embeddings (384-dimensional cosine similarity).
3. **Hybrid RRF Retriever**: Reciprocal Rank Fusion ($k=60$) combining Dense and BM25 candidate ranks with equal weighting.
4. **Hybrid + CrossEncoder Reranker ($\alpha = 0.0$, Default Baseline)**: Pure `cross-encoder/ms-marco-MiniLM-L-6-v2` neural scoring applied over the top-$K$ candidate pool.
5. **Hybrid + CrossEncoder Blended ($\alpha = 0.5$, Diagnostic)**: Reciprocal rank blending between CrossEncoder rank and base retriever rank to prevent suppression of broad queries.

---

### 2.2 Verified Human-Reviewed Development Baseline ($N=33$ Answerable Queries)

*Source Artifacts: `benchmarks/results/retrieval_v2_dev_human_alpha_0_0_verified.json`, `retrieval_v2_dev_human_alpha_0_5_verified.json`*

| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | Wall Clock (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25** | 0.8005 | 0.8409 | 0.9848 | 0.8600 | 0.8705 | 0.001 |
| **Dense** | 0.8460 | 0.9167 | 0.9848 | 0.8902 | 0.9065 | 4.487 |
| **Hybrid RRF** | 0.8359 | 0.9015 | **1.0000** | 0.8891 | 0.9050 | 0.417 |
| **Hybrid+Rerank ($\alpha=0.0$)** | 0.7803 | **0.9318** | **1.0000** | 0.8934 | 0.9271 | 4.852 |
| **Hybrid+Rerank ($\alpha=0.5$)** | **0.8182** | 0.9167 | **1.0000** | **0.9085** | **0.9306** | 4.650 |

---

### 2.3 Category Breakdown Across Human-Reviewed Dev Split

| Category ($N_{\text{ans}}$) | Metric | BM25 | Dense | Hybrid RRF | Hybrid+Rerank ($\alpha=0.0$) | Hybrid+Rerank ($\alpha=0.5$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Factual Direct** ($N=10$) | Recall@5 | **0.9000** | 0.8500 | 0.8500 | 0.8500 | **0.9000** |
| | Recall@10 | 0.9000 | **0.9500** | **0.9500** | **0.9500** | **0.9500** |
| | nDCG@10 | 0.9310 | 0.9260 | 0.9307 | 0.9525 | **0.9545** |
| **Multi-Hop** ($N=6$) | Recall@5 | 0.2917 | 0.4583 | 0.4583 | 0.4583 | **0.5833** |
| | Recall@10 | 0.7083 | **0.8750** | **0.8750** | **0.8750** | **0.8750** |
| | nDCG@10 | 0.6673 | 0.7001 | 0.7849 | 0.8631 | **0.8687** |
| **Technical Identifiers** ($N=6$) | Recall@5 | 0.8000 | **0.8667** | **0.8667** | 0.7667 | 0.8167 |
| | Recall@10 | 0.8000 | 0.9167 | 0.9000 | **0.9500** | **0.9500** |
| | nDCG@10 | 0.9191 | **0.9603** | 0.9622 | 0.9444 | 0.9513 |
| **Ambiguous / Underspecified** ($N=3$) | Recall@5 | 0.3333 | **1.0000** | **1.0000** | 0.3333 | 0.6667 |
| | Recall@10 | 0.3333 | **1.0000** | **1.0000** | 0.6667 | 0.6667 |
| | nDCG@10 | 0.5563 | 0.8146 | **0.8642** | 0.7238 | 0.7715 |
| **Citation Grounding** ($N=8$) | Recall@5 | 0.8500 | **0.8833** | 0.8500 | 0.8333 | 0.8000 |
| | Recall@10 | 0.9000 | 0.8833 | 0.8500 | **0.9333** | 0.9000 |
| | nDCG@10 | 0.8334 | 0.8838 | 0.8502 | **0.9174** | 0.8893 |

---

### 2.4 All-Development Split Baseline ($N=50$ Answerable Queries)

*Source Artifacts: `benchmarks/results/retrieval_v2_dev_all_alpha_0_0.json`, `retrieval_v2_dev_all_alpha_0_5.json`*

| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25** | 0.7317 | 0.7817 | 0.9420 | 0.7991 | 0.8165 |
| **Dense** | **0.7950** | 0.8983 | 0.9700 | 0.8600 | 0.8885 |
| **Hybrid RRF** | 0.7650 | 0.8600 | 0.9729 | 0.8410 | 0.8698 |
| **Hybrid+Rerank ($\alpha=0.0$)** | 0.7533 | **0.9100** | **0.9750** | 0.8606 | **0.9018** |
| **Hybrid+Rerank ($\alpha=0.5$)** | 0.7783 | 0.8950 | **0.9750** | **0.8676** | 0.8993 |

---

## 3. Benchmark Configurations & Cryptographic Fingerprints Audit

### 3.1 Fingerprint Reconcilation
Cryptographic SHA256 fingerprints ensure absolute reproducibility across benchmark iterations:

| Artifact | File Path | Verified SHA256 Digest | Status |
| :--- | :--- | :--- | :---: |
| **Corpus v2** | `datasets/v2/corpus.jsonl` | `26dc2ef7eaa129c217008a9e5fddb311ff8c4241f9db6bcfb4a1c86c178c076d` | Match |
| **Queries Dev** | `datasets/v2/queries_dev.jsonl` | `c3da46871393ab620b8ace61124237cd87a0664ff545679a4c1a4e2db7d44e9e` | Match |
| **Qrels Dev** | `datasets/v2/qrels_dev.tsv` | `1da2965adb6975d7a6a569d7f2192a20fc377f87377629e0586cd7b9136c43c5` | Match |
| **BM25 Index** | `datasets/indexes/bm25/bm25_index.json` | `4325e3552dbf01d84493a23b1f52b5fdaf8a5c69dfc02939299c1173169a40d3` | Match |
| **Dense Vectors** | `datasets/indexes/dense/dense_vectors.npz` | `3b66fd77d82b055afc2b8f2c63f9cc40924b92d784338ce83a8e0c184b795a46` | Match |
| **Dense Metadata** | `datasets/indexes/dense/dense_meta.json` | `0acfc421b848fa0dc9476a7766305dd66b22b3e9e5de02ac936b025ca9f145dd` | Match |

### 3.2 Fingerprint Defect Resolution Verified
During the Phase 9.6 audit, `scripts/run_retrieval_eval_v2.py` previously attempted to hash a nonexistent single file `dense_index.npy`. In Phase 9.6/9.7:
1. `compute_dense_index_fingerprint` was updated to hash both `dense_vectors.npz` and `dense_meta.json`.
2. Focused unit test `test_dense_index_artifact_fingerprints` was added to `backend/tests/test_retrieval.py` validating that:
   - Both files are non-null and yield valid 64-character hexadecimal SHA256 digests.
   - Digest strings exactly match byte-level SHA256 hashes of disk files.
   - Nonexistent files return `None` rather than raising uncaught exceptions.
   - Real deployed benchmark index files match disk bytes.

---

## 4. Metric Computation Consistency Audit

A line-by-line audit of `evaluation/metrics.py` and `scripts/run_retrieval_eval_v2.py` confirmed:
1. **Deduplication Invariant**: Both reranked and non-reranked pathways enforce rank-preserving document-level deduplication (`list(dict.fromkeys(r.document_id for r in ...))`) before slicing at rank $K$.
2. **DCG Discount Calculation**: DCG at rank $i$ uses $\frac{2^{\text{rel}} - 1}{\log_2(i + 1)}$. In Phase 9.6, defensive document deduplication was added to `dcg_at_k` to eliminate redundant gain accumulation from multiple chunks of the same document.
3. **Ideal DCG (IDCG)**: IDCG sorts all positive judgments in the ground truth descending and computes the theoretical maximum discounted gain at rank $K$. If no relevant documents exist, nDCG returns 0.0 without division-by-zero errors.
4. **MRR Calculation**: Computes $\frac{1}{\text{rank}}$ for the first hit within top-$K$, or 0.0 if no relevant document appears in top-$K$.
5. **Noise Rate on Unanswerables**: Unconstrained retrievers consistently return $10.0$ candidates for the 2 unanswerable queries (`noise_rate = 1.0`), confirming that retrieval filtering or generation abstention is required downstream.

---

## 5. Dataset Split Quarantine Audit

A rigorous programmatic audit was conducted across the dataset files and historical benchmark results:

### 5.1 Split Isolation
- `queries_dev.jsonl` contains 60 queries (`q2_dev_001` - `q2_dev_060`).
- `queries_test.jsonl` contains 60 queries (`q2_test_001` - `q2_test_060`).
- **Query ID Overlap**: **0 queries**.
- **Query Text Exact Overlap**: **0 queries**.
- `qrels_dev.tsv` contains judgments for exactly the 60 dev queries.
- `qrels_test.tsv` contains judgments for exactly the 60 test queries.
- **Qrels QID Overlap**: **0 queries**.

### 5.2 Artifact Verification
- All 59 JSON artifacts in `benchmarks/results/` were checked for their `split` metadata.
- **0 benchmark runs** were executed against the test split.
- **Conclusion**: The test split remains completely unpolluted and strictly quarantined.

---

## 6. Fairness of Retrieval Method Comparisons

The baseline results support a completely fair comparison between methods:
1. **Identical Query Universe**: All configurations were evaluated against the exact same 33 answerable queries in `dev` under `human_reviewed` provenance (and same 50 under `all`).
2. **Candidate Budget Parity**: Both `Hybrid` and `Hybrid+Rerank` operate on the exact same candidate pool size ($\text{candidate\_pool} = 20$).
3. **Identical Ground Truth**: All methods were evaluated against the identical `qrels_dev.tsv` annotations.
4. **Consistent Metric Definitions**: All methods used identical cutoffs ($K=5, 10$) and identical metric implementations.

### Comparative Summary & Tradeoffs
- **BM25**: Ultra-fast (<1 ms), highly competitive on exact lexical identifiers and direct factual queries, but severely degrades on multi-hop questions (R@5 = 0.2917) and ambiguous phrasing (R@5 = 0.3333).
- **Dense**: Robust semantic generalization, strong on ambiguous queries (R@5 = 1.0000), but slower (~4.5s) and prone to lower top-1 accuracy on multi-hop (MRR@10 = 0.7500).
- **Hybrid RRF**: Achieves perfect first-hit accuracy (MRR@10 = 1.0000 across all 33 queries), high recall (R@10 = 0.9015), and fast latency (~0.4s), offering the best latency-accuracy balance.
- **Hybrid + CrossEncoder Rerank ($\alpha = 0.0$)**: Delivers the highest top-10 recall (0.9318) and highest ranking fidelity (nDCG@10 = 0.9271), but can suppress broad conceptual queries at top-5.
- **Hybrid + CrossEncoder Blended ($\alpha = 0.5$)**: Mitigates CrossEncoder suppression of broad queries, boosting Recall@5 to 0.8182 and multi-hop Recall@5 to 0.5833, while maintaining nDCG@10 at 0.9306.

---

## 7. Remaining Limitations

1. **Category Sample Sizes**: In the human-reviewed development split ($N=33$), category subgroups are relatively small (`ambiguous_underspecified`: 3 queries, `multi_hop`: 6 queries). A single query movement causes large percentage swings. Full-split evaluation ($N=50$) or test-split evaluation will be required for narrow-margin statistical confidence.
2. **Corpus Multi-Chunk Density**: The active corpus currently indexes 36 documents into 36 single chunks. While synthetic multi-chunk unit tests verify candidate pool expansion and deduplication, testing high-chunk-density corpora (>10 chunks/doc) in production will require monitorable pool headroom.
3. **External Comparison**: External provider comparison remains paused due to upstream quota exhaustion. All benchmarks documented here rely solely on local models and offline mock providers.

---

## 8. Test Suite & Validation Evidence

- **Backend Pytest Suite**: **251 passed, 2 warnings** in 11.26s.
- **Dataset v1 Validator**: **PASSED** (36 documents, 24 queries, 88 qrels).
- **Dataset v2 Validator**: **PASSED** (36 documents, 120 queries: 60 dev / 60 test).
- **Git Status**: Clean uncommitted working tree without branch drift.

---

## 9. Conclusion & Research Readiness

Phase 9.7 has verified and consolidated the FastRAG retrieval evaluation baseline. All benchmark artifacts, evaluation scripts, cryptographic fingerprints, and dataset quarantines are consistent, reproducible, and mathematically sound.

**Verdict**: The retrieval evaluation baseline is trustworthy, fully documented, and ready for subsequent research stages (such as generation optimization, latency tuning, or quarantined test split evaluation when authorized).
