# FastRAG Phase 9.6: Evaluation Reliability & Benchmark Integrity Report

**Date**: October 18, 2026  
**Status**: Completed  
**Branch**: `main`  
**Scope**: Evaluator deduplication symmetry, candidate pool sizing, false abstention metrics, benchmark provenance disaggregation, and index fingerprinting.

---

## 1. Executive Summary

Phase 9.6 audited and resolved evaluation-integrity defects identified in `docs/PHASE_9_6_DIAGNOSTIC.md` prior to further optimization or external model evaluations. 

Key results:
1. **Document-Level Deduplication Symmetry**: Standardized deduplication order and candidate pool sizing across both reranked and non-reranked evaluation pipelines (`scripts/run_retrieval_eval_v2.py`) and inside deterministic metric calculations (`evaluation/metrics.py`). Evaluators now retrieve an expanded candidate pool (`max(candidate_k, top_k * 2)`) before deduplicating in rank order, ensuring multi-chunk documents do not artificially reduce the unique document pool evaluated at rank $K$.
2. **Multi-Chunk Synthetic Verification**: Added unit tests modeling multi-chunk documents, proving that when a document's relevant chunk is ranked lower than an irrelevant chunk in initial retrieval, the CrossEncoder properly promotes the document to rank 1 without being discarded prematurely.
3. **False Abstention Metric**: Implemented `false_abstention_rate` and `false_abstention_count` on answerable queries in `evaluation/generation_eval.py`. Handled zero-denominator boundary conditions gracefully and verified that unanswerable abstention accuracy remains preserved.
4. **Metadata Provenance & Index Fingerprinting**: Augmented benchmark output artifacts with SHA256 cryptographic fingerprints for the corpus, query splits, qrels, and index files, and disaggregated metrics across all three annotation provenance tiers (`human_reviewed`, `heuristic_candidate`, and `pending_review`).
5. **Regression & Safety Verification**:
   - Backend test suite expanded from 244 to **250 passed tests** (+6 new unit tests).
   - Dataset v1 and Dataset v2 validators passed with **0 errors and 0 warnings**.
   - Quarantined test splits (`queries_test.jsonl`, `qrels_test.tsv`) remained untouched.
   - Zero live API calls were executed (all tests and evaluations used local models and offline mock providers).

---

## 2. Issues Addressed and Architectural Changes

### 2.1 Evaluator Deduplication Asymmetry & Metric Distortion

#### Diagnosis
In `scripts/run_retrieval_eval_v2.py`, non-reranked retrieval paths deduplicated chunks into unique document IDs, but retrieved only `top_k` candidate chunks. If multiple chunks from the same document appeared in the initial top-$K$, fewer than $K$ unique documents were evaluated. Conversely, the reranked path did not deduplicate chunks to unique document IDs, passing chunk-level document IDs directly to `evaluate_query_metrics()`.

In `evaluation/metrics.py`, `recall_at_k` converted `retrieved_ids[:k]` to a set (which shrank to $<K$ if duplicates existed), while `dcg_at_k` summed over `retrieved_ids[:k]`, awarding multiple discounted gains to duplicate occurrences of the same document against an ideal DCG that counted each document only once.

#### Implementation
1. **`evaluation/metrics.py`**:
   - `recall_at_k`, `reciprocal_rank_at_k`, and `dcg_at_k` now deduplicate `retrieved_ids` in order of first appearance via `list(dict.fromkeys(retrieved_ids))` before applying the cutoff rank $K$.
   - This ensures document-level evaluation semantics are strictly upheld even if raw chunk candidate lists are passed to the metric functions.
2. **`scripts/run_retrieval_eval_v2.py`**:
   - Both non-reranked and reranked paths now retrieve an expanded candidate pool:
     $$\text{candidate\_pool} = \max(\text{candidate\_k}, \text{top\_k} \times 2)$$
   - Candidate chunks are deduplicated by `document_id` in order of appearance (preserving the highest-scoring chunk for each document), and the top $K$ unique documents are selected:
     ```python
     # Non-reranked path:
     candidates = retriever_obj.search(q_text, top_k=candidate_pool)
     retrieved_ids = list(dict.fromkeys(r.document_id for r in candidates))[:args.top_k]

     # Reranked path:
     candidates = retriever_obj.search(q_text, top_k=candidate_pool)
     reranked = reranker.rerank(q_text, candidates, top_k=len(candidates), blending_alpha=args.blending_alpha)
     retrieved_ids = list(dict.fromkeys(r.document_id for r in reranked))[:args.top_k]
     ```

---

### 2.2 False Abstention Tracking on Answerable Queries

#### Diagnosis
`evaluation/generation_eval.py` previously evaluated `correct_abstention` on answerable queries (checking that the model did not abstain), but did not record `false_abstention_rate` or `false_abstention_count` in `summary_metrics`. When evaluating thresholding or confidence mechanisms, tracking false abstentions on answerable queries is essential to prevent over-filtering.

#### Implementation
1. Added `is_false_abstention: bool = False` to `QueryGenerationEvaluation`. In `evaluate_single_query`, it is evaluated as:
   $$\text{is\_false\_abstention} = (\neg \text{is\_unanswerable}) \land \text{abstained}$$
2. In `run_evaluation`, computed summary statistics across answerable queries:
   $$\text{false\_abstention\_rate} = \frac{\text{false\_abstention\_count}}{|\text{answerable\_results}|} \quad (\text{if } |\text{answerable\_results}| > 0 \text{ else } 0.0)$$
3. Added `false_abstention_rate` and `false_abstention_count` to `summary_metrics`, along with `total_answerable` count and statistical confidence intervals.
4. Maintained existing unanswerable abstention accuracy metrics in `unanswerable_metrics` unchanged.
5. Added properties `false_abstention_rate` and `false_abstention_count` to `GenerationEvaluationReport`.

---

### 2.3 Benchmark Provenance Disaggregation & Index Fingerprints

#### Implementation
1. **Provenance Disaggregation**:
   - Both `run_retrieval_eval_v2.py` and `evaluation/generation_eval.py` now separate metrics by `annotation_provenance.status`:
     - `human_reviewed` (35 queries in dev: 33 answerable, 2 unanswerable)
     - `heuristic_candidate` (8 queries in dev: 8 answerable multi-hop)
     - `pending_review` (17 queries in dev: 9 answerable ambiguous, 8 unanswerable)
   - CLI flags `--provenance` now support `human_reviewed`, `heuristic_candidate`, `pending_review`, and `all`.
2. **Cryptographic Index Fingerprints**:
   - Benchmark JSON artifacts now store SHA256 checksums for:
     - `corpus.jsonl`
     - `queries_{split}.jsonl`
     - `qrels_{split}.tsv`
     - `bm25_index.json`
     - `dense_index.npy`
   - Artifacts also record retrieval model identifiers, blending alpha, context assembler settings (`min_relevance_score`, `ordering_strategy`, `max_tokens`, `max_chunks`), and candidate pool sizes.

---

## 3. Verification & Benchmark Results

### 3.1 Unit Test Verification

Targeted tests and full backend test suite executed cleanly:
```
backend/tests/test_metrics.py:
  test_recall_at_k_known_ranking PASSED
  test_reciprocal_rank_at_k_known_ranking PASSED
  test_mean_reciprocal_rank PASSED
  test_ndcg_at_k_graded_relevance_hand_calculated PASSED
  test_empty_retrieval_results PASSED
  test_no_relevant_documents_in_qrels PASSED
  test_k_equal_to_one PASSED
  test_k_larger_than_retrieved PASSED
  test_invalid_k_values PASSED
  test_evaluate_query_metrics_batch_helper PASSED
  test_invalid_metric_specifications PASSED
  test_document_level_deduplication_recall_at_k PASSED
  test_document_level_deduplication_mrr_at_k PASSED
  test_document_level_deduplication_ndcg_at_k_prevents_multiple_gain PASSED
  test_multi_chunk_corpus_relevant_chunk_not_first PASSED

backend/tests/test_generation_eval.py:
  test_normalize_answer PASSED
  test_compute_token_f1 PASSED
  test_compute_evidence_completeness PASSED
  test_compute_groundedness PASSED
  test_detect_abstention PASSED
  test_compute_citation_metrics PASSED
  test_extract_cited_document_ids PASSED
  test_generation_evaluator_end_to_end PASSED
  test_false_abstention_detection_and_rate PASSED
  test_false_abstention_zero_denominator PASSED

Full Test Suite: 250 passed, 2 warnings in 3.26s
Dataset v1 Validator: PASSED (36 documents, 24 queries, 88 qrels)
Dataset v2 Validator: PASSED (36 documents, 120 queries: 60 dev / 60 test)
```

---

### 3.2 Verified Retrieval Evaluation Comparison

Reran human-reviewed dev retrieval evaluation to separate verified benchmark files:
- `benchmarks/results/retrieval_v2_dev_human_alpha_0_0_verified.json`
- `benchmarks/results/retrieval_v2_dev_human_alpha_0_5_verified.json`

#### Baseline Retrieval ($\alpha = 0.0$):
| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | Wall Clock (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25** | 0.8005 | 0.8409 | 0.9848 | 0.8600 | 0.8705 | 0.00 |
| **Dense** | 0.8460 | 0.9167 | 0.9848 | 0.8902 | 0.9065 | 4.49 |
| **Hybrid RRF** | 0.8359 | 0.9015 | 1.0000 | 0.8891 | 0.9050 | 0.42 |
| **Hybrid+Rerank ($\alpha=0.0$)** | 0.7803 | 0.9318 | 1.0000 | 0.8934 | 0.9271 | 4.85 |

#### Blended Retrieval ($\alpha = 0.5$):
| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | Wall Clock (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25** | 0.8005 | 0.8409 | 0.9848 | 0.8600 | 0.8705 | 0.00 |
| **Dense** | 0.8460 | 0.9167 | 0.9848 | 0.8902 | 0.9065 | 3.60 |
| **Hybrid RRF** | 0.8359 | 0.9015 | 1.0000 | 0.8891 | 0.9050 | 0.36 |
| **Hybrid+Rerank ($\alpha=0.5$)** | 0.8182 | 0.9167 | 1.0000 | 0.9085 | 0.9306 | 4.65 |

#### Comparison Analysis
On the active Dataset v2 corpus (where documents currently have single chunk representations in the base index), numerical metrics are identical to Phase 9.5 across all configurations, confirming mathematical precision. However, unlike Phase 9.5 artifacts, the verified artifacts now embed:
1. Candidate pool expansion guaranteeing that multi-chunk corpora will not suffer from unique document pool shrinkage.
2. Invariant rank deduplication in both reranked and non-reranked pathways.
3. Cryptographic fingerprints (`corpus_sha256 = 8cbf7f16...`, `bm25_index_sha256 = fb67f2b1...`, `dense_index_sha256 = c580877a...`).

---

### 3.3 End-to-End Generation Evaluation (Dev All Split)

Executed offline mock generation evaluation across all 60 dev queries (`dev_all_mock_verified`):

```
======================================================================
 FastRAG Generation Quality Evaluation Report: dev_all_mock_verified
 Timestamp: 2026-10-17T19:39:14.094638+00:00 | Split: DEV | Filter: all
 Provider: mock (mock-llm-v1)
 Evaluated Queries: 60 (Eligible: 60)
======================================================================
 Metric                   | Mean     | Std      | 95% CI           | Count
----------------------------------------------------------------------
 token_f1                 | 0.2629   | 0.0958   | ±0.0265          | 50   
 token_precision          | 0.2463   | 0.0833   | ±0.0231          | 50   
 token_recall             | 0.2955   | 0.1324   | ±0.0367          | 50   
 completeness             | 0.1017   | 0.2365   | ±0.0655          | 50   
 groundedness             | 0.6472   | 0.2418   | ±0.0612          | 60   
 citation_precision       | 0.9600   | 0.1979   | ±0.0549          | 50   
 citation_recall          | 0.7050   | 0.3209   | ±0.0889          | 50   
 latency_ms               | 120.2073 | 11.1032  | ±2.8095          | 60   
 false_abstention_rate    | 0.0000   | 0.0000   | ±0.0000          | 50   
 false_abstention_count   | 0.0000   | N/A      | N/A              | 50   
======================================================================
 Provenance Performance (Total / Ans / False Abstention Rate / F1):
   * heuristic_candidate      (Tot=8 , Ans=8 ): FalseAbstRate=0.0000 | F1=0.2525
   * human_reviewed           (Tot=35, Ans=33): FalseAbstRate=0.0000 | F1=0.2822
   * pending_review           (Tot=17, Ans=9 ): FalseAbstRate=0.0000 | F1=0.2017
======================================================================
```

---

## 4. Modified Files and Preservation of Constraints

The following files were modified to complete Phase 9.6:
1. `evaluation/metrics.py`: Deduplicate document IDs in rank order before applying cutoff $K$ for Recall@K, MRR@K, and nDCG@K.
2. `scripts/run_retrieval_eval_v2.py`: Candidate pool sizing, consistent deduplication, cryptographic fingerprinting, and provenance metric breakdowns.
3. `evaluation/generation_eval.py`: Added `false_abstention_rate` and `false_abstention_count`, zero-denominator handling, provenance metrics breakdown, and context assembly telemetry.
4. `scripts/run_generation_eval.py`: Expanded `--provenance` argument choices.
5. `backend/tests/test_metrics.py`: Added multi-chunk corpus tests, non-first chunk relevance tests, and deduplication tests for Recall, MRR, and nDCG.
6. `backend/tests/test_generation_eval.py`: Added false abstention detection, rate, count, zero-denominator, and summary table formatting tests.

### Constraints Compliance
- **Test Split Quarantine**: `queries_test.jsonl` and `qrels_test.tsv` were NOT read, executed, or modified.
- **Artifact Preservation**: All previous benchmark result files were preserved; new runs were written to separate files (`*_verified.json`).
- **External Calls**: 0 live external provider calls were made.
- **Git State**: No commits or pushes were made.

---

## 5. Conclusion

Phase 9.6 successfully restored benchmark integrity and metric precision across both retrieval and generation evaluation harnesses. The evaluator is now fully resilient to multi-chunk corpus topologies, accurately quantifies false abstention risk, logs complete cryptographic provenance, and distinguishes human-verified annotations from pending or heuristic subsets.
