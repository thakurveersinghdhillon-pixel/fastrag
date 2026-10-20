# FastRAG Phase 9.6 — Evaluation Reliability & Benchmark Integrity Diagnostic Report
**Date**: October 18, 2026
**Author**: FastRAG Engineering Team
**Status**: Stage A & Stage B Complete — Audit & Diagnostic Findings (Awaiting User Approval for Stage C Fixes)

---

## 1. Executive Summary

Phase 9.6 conducts a comprehensive forensic audit of FastRAG's retrieval and generation evaluation harness, metric definitions, dataset annotations, provenance boundaries, and benchmark artifacts. 

The audit verified that the primary retrieval quality metrics (Recall@k, MRR@k, nDCG@k) and generation evaluation metrics are mathematically sound for the current 1-chunk-per-document testbed, and confirmed that repeated runs produce 100% bit-for-bit deterministic results.

However, the audit discovered three structural defects and severe sample size constraints that must be documented and addressed:
1. **Document Deduplication Asymmetry**: Non-reranked configurations (`bm25`, `dense`, `hybrid`) apply `dict.fromkeys(r.document_id)` to deduplicate chunks to documents, while reranked configurations (`hybrid_rerank`) take chunk-level `document_id`s directly without deduplication. On multi-chunk corpora, this causes double-counting in DCG and slot-loss in Recall@k.
2. **Provenance Conflation in All-Development Split**: The all-dev split ($N=60$) mixes 35 `human_reviewed`, 8 `heuristic_candidate`, and 17 `pending_review` queries. Critically, 80% of `multi_hop` queries are automated `heuristic_candidate`s and 90% of `ambiguous_underspecified` queries are unverified `pending_review` queries.
3. **Severe Small-Sample Volatility in Human-Reviewed Split**: Under `provenance == "human_reviewed"`, the dev set contains only **2 multi-hop queries** and **1 broad ambiguous query**. A single document movement causes a 50% or 100% category swing.
4. **Metadata Blind Spots in Benchmark Artifacts**: Output JSON benchmark artifacts omit vital reproducibility parameters, including `candidate_k`, `top_k`, index fingerprints, model names, and context filtering parameters.

---

## 2. Verified Metric Definitions & Mathematical Audit

### 2.1 Retrieval Metrics (`evaluation/metrics.py`)

#### Recall@K:
$$\text{Recall@K} = \frac{|\text{TopK}(\text{retrieved\_docs}) \cap \text{TargetRelevant}|}{|\text{TargetRelevant}|}$$
- **Verification**: If $|\text{TargetRelevant}| = 0$, the metric returns $0.0$. In `run_retrieval_eval_v2.py`, unanswerable queries are partitioned out of `metric_accum`, ensuring that answerable query recall denominators are strictly non-zero positive integers.
- **Defect Risk**: If `retrieved_ids` contains duplicate document IDs (due to multiple chunks from the same document), `set(retrieved_ids[:k])` has fewer than $K$ distinct documents.

#### MRR@K (Reciprocal Rank):
$$\text{RR@K} = \begin{cases} \frac{1}{\text{rank}_i} & \text{if first relevant doc is at rank } i \le K \\ 0.0 & \text{otherwise} \end{cases}$$
- **Verification**: Rank is 1-indexed. If no relevant item appears within cutoff $K$, returns $0.0$. Correctly averaged across queries.

#### nDCG@K (Normalized Discounted Cumulative Gain):
$$\text{DCG@K} = \sum_{i=1}^{\min(K, |\text{retrieved}|)} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$
$$\text{IDCG@K} = \sum_{i=1}^{\min(K, |\text{sorted\_rels}|)} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{nDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
- **Verification**: IDCG correctly sorts ground-truth relevance judgments descending and discounts exponentially. Clamped to $[0.0, 1.0]$.
- **Defect Risk**: If `retrieved_ids` contains duplicate document IDs, DCG counts gain for the same document multiple times, while IDCG only counts it once.

### 2.2 Generation & Groundedness Metrics (`evaluation/generation_eval.py`)

#### Citation Precision & Recall:
- **Citation Recall**: Fraction of ground-truth `source_doc_ids` cited in generated text. Denominator is $|\text{source\_doc\_ids}|$.
- **Citation Precision**: Fraction of cited document IDs that are in `relevant_doc_ids`. Denominator is $|\text{cited\_doc\_ids}|$.
- **Verification**: Averaged only over `answerable_results`. If no citations are emitted for an answerable query, precision is $0.0$.

#### Abstention Accuracy:
- **Formula**: $\frac{\text{abstained\_unanswerables}}{|\text{unanswerable\_queries}|}$.
- **Verification**: Tested with `min_relevance_score = -5.0`; correctly identifies 2/2 unanswerables in the human-reviewed dev split.
- **Defect Identified**: The evaluator does **not** track or report `false_abstention_rate` (abstentions on answerable queries) in `summary_metrics`. A completely non-responsive system that abstains on all queries would show 100% abstention accuracy while failing all answerable queries.

---

## 3. Implementation Inconsistencies & Defect Analysis

### Defect 1: Chunk-Level vs Document-Level Deduplication Asymmetry
In `scripts/run_retrieval_eval_v2.py` (lines 183–191):
```python
if use_reranker:
    candidates = retriever_obj.search(q_text, top_k=args.candidate_k)
    reranked = reranker.rerank(
        q_text, candidates, top_k=args.top_k, blending_alpha=args.blending_alpha
    )
    retrieved_ids = [r.document_id for r in reranked]
else:
    candidates = retriever_obj.search(q_text, top_k=args.top_k)
    # Deduplicate chunks to doc_ids
    retrieved_ids = list(dict.fromkeys(r.document_id for r in candidates))
```

#### Discrepancies:
1. `hybrid_rerank` does **not** deduplicate by `document_id`. If multiple chunks from `doc_A` appear in the top-K, `retrieved_ids` contains duplicate entries.
2. In the non-reranked path, `search` is called with `top_k=args.top_k` (10). If the top 10 chunks contain 3 chunks from `doc_A`, `dict.fromkeys` collapses them to 8 unique documents. The non-reranked evaluator therefore only evaluates 8 documents instead of 10!
3. **Current Testbed Reality**: In `datasets/indexes/`, there are 36 documents and 36 chunks (1:1 mapping). Thus, in the current testbed, this defect remained dormant. However, as soon as multi-chunk document splitting is evaluated, benchmark comparisons will be mathematically invalid.

### Defect 2: Incomplete Telemetry & Metadata Capture in Benchmark JSON
- `run_retrieval_eval_v2.py` output JSON records `blending_alpha`, `split`, and `provenance_filter`, but **omits**:
  - `candidate_k` (initial pool size).
  - `top_k` (evaluation cutoff).
  - Index paths and corpus fingerprints (`corpus_fingerprint`).
  - Model names for dense encoder and CrossEncoder reranker.
- `run_generation_eval.py` output JSON records `retrieval_configuration`, but **omits**:
  - `blending_alpha`.
  - `min_relevance_score`.
  - `ordering_strategy`.
  - `context_max_tokens` and `context_max_chunks`.

---

## 4. Dataset Provenance & Category Coverage Analysis

### 4.1 Development Split Breakdown ($N=60$ Queries)

| Category | Human Reviewed | Heuristic Candidate | Pending Review | Total Dev Queries |
| :--- | :---: | :---: | :---: | :---: |
| `factual_direct` | 10 | 0 | 0 | 10 |
| `technical_identifiers` | 10 | 0 | 0 | 10 |
| `citation_grounding` | 10 | 0 | 0 | 10 |
| `multi_hop` | **2** (`q2_dev_011`, `q2_dev_012`) | **8** | 0 | 10 |
| `ambiguous_underspecified` | **1** (`q2_dev_041`) | 0 | **9** | 10 |
| `unanswerable_negative` | 2 | 0 | 8 | 10 |
| **Total** | **35** (33 answerable) | **8** | **17** | **60** (50 answerable) |

### 4.2 Test Split Breakdown ($N=60$ Queries — Quarantined)

| Category | Human Reviewed | Heuristic Candidate | Pending Review | Total Test Queries |
| :--- | :---: | :---: | :---: | :---: |
| `factual_direct` | 10 | 0 | 0 | 10 |
| `technical_identifiers` | 10 | 0 | 0 | 10 |
| `citation_grounding` | 10 | 0 | 0 | 10 |
| `multi_hop` | **0** | **10** | 0 | 10 |
| `ambiguous_underspecified` | **0** | 0 | **10** | 10 |
| `unanswerable_negative` | 2 | 0 | 8 | 10 |
| **Total** | **32** (30 answerable) | **10** | **18** | **60** (50 answerable) |

### 4.3 Provenance Conflation Risks
1. In the `human_reviewed` dev slice, `multi_hop` ($N=2$) and `ambiguous_underspecified` ($N=1$) represent only **9%** of answerable queries.
2. In the `all` dev slice ($N=50$ answerable), `multi_hop` and `ambiguous` expand to 20 queries (**40%** of answerable queries), but **17 of those 20 queries (85%) have never been reviewed by a human annotator**.
3. In the held-out `test` split, **100% of multi-hop queries are heuristic candidates and 100% of ambiguous queries are pending review**.
4. **Conclusion**: Tuning algorithms on `--provenance all` risks optimizing against automated heuristics and unreviewed annotations. All benchmark claims must report `human_reviewed` as the primary benchmark and clearly label `heuristic_candidate` / `pending_review` slices as exploratory.

---

## 5. Granular Per-Query Audit: Alpha 0.0 vs Alpha 0.5

Across the 33 answerable queries in `human_reviewed`:
- **4 queries improved** on Recall@5:
  1. `q2_dev_005` (Factual direct): 0.5000 $\rightarrow$ 1.0000 (+0.5000)
  2. `q2_dev_012` (Multi-hop RRF): 0.2500 $\rightarrow$ 0.5000 (+0.2500)
  3. `q2_dev_027` (Technical ID): 0.5000 $\rightarrow$ 1.0000 (+0.5000)
  4. `q2_dev_041` (Broad ambiguous): 0.3333 $\rightarrow$ 0.6667 (+0.3333)
- **1 query degraded** on Recall@5:
  1. `q2_dev_056` (Citation grounding): 1.0000 $\rightarrow$ 0.6667 (-0.3333)
- **28 queries unchanged** on Recall@5.

### Root-Cause Analysis of the Single Degradation (`q2_dev_056`):
- Query: *"What is the computational complexity of CrossEncoder reranking and why does it differ from bi-encoders? Cite the source documents."*
- Relevant documents: `doc_017` (rel=2), `doc_005` (rel=2), `doc_018` (rel=1).
- In Base Hybrid retrieval:
  - `doc_017` was rank 1.
  - `doc_018` was rank 3.
  - `doc_005` had poor hybrid rank: **rank 16**.
  - Distractor `doc_011` (irrelevant) had strong hybrid rank: **rank 5**.
- In CrossEncoder reranking ($\alpha = 0.0$):
  - CrossEncoder scored `doc_005` at $-0.963$ (**CE rank 4**), placing all 3 gold documents in the top 5 (Recall@5 = 1.0000).
- In Blended reranking ($\alpha = 0.5$):
  - Distractor `doc_011` (CE rank 6, base rank 5) received blended score:
    $$S_{\text{blended}}(\text{doc\_011}) = \frac{1}{60 + 6} + \frac{0.5}{60 + 5} = 0.01515 + 0.00769 = 0.02284$$
  - Gold `doc_005` (CE rank 4, base rank 16) received blended score:
    $$S_{\text{blended}}(\text{doc\_005}) = \frac{1}{60 + 4} + \frac{0.5}{60 + 16} = 0.01562 + 0.00658 = 0.02220$$
  - Distractor `doc_011` edged out `doc_005` into rank 5 by a margin of $+0.00064$, pushing `doc_005` to rank 6 and dropping Recall@5 to 0.6667.
- **Architectural Takeaway**: When a relevant passage has very poor candidate generation rank ($\ge 16$) but strong CrossEncoder affinity, high $\alpha$ blending can disadvantage it against strong lexical distractors. Blending $\alpha = 0.5$ represents the Pareto frontier between broad-query recovery and single-passage retention.

---

## 6. Baseline Reproducibility Results

We verified reproducibility by comparing consecutive evaluations with identical configurations:
- `retrieval_v2_dev_human_alpha_0_5.json` per-query scores reproduced with 0.000000% variance.
- Deterministic sorting by `(-round(score, 8), chunk_id)` completely eliminates non-deterministic tie-breaking order shifts.
- Offline `MockLLMProvider` generation evaluation produced identical 1.0000 citation precision and 100% unanswerable abstention accuracy across runs.

---

## 7. Statistical Limitations & Sample Size Caveats

1. **Extreme Sensitivity of Small Slices**:
   In `human_reviewed`, category size $N=2$ means a single missed document (e.g. `q2_dev_012`) drops category Recall@5 by 25–50%. A claim like *"Multi-hop Recall@5 is 0.4583 vs 0.5833"* is based on **two questions**.
2. **Confidence Intervals**:
   - For $N=33$ answerable queries, the 95% confidence interval half-width on Recall@5 is $\pm 0.045$.
   - For $N=2$ multi-hop queries, the 95% confidence interval is essentially uninformative ($\pm 0.400$).
3. **Quarantine Protection**:
   The held-out `test` split must not be accessed or evaluated during this phase.

---

## 8. Proposed Corrective Actions (For Stage C)

We recommend four targeted, low-risk corrections to guarantee benchmark integrity before Phase 9.7:

1. **Fix Document-Level Deduplication in `run_retrieval_eval_v2.py`**:
   - Ensure `retrieved_ids` is deduplicated to unique document IDs consistently for **both** non-reranked and reranked paths:
     `unique_doc_ids = list(dict.fromkeys(r.document_id for r in results))[:top_k]`
   - Retrieve sufficient candidate depth ($k \ge \text{top\_k} \times 2$) in non-reranked baselines before deduplicating, so deduplication never starves the cutoff.
2. **Add False-Abstention Rate to Generation Evaluation**:
   - In `evaluation/generation_eval.py`, track and report `false_abstention_rate` (the percentage of answerable queries where the model abstained) alongside unanswerable abstention accuracy.
3. **Capture Complete Reproducibility Metadata**:
   - Include `blending_alpha`, `candidate_k`, `top_k`, `min_relevance_score`, `ordering_strategy`, `corpus_fingerprint`, and model names in benchmark report headers.
4. **Explicit Provenance Breakdown in Benchmarks**:
   - Update `run_retrieval_eval_v2.py` to output metrics disaggregated by provenance (`human_reviewed` vs `heuristic_candidate` vs `pending_review`) in addition to category.
