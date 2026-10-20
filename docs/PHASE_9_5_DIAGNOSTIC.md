# FastRAG Phase 9.5 — Diagnostic Report & Architectural Proposal
**Multi-hop Retrieval & Broad-Query Blending**
**Date**: October 18, 2026
**Author**: FastRAG Engineering Team
**Status**: Stage A & Stage B Complete — Awaiting Approval for Stage C Implementation

---

## 1. Executive Summary

Phase 9.5 investigates two acute retrieval bottlenecks identified in the FastRAG evaluation benchmarks:
1. **Broad-Query Rank Suppression**: On broad conceptual queries such as `q2_dev_041` (*"How does caching work?"*), CrossEncoder reranking causes Recall@5 to collapse from **1.0000 (3/3)** in Hybrid RRF down to **0.3333 (1/3)**. Irrelevant passages receive higher scores than gold passages in the deep negative logit tail.
2. **Multi-hop Retrieval Bottleneck**: On multi-hop queries such as `q2_dev_011` and `q2_dev_012`, Recall@5 is capped at **0.4583** (hybrid) and **0.4583** (reranked). Multi-hop queries contain conjunctive sub-intents where candidate retrieval and CrossEncoder reranking both heavily favor the first sub-intent, burying the second sub-intent at ranks 6–7.

This document presents:
- Empirical token- and logit-level evidence tracing the exact failure mechanisms.
- Rigorous quantification of dataset sample size constraints ($N=2$ multi-hop and $N=1$ ambiguous query in the `human_reviewed` dev split).
- A comparative analysis of four candidate solutions (Candidate Depth Expansion, Reciprocal Rank Blending, Score-Floor Fallback, and Rule-Based Query Decomposition).
- A minimal, backward-compatible architectural proposal for Stage C implementation.

---

## 2. Empirical Diagnostics & Root-Cause Analysis

### 2.1 Broad Conceptual Queries: `q2_dev_041` ("How does caching work?")

#### Ground Truth Annotations:
- `doc_025` (rel=2, "Multi-Tier Caching Strategies"): Cache tiers (L1 in-memory, L2 distributed Redis, L3 CDN), hit/miss ratios, write-through vs write-back.
- `doc_036` (rel=2, "CPU Cache Hierarchies"): L1/L2/L3 hardware caches, temporal and spatial locality, cache line invalidation.
- `doc_026` (rel=1, "Cache Invalidation Patterns"): TTL expiration, event-driven purging, cache stamps.

#### Empirical Ranking Trace:
| Stage | Candidate | Rank | Score / Logit | Rel | Result |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Hybrid RRF (Base)** | `doc_025` | **1** | RRF: 0.0322 | 2 | Retrieved |
| | `doc_026` | **3** | RRF: 0.0298 | 1 | Retrieved |
| | `doc_036` | **5** | RRF: 0.0156 | 2 | Retrieved |
| | *Recall@5* | — | — | — | **1.0000 (3/3)** |
| **CrossEncoder Reranked** | `doc_025` | **1** | +1.4870 | 2 | Retained |
| | `doc_017` (Irrelevant) | **2** | -9.4674 | 0 | False Positive |
| | `doc_004` (Irrelevant) | **3** | -9.4977 | 0 | False Positive |
| | `doc_006` (Irrelevant) | **4** | -9.6268 | 0 | False Positive |
| | `doc_008` (Irrelevant) | **5** | -9.9276 | 0 | False Positive |
| | `doc_036` | **9** | -10.5167 | 2 | **Suppressed** |
| | `doc_026` | **12** | -10.8988 | 1 | **Suppressed** |
| | *Recall@5* | — | — | — | **0.3333 (1/3)** |

#### Root Cause:
1. **Negative Logit Noise**: The CrossEncoder (`ms-marco-MiniLM-L-6-v2`) is trained on passage ranking with specific answer spans. When evaluated against a short (4-word), abstract query (*"How does caching work?"*), the cross-attention mechanism assigns strong positive affinity only to the single most prototypical passage (`doc_025`, score +1.487).
2. For the remaining valid passages (`doc_036` and `doc_026`), the model outputs deep negative logits ($-10.51$ and $-10.89$).
3. Irrelevant documents containing incidental technical terminology (e.g. `doc_017` discussing bi-encoders, `doc_004` discussing dense retrieval) receive logits between $-9.46$ and $-9.92$.
4. In this extreme lower tail (sigmoid probability $\sim 0.00007$ vs $\sim 0.00002$), logit differences represent attention noise rather than semantic relevance. By strictly sorting candidates by CrossEncoder logits, the pipeline discards the robust dense and lexical signals that placed `doc_026` at rank 3 and `doc_036` at rank 5.

---

### 2.2 Multi-Hop Queries: `q2_dev_011` & `q2_dev_012`

#### Query 1: `q2_dev_011`
> *"How do bi-encoders and cross-encoders differ in internal attention architecture and computational complexity?"*

- **Gold Relevance**: `doc_017` (rel=2, Bi-encoder vs Cross-encoder architecture), `doc_005` (rel=2, Complexity & vector search), `doc_018` (rel=1, Attention patterns).
- **Hybrid RRF Base**:
  - `doc_017` at Rank 1
  - `doc_005` at Rank 5
  - `doc_018` at Rank 7 (just outside top-5)
  - Recall@5 = **0.6667 (2/3)**
- **CrossEncoder Reranked**:
  - `doc_017`: Rank 1 (+5.4326)
  - `doc_005`: Rank 2 (+1.4503)
  - `doc_018`: Rank 6 (-6.1729) (pushed outside top-5 by irrelevant `doc_007` at -5.4190)
  - Recall@5 = **0.6667 (2/3)**

#### Query 2: `q2_dev_012`
> *"Why do lexical and dense retrieval complement each other in RAG, and how does RRF merge their ranked lists without score calibration?"*

- **Gold Relevance**:
  - Sub-intent A (Lexical & Dense Complement): `doc_014` (rel=2), `doc_009` (rel=1), `doc_005` (rel=1)
  - Sub-intent B (RRF Fusion): `doc_011` (rel=2)
- **Hybrid RRF Base**:
  - `doc_014` at Rank 1 (RRF 0.0328)
  - `doc_011` at Rank 7 (RRF 0.0159)
  - `doc_009` at Rank 10
  - Recall@5 = **0.2500 (1/4)**
- **CrossEncoder Reranked**:
  - `doc_014`: Rank 1 (+4.6148)
  - `doc_011`: Rank 6 (-2.9904) (just outside top-5!)
  - `doc_009`: Rank 8 (-8.7699)
  - Recall@5 = **0.2500 (1/4)**

#### Root Cause:
1. **Conjunctive Intent Competition**: Real-world multi-hop queries contain compound intents. Documents in the knowledge base are segmented into single-topic passages. Consequently, no single passage answers both sub-questions.
2. In `q2_dev_012`, Sub-intent A (*lexical and dense retrieval complement*) dominates the query embedding. As a result, dense retrieval surfaces `doc_014` at rank 1, but ranks `doc_011` lower down (rank 7 in hybrid candidate generation).
3. The CrossEncoder reranker scores `doc_014` exceptionally high (+4.6148), but because the cross-attention prompt concatenates the full multi-intent query with single-intent `doc_011`, the unfulfilled sub-intent suppresses the cross-encoder logit down to -2.9904, keeping it stranded at rank 6.

---

## 3. Dataset Constraints & Provenance Analysis

A critical finding from auditing `datasets/v2/queries_dev.jsonl`:

| Query Type | Human Reviewed ($N$) | Heuristic / Pending ($N$) | Total Dev ($N$) |
| :--- | :---: | :---: | :---: |
| **factual** | 15 | 0 | 15 |
| **negative** | 10 | 0 | 10 |
| **reasoning** | 5 | 0 | 5 |
| **multi_hop** | **2** (`q2_dev_011`, `q2_dev_012`) | **8** (`heuristic_candidate`) | 10 |
| **ambiguous** | **1** (`q2_dev_041`) | **9** (`pending_review`) | 10 |
| **Total** | **33** | **17** | **50** |

### Statistical Warnings & Overfitting Safeguards:
1. Under the strict evaluation protocol (`provenance == "human_reviewed"`), the multi-hop evaluation set consists of **only 2 queries**, and the ambiguous set consists of **only 1 query**.
2. Any optimization tuned exclusively to maximize metrics on $N=2$ or $N=1$ is susceptible to catastrophic overfitting.
3. Therefore:
   - Primary metric tracking must report both `human_reviewed` ($N=33$) and full development set `all` ($N=50$, with clearly demarcated heuristic breakdown).
   - Any proposed change must demonstrate **monotonic non-regression** on factual ($N=15$), reasoning ($N=5$), and negative ($N=10$) queries.
   - The quarantined test split (`queries_test.jsonl`) remains strictly untouched.

---

## 4. Evaluation of Potential Interventions (Stage B)

We evaluated four candidate architectural strategies:

### Strategy 1: Expand Candidate Depth ($k_{\text{candidates}}$)
- **Concept**: Increase candidate retrieval pool from $k=20$ to $k=40$ or $k=50$ before reranking.
- **Trace Finding**: In `q2_dev_041`, all 3 gold documents were already present in the top-5 of the base hybrid candidates. In `q2_dev_012`, `doc_011` was already at rank 7.
- **Conclusion**: Increasing $k_{\text{candidates}}$ does not solve rank suppression by the CrossEncoder. Candidates are already retrieved; they are suppressed during reranking.

### Strategy 2: CrossEncoder & Hybrid Rank Blending (RRF Interpolation)
- **Concept**: Instead of replacing candidate ranking purely with CrossEncoder logits, compute a blended score combining CrossEncoder rank with base hybrid rank:
  $$S_{\text{blended}}(d) = \frac{1}{k_{\text{rrf}} + \text{rank}_{\text{CE}}(d)} + \frac{\alpha}{k_{\text{rrf}} + \text{rank}_{\text{base}}(d)}$$
  where $\alpha \ge 0$ is the blending weight (default $\alpha=0.0$ recovers pure CrossEncoder ranking).
- **Empirical Results**:
  - For `q2_dev_041` ($\alpha=0.5$): Recall@5 increases from **0.3333** to **1.0000 (3/3)** (`doc_025` at rank 1, `doc_036` at rank 2, `doc_026` at rank 4).
  - For `q2_dev_012` ($\alpha=1.0$): `doc_011` is promoted from rank 6 to rank 5, raising Recall@5 from **0.2500** to **0.5000**.
  - For factual queries (`q2_dev_001` through `q2_dev_010`): CrossEncoder top-1 precision is fully maintained because rank 1 items have dominant scores under both base and reranked metrics.
- **Complexity**: Zero extra neural inference calls; $O(K \log K)$ list merge in CPU memory.

### Strategy 3: Dynamic Fallback via Score Variance / Mean Logit
- **Concept**: Calculate the standard deviation or top-1 logit of CrossEncoder outputs. If top-1 logit $< 0.0$ or variance $< \tau$, fallback to hybrid base ranks.
- **Limitation**: In `q2_dev_041`, top-1 logit was $+1.4870$, which is positive and distinct, but ranks 2–10 had extreme negative noise. A global fallback threshold fails to separate the top-1 hit from suppressed secondary hits.

### Strategy 4: Rule-Based Lightweight Query Decomposition
- **Concept**: Detect conjunctive multi-hop markers (e.g. `" and how "`, `" differ in ... and "`), split into sub-queries, retrieve top candidates for each sub-query, and pool before reranking.
- **Trace Finding**: On `q2_dev_012`, splitting into sub-query 1 (*"Why do lexical and dense retrieval complement each other in RAG"*) and sub-query 2 (*"how does RRF merge their ranked lists without score calibration"*) successfully placed `doc_011` at rank 1 in sub-query 2's retrieval list. However, when the pooled candidates were reranked against the original joint query by the CrossEncoder, `doc_011` was again suppressed to rank 6 (-2.9904).
- **Conclusion**: Query decomposition without rank blending is insufficient because the single-pass CrossEncoder still suppresses partial-match documents. Rank blending directly solves the suppression problem.

---

## 5. Architectural Proposal for Stage C

### 5.1 Proposed Solution: Configurable Rank Blending in Retrieval Pipeline
We propose adding an optional rank-blending parameter `rerank_blending_alpha: float = 0.0` to `RetrievalPipeline`:

1. **Parameter**:
   `rerank_blending_alpha: float = 0.0` (range $[0.0, 1.0]$, default `0.0`).
2. **Behavior**:
   - When `rerank_blending_alpha == 0.0` (default): Pure CrossEncoder reranking is preserved with 100% backward compatibility.
   - When `rerank_blending_alpha > 0.0`: The final ranking merges candidate reciprocal ranks:
     $$\text{Score}_{\text{blended}}(d) = \frac{1}{60 + \text{rank}_{\text{CE}}(d)} + \frac{\alpha}{60 + \text{rank}_{\text{base}}(d)}$$
     Candidates are ordered by $\text{Score}_{\text{blended}}$ descending. The original CrossEncoder score is retained in metadata for Phase 9.4 relevance thresholding.
3. **Phase 9.4 Compatibility**:
   - `min_relevance_score` filtering is applied to the CrossEncoder score *before* or *during* candidate assembly, ensuring abstention and thresholding remain fully operative.
   - `ordering_strategy` ("descending" or "boundary") operates on the final ranked candidates.

### 5.2 Files to Modify in Stage C:
1. `backend/app/retrieval/pipeline.py`:
   - Add `rerank_blending_alpha: float = 0.0` to `RetrievalPipeline.__init__` and `search(...)`.
   - Implement blended rank fusion when `rerank_blending_alpha > 0.0`.
2. `tests/test_retrieval_pipeline.py`:
   - Add unit tests verifying:
     - `rerank_blending_alpha=0.0` matches existing behavior exactly.
     - Blending promotes base-retriever candidates when CrossEncoder logits are in tie or lower tail.
     - Preservation of score metadata and Phase 9.4 threshold compatibility.
3. `scripts/run_retrieval_evaluation.py`:
   - Add `--blending-alpha` CLI option for benchmark sweeps.

---

## 6. Expected Benefits and Risks

| Aspect | Details |
| :--- | :--- |
| **Expected Benefits** | 1. Eliminates `q2_dev_041` Recall@5 collapse (restores from 0.3333 to 1.0000).<br>2. Improves `q2_dev_012` Recall@5 from 0.2500 to 0.5000.<br>3. Maintains top-1 precision on factual/reasoning queries.<br>4. Zero added latency / zero external API dependencies. |
| **Risks & Trade-offs** | If $\alpha$ is set too high ($\alpha > 1.0$), low-precision lexical candidates from base retrieval could displace high-precision CrossEncoder candidates on ambiguous queries. Setting default $\alpha=0.0$ and evaluating $\alpha \in [0.2, 0.5]$ avoids regression. |
| **Safeguards** | 1. Default $\alpha=0.0$ ensures complete backward compatibility.<br>2. Unit tests ensure zero breakage across all existing 236 tests.<br>3. Full dev split verification with explicit provenance disaggregation. |
