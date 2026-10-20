# FastRAG Phase 9.2 Design: Retrieval and Generation Quality Evaluation Harness

## 1. Overview and Objectives
Phase 9.2 establishes a reproducible, deterministic evaluation harness for measuring retrieval precision and grounded answer generation in FastRAG.
The harness evaluates FastRAG's end-to-end pipeline against the versioned research benchmark **Dataset v2** (`datasets/v2/`), decoupling retrieval ranking performance from LLM generation quality while enforcing strict provenance-based eligibility rules.

---

## 2. Provenance Eligibility & Anti-Fabrication Principles

To ensure academic and experimental integrity, the harness implements explicit filtering based on `annotation_provenance`:

1. **Eligibility Rule**:
   - Only records with `annotation_provenance.status == "human_reviewed"` are eligible for definitive accuracy, F1, and retrieval ranking benchmarks.
   - Records marked `heuristic_candidate` (18 queries) or `pending_review` (35 queries) are excluded from primary benchmarks to prevent treating unverified annotations as ground truth.
2. **Split Isolation**:
   - All evaluation and tuning during pipeline development are restricted strictly to the **Development Split** (`dev`, 60 queries total; 35 `human_reviewed`).
   - The **Test Split** (`test`, 60 queries total) remains strictly held-out and is never tuned against.
3. **Verified Sample Size Reporting**:
   - In the dev split, exactly 35 queries have `human_reviewed` status:
     - 33 **Answerable Queries** (with positive relevance judgments and verified reference answers)
     - 2 **Unanswerable / Negative Queries** (with 0 relevant documents and reference answer = `null`)

---

## 3. Retrieval Evaluation Methodology

### 3.1 Four Benchmark Configurations
The harness systematically evaluates the four standard retrieval pipelines:
1. **BM25 Lexical Retrieval**: Inverted index posting accumulation with BM25 term frequency saturation.
2. **Dense Bi-Encoder Retrieval**: Exact / ANN vector search using normalized `all-MiniLM-L6-v2` embeddings.
3. **Hybrid RRF**: Concurrent dispatch of BM25 and Dense branches merged via Reciprocal Rank Fusion ($k=60$).
4. **Hybrid RRF + CrossEncoder Reranking**: Two-stage funnel fetching top-20 candidates via Hybrid RRF, followed by cross-attention scoring with `cross-encoder/ms-marco-MiniLM-L-6-v2` to select top-10 passages.

### 3.2 Metric Definitions for Answerable Queries ($R > 0$)
- **Recall@K**: Proportion of all target relevant documents retrieved within the top-$K$ candidates.
- **MRR@K**: Reciprocal rank ($1/\text{rank}$) of the first relevant document in the top-$K$ list.
- **nDCG@K**: Normalized Discounted Cumulative Gain over graded relevance levels ($\{0, 1, 2\}$) with logarithmic position decay.

### 3.3 Separate Treatment of Unanswerable Queries ($R = 0$)
- In classical IR, when a query has zero relevant documents in the corpus, Recall and nDCG are undefined or mathematically zero.
- Mixing $R=0$ queries into standard retrieval recall averages artificially penalizes the retriever for an unanswerable query.
- Therefore, the harness separates queries into **Answerable** and **Unanswerable** subsets:
  - Answerable queries receive standard IR metrics (Recall, MRR, nDCG).
  - Unanswerable queries receive **Retrieval Noise / False Discovery Analysis**: measuring how many unconstrained candidates were retrieved and their confidence score distribution.

---

## 4. Generation Evaluation Methodology

Generation evaluation assesses the quality of synthesized answers produced by `RAGPipeline.execute_async()`:

### 4.1 Answer Correctness (Token-Level F1)
Normalized token overlap between the generated answer $P$ and reference answer $G$:
$$\text{Precision} = \frac{|P \cap G|}{|P|}, \quad \text{Recall} = \frac{|P \cap G|}{|G|}, \quad \text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
Text normalization removes punctuation, articles (*"a", "an", "the"*), and collapses case and whitespace.

### 4.2 Answer Completeness
Measures the coverage of verified evidence spans:
$$\text{Completeness} = \frac{\sum_{s \in \text{Evidence}} \mathbf{1}(\text{coverage}(s, P) \ge 0.5)}{|\text{Evidence}|}$$

### 4.3 Context Groundedness / Faithfulness
Measures the fraction of generated sentences whose claims are attested by the retrieved context passages:
$$\text{Groundedness} = \frac{\text{Grounded Sentences in Answer}}{\text{Total Sentences in Answer}}$$
A sentence is grounded if at least 60% of its content words appear in the assembled context block.

### 4.4 Citation Precision & Recall
- **Citation Extraction**: Resolves explicit `[Doc-N]` tags via `AssembledContext.selected_chunks` mapping to underlying `doc_id`, as well as direct `doc_XXX` references.
- **Citation Recall**: Fraction of ground-truth source documents (`source_doc_ids`) cited in the answer:
  $$\text{Citation Recall} = \frac{|\text{Cited Docs} \cap \text{Source Docs}|}{|\text{Source Docs}|}$$
- **Citation Precision**: Fraction of cited documents that are genuinely relevant (relevance grade $\ge 1$):
  $$\text{Citation Precision} = \frac{|\text{Cited Docs} \cap \text{Relevant Docs}|}{|\text{Cited Docs}|}$$

### 4.5 Correct Abstention on Unanswerable Queries
- For `unanswerable_negative` queries, the model should recognize that the corpus lacks evidence and explicitly abstain.
- The harness checks for standard abstention patterns (*"does not contain sufficient information"*, *"cannot be answered based on the provided context"*, etc.) and the `context_insufficient` flag.
- **Abstention Accuracy**: $1.0$ if the model abstains on an unanswerable query, $0.0$ if it attempts to answer.

---

## 5. Scoring Reproducibility & Uncertainty Quantification

1. **Deterministic Foundations**: All primary metrics (Token F1, Completeness, Groundedness, Citation Precision/Recall, Abstention Accuracy) are computed via exact, reproducible Python algorithms with zero non-deterministic dependencies.
2. **Uncertainty Estimates**: Every aggregate metric is reported with:
   - Sample Mean ($\mu$)
   - Sample Standard Deviation ($\sigma$)
   - 95% Confidence Interval: $\mu \pm 1.96 \cdot \frac{\sigma}{\sqrt{N}}$
3. **Machine-Readable Raw Telemetry**: All per-query predictions, generated answers, retrieved IDs, cited IDs, latencies, and token counts are serialized to JSON in `benchmarks/results/`.
4. **Provider Isolation**:
   - `MockLLMProvider` is evaluated for local, zero-network verification.
   - `OpenAICompatibleProvider` (configured for Gemini `gemini-3.8-flash`) is separated and clearly distinguished. Mock latencies are never reported as real LLM performance.
