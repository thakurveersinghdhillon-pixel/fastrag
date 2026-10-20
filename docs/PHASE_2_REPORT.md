# FastRAG Phase 2: Dataset Preparation & Baseline Evaluation — Report

## 1. Executive Summary

Phase 2 establishes the baseline evaluation infrastructure for FastRAG as a completely self-contained, reproducible test and validation framework. No retrieval models, vector databases, or LLMs were implemented prematurely. Instead, this phase guarantees that once dense, lexical, and hybrid retrieval engines are implemented in subsequent phases, their retrieval precision, recall, and rank ordering can be evaluated with mathematical rigor.

---

## 2. Dataset Design & Taxonomy

The dataset was generated deterministically via `scripts/generate_dataset.py` and is organized in `datasets/`:

### A. Document Corpus (`corpus.jsonl`)
- **Total Documents**: 36
- **Corpus Taxonomy**:
  1. *Chunking & Text Segmentation* (`doc_001` - `doc_004`): Recursive character splitting, semantic chunking, token count boundaries, sliding windows.
  2. *Dense Vector Retrieval & Embeddings* (`doc_005` - `doc_008`): Bi-encoders, vector normalization, HNSW graphs, IVF-PQ.
  3. *Lexical Search & BM25* (`doc_009` - `doc_012`): BM25 formulation, inverted indexes, lexical-dense complementarity, parameter tuning ($k_1, b$).
  4. *Hybrid Concurrency & Reciprocal Rank Fusion* (`doc_013` - `doc_016`): AsyncIO gather dispatch, RRF formulas, smoothing parameter $k$, candidate deduplication.
  5. *CrossEncoder Reranking* (`doc_017` - `doc_020`): Full cross-attention complexity $O(N \cdot L^2)$, two-stage funneling, batching, lightweight models.
  6. *Context Assembly & Citations* (`doc_021` - `doc_024`): Token budgeting, lost-in-the-middle mitigation, grounded citation tags `[Doc-1]`, context deduplication.
  7. *Caching & Latency Optimization* (`doc_025` - `doc_028`): L1/L2 multi-tier caching, cache invalidation, Time to First Token (TTFT), non-blocking I/O.
  8. *Evaluation & Metrics* (`doc_029` - `doc_032`): MRR@K, nDCG@K, Recall@K, RAG Triad.
  9. *Distractor & Negative Documents* (`doc_033` - `doc_036`): Relational B-Tree indexes, Elasticsearch e-commerce catalog search, standalone LLM summarization without retrieval, hardware CPU cache lines (L1/L2/L3).

### B. Evaluation Queries (`queries.jsonl`)
- **Total Queries**: 24
- **Query Categories**:
  - `exact_keyword` (`q_001` - `q_004`): Queries targeting specific terminology and parameter names (e.g. `HNSW efConstruction and M hyperparameters`).
  - `semantic_paraphrase` (`q_005` - `q_008`): Conceptual natural language questions testing semantic bi-encoder capture without exact keyword matches.
  - `technical_distinction` (`q_009` - `q_012`): Questions contrasting similar architectures (e.g. bi-encoder vs cross-encoder; MRR vs nDCG).
  - `acronym_lookup` (`q_013` - `q_016`): Acronyms and technical identifiers (RRF, TTFT, IVF-PQ, BPE).
  - `pipeline_optimization` (`q_017` - `q_020`): High-level operational queries (two-stage funnel sizing, cache invalidation).
  - `ambiguous_distinction` (`q_021` - `q_022`): Queries testing discrimination between hardware/relational concepts and RAG concepts.
  - `distractor_evaluation` (`q_023` - `q_024`): Negative verification queries where general search documents are relevant, but RAG documents must not be falsely retrieved.

### C. Ground-Truth Relevance Judgments (`qrels.tsv`)
- **Total Judgments**: 88 labeled query-document pairs in standard TREC format (`query_id \t 0 \t doc_id \t relevance`).
- **Relevance Scale**:
  - `0`: Not relevant (explicitly verified distractor / negative).
  - `1`: Partially relevant (mentions related principles or contextual background).
  - `2`: Highly relevant (directly answers the query).

---

## 3. Dataset Validation Architecture

The dataset validator (`evaluation/dataset_validation.py` and CLI `scripts/validate_dataset.py`) enforces 10 strict consistency guarantees:

1. **File Presence**: Verifies `corpus.jsonl`, `queries.jsonl`, `qrels.tsv`, and `dataset_manifest.json`.
2. **JSON Syntax**: Parses each line of JSONL files individually.
3. **Schema Compliance**: Enforces required types (`doc_id: str`, `title: str`, `text: str`, `source: str`, `query_id: str`, `query: str`, etc.) and non-empty strings.
4. **Identifier Uniqueness**: Rejects duplicate document IDs or query IDs.
5. **Referential Integrity**: Guarantees that every `query_id` and `doc_id` in `qrels.tsv` exists in `queries.jsonl` and `corpus.jsonl`.
6. **Relevance Scale Validation**: Rejects any judgment not in `{0, 1, 2}`.
7. **Malformed Record Prevention**: Does not silently drop bad rows; returns precise line numbers and reasons.
8. **Count Verification**: Compares manifest counts against physical file counts.
9. **Cryptographic Checksum Verification**: Re-computes SHA-256 for all dataset files and asserts exact match against manifest hashes.
10. **Zero-Warning Clean Execution**: Passed validation with 0 errors and 0 warnings.

---

## 4. Information Retrieval Metric Implementations

Implemented in `evaluation/metrics.py`:

### A. Recall@K
- Evaluates binary relevance ($\text{relevance} > 0$).
- Measures the proportion of ground-truth relevant documents present within the top-$K$ candidate list.
- Properly handles edge cases: returns `0.0` when no relevant documents exist in ground truth, when the retrieved list is empty, or when $K$ boundaries are exceeded.

### B. Reciprocal Rank / MRR@K
- Identifies the 1-based rank position of the first relevant document ($\text{relevance} > 0$).
- Score is $1 / \text{rank}$ if $\text{rank} \le K$, otherwise $0.0$.
- Mean Reciprocal Rank averages reciprocal ranks across all evaluation queries.

### C. Normalized Discounted Cumulative Gain (nDCG@K)
- Uses graded relevance levels: $0$, $1$, and $2$.
- Exponential gain formulation: $\text{gain}(\text{rel}) = 2^{\text{rel}} - 1$.
- Logarithmic discount: $\log_2(\text{rank} + 1)$.
- Normalizes actual DCG by Ideal DCG (IDCG), yielding scores strictly bounded in $[0.0, 1.0]$. Returns $0.0$ if $\text{IDCG} = 0.0$.

---

## 5. Evaluation Runner & Experiment Tracking

Implemented in `evaluation/runner.py` and `evaluation/experiment.py`:

- **Batch Metric Evaluation**: Accepts arbitrary candidate ranking dictionaries `{query_id: [doc_id, ...]}` and evaluates configured cutoffs (e.g. Recall@5, Recall@10, MRR@10, nDCG@5, nDCG@10).
- **Statistical Aggregation**: Computes mean, min, max, and standard deviation across all queries.
- **Reproducibility Lock**: `ExperimentMetadata` inspects `datasets/dataset_manifest.json` and records the manifest file's SHA-256 checksum in every output report. This ensures that any experiment can be traced back to the exact dataset revision.
- **Machine-Readable Output**: Saves structured JSON reports to `benchmarks/results/` while `.gitignore` prevents uncommitted benchmark outputs from polluting git status.

---

## 6. Synthetic Ranking Fixtures

Implemented in `evaluation/fixtures.py`:
- `get_synthetic_perfect_rankings()`: Ranks all relevant documents at the top (achieves 1.0 on all metrics).
- `get_synthetic_partial_rankings()`: Simulates realistic rankings with mixed relevance.
- `get_synthetic_inverse_rankings()`: Adversarial ranking putting irrelevant documents first.
- `get_synthetic_empty_rankings()`: Empty list returning 0.0.

> [!IMPORTANT]
> These rankings are purely unit-test fixtures. They do not represent benchmark measurements of FastRAG retrieval algorithms, which will be built in Phases 4–6.

---

## 7. Automated Test Execution Results

Ran `pytest -v` across Phase 1 and Phase 2 test suites:

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.9, pytest-8.4.2, pluggy-1.6.0 -- fastrag/.venv/bin/python3.11
cachedir: .pytest_cache
rootdir: /Users/thakurveersingh/Documents/fastrag
configfile: pytest.ini
testpaths: backend/tests
plugins: asyncio-0.26.0, anyio-4.15.1
collecting ... collected 39 items

backend/tests/test_config.py::test_default_configuration PASSED          [  2%]
backend/tests/test_config.py::test_environment_variable_overrides PASSED [  5%]
backend/tests/test_config.py::test_case_insensitive_and_whitespace_normalization PASSED [  7%]
backend/tests/test_config.py::test_invalid_environment_validation PASSED [ 10%]
backend/tests/test_config.py::test_invalid_log_level_validation PASSED   [ 12%]
backend/tests/test_config.py::test_invalid_port_validation PASSED        [ 15%]
backend/tests/test_config.py::test_get_settings_caching PASSED           [ 17%]
backend/tests/test_config.py::test_logger_setup PASSED                   [ 20%]
backend/tests/test_config.py::test_measure_latency_context_manager PASSED [ 23%]
backend/tests/test_dataset.py::test_production_dataset_is_valid PASSED   [ 25%]
backend/tests/test_dataset.py::test_missing_required_file PASSED         [ 28%]
backend/tests/test_dataset.py::test_invalid_jsonl_parsing PASSED         [ 30%]
backend/tests/test_dataset.py::test_missing_required_fields_in_corpus PASSED [ 33%]
backend/tests/test_dataset.py::test_duplicate_document_ids PASSED        [ 35%]
backend/tests/test_dataset.py::test_duplicate_query_ids PASSED           [ 38%]
backend/tests/test_dataset.py::test_invalid_qrels_references PASSED      [ 41%]
backend/tests/test_dataset.py::test_invalid_relevance_labels PASSED      [ 43%]
backend/tests/test_dataset.py::test_incorrect_manifest_counts PASSED     [ 46%]
backend/tests/test_dataset.py::test_incorrect_manifest_checksum PASSED   [ 48%]
backend/tests/test_health.py::test_application_import PASSED             [ 51%]
backend/tests/test_health.py::test_health_endpoint_default PASSED        [ 53%]
backend/tests/test_health.py::test_health_endpoint_custom_settings PASSED [ 56%]
backend/tests/test_health.py::test_health_endpoint_not_found_on_invalid_route PASSED [ 58%]
backend/tests/test_metrics.py::test_recall_at_k_known_ranking PASSED     [ 61%]
backend/tests/test_metrics.py::test_reciprocal_rank_at_k_known_ranking PASSED [ 64%]
backend/tests/test_metrics.py::test_mean_reciprocal_rank PASSED          [ 66%]
backend/tests/test_metrics.py::test_ndcg_at_k_graded_relevance_hand_calculated PASSED [ 69%]
backend/tests/test_metrics.py::test_empty_retrieval_results PASSED       [ 71%]
backend/tests/test_metrics.py::test_no_relevant_documents_in_qrels PASSED [ 74%]
backend/tests/test_metrics.py::test_k_equal_to_one PASSED                [ 76%]
backend/tests/test_metrics.py::test_k_larger_than_retrieved PASSED       [ 79%]
backend/tests/test_metrics.py::test_invalid_k_values PASSED              [ 82%]
backend/tests/test_metrics.py::test_evaluate_query_metrics_batch_helper PASSED [ 84%]
backend/tests/test_metrics.py::test_invalid_metric_specifications PASSED [ 87%]
backend/tests/test_runner.py::test_load_qrels_format PASSED              [ 89%]
backend/tests/test_runner.py::test_experiment_metadata_creation_and_checksum PASSED [ 92%]
backend/tests/test_runner.py::test_runner_perfect_fixture_yields_perfect_scores PASSED [ 94%]
backend/tests/test_runner.py::test_runner_empty_rankings_yields_zero_scores PASSED [ 97%]
backend/tests/test_runner.py::test_runner_summary_table_formatting PASSED [100%]

======================== 39 passed, 1 warning in 0.20s =========================
```

---

## 8. Known Limitations

1. **Corpus Scale**: 36 documents and 24 queries provide an initial evaluation harness for unit and integration testing. In Phase 9, this harness will be supplemented with larger benchmark corpora (e.g. BEIR subsets).
2. **Absence of Retrieval Implementation**: Because Phase 2 strictly builds evaluation infrastructure, the metric scores currently generated in tests reflect deterministic synthetic fixtures. Actual algorithmic retrieval performance will be recorded in Phase 9.

---

## 9. Next Steps for Phase 3

Development is halted pending explicit user approval to proceed to **Phase 3: Document Ingestion, Chunking, and Indexing**:
- Implement `backend/app/ingestion/parser.py` (text normalization and format handlers).
- Implement `backend/app/ingestion/chunker.py` (token-aware recursive sliding window splitters).
- Implement `backend/app/ingestion/pipeline.py` (batch ingestion coordinator).
- Implement CLI ingestion script `scripts/ingest_corpus.py`.
