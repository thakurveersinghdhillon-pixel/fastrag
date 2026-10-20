# FastRAG Evaluation Dataset

This directory contains the self-contained baseline evaluation dataset for FastRAG, covering technical concepts in RAG architecture, retrieval algorithms, indexing structures, latency optimization, and distractor topics.

---

## 1. Directory Structure

```
datasets/
├── corpus.jsonl          # Document collection (36 technical documents)
├── queries.jsonl         # Evaluation query set (24 diverse evaluation queries)
├── qrels.tsv             # Ground-truth relevance judgments (88 labeled query-doc pairs)
├── dataset_manifest.json # Programmatic manifest with counts and SHA-256 checksums
└── README.md             # Dataset documentation and schema specifications
```

---

## 2. File Schemas & Field Definitions

### A. Document Corpus (`corpus.jsonl`)
Stored as one JSON record per line.

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `doc_id` | `str` | Stable, unique document identifier | `"doc_001"` |
| `title` | `str` | Document title | `"Recursive Character Text Chunking for RAG"` |
| `text` | `str` | Document passage text | `"Recursive character chunking splits documents using..."` |
| `source` | `str` | Provenance / path origin | `"fastrag-docs/ingestion/chunking"` |
| `metadata` | `dict` | Structured taxonomy attributes | `{"topic": "chunking", "category": "ingestion", "difficulty": "intermediate"}` |

### B. Evaluation Queries (`queries.jsonl`)
Stored as one JSON record per line.

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `query_id` | `str` | Stable, unique query identifier | `"q_001"` |
| `query` | `str` | Query string / question text | `"recursive character text chunking separator hierarchy"` |
| `category` | `str` | Query taxonomy / evaluation intent | `"exact_keyword"`, `"semantic_paraphrase"`, `"technical_distinction"`, `"acronym_lookup"`, `"pipeline_optimization"`, `"ambiguous_distinction"`, `"distractor_evaluation"` |
| `expected_behavior` | `str` | Expected retrieval behavior description | `"Should retrieve doc_001 explaining separator hierarchies..."` |

### C. Ground-Truth Relevance Judgments (`qrels.tsv`)
Standard TREC-formatted tab-separated values without header:
```
<query_id>\t0\t<doc_id>\t<relevance>
```

#### Relevance Scale Semantics:
- `0` (**Not Relevant / Explicit Negative**): The document covers overlapping keywords (e.g. B-Trees, CPU hardware caches, e-commerce catalog search, standalone summarization) but does NOT provide the information sought by the query.
- `1` (**Partially Relevant**): The document provides related background, adjacent principles, or secondary context.
- `2` (**Highly Relevant**): The document directly answers the query or provides the core requested technical explanation.

---

## 3. Dataset Regeneration

The dataset is generated deterministically by `scripts/generate_dataset.py`. Counts, file sizes, and SHA-256 hashes are automatically updated in `dataset_manifest.json`:

```bash
# From fastrag/ root directory:
python3 scripts/generate_dataset.py
```

---

## 4. Dataset Validation

Run the dataset validation suite to verify schema compliance, referential integrity, and checksum agreement:

```bash
# Command-line validation script:
python3 scripts/validate_dataset.py

# Or via pytest:
pytest backend/tests/test_dataset.py -v
```

The validator checks:
1. File existence for all required components.
2. JSON syntax validity on every line.
3. Required fields, types, and non-empty strings.
4. Identifier uniqueness (`doc_id`, `query_id`).
5. Referential integrity (`qrels.tsv` queries and documents must exist in the corpus and query set).
6. Relevance scale compliance (`0`, `1`, `2`).
7. Programmatic count agreement with manifest.
8. SHA-256 cryptographic checksum matching.

---

## 5. Dataset Limitations

1. **Synthetic Scale**: The initial collection contains 36 documents and 24 queries. It is designed for fast, reliable unit and regression testing during development phases rather than large-scale web search evaluations.
2. **Domain Focus**: The corpus focuses on RAG system architecture, vector indexing, lexical retrieval, and latency optimization.
3. **No Dynamic Annotations**: Ground-truth relevance is fixed upfront independently of any retrieval algorithm to prevent evaluation bias.
