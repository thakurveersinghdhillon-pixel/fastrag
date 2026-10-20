# FastRAG Dataset v2: Dataset Card

## 1. Summary
**FastRAG Research Evaluation Dataset v2** is a versioned, multi-category benchmark designed for rigorous evaluation of retrieval-augmented generation (RAG) pipelines, dense/lexical search, reciprocal rank fusion, CrossEncoder reranking, and grounded response generation.

- **Dataset Version**: `2.0.0`
- **Corpus**: 36 documents (32 FastRAG technical reference documents, 4 external distractor documents)
- **Total Queries**: 120
  - **Dev Split**: 60 queries
  - **Test Split**: 60 held-out queries
- **Total Qrels**: 411 relevance judgments (scale 0, 1, 2)
  - **Dev Qrels**: 206
  - **Test Qrels**: 205
- **Manifest Location**: `datasets/v2/dataset_manifest.json`

---

## 2. Question Categories & Distribution

Dataset v2 balances 6 core research categories (20 queries each; 10 dev / 10 test):

| Category | Description | Dev | Test | Total |
| :--- | :--- | :---: | :---: | :---: |
| `factual_direct` | Direct single-passage technical questions with exact factual answers | 10 | 10 | 20 |
| `multi_hop` | Multi-document synthesis questions requiring evidence across 2+ passages | 10 | 10 | 20 |
| `technical_identifiers` | Exact parameter names, hyperparameters, math formulas, and code tokens | 10 | 10 | 20 |
| `unanswerable_negative` | Queries on out-of-corpus RAG/ML concepts testing hallucination suppression | 10 | 10 | 20 |
| `ambiguous_underspecified`| Underspecified queries testing disambiguation across multiple technical topics | 10 | 10 | 20 |
| `citation_grounding` | Queries requiring explicit source provenance citation tags (`[doc_XXX]`) | 10 | 10 | 20 |
| **Total** | | **60** | **60** | **120** |

---

## 3. Annotation Provenance and Status

To uphold academic and research integrity, every query includes an `annotation_provenance` descriptor. Unreviewed heuristic or candidate labels are **never** presented as verified ground truth:

| Provenance Status | Count | Description |
| :--- | :---: | :--- |
| `human_reviewed` | 67 | Manually curated, checked against source text, and reviewed by research engineers. |
| `heuristic_candidate` | 18 | Multi-topic pairings generated from topic taxonomy, verified for referential integrity. |
| `pending_review` | 35 | Synthetic-assisted questions awaiting second-pass independent annotator sign-off. |

---

## 4. Relevance Judgment Scale

Relevance judgments follow the standard 3-tier TREC convention:
- **`0` (Not Relevant / Negative / Distractor)**: Passage does not contain relevant information, is a distractor, or query is unanswerable.
- **`1` (Partially Relevant)**: Passage provides related background, architectural context, or partial evidence.
- **`2` (Highly Relevant)**: Passage directly addresses the query or provides required core evidence.

---

## 5. Scope & Limitations

1. **Corpus Size Bound**: FastRAG's current corpus contains 36 documents (~120–180 words each). While sufficient for 120 distinct queries across the 6 categories, expanding beyond 200 questions without artificial overlap would require ingesting additional documentation domains.
2. **Deterministic Evaluation**: The dataset is deterministic, zero-network dependent, and cryptographically verified with SHA-256 hashes in `dataset_manifest.json`.
3. **Negative Queries**: Unanswerable negative questions intentionally contain zero relevant passages in the corpus. This provides a clean benchmark for measuring model abstention and hallucination rates.

---

## 6. Verification Commands

```bash
# Validate Dataset v2 integrity and schema
.venv/bin/python scripts/validate_dataset_v2.py

# Run unit tests covering dataset schemas and error cases
.venv/bin/pytest backend/tests/test_dataset_v2.py -v
```
