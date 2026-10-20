# FastRAG Phase 3: Document Ingestion, Chunking, and Indexing Preparation — Report

## 1. Executive Summary

Phase 3 establishes the document preprocessing, text normalization, boundary-aware chunking, integrity validation, and atomic export pipeline for FastRAG. The ingestion pipeline provides a deterministic and verifiable bridge between raw source documents and future vector/lexical indexing engines.

In strict adherence to project scope boundaries, **no embedding models, vector stores, BM25 indices, CrossEncoders, or LLM generation components** were introduced in this phase.

---

## 2. Ingestion Subsystem Architecture & Modules

The ingestion subsystem resides under `backend/app/ingestion/`:

```
backend/app/ingestion/
├── __init__.py           # Package exports and public API
├── models.py             # Strongly typed Document, Chunk, and IngestionSummary data models
├── loaders.py            # Local JSONL (streaming/batch), plain-text (.txt), and directory loaders
├── normalization.py      # Conservative, non-destructive Unicode and whitespace normalizer
├── chunking.py           # Configurable, boundary-aware deterministic chunker
├── validation.py         # Document schema, chunk offset alignment, and batch validation
└── pipeline.py           # Ingestion orchestrator with atomic JSONL file export
```

---

## 3. Input & Output Schemas

### Input Document Schema (`Document`)
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `document_id` | `str` | Yes | Stable, unique, non-empty identifier (supports `doc_id` alias) |
| `text` | `str` | Yes | Document body text |
| `title` | `str` | No | Document title or heading (defaults to filename if omitted) |
| `source` | `str` | No | Provenance URI or file path |
| `metadata` | `dict` | No | Arbitrary JSON-serializable structured metadata |
| `raw_text` | `str \| None` | No | Original unmodified text preserved for auditability |

### Output Chunk Schema (`Chunk`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `chunk_id` | `str` | Deterministic content-addressed identifier |
| `document_id` | `str` | Foreign key referencing parent `Document` |
| `chunk_index` | `int` | Zero-indexed sequential position within document (0, 1, 2, ...) |
| `text` | `str` | Extracted chunk body text (without boundary whitespace) |
| `source` | `str` | Inherited from parent document |
| `start_offset` | `int` | Zero-indexed character start position in normalized document text |
| `end_offset` | `int` | Character end position in normalized document text |
| `metadata` | `dict` | Inherited document metadata |

---

## 4. Chunking Strategy & Defaults

### Hyperparameter Defaults
- `chunk_size`: 1,000 characters (strictly positive).
- `chunk_overlap`: 150 characters (non-negative and strictly smaller than `chunk_size`).
- `preserve_paragraphs`: `True` (boundary-seeking enabled).

### Breakpoint Selection Hierarchy
To prevent splitting words or sentences across chunk boundaries, the chunker evaluates delimiters in the second half of the chunk window (`[start + chunk_size//2, target_end]`) in descending priority order:
1. Double newlines: `\n\n` (paragraph boundaries)
2. Single newlines: `\n` (line/list item boundaries)
3. Sentence terminations: `". "`, `"? "`, `"! "`
4. Word boundaries: `" "` (whitespace)
5. Hard character boundary fallback if no natural delimiter occurs.

### Invariant Verification
The pipeline strictly enforces the offset alignment invariant for every generated chunk:
$$\text{normalized\_doc\_text}[\text{start\_offset} : \text{end\_offset}] == \text{chunk.text}$$

---

## 5. Deterministic Chunk Identifier Design

Chunk identifiers follow a structured, content-addressed convention:
$$\text{chunk\_id} = \{\text{document\_id}\}\text{\#c}\{\text{chunk\_index}:04d\}\_\{\text{sha256}[:12]\}$$
where $\text{sha256}$ fingerprints `f"{document_id}:{chunk_index}:{chunk_size}:{chunk_overlap}:{chunk_text}"`.

**Guarantees**:
1. **Human Traceability**: Document ID and sequential index are immediately legible.
2. **Deterministic Reproducibility**: Identical inputs and hyperparameters yield identical chunk IDs across runs.
3. **Hyperparameter Sensitivity**: Altering chunk size, overlap, or text content automatically updates the hash component, preventing stale chunk references in downstream indexes.

---

## 6. Execution Summary: Ingestion of Evaluation Corpus

Ran `scripts/process_documents.py --input datasets/corpus.jsonl --output datasets/processed/chunks.jsonl`:

```text
============================================================
 FastRAG Document Ingestion Summary
 Timestamp (UTC): 2026-10-14T18:46:34.137957+00:00
============================================================
 Input Documents:        36
 Successfully Processed: 36
 Rejected Documents:     0
 Empty Documents:        0
 Total Chunks Output:    36
 Avg Chunks / Document:  1.00
 Ingestion Duration:     0.0013s
------------------------------------------------------------
 Chunking Config:        size=1000, overlap=150, preserve_paragraphs=True
 Output Location:        datasets/processed/chunks.jsonl
============================================================
```

Running with fine-grained chunk sizes (`--chunk-size 250 --chunk-overlap 50`) yields 81 chunks across the 36 documents (2.25 chunks/doc) in 0.0019s, confirming robust multi-chunk sliding window behavior.

---

## 7. Automated Test Suite Results

Ran `pytest -v` across all Phase 1, Phase 2, and Phase 3 tests:

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.9, pytest-8.4.2, pluggy-1.6.0 -- fastrag/.venv/bin/python3.11
cachedir: .pytest_cache
rootdir: /Users/thakurveersingh/Documents/fastrag
configfile: pytest.ini
testpaths: backend/tests
plugins: asyncio-0.26.0, anyio-4.15.1
collecting ... collected 69 items

backend/tests/test_config.py (9 tests) .................... PASSED [ 13%]
backend/tests/test_dataset.py (10 tests) .................. PASSED [ 27%]
backend/tests/test_health.py (4 tests) .................... PASSED [ 33%]
backend/tests/test_ingestion.py (30 tests) ................ PASSED [ 76%]
backend/tests/test_metrics.py (11 tests) .................. PASSED [ 92%]
backend/tests/test_runner.py (5 tests) .................... PASSED [100%]

======================== 69 passed, 1 warning in 0.19s =========================
```

All 69 unit tests passed with 100% success rate. The original Phase 2 dataset integrity check (`scripts/validate_dataset.py`) was verified post-run and confirmed zero changes to `datasets/corpus.jsonl`.

---

## 8. Known Limitations

1. **Character-Based Chunking**: The current chunker operates on character counts with natural delimiter snapping. Token-based chunking (e.g., using TikToken or HuggingFace tokenizers) will be introduced in Phase 4 when embedding models and tokenizers are officially selected.
2. **File Formats**: Phase 3 supports JSONL and `.txt`. Binary formats such as PDF, DOCX, and HTML parsing will be supported as future extensions using lightweight isolated parsers.

---

## 9. Next Steps for Phase 4

Phase 3 is complete. Development is halted pending explicit user approval to proceed to **Phase 4: Dense Retrieval and BM25 Retrieval**:
- Select and integrate an efficient embedding model for batched vector generation.
- Build local/in-memory dense vector index (e.g. HNSW).
- Implement BM25 lexical inverted index with term frequency saturation ($k_1$) and length normalization ($b$).
- Expose retriever query interfaces with stage-by-stage latency logging.
