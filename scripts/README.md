# FastRAG Operations & Utility Scripts

This directory houses executable operational scripts, command-line runners, ingestion jobs, and evaluation harnesses for FastRAG.

## Script Inventory

| Script | Purpose | Status | Usage Example |
| :--- | :--- | :--- | :--- |
| `generate_dataset.py` | Generate deterministic evaluation corpus, queries, and qrels | **Implemented** | `python3 scripts/generate_dataset.py` |
| `validate_dataset.py` | Verify schema, uniqueness, and SHA-256 hashes of evaluation dataset | **Implemented** | `python3 scripts/validate_dataset.py` |
| `run_eval.py` | Compute Recall@K, MRR@K, nDCG@K on retrievers or fixture rankings | **Implemented** | `python3 scripts/run_eval.py --retriever bm25` |
| `process_documents.py` | Ingest, normalize, chunk, and export documents to JSONL | **Implemented** | `python3 scripts/process_documents.py --input datasets/corpus.jsonl` |
| `build_indexes.py` | Compute vector embeddings and construct dense & BM25 indexes | **Implemented** | `python3 scripts/build_indexes.py --type all` |
| `query_index.py` | Interactive CLI query search against BM25 or Dense indexes | **Implemented** | `python3 scripts/query_index.py "search text" --type bm25` |
| `query_hybrid.py` | Interactive CLI concurrent hybrid search with RRF | **Implemented** | `python3 scripts/query_hybrid.py "search text" --top-k 5` |
| `query_reranked.py` | Interactive CLI for Hybrid/Dense/BM25 retrieval + CrossEncoder reranking + context assembly | **Implemented** | `python3 scripts/query_reranked.py "search text" --mode hybrid --top-k 5` |
| `benchmark_concurrency.py` | Measure throughput (QPS) and latency percentiles | **Implemented** | `python3 scripts/benchmark_concurrency.py --iterations 2` |
| `run_benchmarks.py` | Execute latency and throughput load benchmarks | *Phase 9* | `python3 scripts/run_benchmarks.py` |

## Execution Guidelines

Always activate the project virtual environment before running scripts:

```bash
source .venv/bin/activate
python3 scripts/<script_name>.py [arguments]
```
