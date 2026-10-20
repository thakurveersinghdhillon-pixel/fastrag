# FastRAG

> **FastRAG** is a high-performance, modular Retrieval-Augmented Generation (RAG) system engineered from scratch for minimal end-to-end latency, high retrieval recall and precision, and verifiable factual grounding.

---

## 1. Project Overview

FastRAG provides an asynchronous, performance-first architecture designed to solve the common latency bottlenecks of traditional RAG pipelines. Rather than executing retrieval and generation steps sequentially, FastRAG pairs lexical and dense retrieval with asynchronous dispatch, fuses candidate rank distributions with Reciprocal Rank Fusion (RRF), selectively reranks top candidates with CrossEncoders, and streams answers with token-level citations.

### Key Architectural Pillars
- **Asynchronous Hybrid Retrieval**: Concurrent execution of dense vector search and BM25 lexical search (`asyncio.gather`).
- **Reciprocal Rank Fusion (RRF)**: Scale-invariant candidate rank fusion.
- **Layered Funneling**: High-recall candidate filtering followed by precision CrossEncoder reranking on a small candidate pool.
- **Token-Budgeted Context Assembly**: Strict token bounds preventing context overflow and reducing generation latency.
- **Microsecond Latency Profiling**: Stage-by-stage latency tracking hooks built into application telemetry.

---

## 2. Implementation Status

| Stage | Scope | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Greenfield Project Foundation** | **Completed** | Modular structure, FastAPI core, configuration, logging, tests |
| **Phase 2** | **Dataset Preparation & Baseline Eval** | **Completed** | Corpus, queries, qrels, manifest, validator, IR metrics, runner |
| **Phase 3** | **Document Ingestion & Chunking** | **Completed** | Loaders, conservative normalization, boundary-aware chunking, pipeline |
| **Phase 4** | **Dense & BM25 Retrieval** | **Completed** | Cosine dot-product, SentenceTransformers, Okapi BM25, query CLI, eval adapter |
| **Phase 5** | **Concurrent Hybrid Retrieval & RRF** | **Completed** | Async branch dispatch (`asyncio.gather`), RRF fusion ($k=60$), strict/degraded modes, CLI |
| **Phase 6** | **CrossEncoder Reranking & Context Assembly** | **Completed** | Cross-attention scoring, token budgeting, citation tagging, 121 tests passing |
| **Phase 7** | **LLM Generation & Streaming** | **Completed** | Grounded answer synthesis, SSE streaming, pluggable providers, 146 tests passing |
| Phase 8 | Caching & Orchestration Optimization | *Pending Approval* | Level 1 query cache, Level 2 embedding vector cache |
| Phase 9 | Performance Benchmarking & Load Testing | *Pending Approval* | P50–P99 latency profiling, throughput saturation analysis |
| Phase 10 | Containerization & Documentation | *Pending Approval* | Docker, Docker Compose, technical report |

---

## 3. Planned Architecture

```
                  ┌────────────────────────┐
                  │      Client Query      │
                  └───────────┬────────────┘
                              │
                    [ Cache Check (L1/L2) ]
                              │ (miss)
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │ Dense Vector     │            │ BM25 Lexical     │
    │ Retrieval (top-k)│            │ Retrieval (top-k)│
    └─────────┬────────┘            └─────────┬────────┘
              └───────────────┬───────────────┘
                              │ (concurrent asyncio.gather)
                              ▼
                 [ Reciprocal Rank Fusion ]
                              │
                 [ Optional CrossEncoder ]
                              │
               [ Token-Budget Context Assembler ]
                              │
               [ LLM Generator (Async Stream) ]
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Streamed Response +    │
                  │ Grounded Citations     │
                  └────────────────────────┘
```

For detailed specifications, sequence flows, and component dependency analyses, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 4. Prerequisites

- **Operating System**: macOS (Apple Silicon or Intel) or Linux.
- **Python**: Python 3.11 or 3.12 (tested and verified with Python 3.11.9).
- **Git**: Installed for version tracking.

---

## 5. macOS Setup Instructions

Follow these steps from the root directory of the project (`fastrag/`):

### Step 1: Clone or Navigate to the Repository

```bash
cd /Users/thakurveersingh/Documents/fastrag
```

### Step 2: Create and Activate Virtual Environment

```bash
# Create a dedicated virtual environment using Python 3.11 or 3.12
python3.11 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### Step 3: Install Phase 1 Dependencies

```bash
# Install core dependencies and test runner
pip install -r backend/requirements.txt
```

### Step 4: Configure Environment Variables

```bash
# Copy example configuration to .env (adjust if port 8000 is occupied)
cp backend/.env.example backend/.env
```

---

## 6. Running the Server

Start the FastRAG ASGI server using Uvicorn:

```bash
# From fastrag/ root directory:
PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

*Note: If port 8000 is already in use by another local service on your machine, launch on an alternative port (e.g., 8008):*
```bash
PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8008 --reload
```

### Verify Service Health

Open a separate terminal or run `curl`:

```bash
curl http://localhost:8000/health
```

Expected output:
```json
{"status":"healthy","app":"FastRAG","environment":"development","version":"0.1.0"}
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

---

## 7. Running the Test Suite

Run the full automated test suite using `pytest`:

```bash
# Run all 96 unit and concurrency tests across configuration, health, validation, metrics, runner, ingestion, retrieval, and hybrid fusion:
pytest -v
```

All 96 unit tests execute in under 0.6s and verify:
- Application factory, lifespan, and `/health` response.
- Default settings, environment overrides, and invalid configuration guards.
- Structured logging initialization and latency measurement context managers.
- Dataset file integrity, JSON schema validation, unique IDs, referential integrity, and SHA-256 manifest matching.
- Mathematical precision and edge-case behavior for Recall@K, MRR@K, and nDCG@K.
- Evaluation runner batch metric calculation and experiment metadata serialization.
- Document loaders (JSONL and plain text), streaming, and directory traversal.
- Conservative text normalization preserving Unicode NFC, technical identifiers, code, and paragraph structure.
- Boundary-aware chunking, character offset alignment, deterministic chunk IDs, and atomic JSONL export.
- Dense cosine similarity, exact vector projection, mock offline embedding encoders, tie-breaking, and NPZ persistence.
- BM25 Okapi term frequency/IDF scoring, technical tokenization, tie-breaking, and atomic JSON persistence.
- Reciprocal Rank Fusion mathematical accuracy, scale-invariance, deduplication, and deterministic tie-breaking.
- Concurrent branch execution overlap, worker thread dispatch, timeout handling, strict/degraded error modes.
- Evaluation adapter latency profiling (cold start vs warm percentiles) and runner bridge.

---

## 8. Dataset Validation & Baseline Evaluation

Phase 2 establishes the reproducible dataset and IR metric evaluation harness:

### Validate Dataset Schema & Cryptographic Hashes
```bash
python3 scripts/validate_dataset.py
```
Checks `corpus.jsonl`, `queries.jsonl`, `qrels.tsv`, and `dataset_manifest.json` for schema adherence, referential consistency, and SHA-256 agreement.

### Run Baseline Evaluation Runner
```bash
# Verify evaluation runner with synthetic fixtures:
python3 scripts/run_eval.py --fixture perfect --no-save
python3 scripts/run_eval.py --fixture partial --no-save

# Evaluate custom pre-computed rankings JSON against ground-truth qrels:
python3 scripts/run_eval.py --rankings path/to/rankings.json --experiment-id my_experiment
```

Results are saved as structured JSON reports in `benchmarks/results/`.

---

## 9. Document Ingestion & Chunking Pipeline

Phase 3 establishes the document preprocessing and chunking pipeline:

### Supported Input Formats
- **JSONL Corpora**: Files containing one JSON record per line with `document_id` (or `doc_id`), `text`, optional `title`, `source`, and `metadata`.
- **Plain Text (`.txt`)**: Single files or directories of UTF-8 encoded text files.

### Conservative Normalization
- Standardizes Unicode to NFC without mangling accents or code symbols.
- Strips null characters (`\x00`).
- Normalizes CRLF/CR to Unix LF (`\n`).
- Preserves paragraph separations (`\n\n`), code indentation, math notation, and URLs without lowercase distortion.

### Boundary-Aware Chunking
- Configurable `chunk_size` (default: 1,000 characters) and `chunk_overlap` (default: 150 characters).
- Seeks natural boundaries: Paragraph $\to$ Line $\to$ Sentence $\to$ Word.
- Exact character offset tracking guaranteeing `doc_text[start:end] == chunk.text`.
- Content-addressed deterministic chunk IDs: `{document_id}#c{chunk_index:04d}_{sha256[:12]}`.

### CLI Execution
```bash
# Ingest the baseline corpus with default parameters:
python3 scripts/process_documents.py --input datasets/corpus.jsonl --output datasets/processed/chunks.jsonl

# Ingest with customized chunking parameters:
python3 scripts/process_documents.py --input datasets/corpus.jsonl --chunk-size 500 --chunk-overlap 80
```

---

## 10. Retrieval Engines (BM25 & Dense Vector)

Phase 4 establishes independent lexical and dense vector retrieval engines, index generation, interactive querying, and evaluation bridging:

### Build Retrieval Indexes
```bash
# Build both BM25 and Dense indexes from processed chunks:
python3 scripts/build_indexes.py --type all

# Or build individual indexes:
python3 scripts/build_indexes.py --type bm25
python3 scripts/build_indexes.py --type dense
```

Indexes are persisted atomically under `datasets/indexes/`:
- `datasets/indexes/bm25/bm25_index.json`: Vocabulary, document frequencies, term frequencies, and chunk payloads.
- `datasets/indexes/dense/dense_vectors.npz` & `dense_meta.json`: Compressed 384-dimensional embeddings and chunk metadata.

### Interactive Search CLI
```bash
# Query the BM25 index:
python3 scripts/query_index.py "What is Reciprocal Rank Fusion RRF?" --type bm25 --top-k 3

# Query the Dense index:
python3 scripts/query_index.py "What is Reciprocal Rank Fusion RRF?" --type dense --top-k 3

# Output machine-readable JSON:
python3 scripts/query_index.py "cache stampede mitigation" --type bm25 --top-k 3 --json
```

### Retrieval Evaluation & Latency Profiling
```bash
# Evaluate BM25 against the 24 evaluation queries and 88 ground-truth qrels:
python3 scripts/run_eval.py --retriever bm25 --experiment-id bm25_baseline

# Evaluate Dense retrieval with cold-start and warm query latency stats:
python3 scripts/run_eval.py --retriever dense --experiment-id dense_baseline
```

---

## 11. Concurrent Hybrid Retrieval & Reciprocal Rank Fusion (RRF)

Phase 5 establishes concurrent hybrid retrieval pairing Dense vector search and BM25 lexical search with Reciprocal Rank Fusion:

### Interactive Hybrid Search CLI
```bash
# Execute concurrent hybrid search:
python3 scripts/query_hybrid.py "What is Reciprocal Rank Fusion RRF?" --top-k 5

# Custom candidate pools and smoothing constant k:
python3 scripts/query_hybrid.py "asynchronous cache stampede" --dense-candidates 30 --bm25-candidates 30 --rrf-k 60 --top-k 5

# JSON output with branch execution diagnostics:
python3 scripts/query_hybrid.py "HNSW indexing" --top-k 3 --json
```

### Hybrid Evaluation & Concurrency Benchmarking
```bash
# Evaluate concurrent Hybrid RRF against the benchmark dataset:
python3 scripts/run_eval.py --retriever hybrid --experiment-id hybrid_concurrent_baseline

# Evaluate sequential Hybrid RRF control baseline:
python3 scripts/run_eval.py --retriever hybrid --sequential --experiment-id hybrid_sequential_baseline

# Run QPS throughput and latency percentile benchmark:
python3 scripts/benchmark_concurrency.py --iterations 2
```

---

## 12. CrossEncoder Reranking & Context Assembly (Phase 6)

### End-to-End Pipeline CLI (`scripts/query_reranked.py`)
```bash
# Execute Hybrid retrieval -> neural CrossEncoder reranking -> token-budgeted context assembly:
python3 scripts/query_reranked.py "What is Reciprocal Rank Fusion RRF?" --mode hybrid --top-k 5

# Custom candidate count, token budget, and chunk limit:
python3 scripts/query_reranked.py "vector indexing" --candidates 30 --top-k 5 --context-max-tokens 600 --context-max-chunks 3

# Fast/offline deterministic test execution (no neural weights downloaded):
python3 scripts/query_reranked.py "BM25 scoring" --mode bm25 --mock-reranker --top-k 3

# Structured JSON output with complete timing breakdown and assembled context:
python3 scripts/query_reranked.py "HNSW indexing" --top-k 3 --json
```

### Full Benchmark Evaluation with CrossEncoder Reranking
```bash
# Benchmark Hybrid RRF + CrossEncoder reranker against 24 queries and 88 qrels:
python3 scripts/run_eval.py --retriever hybrid-rerank --experiment-id hybrid_cross_encoder_phase6
```

---

## 13. LLM Generation & Real-Time Streaming (Phase 7)

Phase 7 establishes end-to-end grounded question answering, real-time SSE streaming, pluggable provider architecture (OpenAI-compatible and mock), and FastAPI REST/SSE endpoints:

### End-to-End Generation CLI (`scripts/query_generation.py`)
```bash
# Interactive non-streaming question answering with mock provider:
python3 scripts/query_generation.py "What is Reciprocal Rank Fusion RRF?" --mock

# Live real-time token streaming to terminal stdout:
python3 scripts/query_generation.py "What is Reciprocal Rank Fusion RRF?" --mock --stream

# Execute with real neural models (Dense + CrossEncoder) and mock generator:
python3 scripts/query_generation.py "What is Reciprocal Rank Fusion RRF?" --provider mock

# Execute with hosted OpenAI-compatible provider (requires API key):
python3 scripts/query_generation.py "What is Reciprocal Rank Fusion RRF?" --provider openai --model gpt-4o-mini --api-key sk-...

# Structured machine-readable JSON output:
python3 scripts/query_generation.py "HNSW indexing" --mock --json
```

### FastAPI Endpoints
```bash
# Start FastAPI service:
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000

# Non-streaming JSON answer:
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Reciprocal Rank Fusion RRF?", "candidate_k": 20, "rerank_top_k": 5}'

# Real-time SSE streaming:
curl -N -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Reciprocal Rank Fusion RRF?", "candidate_k": 20, "rerank_top_k": 5}'
```

---

## 14. Directory Structure

```
fastrag/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI ASGI application, /health & /api/v1 mounting
│   │   ├── api/                     # API route endpoints & dependencies
│   │   │   ├── __init__.py
│   │   │   ├── dependencies.py      # Shared RAGPipeline dependency factory
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       └── routes.py        # POST /api/v1/query & POST /api/v1/query/stream
│   │   ├── core/                    # Settings, logging, and error hierarchy
│   │   │   ├── config.py            # Pydantic Settings & LLM configurations
│   │   │   ├── logging_config.py
│   │   │   └── exceptions.py        # Domain exceptions (Generation, Provider, Stream)
│   │   ├── schemas/                 # Pydantic schemas (QueryRequest, QueryResponse)
│   │   │   ├── __init__.py
│   │   │   └── query.py
│   │   ├── ingestion/               # Document ingestion & chunking (Phase 3)
│   │   │   ├── __init__.py
│   │   │   ├── models.py            # Document, Chunk, and IngestionSummary
│   │   │   ├── loaders.py           # JSONL and plain-text loaders
│   │   │   ├── normalization.py     # Conservative text normalizer
│   │   │   ├── chunking.py          # Boundary-aware deterministic chunker
│   │   │   ├── validation.py        # Document & chunk schema validators
│   │   │   └── pipeline.py          # Atomic pipeline orchestrator
│   │   ├── retrieval/               # Retrieval & fusion subsystem (Phases 4-6)
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # BaseRetriever, RetrievalResult, IndexInfo
│   │   │   ├── bm25.py              # Okapi BM25 retriever & tokenizer
│   │   │   ├── dense.py             # Cosine similarity dense retriever & encoders
│   │   │   ├── fusion.py            # Reciprocal Rank Fusion (RRF) algorithm
│   │   │   ├── hybrid.py            # Concurrent HybridRetriever orchestrator
│   │   │   ├── reranker.py          # CrossEncoder reranker & mock scorers (Phase 6)
│   │   │   ├── context_assembler.py # Context assembly & token budgeting (Phase 6)
│   │   │   ├── pipeline.py          # End-to-end RAGPipeline coordinator (Phases 6-7)
│   │   │   └── indexer.py           # Index building & persistence pipelines
│   │   ├── generation/              # Grounded LLM generation & streaming (Phase 7)
│   │   │   ├── __init__.py
│   │   │   ├── models.py            # UsageStats, StreamChunk, StreamEvent, Request/Response
│   │   │   ├── prompts.py           # Grounded system prompts & injection defense
│   │   │   ├── providers.py         # MockLLMProvider & OpenAICompatibleProvider
│   │   │   └── generator.py         # RAGGenerator coordinator & citation resolver
│   │   └── caching/                 # L1 query & L2 embedding cache (Phase 8)
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_health.py           # Application and /health endpoint test suite
│   │   ├── test_config.py           # Configuration, validation & logging test suite
│   │   ├── test_dataset.py          # Dataset schema and tamper validation tests
│   │   ├── test_metrics.py          # Recall@K, MRR@K, nDCG@K metric unit tests
│   │   ├── test_runner.py           # Evaluation runner & experiment metadata tests
│   │   ├── test_ingestion.py        # Loading, normalization, chunking & pipeline tests
│   │   ├── test_retrieval.py        # BM25, dense, tie-breaking, persistence & adapter tests
│   │   ├── test_hybrid.py           # RRF numerical, concurrency, timeout & degraded tests
│   │   ├── test_reranker.py         # CrossEncoder reranking & tie-breaking tests (Phase 6)
│   │   ├── test_context_assembler.py# Token budgeting & citation assembly tests (Phase 6)
│   │   ├── test_pipeline.py         # End-to-end pipeline retrieval integration tests (Phase 6)
│   │   ├── test_generation.py       # LLM generation models, prompts & providers (Phase 7)
│   │   └── test_pipeline_generation.py # RAG answer generation & API endpoint tests (Phase 7)
│   ├── requirements.txt             # Backend dependencies
│   └── .env.example                 # Environment configuration template
├── datasets/
│   ├── corpus.jsonl                 # Document collection (36 technical documents)
│   ├── queries.jsonl                # Evaluation query set (24 queries across 7 types)
│   ├── qrels.tsv                    # Ground-truth judgments (88 pairs; scale 0, 1, 2)
│   ├── dataset_manifest.json        # Programmatic metadata and SHA-256 hashes
│   ├── processed/                   # Ingested chunk outputs (e.g. chunks.jsonl)
│   ├── indexes/                     # Built retriever indexes (bm25, dense)
│   └── README.md                    # Dataset schema & relevance guidelines
├── evaluation/
│   ├── __init__.py
│   ├── dataset_validation.py        # Schema, uniqueness & checksum validator
│   ├── metrics.py                   # IR metrics: Recall@K, MRR@K, nDCG@K
│   ├── runner.py                    # Batch metric runner & report generator
│   ├── experiment.py                # Reproducible experiment metadata
│   ├── fixtures.py                  # Synthetic ranking fixtures for verification
│   ├── retriever_adapter.py         # Adapter bridging retrievers to evaluation runner
│   └── README.md                    # Evaluation framework documentation
├── benchmarks/
│   ├── results/                     # Machine-readable evaluation reports (gitignored)
│   └── README.md                    # Benchmarking methodology
├── scripts/
│   ├── generate_dataset.py          # Deterministic dataset generator
│   ├── validate_dataset.py          # CLI dataset validation runner
│   ├── run_eval.py                  # CLI evaluation runner (BM25, Dense, Hybrid, Reranked)
│   ├── process_documents.py         # CLI document ingestion runner
│   ├── build_indexes.py             # CLI index builder
│   ├── query_index.py               # CLI single-index retriever search
│   ├── query_hybrid.py              # CLI concurrent hybrid retrieval search
│   ├── query_reranked.py            # CLI reranked retrieval + context assembly (Phase 6)
│   ├── benchmark_concurrency.py     # CLI throughput & concurrency benchmark
│   └── README.md                    # Operational script inventory
├── docs/
│   ├── ARCHITECTURE.md              # Greenfield system architecture specification
│   ├── DEVELOPMENT_PLAN.md          # Multi-phase engineering roadmap
│   ├── PHASE_2_REPORT.md            # Phase 2 implementation report
│   ├── PHASE_3_REPORT.md            # Phase 3 implementation report
│   ├── PHASE_4_REPORT.md            # Phase 4 implementation report
│   ├── PHASE_5_REPORT.md            # Phase 5 implementation report
│   └── PHASE_6_REPORT.md            # Phase 6 implementation report
├── pytest.ini                       # Test path and pythonpath configuration
├── .gitignore                       # Git exclusions for secrets, venv, data, and cache
└── README.md                        # Project documentation
```

---

## 13. Future Development Roadmap

See [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) for the detailed phase breakdown, milestones, and acceptance criteria.



