# FastRAG Engineering Roadmap & Development Plan

This document establishes the multi-phase engineering plan for building, evaluating, and deploying **FastRAG**, an end-to-end, high-performance Retrieval-Augmented Generation system.

---

## Roadmap Overview

| Phase | Title | Focus Area | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Greenfield Setup and Architecture | Foundation, Config, Health Endpoint, Tests, Docs | **Completed** |
| **Phase 2** | Dataset Preparation and Baseline Evaluation | Corpora, Evaluation queries, Ground truth, Metric harnesses | **Completed** |
| **Phase 3** | Document Ingestion, Chunking, and Indexing | Loaders, Normalization, Chunking, Ingestion Pipeline | **Completed** |
| **Phase 4** | Dense Retrieval and BM25 Retrieval | Vector store integration, BM25 inverted index | *Pending Approval* |
| **Phase 5** | Concurrent Hybrid Retrieval and RRF | Async dispatch, Reciprocal Rank Fusion algorithm | *Pending Approval* |
| **Phase 6** | CrossEncoder Reranking and Context Assembly | Transformer cross-attention, Token budgeting, Citations | *Pending Approval* |
| **Phase 7** | LLM Generation and Streaming | LLM client, Prompt templates, SSE streaming | *Pending Approval* |
| **Phase 8** | Caching and Orchestration Optimization | Multilevel query/embedding caching, Pipeline coordination | *Pending Approval* |
| **Phase 9** | Performance Benchmarks, Load Testing, Quality Eval | P50-P99 profiling, Throughput load tests, RAG quality | *Pending Approval* |
| **Phase 10** | Deployment, Documentation, and Research Paper | Containerization, Production CI/CD, Technical report | *Pending Approval* |

---

## Phase Details

### Phase 1: Greenfield Setup and Architecture
- **Objectives**:
  - Establish a completely independent, performance-oriented RAG project foundation.
  - Implement a clean FastAPI application with standard logging, Pydantic settings, and health check monitoring.
  - Define system architecture, data flow diagrams, and a comprehensive development roadmap.
  - Verify environment reproducibility with automated tests.
- **Deliverables**:
  - Modular project structure (`backend/`, `datasets/`, `evaluation/`, `benchmarks/`, `scripts/`, `docs/`).
  - `backend/app/main.py` with `GET /health`.
  - `backend/app/core/config.py` with Pydantic settings and environment variable validation.
  - `backend/app/core/logging_config.py` with latency measurement utilities.
  - `backend/app/core/exceptions.py` with domain error hierarchy.
  - Automated test suite (`test_health.py`, `test_config.py`).
  - `docs/ARCHITECTURE.md` and `docs/DEVELOPMENT_PLAN.md`.
  - Root `README.md` and subsystem README files.
- **Dependencies**: Python 3.11/3.12, virtual environment, Git.
- **Acceptance Criteria**:
  - [x] Application imports and starts cleanly without external infrastructure.
  - [x] `GET /health` returns HTTP 200 with predictable JSON payload.
  - [x] Configuration properly resolves defaults and environment overrides.
  - [x] Automated test suite runs and passes 100% of tests.
  - [x] Complete architecture documentation with Mermaid diagrams.
  - [x] Zero external code imported from legacy projects.

---

### Phase 2: Dataset Preparation and Baseline Evaluation
- **Objectives**:
  - Author and validate a self-contained, reproducible evaluation dataset covering RAG architectural and retrieval topics.
  - Define evaluation query taxonomy and independent ground-truth relevance judgements (`qrels.tsv`).
  - Establish automated calculation and unit testing for IR retrieval quality metrics (Recall@K, MRR@K, nDCG@K).
  - Implement an evaluation runner producing machine-readable JSON reports with experiment metadata and checksum locking.
- **Deliverables**:
  - Deterministic dataset generator: `scripts/generate_dataset.py`.
  - Dataset corpus: `datasets/corpus.jsonl` (36 technical documents).
  - Evaluation queries: `datasets/queries.jsonl` (24 queries across 7 categories).
  - Ground-truth relevance judgments: `datasets/qrels.tsv` (88 labeled query-doc pairs; scale 0, 1, 2).
  - Dataset manifest with SHA-256 hashes: `datasets/dataset_manifest.json`.
  - Dataset validator & CLI: `evaluation/dataset_validation.py`, `scripts/validate_dataset.py`.
  - IR metrics engine: `evaluation/metrics.py` (Recall@K, MRR@K, nDCG@K).
  - Batch evaluation runner: `evaluation/runner.py`, `scripts/run_eval.py`.
  - Experiment metadata tracking: `evaluation/experiment.py`.
  - Synthetic test fixtures: `evaluation/fixtures.py`.
  - Automated tests: `backend/tests/test_dataset.py`, `test_metrics.py`, `test_runner.py`.
  - Documentation: `datasets/README.md`, `docs/PHASE_2_REPORT.md`.
- **Dependencies**: Phase 1 completion and approval.
- **Acceptance Criteria**:
  - [x] Corpus, queries, qrels, and manifest present and internally consistent.
  - [x] Dataset validation passes with zero errors, verifying counts and SHA-256 checksums.
  - [x] Recall@K, MRR@K, and nDCG@K have deterministic unit tests and handle all edge cases.
  - [x] Evaluation runner processes predetermined rankings and generates machine-readable JSON reports.
  - [x] Experiment metadata accurately records dataset version and manifest hash.
  - [x] All automated tests pass (39/39 passing).
  - [x] Zero external cloud dependencies, LLMs, or retrieval engines prematurely introduced.

---

### Phase 3: Document Ingestion, Chunking, and Indexing Preparation
- **Objectives**:
  - Implement a modular, deterministic document ingestion pipeline for JSONL and plain-text (`.txt`) documents.
  - Implement conservative text normalization preserving Unicode NFC, code snippets, URLs, and paragraph structures.
  - Develop configurable, boundary-aware text chunking with exact character offset tracking and deterministic chunk IDs.
  - Implement pre-ingestion document validation and post-chunking integrity verification.
  - Export processed chunks atomically to JSONL format.
- **Deliverables**:
  - `backend/app/ingestion/models.py` (`Document`, `Chunk`, `IngestionSummary`).
  - `backend/app/ingestion/loaders.py` (`load_jsonl`, `load_jsonl_stream`, `load_text_file`, `load_directory`).
  - `backend/app/ingestion/normalization.py` (`normalize_text`).
  - `backend/app/ingestion/chunking.py` (`Chunker`, `generate_chunk_id`).
  - `backend/app/ingestion/validation.py` (`validate_document`, `validate_chunk`, `validate_documents_batch`).
  - `backend/app/ingestion/pipeline.py` (`IngestionPipeline`).
  - CLI ingestion runner: `scripts/process_documents.py`.
  - Comprehensive unit test suite: `backend/tests/test_ingestion.py`.
  - Engineering report: `docs/PHASE_3_REPORT.md`.
- **Dependencies**: Phase 2 evaluation dataset and validation harness.
- **Acceptance Criteria**:
  - [x] Phase 2 prerequisites verified (corpus, queries, qrels, manifest, validation pass).
  - [x] Document loaders successfully process JSONL corpora and plain-text files.
  - [x] Conservative normalization preserves technical identifiers, code, and paragraph structure.
  - [x] Chunking is configurable, boundary-aware, and deterministic.
  - [x] Chunk IDs are content-addressed and reproducible across runs.
  - [x] Exact character offsets verified (`doc_text[start:end] == chunk.text`).
  - [x] Source metadata and provenance are preserved.
  - [x] Output exported atomically to `datasets/processed/chunks.jsonl`.
  - [x] CLI runner operational with `--chunk-size` and `--chunk-overlap` options.
  - [x] Complete test suite passes (69/69 tests passing).
  - [x] Original evaluation dataset remains unmodified and verified.

---

### Phase 4: Dense Retrieval and BM25 Retrieval
- **Status**: **Completed** (Report: [docs/PHASE_4_REPORT.md](file:///Users/thakurveersingh/Documents/fastrag/docs/PHASE_4_REPORT.md))
- **Objectives**:
  - Implement batched embedding generation using an efficient embedding model.
  - Implement an in-memory/disk dense vector index with exact cosine similarity.
  - Implement an in-memory/disk BM25 lexical inverted index.
  - Expose independent retriever interfaces with latency profiling and evaluation bridging.
- **Deliverables**:
  - `backend/app/retrieval/base.py` (`BaseRetriever`, `RetrievalResult`, `IndexInfo`, `validate_top_k`, `compute_corpus_fingerprint`).
  - `backend/app/retrieval/dense.py` (`DenseRetriever`, `SentenceTransformerEncoder`, `MockEmbeddingEncoder`).
  - `backend/app/retrieval/bm25.py` (`BM25Retriever`, `default_bm25_tokenizer`).
  - `backend/app/retrieval/indexer.py` (`build_and_save_bm25_index`, `build_and_save_dense_index`).
  - `evaluation/retriever_adapter.py` (`evaluate_retriever`, `RetrievalLatencyStats`).
  - CLI tools: `scripts/build_indexes.py`, `scripts/query_index.py`, `scripts/run_eval.py`.
  - Comprehensive unit and integration test suite: `backend/tests/test_retrieval.py` (15 tests).
  - Engineering report: `docs/PHASE_4_REPORT.md`.
- **Dependencies**: Phase 3 chunked corpus (`datasets/processed/chunks.jsonl`).
- **Acceptance Criteria**:
  - [x] Dense retriever correctly indexes and searches top-$k$ embeddings using cosine similarity.
  - [x] BM25 retriever correctly indexes terms and executes keyword lookup with Okapi formulation.
  - [x] Shared typed retrieval interface implemented with deterministic tie-breaking.
  - [x] Index persistence formats are safe and reproducible (NPZ + JSON for dense, JSON for BM25).
  - [x] Retrieval latency measured and segregated into cold-start and warm query percentiles.
  - [x] Retrievers evaluated against Phase 2 dataset queries and qrels with full reports generated.
  - [x] Full test suite passes (84/84 tests passing).
  - [x] No hybrid fusion, CrossEncoder, or LLM generation implemented (strict Phase 4 boundary).

---

### Phase 5: Concurrent Hybrid Retrieval and RRF
- **Status**: **Completed** (Report: [docs/PHASE_5_REPORT.md](file:///Users/thakurveersingh/Documents/fastrag/docs/PHASE_5_REPORT.md))
- **Objectives**:
  - Build the concurrent retrieval orchestrator utilizing Python `asyncio.to_thread` and `asyncio.gather`.
  - Implement standard Reciprocal Rank Fusion (RRF) to merge dense and lexical candidate lists.
  - Eliminate score distribution disparities through rank-based fusion.
  - Support configurable retrieval parameters ($K_{\text{dense}}$, $K_{\text{bm25}}$, $RRF_k$).
  - Measure per-branch latency and evaluate throughput under concurrency.
- **Deliverables**:
  - `backend/app/retrieval/fusion.py` (`reciprocal_rank_fusion`, `validate_rrf_k`).
  - `backend/app/retrieval/hybrid.py` (`HybridRetriever`, `HybridRetrievalResponse`, `HybridRetrievalDiagnostics`).
  - `scripts/query_hybrid.py` (interactive CLI query search for hybrid retrieval).
  - `scripts/benchmark_concurrency.py` (concurrency and throughput benchmark runner).
  - Comprehensive unit and concurrency tests: `backend/tests/test_hybrid.py` (12 tests).
  - Engineering and evaluation report: `docs/PHASE_5_REPORT.md`.
- **Dependencies**: Phase 4 retrievers (`BM25Retriever`, `DenseRetriever`).
- **Acceptance Criteria**:
  - [x] Dense and lexical searches execute concurrently, verified with temporal overlap test doubles.
  - [x] RRF correctly fuses rankings without floating-point scale bias and breaks ties deterministically.
  - [x] Strict and degraded error handling modes support branch timeouts and isolated failures.
  - [x] Results are deduplicated by `chunk_id` with preserved document provenance and metadata.
  - [x] CLI script `query_hybrid.py` operational with diagnostic branch timings.
  - [x] Hybrid retriever evaluated against benchmark queries and qrels for both sequential and concurrent modes.
  - [x] Full test suite passes (96/96 tests passing).
  - [x] Strict scope preserved: no CrossEncoder, no LLM generation, no caching.

---

### Phase 6: CrossEncoder Reranking and Context Assembly
- **Status**: **Completed** (Report: [docs/PHASE_6_REPORT.md](file:///Users/thakurveersingh/Documents/fastrag/docs/PHASE_6_REPORT.md))
- **Objectives**:
  - Integrate a configurable CrossEncoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for fine-grained re-scoring of top-$N$ candidate passages.
  - Implement dynamic reranking bypass for latency-sensitive queries or offline evaluation.
  - Develop token-budgeted context assembly with deduplication and source diversity.
  - Format passages with structured citation identifiers (`[Doc-1]`, `[Doc-2]`).
  - Provide unified end-to-end `RAGPipeline` coordinating Candidate Retrieval -> Reranking -> Assembly.
- **Deliverables**:
  - `backend/app/retrieval/reranker.py` (`CrossEncoderReranker`, `SentenceTransformerCrossEncoder`, `MockCrossEncoderScorer`).
  - `backend/app/retrieval/context_assembler.py` (`ContextAssembler`, `AssembledContext`, `ContextChunk`, `approximate_token_counter`).
  - `backend/app/retrieval/pipeline.py` (`RAGPipeline`, `PipelineResponse`, `PipelineDiagnostics`).
  - `scripts/query_reranked.py` (interactive CLI for reranked search and context assembly).
  - Extended evaluation adapter and CLI in `evaluation/retriever_adapter.py` and `scripts/run_eval.py`.
  - Comprehensive unit and integration test suite (25 new tests, 121/121 total passing): `backend/tests/test_reranker.py`, `backend/tests/test_context_assembler.py`, `backend/tests/test_pipeline.py`.
  - Engineering and evaluation report: `docs/PHASE_6_REPORT.md`.
- **Dependencies**: Phase 5 hybrid retrieval.
- **Acceptance Criteria**:
  - [x] CrossEncoder scores query-passage pairs accurately and re-orders candidates.
  - [x] Neural weights lazy loaded with local cache precedence and instance reuse across queries.
  - [x] Deterministic tie-breaking by `chunk_id` ascending and deduplication prior to neural scoring.
  - [x] Token budgeter strictly truncates/prunes context within configured limit.
  - [x] Citation tags consistently reference correct source document metadata.
  - [x] End-to-end pipeline operational and exposed via CLI (`scripts/query_reranked.py`).
  - [x] Full evaluation performed across all four configurations on 24 queries and 88 qrels.
  - [x] Full test suite passes (121/121 tests passing).
  - [x] Zero LLM generation, answer synthesis, streaming, or caching prematurely introduced.

---

### Phase 7: LLM Generation and Streaming
- **Status**: **Completed** (Report: [docs/PHASE_7_REPORT.md](file:///Users/thakurveersingh/Documents/fastrag/docs/PHASE_7_REPORT.md))
- **Objectives**:
  - Implement an asynchronous grounded LLM answer generator strictly consuming Phase 6 `AssembledContext`.
  - Provide pluggable provider abstraction supporting OpenAI-compatible endpoints and deterministic mock generators.
  - Implement real-time response streaming using standard Server-Sent Events (SSE).
  - Track generation latency (TTFT and generation duration) and distinguish reported vs estimated token usage.
  - Map citation identifiers (`[Doc-N]`) back to source document provenance.
  - Expose FastAPI endpoints (`POST /api/v1/query` and `POST /api/v1/query/stream`) and CLI (`scripts/query_generation.py`).
- **Deliverables**:
  - `backend/app/generation/models.py` (`UsageStats`, `GenerationDiagnostics`, `GenerationRequest`, `GenerationResponse`, `StreamChunk`, `StreamEvent`).
  - `backend/app/generation/prompts.py` (`DEFAULT_SYSTEM_PROMPT`, `build_grounded_prompt`).
  - `backend/app/generation/providers.py` (`LLMProvider`, `MockLLMProvider`, `OpenAICompatibleProvider`).
  - `backend/app/generation/generator.py` (`RAGGenerator`).
  - `backend/app/retrieval/pipeline.py` (extended with `execute_stream_async` and generation parameters).
  - `backend/app/api/v1/routes.py` (FastAPI endpoints `POST /api/v1/query` and `POST /api/v1/query/stream`).
  - `scripts/query_generation.py` (interactive CLI for non-streaming, live streaming, mock offline, and JSON modes).
  - Comprehensive unit and integration test suite (25 new tests, 146/146 total passing): `backend/tests/test_generation.py`, `backend/tests/test_pipeline_generation.py`.
  - Engineering report: `docs/PHASE_7_REPORT.md`.
- **Dependencies**: Phase 6 context assembly.
- **Acceptance Criteria**:
  - [x] End-to-end question answering pipeline produces coherent, grounded answers.
  - [x] Streaming endpoint pushes incremental tokens to clients with minimal TTFT via SSE.
  - [x] Answers cite relevant source identifiers (`[Doc-N]`) and map back to source document IDs.
  - [x] Prompt injection defense and explicit refusal when context is missing or insufficient.
  - [x] Pluggable provider interface supports both OpenAI-compatible APIs and mock providers for CI/CD.
  - [x] Full test suite passes (146/146 tests passing in <1s).
  - [x] Retrieval quality preserved with zero regression across all metrics.
  - [x] Zero premature caching (Phase 8) introduced.

---

### Phase 8: Caching and Orchestration Optimization
- **Objectives**:
  - Implement Level 1 Query Cache (exact/semantic query matching) to bypass execution for warm queries.
  - Implement Level 2 Embedding Cache (in-memory LRU/TTL) for query strings.
  - Optimize pipeline orchestration and error recovery.
- **Deliverables**:
  - `backend/app/caching/query_cache.py`.
  - `backend/app/caching/embedding_cache.py`.
  - `backend/app/orchestration/pipeline.py` (central coordinator with caching hooks).
- **Dependencies**: Phases 5–7.
- **Acceptance Criteria**:
  - Repeated queries hit cache with sub-10ms response times.
  - Embedding cache prevents redundant encoder inference.
  - Cache hit/miss metrics logged in application telemetry.

---

### Phase 9: Performance Benchmarks, Load Testing, and Quality Evaluation
- **Objectives**:
  - Execute rigorous latency and throughput load tests (single request vs. concurrent clients).
  - Profile P50, P90, P95, and P99 latency percentiles across all pipeline stages.
  - Measure retrieval quality (MRR@k, NDCG@k, Recall@k) and generation faithfulness against Phase 2 baselines.
  - Analyze trade-offs: Dense vs. BM25 vs. Hybrid vs. CrossEncoder.
- **Deliverables**:
  - `benchmarks/load_tests/` (concurrency load test scripts).
  - `scripts/run_benchmarks.py` and `scripts/run_eval.py`.
  - Comprehensive performance benchmark report with graphs and metric tables.
- **Dependencies**: Phases 1–8.
- **Acceptance Criteria**:
  - End-to-end latency breakdowns published with empirical percentile distributions.
  - Retrieval quality metrics quantified across all configurations.
  - Reproducible benchmark harness with automated test execution.

---

### Phase 10: Deployment, Documentation, and Research-Paper Preparation
- **Objectives**:
  - Package FastRAG into lightweight production Docker containers with multi-stage builds.
  - Provide Docker Compose configuration for development and production profiles.
  - Finalize production documentation, operational runbooks, and API specifications.
  - Author technical whitepaper / research report detailing architectural optimizations and empirical results.
- **Deliverables**:
  - `Dockerfile` and `docker-compose.yml`.
  - Complete API documentation with OpenAPI specifications.
  - Technical architecture and evaluation paper (`docs/TECHNICAL_REPORT.md`).
- **Dependencies**: Phase 9 benchmark results.
- **Acceptance Criteria**:
  - Containerized deployment starts cleanly and passes health check.
  - Reproducible setup verified on fresh environment.
  - Comprehensive technical paper prepared with empirical evidence.
