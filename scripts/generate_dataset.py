"""FastRAG Dataset Generator.

Generates a deterministic, self-contained evaluation dataset for FastRAG:
- datasets/corpus.jsonl: 36 technical documents across RAG and retrieval domains
- datasets/queries.jsonl: 24 evaluation queries covering exact, semantic, acronym, and distractor types
- datasets/qrels.tsv: Ground-truth relevance judgments (scale: 0=irrelevant, 1=partially relevant, 2=highly relevant)
- datasets/dataset_manifest.json: Dataset metadata, counts, and SHA-256 checksums
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


DOCUMENTS = [
    # Topic 1: Chunking & Text Segmentation (doc_001 - doc_004)
    {
        "doc_id": "doc_001",
        "title": "Recursive Character Text Chunking for RAG",
        "text": "Recursive character chunking splits documents using a hierarchy of separators such as double newlines, single newlines, punctuation, and spaces. By prioritizing semantic boundaries before falling back to arbitrary character cuts, it preserves paragraph and sentence coherence. Typical chunk sizes range from 256 to 512 tokens with a 10% to 20% token overlap to maintain context across chunk boundaries.",
        "source": "fastrag-docs/ingestion/chunking",
        "metadata": {"topic": "chunking", "category": "ingestion", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_002",
        "title": "Semantic Chunking via Sentence Embedding Similarity",
        "text": "Semantic chunking determines split points dynamically by computing embedding distance between consecutive sentences. When cosine distance between sequential sentences exceeds a predefined threshold, a new chunk boundary is established. While computationally heavier than fixed-size sliding windows, semantic chunking avoids splitting tightly coupled arguments across separate chunks.",
        "source": "fastrag-docs/ingestion/semantic_chunking",
        "metadata": {"topic": "chunking", "category": "ingestion", "difficulty": "advanced"},
    },
    {
        "doc_id": "doc_003",
        "title": "Token Count Boundaries vs Character Length Splitting",
        "text": "Splitting text by raw character length frequently leads to token count mismatches with LLM tokenizers like Byte-Pair Encoding (BPE) or WordPiece. Code snippets, technical identifiers, and non-ASCII scripts produce significantly more tokens per character than standard English prose. Robust chunkers count tokens directly using model-specific tokenizers to prevent truncation during prompt assembly.",
        "source": "fastrag-docs/ingestion/tokenization",
        "metadata": {"topic": "chunking", "category": "ingestion", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_004",
        "title": "Sliding Window Chunking and Overlap Strategies",
        "text": "Sliding window chunking advances a fixed token window across source text with a predefined stride. The overlap parameter (typically 32 to 64 tokens) ensures that query-matching phrases straddling a chunk boundary appear intact in at least one retrieved passage. Excessive overlap increases index storage footprint and duplicates candidate tokens in downstream prompts.",
        "source": "fastrag-docs/ingestion/sliding_window",
        "metadata": {"topic": "chunking", "category": "ingestion", "difficulty": "introductory"},
    },

    # Topic 2: Dense Vector Retrieval & Embeddings (doc_005 - doc_008)
    {
        "doc_id": "doc_005",
        "title": "Dense Vector Embeddings and Bi-Encoder Architecture",
        "text": "Dense vector retrieval maps unstructured text into continuous vector representations in a shared latent space. Bi-encoder architectures encode documents and queries independently using twin Transformer encoders, allowing document vectors to be pre-computed offline. Query vectors are compared against document vectors using dot product or cosine similarity during online search.",
        "source": "fastrag-docs/retrieval/dense_embeddings",
        "metadata": {"topic": "embeddings", "category": "retrieval", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_006",
        "title": "Cosine Similarity and Vector Normalization",
        "text": "Cosine similarity measures the angle between two multi-dimensional vectors, invariant to vector magnitude. By pre-normalizing all document embeddings to unit Euclidean length (L2 norm = 1.0) during offline ingestion, cosine similarity simplifies computationally to a fast vector dot product, substantially accelerating nearest neighbor search.",
        "source": "fastrag-docs/retrieval/vector_math",
        "metadata": {"topic": "embeddings", "category": "retrieval", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_007",
        "title": "Hierarchical Navigable Small World (HNSW) Graphs",
        "text": "HNSW is an Approximate Nearest Neighbor (ANN) vector indexing structure based on multi-layer skip-list graphs. The upper layers provide long-range exploration, while lower layers refine search to local neighborhoods. Key hyperparameters include M (number of bi-directional links per node) and efConstruction (size of dynamic candidate list during graph building). HNSW delivers sub-millisecond retrieval latency with high recall.",
        "source": "fastrag-docs/retrieval/hnsw_indexing",
        "metadata": {"topic": "vector_index", "category": "retrieval", "difficulty": "advanced"},
    },
    {
        "doc_id": "doc_008",
        "title": "Inverted File Indexing with Product Quantization (IVF-PQ)",
        "text": "IVF-PQ partitions vector space into Voronoi cells via k-means clustering and compresses high-dimensional vectors into compact byte codes using product quantization. During query execution, only vectors belonging to the nearest Voronoi centroids (nprobe) are decompressed and scored. IVF-PQ significantly reduces RAM consumption at the cost of slight recall degradation compared to exact Flat indexing.",
        "source": "fastrag-docs/retrieval/ivf_quantization",
        "metadata": {"topic": "vector_index", "category": "retrieval", "difficulty": "advanced"},
    },

    # Topic 3: Lexical Search & BM25 (doc_009 - doc_012)
    {
        "doc_id": "doc_009",
        "title": "BM25 Probabilistic Lexical Scoring Formulation",
        "text": "BM25 (Best Matching 25) scores document relevance based on term frequency (TF) and inverse document frequency (IDF). The term frequency saturation parameter k1 (typically 1.2 to 2.0) controls how quickly additional occurrences of a term reach diminishing returns. The parameter b (typically 0.75) penalizes long documents by normalizing term frequency relative to average document length in the collection.",
        "source": "fastrag-docs/retrieval/bm25_formulation",
        "metadata": {"topic": "bm25", "category": "retrieval", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_010",
        "title": "Lexical Inverted Index Posting Lists and Tokenization",
        "text": "An inverted index maps distinct vocabulary terms to posting lists containing document IDs, term frequencies, and positional offsets. Text preprocessing typically includes lowercasing, punctuation stripping, stop word filtering, and Porter or Snowball stemming. Inverted indexes excel at exact identifier lookups, code token queries, and rare acronym matching where embedding models frequently suffer out-of-vocabulary distortion.",
        "source": "fastrag-docs/retrieval/inverted_index",
        "metadata": {"topic": "bm25", "category": "retrieval", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_011",
        "title": "Lexical vs Dense Retrieval Complementarity in RAG",
        "text": "BM25 relies on exact surface-form token matching, making it resilient to rare entity names, serial numbers, and technical terminology. In contrast, dense bi-encoders capture semantic meaning, synonyms, and conceptual intent but struggle with exact alphanumeric strings. Combining both retrieval paradigms overcomes individual blind spots, boosting overall system recall.",
        "source": "fastrag-docs/retrieval/lexical_dense_hybrid",
        "metadata": {"topic": "hybrid_retrieval", "category": "retrieval", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_012",
        "title": "Tuning BM25 Parameters k1 and b for Technical Documentation",
        "text": "For technical manuals containing concise reference sections alongside verbose tutorials, default BM25 parameters may underperform. Setting b=0.5 dampens excessive length penalties, preventing brief function signatures from drowning out detailed architectural guides. Lowering k1 toward 1.2 curbs the influence of repeated keywords in repetitive log entries or API specifications.",
        "source": "fastrag-docs/retrieval/bm25_tuning",
        "metadata": {"topic": "bm25", "category": "retrieval", "difficulty": "advanced"},
    },

    # Topic 4: Hybrid Concurrency & Reciprocal Rank Fusion (doc_013 - doc_016)
    {
        "doc_id": "doc_013",
        "title": "Concurrent Retrieval Dispatch Using AsyncIO Gather",
        "text": "Sequential execution of dense ANN search and BM25 search accumulates latency: total retrieval time equals T_dense + T_lexical. By dispatching both retrievers concurrently via Python's asyncio.gather, total wall-clock retrieval time is reduced to max(T_dense, T_lexical) plus negligible non-blocking orchestration overhead.",
        "source": "fastrag-docs/orchestration/async_retrieval",
        "metadata": {"topic": "concurrency", "category": "orchestration", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_014",
        "title": "Reciprocal Rank Fusion (RRF) Mathematical Principles",
        "text": "Reciprocal Rank Fusion (RRF) merges multiple ranked lists without requiring score calibration across disparate scoring scales. For each document d, RRF score is computed as sum(1 / (k + rank_m(d))) across all retrieval systems m, where k is a smoothing constant (traditionally k=60). RRF prevents high absolute score outliers in one system from overwhelming consensus rankings.",
        "source": "fastrag-docs/reranking/rrf_principles",
        "metadata": {"topic": "rrf", "category": "reranking", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_015",
        "title": "RRF Smoothing Constant k Tuning and Sensitivity",
        "text": "The RRF constant k regulates the ranking penalty between top-ranked and lower-ranked candidates. A small k (e.g., k=10) aggressively favors documents ranked in the top 3 positions of either list, whereas a large k (e.g., k=100) smooths the distribution, giving balanced weight to documents appearing moderately well across both dense and lexical lists.",
        "source": "fastrag-docs/reranking/rrf_tuning",
        "metadata": {"topic": "rrf", "category": "reranking", "difficulty": "advanced"},
    },
    {
        "doc_id": "doc_016",
        "title": "Deduplication and Score Normalization in Hybrid Pipelines",
        "text": "When dense and BM25 retrievers return overlapping candidate pools, naive concatenation causes document duplication and skewed downstream rankings. Rank fusion methods like RRF naturally handle deduplication by summing reciprocal rank bonuses whenever a document identifier appears across multiple retriever result sets.",
        "source": "fastrag-docs/reranking/deduplication",
        "metadata": {"topic": "rrf", "category": "reranking", "difficulty": "intermediate"},
    },

    # Topic 5: CrossEncoder Reranking (doc_017 - doc_020)
    {
        "doc_id": "doc_017",
        "title": "CrossEncoder vs Bi-Encoder Architecture and Complexity",
        "text": "Unlike bi-encoders which process query and document in isolation, a CrossEncoder feeds the concatenated query and candidate passage into full self-attention layers simultaneously. This allows every query token to attend to every document token, capturing nuanced syntactic and semantic dependencies. CrossEncoders have O(N * L^2) computational cost, making them suitable only for small reranking candidate windows.",
        "source": "fastrag-docs/reranking/cross_encoder",
        "metadata": {"topic": "reranking", "category": "reranking", "difficulty": "advanced"},
    },
    {
        "doc_id": "doc_018",
        "title": "Two-Stage Retrieval: High Recall Funnel Followed by High Precision Rerank",
        "text": "Modern high-performance RAG pipelines adopt a two-stage funnel architecture. First, fast hybrid retrieval (Dense + BM25) fetches 50 to 100 candidate documents in tens of milliseconds to maximize candidate recall. Second, a lightweight CrossEncoder reranks only the top 10 to 20 candidates, optimizing top-k precision while keeping total query latency bounded.",
        "source": "fastrag-docs/architecture/two_stage_funnel",
        "metadata": {"topic": "reranking", "category": "architecture", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_019",
        "title": "CrossEncoder Batching and Inference Latency Optimization",
        "text": "Evaluating CrossEncoder pairs sequentially introduces unacceptable tail latency. Batching candidate pairs into a single tensor evaluation leverages SIMD and GPU matrix acceleration. For latency-sensitive applications, dynamic early-exit or bypassing CrossEncoder reranking when RRF confidence is exceptionally high preserves tight SLA targets.",
        "source": "fastrag-docs/reranking/inference_optimization",
        "metadata": {"topic": "reranking", "category": "performance", "difficulty": "advanced"},
    },
    {
        "doc_id": "doc_020",
        "title": "Distillation and Lightweight Rerankers (MiniLM, BGE-Reranker-Base)",
        "text": "Deploying massive 3B+ parameter CrossEncoders in the serving path creates latency bottlenecks. Knowledge-distilled models such as MS-MARCO MiniLM-L6 or BGE-Reranker-Base offer substantial precision improvements over bi-encoders with inference latencies under 15ms for batches of 10 candidate passages on modern hardware.",
        "source": "fastrag-docs/reranking/lightweight_models",
        "metadata": {"topic": "reranking", "category": "models", "difficulty": "intermediate"},
    },

    # Topic 6: Context Assembly, Citations & Prompt Engineering (doc_021 - doc_024)
    {
        "doc_id": "doc_021",
        "title": "Token Budgeting and Context Window Optimization",
        "text": "Context assembly allocates a strict token budget among system instructions, conversation history, retrieved passages, and anticipated output tokens. Greedy knapsack packing selects top-ranked passages until the context budget is exhausted, preventing prompt overflow errors and curbing LLM time-to-first-token latency.",
        "source": "fastrag-docs/generation/token_budgeting",
        "metadata": {"topic": "context_assembly", "category": "generation", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_022",
        "title": "Mitigating the Lost-in-the-Middle Effect in RAG Contexts",
        "text": "Empirical studies reveal that LLM attention mechanisms preferentially attend to tokens located at the extreme beginning and end of long input prompts, frequently overlooking information in the middle. Strategic context ordering places the highest-relevance passages at the prompt boundaries to maximize factual utilization.",
        "source": "fastrag-docs/generation/lost_in_middle",
        "metadata": {"topic": "context_assembly", "category": "generation", "difficulty": "advanced"},
    },
    {
        "doc_id": "doc_023",
        "title": "Source Provenance and Grounded Citation Tagging",
        "text": "To prevent hallucinations and facilitate verification, context assemblers format each retrieved passage with explicit source tags like [Doc-1] or [Ref-A]. The prompt instructs the LLM to ground factual assertions using these tags, providing traceable citations linking answer statements directly back to source documents.",
        "source": "fastrag-docs/generation/citation_grounding",
        "metadata": {"topic": "citations", "category": "generation", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_024",
        "title": "Deduplicating Context Passages Across Multi-Chunk Results",
        "text": "When multiple consecutive chunks from the same document appear in the top-k ranking, redundant introductory headers and overlapping boundaries waste precious prompt tokens. A context deduplicator merges overlapping chunks into a single contiguous context window before final prompt synthesis.",
        "source": "fastrag-docs/generation/context_deduplication",
        "metadata": {"topic": "context_assembly", "category": "generation", "difficulty": "intermediate"},
    },

    # Topic 7: Caching & Latency Optimization (doc_025 - doc_028)
    {
        "doc_id": "doc_025",
        "title": "Multi-Tier Caching Architecture for RAG Systems",
        "text": "FastRAG uses a two-tier caching architecture: Level 1 Query Cache stores completed end-to-end responses for exact or near-identical queries, bypassing retrieval and generation entirely. Level 2 Embedding Cache stores query embedding vectors in-memory, avoiding redundant Transformer encoder passes when queries share common substrings.",
        "source": "fastrag-docs/caching/multi_tier_cache",
        "metadata": {"topic": "caching", "category": "performance", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_026",
        "title": "Cache Invalidation and Index Mutation Versioning",
        "text": "Serving stale answers after underlying documents have changed compromises RAG reliability. In FastRAG, cache entries are tagged with the active corpus version hash. Any document insertion, update, or deletion increments the index version, invalidating outdated cache entries automatically.",
        "source": "fastrag-docs/caching/cache_invalidation",
        "metadata": {"topic": "caching", "category": "performance", "difficulty": "advanced"},
    },
    {
        "doc_id": "doc_027",
        "title": "Time to First Token (TTFT) and Streaming Generation",
        "text": "Time to First Token (TTFT) measures latency from request submission until the initial generated token arrives at the client. Streaming LLM output using Server-Sent Events (SSE) or HTTP chunked transfer dramatically reduces perceived latency, providing interactive feedback while subsequent tokens generate.",
        "source": "fastrag-docs/generation/streaming_ttft",
        "metadata": {"topic": "streaming", "category": "generation", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_028",
        "title": "Asynchronous Non-Blocking IO in Python RAG Backends",
        "text": "FastAPI combined with Uvicorn and uvloop provides an asynchronous event loop capable of handling thousands of concurrent connections. By delegating vector search and model inference to non-blocking worker threads or external async clients, the main thread remains responsive to incoming health checks and requests.",
        "source": "fastrag-docs/architecture/async_io",
        "metadata": {"topic": "concurrency", "category": "architecture", "difficulty": "intermediate"},
    },

    # Topic 8: Retrieval Evaluation & Metrics (doc_029 - doc_032)
    {
        "doc_id": "doc_029",
        "title": "Mean Reciprocal Rank (MRR) in Question-Answering Evaluation",
        "text": "MRR measures how early the first relevant document appears in the retrieved ranking. Reciprocal rank is 1/rank for the first relevant hit, and 0 if no relevant hit appears within top-k. In QA scenarios where a single comprehensive passage suffices to answer the question, MRR serves as a sensitive indicator of top-rank retrieval precision.",
        "source": "fastrag-docs/evaluation/mrr_metrics",
        "metadata": {"topic": "evaluation", "category": "metrics", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_030",
        "title": "Normalized Discounted Cumulative Gain (nDCG) for Graded Relevance",
        "text": "nDCG evaluates ranked retrieval lists using graded relevance judgments (e.g., 0=not relevant, 1=partially relevant, 2=highly relevant). Discounted Cumulative Gain (DCG) applies logarithmic decay to lower ranks: sum((2^rel - 1) / log2(rank + 1)). Dividing DCG by Ideal DCG (IDCG) normalizes the metric to [0.0, 1.0].",
        "source": "fastrag-docs/evaluation/ndcg_metrics",
        "metadata": {"topic": "evaluation", "category": "metrics", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_031",
        "title": "Recall@K and Candidate Funnel Sizing",
        "text": "Recall@K calculates the proportion of all relevant documents captured within the top-k retrieved results. In two-stage retrieval architectures, high Recall@K at stage one (e.g., k=50 or k=100) is critical because downstream rerankers cannot score relevant passages that failed to pass the initial retrieval filter.",
        "source": "fastrag-docs/evaluation/recall_metrics",
        "metadata": {"topic": "evaluation", "category": "metrics", "difficulty": "introductory"},
    },
    {
        "doc_id": "doc_032",
        "title": "RAG Triad Metrics: Faithfulness, Answer Relevance, and Context Relevance",
        "text": "Comprehensive RAG evaluation extends beyond IR metrics to assess the generation phase. Context Relevance verifies that retrieved passages are pertinent to the query. Faithfulness measures whether generated claims are strictly supported by context. Answer Relevance verifies that the response addresses the user question.",
        "source": "fastrag-docs/evaluation/rag_triad",
        "metadata": {"topic": "evaluation", "category": "metrics", "difficulty": "advanced"},
    },

    # Topic 9: Distractor & Negative Documents (doc_033 - doc_036)
    # These documents contain overlapping keywords (index, b-tree, sql, event loop, ecommerce, summarization)
    # but do NOT describe RAG architecture or retrieval mechanics!
    {
        "doc_id": "doc_033",
        "title": "Relational Database B-Tree Indexing and Query Optimization",
        "text": "Relational database engines use balanced trees (B-Trees and B+Trees) to index primary keys and secondary columns. B-Trees keep keys sorted, enabling logarithmic O(log N) point lookups and sequential range scans. Database query optimizers evaluate index cardinality, statistics, and selectivity before choosing between index scans and full table scans.",
        "source": "external/database_systems/btree_indexes",
        "metadata": {"topic": "database", "category": "distractor", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_034",
        "title": "Elasticsearch E-Commerce Catalog Product Search and Faceting",
        "text": "Online retail platforms configure Elasticsearch clusters with specialized analyzers for catalog searches. Products feature structured attributes like category, brand, color, and price. Queries combine multi-match full-text scoring with faceted aggregations, price-range filters, and inventory stock availability checks.",
        "source": "external/ecommerce/catalog_search",
        "metadata": {"topic": "ecommerce", "category": "distractor", "difficulty": "intermediate"},
    },
    {
        "doc_id": "doc_035",
        "title": "Standalone Text Summarization Without External Retrieval",
        "text": "Zero-shot document summarization feeds raw text directly into an instruction-tuned LLM without consulting external vector databases or retrieval indexes. Prompting strategies such as MapReduce or Refine break long transcripts into sections, asking the model to synthesize key themes and generate executive summaries solely from prompt contents.",
        "source": "external/nlp/standalone_summarization",
        "metadata": {"topic": "nlp", "category": "distractor", "difficulty": "introductory"},
    },
    {
        "doc_id": "doc_036",
        "title": "Hardware CPU Cache Hierarchies (L1, L2, L3) and Cache Lines",
        "text": "Modern multi-core processors implement multi-level hardware caches to bridge the CPU-DRAM latency gap. L1 caches reside closest to the execution units with sub-nanosecond access, followed by larger L2 and shared L3 caches. Memory is transferred in fixed 64-byte cache lines, and cache miss penalties cause processor pipeline stalls.",
        "source": "external/hardware/cpu_cache_lines",
        "metadata": {"topic": "hardware", "category": "distractor", "difficulty": "intermediate"},
    },
]


QUERIES = [
    # 1. Exact keyword queries
    {
        "query_id": "q_001",
        "query": "recursive character text chunking separator hierarchy",
        "category": "exact_keyword",
        "expected_behavior": "Should retrieve doc_001 explaining separator hierarchies (double newlines, single newlines, punctuation) for chunking.",
    },
    {
        "query_id": "q_002",
        "query": "BM25 parameter k1 term frequency saturation",
        "category": "exact_keyword",
        "expected_behavior": "Should retrieve doc_009 and doc_012 explaining BM25 term frequency saturation and tuning parameter k1.",
    },
    {
        "query_id": "q_003",
        "query": "Reciprocal Rank Fusion smoothing constant k formula",
        "category": "exact_keyword",
        "expected_behavior": "Should retrieve doc_014 and doc_015 detailing the RRF formula sum(1 / (k + rank)) and tuning constant k.",
    },
    {
        "query_id": "q_004",
        "query": "HNSW efConstruction and M hyperparameters",
        "category": "exact_keyword",
        "expected_behavior": "Should retrieve doc_007 explaining Hierarchical Navigable Small World graphs and M/efConstruction parameters.",
    },

    # 2. Semantic & paraphrased queries
    {
        "query_id": "q_005",
        "query": "How to avoid cutting sentences and paragraphs in half when splitting documents?",
        "category": "semantic_paraphrase",
        "expected_behavior": "Should retrieve doc_001 (recursive character chunking) and doc_002 (semantic chunking based on sentence distance).",
    },
    {
        "query_id": "q_006",
        "query": "Techniques for combining keyword search and semantic vector search in parallel to reduce latency",
        "category": "semantic_paraphrase",
        "expected_behavior": "Should retrieve doc_013 (concurrent retrieval dispatch with asyncio.gather) and doc_011 (lexical vs dense complementarity).",
    },
    {
        "query_id": "q_007",
        "query": "Why does full query-document self-attention cost more than vector dot product?",
        "category": "semantic_paraphrase",
        "expected_behavior": "Should retrieve doc_017 explaining CrossEncoder O(N*L^2) complexity vs bi-encoder dot product.",
    },
    {
        "query_id": "q_008",
        "query": "Methods to stop language models from making up facts by referencing original sources",
        "category": "semantic_paraphrase",
        "expected_behavior": "Should retrieve doc_023 (source provenance and citation tags) and doc_032 (faithfulness metrics).",
    },

    # 3. Technical distinction & comparison queries
    {
        "query_id": "q_009",
        "query": "Difference between bi-encoder and cross-encoder architectures",
        "category": "technical_distinction",
        "expected_behavior": "Should retrieve doc_005 (bi-encoder twin networks) and doc_017 (cross-encoder full cross-attention).",
    },
    {
        "query_id": "q_010",
        "query": "HNSW graph search versus IVF-PQ vector quantization trade-offs",
        "category": "technical_distinction",
        "expected_behavior": "Should retrieve doc_007 (HNSW multi-layer graph) and doc_008 (IVF-PQ Voronoi clustering and compression).",
    },
    {
        "query_id": "q_011",
        "query": "Difference between Mean Reciprocal Rank and Normalized Discounted Cumulative Gain",
        "category": "technical_distinction",
        "expected_behavior": "Should retrieve doc_029 (MRR for first relevant rank) and doc_030 (nDCG for graded multi-item relevance).",
    },
    {
        "query_id": "q_012",
        "query": "Level 1 query response cache vs Level 2 embedding vector cache in RAG",
        "category": "technical_distinction",
        "expected_behavior": "Should retrieve doc_025 explaining FastRAG two-tier caching architecture.",
    },

    # 4. Acronym and identifier queries
    {
        "query_id": "q_013",
        "query": "RRF ranking algorithm",
        "category": "acronym_lookup",
        "expected_behavior": "Should retrieve doc_014, doc_015, and doc_016 covering Reciprocal Rank Fusion principles.",
    },
    {
        "query_id": "q_014",
        "query": "TTFT latency reduction with Server-Sent Events",
        "category": "acronym_lookup",
        "expected_behavior": "Should retrieve doc_027 explaining Time to First Token and SSE streaming.",
    },
    {
        "query_id": "q_015",
        "query": "IVF-PQ nprobe parameter and Voronoi cells",
        "category": "acronym_lookup",
        "expected_behavior": "Should retrieve doc_008 detailing Inverted File with Product Quantization and nprobe scanning.",
    },
    {
        "query_id": "q_016",
        "query": "BPE token boundaries in text splitters",
        "category": "acronym_lookup",
        "expected_behavior": "Should retrieve doc_003 explaining Byte-Pair Encoding and tokenizer mismatches with character chunking.",
    },

    # 5. Pipeline optimization & operational queries
    {
        "query_id": "q_017",
        "query": "How to optimize two-stage retrieval with high recall followed by reranking?",
        "category": "pipeline_optimization",
        "expected_behavior": "Should retrieve doc_018 (two-stage funnel), doc_019 (batching), and doc_031 (Recall@K funnel sizing).",
    },
    {
        "query_id": "q_018",
        "query": "How to handle cache invalidation when knowledge documents are updated?",
        "category": "pipeline_optimization",
        "expected_behavior": "Should retrieve doc_026 explaining index mutation versioning and cache invalidation.",
    },
    {
        "query_id": "q_019",
        "query": "Preventing token budget overflow and lost-in-the-middle issues during prompt construction",
        "category": "pipeline_optimization",
        "expected_behavior": "Should retrieve doc_021 (token budgeting) and doc_022 (lost-in-the-middle context placement).",
    },
    {
        "query_id": "q_020",
        "query": "Why normalize vector embeddings to unit length L2 norm?",
        "category": "pipeline_optimization",
        "expected_behavior": "Should retrieve doc_006 explaining cosine similarity equivalence to dot product for unit vectors.",
    },

    # 6. Ambiguous & negative queries (testing discrimination against distractors)
    {
        "query_id": "q_021",
        "query": "B-Tree index vs vector index search performance",
        "category": "ambiguous_distinction",
        "expected_behavior": "Should retrieve doc_033 (B-Tree relational indexing) and doc_007 (HNSW vector indexing).",
    },
    {
        "query_id": "q_022",
        "query": "Hardware CPU cache lines L1 L2 vs software RAG query caching",
        "category": "ambiguous_distinction",
        "expected_behavior": "Should retrieve doc_036 (hardware CPU cache lines) and doc_025 (RAG query cache).",
    },
    {
        "query_id": "q_023",
        "query": "E-commerce faceted search with Elasticsearch",
        "category": "distractor_evaluation",
        "expected_behavior": "Should retrieve doc_034 (e-commerce catalog search). Must not mark RAG documents as relevant.",
    },
    {
        "query_id": "q_024",
        "query": "Document summarization without retrieval using direct LLM prompts",
        "category": "distractor_evaluation",
        "expected_behavior": "Should retrieve doc_035 (standalone summarization without retrieval). Must distinguish from RAG.",
    },
]


# Ground-truth relevance judgments (query_id, 0, doc_id, relevance)
# Scale: 2 = highly relevant, 1 = partially relevant, 0 = explicit non-relevant / distractor
QRELS = [
    # q_001: recursive character text chunking separator hierarchy
    ("q_001", "doc_001", 2),
    ("q_001", "doc_004", 1),
    ("q_001", "doc_003", 1),
    ("q_001", "doc_033", 0),

    # q_002: BM25 parameter k1 term frequency saturation
    ("q_002", "doc_009", 2),
    ("q_002", "doc_012", 2),
    ("q_002", "doc_010", 1),
    ("q_002", "doc_034", 0),

    # q_003: Reciprocal Rank Fusion smoothing constant k formula
    ("q_003", "doc_014", 2),
    ("q_003", "doc_015", 2),
    ("q_003", "doc_016", 1),
    ("q_003", "doc_007", 0),

    # q_004: HNSW efConstruction and M hyperparameters
    ("q_004", "doc_007", 2),
    ("q_004", "doc_008", 1),
    ("q_004", "doc_005", 1),
    ("q_004", "doc_033", 0),

    # q_005: How to avoid cutting sentences and paragraphs in half
    ("q_005", "doc_001", 2),
    ("q_005", "doc_002", 2),
    ("q_005", "doc_004", 1),
    ("q_005", "doc_035", 0),

    # q_006: Combining keyword search and semantic vector search in parallel
    ("q_006", "doc_013", 2),
    ("q_006", "doc_011", 2),
    ("q_006", "doc_014", 1),
    ("q_006", "doc_028", 1),
    ("q_006", "doc_034", 0),

    # q_007: Why does full query-document self-attention cost more than vector dot product?
    ("q_007", "doc_017", 2),
    ("q_007", "doc_005", 1),
    ("q_007", "doc_019", 1),
    ("q_007", "doc_006", 1),

    # q_008: Methods to stop language models from making up facts
    ("q_008", "doc_023", 2),
    ("q_008", "doc_032", 2),
    ("q_008", "doc_021", 1),
    ("q_008", "doc_035", 0),

    # q_009: Difference between bi-encoder and cross-encoder architectures
    ("q_009", "doc_017", 2),
    ("q_009", "doc_005", 2),
    ("q_009", "doc_018", 1),
    ("q_009", "doc_020", 1),

    # q_010: HNSW graph search versus IVF-PQ vector quantization trade-offs
    ("q_010", "doc_007", 2),
    ("q_010", "doc_008", 2),
    ("q_010", "doc_005", 1),
    ("q_010", "doc_033", 0),

    # q_011: Difference between Mean Reciprocal Rank and nDCG
    ("q_011", "doc_029", 2),
    ("q_011", "doc_030", 2),
    ("q_011", "doc_031", 1),

    # q_012: Level 1 query response cache vs Level 2 embedding vector cache
    ("q_012", "doc_025", 2),
    ("q_012", "doc_026", 1),
    ("q_012", "doc_036", 0),

    # q_013: RRF ranking algorithm
    ("q_013", "doc_014", 2),
    ("q_013", "doc_015", 2),
    ("q_013", "doc_016", 2),
    ("q_013", "doc_011", 1),

    # q_014: TTFT latency reduction with Server-Sent Events
    ("q_014", "doc_027", 2),
    ("q_014", "doc_028", 1),
    ("q_014", "doc_021", 1),

    # q_015: IVF-PQ nprobe parameter and Voronoi cells
    ("q_015", "doc_008", 2),
    ("q_015", "doc_007", 1),
    ("q_015", "doc_005", 1),

    # q_016: BPE token boundaries in text splitters
    ("q_016", "doc_003", 2),
    ("q_016", "doc_001", 1),
    ("q_016", "doc_004", 1),

    # q_017: How to optimize two-stage retrieval with high recall followed by reranking?
    ("q_017", "doc_018", 2),
    ("q_017", "doc_031", 2),
    ("q_017", "doc_019", 1),
    ("q_017", "doc_017", 1),
    ("q_017", "doc_020", 1),

    # q_018: How to handle cache invalidation when knowledge documents are updated?
    ("q_018", "doc_026", 2),
    ("q_018", "doc_025", 1),
    ("q_018", "doc_036", 0),

    # q_019: Preventing token budget overflow and lost-in-the-middle issues
    ("q_019", "doc_021", 2),
    ("q_019", "doc_022", 2),
    ("q_019", "doc_024", 1),

    # q_020: Why normalize vector embeddings to unit length L2 norm?
    ("q_020", "doc_006", 2),
    ("q_020", "doc_005", 1),
    ("q_020", "doc_007", 1),

    # q_021: B-Tree index vs vector index search performance
    ("q_021", "doc_033", 2),
    ("q_021", "doc_007", 2),
    ("q_021", "doc_008", 1),

    # q_022: Hardware CPU cache lines L1 L2 vs software RAG query caching
    ("q_022", "doc_036", 2),
    ("q_022", "doc_025", 2),
    ("q_022", "doc_026", 1),

    # q_023: E-commerce faceted search with Elasticsearch
    ("q_023", "doc_034", 2),
    ("q_023", "doc_010", 1),
    ("q_023", "doc_001", 0),
    ("q_023", "doc_005", 0),

    # q_024: Document summarization without retrieval using direct LLM prompts
    ("q_024", "doc_035", 2),
    ("q_024", "doc_021", 1),
    ("q_024", "doc_001", 0),
    ("q_024", "doc_005", 0),
]


def sha256_file(filepath: Path) -> str:
    """Compute the SHA-256 hexadecimal hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_dataset(output_dir: Path = DATASETS_DIR) -> dict:
    """Generate all dataset files deterministically and return manifest metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = output_dir / "corpus.jsonl"
    queries_file = output_dir / "queries.jsonl"
    qrels_file = output_dir / "qrels.tsv"
    manifest_file = output_dir / "dataset_manifest.json"

    # 1. Write corpus.jsonl
    with open(corpus_file, "w", encoding="utf-8") as f:
        for doc in DOCUMENTS:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # 2. Write queries.jsonl
    with open(queries_file, "w", encoding="utf-8") as f:
        for query in QUERIES:
            f.write(json.dumps(query, ensure_ascii=False) + "\n")

    # 3. Write qrels.tsv
    with open(qrels_file, "w", encoding="utf-8") as f:
        for query_id, doc_id, relevance in QRELS:
            f.write(f"{query_id}\t0\t{doc_id}\t{relevance}\n")

    # 4. Compute file stats and checksums
    file_metadata = {}
    for name, path in [
        ("corpus.jsonl", corpus_file),
        ("queries.jsonl", queries_file),
        ("qrels.tsv", qrels_file),
    ]:
        file_metadata[name] = {
            "relative_path": f"datasets/{name}",
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "line_count": sum(1 for _ in open(path, "r", encoding="utf-8")),
        }

    # 5. Build and write manifest
    manifest = {
        "dataset_name": "FastRAG Technical Evaluation Dataset",
        "version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Deterministic, self-contained technical evaluation dataset covering "
            "RAG architecture, chunking, dense/lexical retrieval, rank fusion, reranking, "
            "caching, and distractor concepts."
        ),
        "counts": {
            "documents": len(DOCUMENTS),
            "queries": len(QUERIES),
            "qrels": len(QRELS),
        },
        "relevance_scale": {
            "0": "not relevant (explicit negative / distractor)",
            "1": "partially relevant (mentions related principles or contextual background)",
            "2": "highly relevant (directly addresses the query topic)",
        },
        "files": file_metadata,
        "generation_notes": (
            "Generated deterministically by scripts/generate_dataset.py. "
            "Do not manually edit checksums or counts."
        ),
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    manifest = generate_dataset()
    print("Dataset generated successfully:")
    print(f"  - Documents: {manifest['counts']['documents']}")
    print(f"  - Queries:   {manifest['counts']['queries']}")
    print(f"  - Qrels:     {manifest['counts']['qrels']}")
    for fname, meta in manifest["files"].items():
        print(f"  - {fname}: {meta['sha256']} ({meta['size_bytes']} bytes)")
