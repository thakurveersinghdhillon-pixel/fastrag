"""FastRAG Retrieval Package.

Provides dense semantic retrieval, BM25 lexical retrieval, concurrent hybrid retrieval,
Reciprocal Rank Fusion (RRF), shared result schemas, and index management utilities.
"""

from app.retrieval.base import (
    BaseRetriever,
    IndexInfo,
    RetrievalResult,
    compute_corpus_fingerprint,
    validate_top_k,
)
from app.retrieval.bm25 import BM25Retriever, default_bm25_tokenizer
from app.retrieval.dense import (
    DenseRetriever,
    EmbeddingEncoder,
    MockEmbeddingEncoder,
    SentenceTransformerEncoder,
)
from app.retrieval.context_assembler import (
    AssembledContext,
    ContextAssembler,
    ContextChunk,
    approximate_token_counter,
)
from app.retrieval.fusion import (
    DEFAULT_RRF_SMOOTHING_K,
    reciprocal_rank_fusion,
    validate_rrf_k,
)
from app.retrieval.hybrid import (
    BranchDiagnostics,
    HybridRetrievalDiagnostics,
    HybridRetrievalResponse,
    HybridRetriever,
)
from app.retrieval.pipeline import (
    PipelineDiagnostics,
    PipelineResponse,
    RAGPipeline,
    RetrievalPipeline,
)
from app.retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    CrossEncoderScorer,
    MockCrossEncoderScorer,
    RerankDiagnostics,
    SentenceTransformerCrossEncoder,
)

__all__ = [
    "AssembledContext",
    "BaseRetriever",
    "BM25Retriever",
    "BranchDiagnostics",
    "ContextAssembler",
    "ContextChunk",
    "CrossEncoderReranker",
    "CrossEncoderScorer",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_RRF_SMOOTHING_K",
    "DenseRetriever",
    "EmbeddingEncoder",
    "HybridRetrievalDiagnostics",
    "HybridRetrievalResponse",
    "HybridRetriever",
    "IndexInfo",
    "MockCrossEncoderScorer",
    "MockEmbeddingEncoder",
    "PipelineDiagnostics",
    "PipelineResponse",
    "RAGPipeline",
    "RerankDiagnostics",
    "RetrievalPipeline",
    "RetrievalResult",
    "SentenceTransformerCrossEncoder",
    "SentenceTransformerEncoder",
    "approximate_token_counter",
    "compute_corpus_fingerprint",
    "default_bm25_tokenizer",
    "reciprocal_rank_fusion",
    "validate_rrf_k",
    "validate_top_k",
]

