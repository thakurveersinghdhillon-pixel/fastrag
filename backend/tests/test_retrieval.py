"""Tests for FastRAG Retrieval Module (Phase 4).

Covers:
- BaseRetriever validation (top_k, empty queries, search without index)
- BM25Retriever (tokenization, indexing, deterministic tie-breaking, persistence roundtrip)
- DenseRetriever (exact cosine similarity, MockEmbeddingEncoder, deterministic tie-breaking, persistence roundtrip)
- Retriever Evaluation Adapter (latency metrics, EvaluationReport bridge)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from app.ingestion.models import Chunk
from app.retrieval.base import BaseRetriever, IndexInfo, RetrievalResult, compute_corpus_fingerprint, validate_top_k
from app.retrieval.bm25 import BM25Retriever, default_bm25_tokenizer
from app.retrieval.dense import DenseRetriever, MockEmbeddingEncoder
from evaluation.retriever_adapter import RetrievalLatencyStats, evaluate_retriever


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="chunk_001_0000",
            document_id="doc_001",
            chunk_index=0,
            text="FastRAG provides high-throughput retrieval-augmented generation and asynchronous caching.",
            source="test_source",
            start_offset=0,
            end_offset=89,
            metadata={"title": "FastRAG Architecture", "domain": "systems"},
        ),
        Chunk(
            chunk_id="chunk_002_0000",
            document_id="doc_002",
            chunk_index=0,
            text="BM25 lexical search ranks documents based on Robertson-Spärck Jones Okapi term frequency and IDF.",
            source="test_source",
            start_offset=0,
            end_offset=98,
            metadata={"title": "BM25 Scoring", "domain": "ir"},
        ),
        Chunk(
            chunk_id="chunk_003_0000",
            document_id="doc_003",
            chunk_index=0,
            text="Dense semantic retrieval maps queries and documents into continuous vector embeddings using cosine similarity.",
            source="test_source",
            start_offset=0,
            end_offset=111,
            metadata={"title": "Dense Retrieval", "domain": "vector_search"},
        ),
    ]


# ==========================================
# BaseRetriever & Utilities Tests
# ==========================================

def test_validate_top_k() -> None:
    validate_top_k(1)
    validate_top_k(10)
    with pytest.raises(ValueError, match="top_k must be strictly positive"):
        validate_top_k(0)
    with pytest.raises(ValueError, match="top_k must be strictly positive"):
        validate_top_k(-5)
    with pytest.raises(TypeError, match="top_k must be an integer"):
        validate_top_k("10")  # type: ignore[arg-type]


def test_corpus_fingerprint(sample_chunks: list[Chunk]) -> None:
    fp1 = compute_corpus_fingerprint(sample_chunks)
    assert len(fp1) == 64
    # Determinism check
    fp2 = compute_corpus_fingerprint(sample_chunks)
    assert fp1 == fp2


def test_bm25_tokenizer() -> None:
    tokens = default_bm25_tokenizer("FastRAG v1.2.3 uses fast-rag & cache_layer!")
    assert "fastrag" in tokens
    assert "v1.2.3" in tokens
    assert "fast-rag" in tokens
    assert "cache_layer" in tokens


# ==========================================
# BM25 Retriever Tests
# ==========================================

def test_bm25_search_without_index() -> None:
    retriever = BM25Retriever()
    assert retriever.search("test query") == []


def test_bm25_build_and_search(sample_chunks: list[Chunk]) -> None:
    retriever = BM25Retriever()
    info = retriever.build_index(sample_chunks)

    assert info.chunk_count == 3
    assert info.retriever_type == "bm25"
    assert retriever.is_indexed

    # Exact term match query
    results = retriever.search("BM25 lexical search Okapi", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "chunk_002_0000"
    assert results[0].document_id == "doc_002"
    assert results[0].rank == 1
    assert results[0].score > 0.0
    assert results[0].metadata["domain"] == "ir"


def test_bm25_empty_query(sample_chunks: list[Chunk]) -> None:
    retriever = BM25Retriever()
    retriever.build_index(sample_chunks)

    # Empty string or whitespace
    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_bm25_tie_breaking() -> None:
    """When chunks have identical scores, sorting must deterministically break ties by chunk_id."""
    chunks = [
        Chunk(
            chunk_id="chunk_z_000",
            document_id="doc_z",
            chunk_index=0,
            text="Shared identical text token pattern",
            source="s",
            start_offset=0,
            end_offset=35,
        ),
        Chunk(
            chunk_id="chunk_a_000",
            document_id="doc_a",
            chunk_index=0,
            text="Shared identical text token pattern",
            source="s",
            start_offset=0,
            end_offset=35,
        ),
    ]
    retriever = BM25Retriever()
    retriever.build_index(chunks)

    results = retriever.search("Shared identical", top_k=2)
    assert len(results) == 2
    # chunk_a_000 should come before chunk_z_000 due to alphabetical tie-breaking
    assert results[0].chunk_id == "chunk_a_000"
    assert results[1].chunk_id == "chunk_z_000"


def test_bm25_persistence_roundtrip(sample_chunks: list[Chunk], tmp_path: Path) -> None:
    retriever = BM25Retriever(k1=1.2, b=0.8)
    original_info = retriever.build_index(sample_chunks)
    original_results = retriever.search("FastRAG caching", top_k=3)

    index_dir = tmp_path / "bm25_index"
    retriever.save_index(index_dir)

    assert (index_dir / "bm25_index.json").exists()

    # Load into fresh retriever
    loaded_retriever = BM25Retriever()
    loaded_info = loaded_retriever.load_index(index_dir)

    assert loaded_info.chunk_count == original_info.chunk_count
    assert loaded_info.corpus_fingerprint == original_info.corpus_fingerprint
    assert loaded_info.hyperparameters["k1"] == 1.2
    assert loaded_info.hyperparameters["b"] == 0.8

    loaded_results = loaded_retriever.search("FastRAG caching", top_k=3)
    assert len(loaded_results) == len(original_results)
    for orig, loaded in zip(original_results, loaded_results):
        assert orig.chunk_id == loaded.chunk_id
        assert orig.rank == loaded.rank
        assert pytest.approx(orig.score, 1e-5) == loaded.score


# ==========================================
# Dense Retriever Tests
# ==========================================

def test_dense_search_without_index() -> None:
    retriever = DenseRetriever(encoder=MockEmbeddingEncoder())
    assert retriever.search("test query") == []


def test_dense_mock_build_and_search(sample_chunks: list[Chunk]) -> None:
    encoder = MockEmbeddingEncoder(dimension=64)
    retriever = DenseRetriever(encoder=encoder)
    info = retriever.build_index(sample_chunks)

    assert info.chunk_count == 3
    assert info.retriever_type == "dense"
    assert info.hyperparameters["dimension"] == 64
    assert retriever.is_indexed

    results = retriever.search("FastRAG architecture", top_k=2)
    assert len(results) == 2
    assert results[0].rank == 1
    assert -1.0 <= results[0].score <= 1.0001
    assert results[0].chunk_id in [c.chunk_id for c in sample_chunks]


def test_dense_empty_query(sample_chunks: list[Chunk]) -> None:
    retriever = DenseRetriever(encoder=MockEmbeddingEncoder())
    retriever.build_index(sample_chunks)

    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_dense_cosine_exact_match() -> None:
    """Test with controlled custom vectors to verify exact cosine dot product."""
    from app.retrieval.dense import EmbeddingEncoder

    class ControlledEncoder(EmbeddingEncoder):
        @property
        def model_name(self) -> str:
            return "controlled"

        @property
        def dimension(self) -> int:
            return 3

        def encode(self, texts: Sequence[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
            vecs = []
            for t in texts:
                if "first" in t:
                    vecs.append([1.0, 0.0, 0.0])
                elif "second" in t:
                    vecs.append([0.0, 1.0, 0.0])
                else:
                    vecs.append([0.0, 0.0, 1.0])
            return np.array(vecs, dtype=np.float32)

        def encode_query(self, query: str) -> np.ndarray:
            return self.encode([query])[0]

    chunks = [
        Chunk(chunk_id="c1", document_id="d1", chunk_index=0, text="first item", source="s", start_offset=0, end_offset=10),
        Chunk(chunk_id="c2", document_id="d2", chunk_index=0, text="second item", source="s", start_offset=0, end_offset=11),
    ]

    retriever = DenseRetriever(encoder=ControlledEncoder())
    retriever.build_index(chunks)

    results = retriever.search("query for first item", top_k=2)
    assert results[0].chunk_id == "c1"
    assert pytest.approx(results[0].score, 1e-5) == 1.0
    assert results[1].chunk_id == "c2"
    assert pytest.approx(results[1].score, 1e-5) == 0.0


def test_dense_tie_breaking() -> None:
    """When chunks have identical embeddings, ranking breaks ties deterministically on chunk_id."""
    from app.retrieval.dense import EmbeddingEncoder

    class ConstantEncoder(EmbeddingEncoder):
        @property
        def model_name(self) -> str:
            return "constant"

        @property
        def dimension(self) -> int:
            return 2

        def encode(self, texts: Sequence[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
            return np.ones((len(texts), 2), dtype=np.float32) / np.sqrt(2.0, dtype=np.float32)

        def encode_query(self, query: str) -> np.ndarray:
            return np.ones(2, dtype=np.float32) / np.sqrt(2.0, dtype=np.float32)

    chunks = [
        Chunk(chunk_id="chunk_z", document_id="dz", chunk_index=0, text="text z", source="s", start_offset=0, end_offset=6),
        Chunk(chunk_id="chunk_a", document_id="da", chunk_index=0, text="text a", source="s", start_offset=0, end_offset=6),
    ]
    retriever = DenseRetriever(encoder=ConstantEncoder())
    retriever.build_index(chunks)

    results = retriever.search("any query", top_k=2)
    assert results[0].chunk_id == "chunk_a"
    assert results[1].chunk_id == "chunk_z"


def test_dense_persistence_roundtrip(sample_chunks: list[Chunk], tmp_path: Path) -> None:
    encoder = MockEmbeddingEncoder(dimension=32)
    retriever = DenseRetriever(encoder=encoder)
    original_info = retriever.build_index(sample_chunks)
    original_results = retriever.search("FastRAG continuous vector", top_k=3)

    index_dir = tmp_path / "dense_index"
    retriever.save_index(index_dir)

    assert (index_dir / "dense_meta.json").exists()
    assert (index_dir / "dense_vectors.npz").exists()

    loaded_retriever = DenseRetriever(encoder=MockEmbeddingEncoder(dimension=32))
    loaded_info = loaded_retriever.load_index(index_dir)

    assert loaded_info.chunk_count == original_info.chunk_count
    assert loaded_info.corpus_fingerprint == original_info.corpus_fingerprint
    assert loaded_info.hyperparameters["dimension"] == 32

    loaded_results = loaded_retriever.search("FastRAG continuous vector", top_k=3)
    assert len(loaded_results) == len(original_results)
    for orig, loaded in zip(original_results, loaded_results):
        assert orig.chunk_id == loaded.chunk_id
        assert orig.rank == loaded.rank
        assert pytest.approx(orig.score, 1e-5) == loaded.score


# ==========================================
# Evaluation Adapter Integration Tests
# ==========================================

def test_evaluate_retriever_adapter(sample_chunks: list[Chunk], tmp_path: Path) -> None:
    # Create temporary queries and qrels
    queries_file = tmp_path / "test_queries.jsonl"
    qrels_file = tmp_path / "test_qrels.tsv"

    queries_data = [
        {"query_id": "q1", "query": "FastRAG architecture", "category": "architecture"},
        {"query_id": "q2", "query": "BM25 scoring formula", "category": "lexical"},
    ]
    with open(queries_file, "w", encoding="utf-8") as f:
        for q in queries_data:
            f.write(json.dumps(q) + "\n")

    with open(qrels_file, "w", encoding="utf-8") as f:
        f.write("query_id\titeration\tdoc_id\trelevance\n")
        f.write("q1\t0\tdoc_001\t2\n")
        f.write("q2\t0\tdoc_002\t2\n")

    bm25 = BM25Retriever()
    bm25.build_index(sample_chunks)

    report, latency_stats = evaluate_retriever(
        retriever=bm25,
        queries_path=queries_file,
        qrels_path=qrels_file,
        experiment_id="test_adapter_exp",
        top_k=5,
        metrics=["recall@5", "mrr@10"],
        save_results=False,
    )

    assert report.num_queries == 2
    assert "recall@5" in report.aggregate_metrics
    assert "mrr@10" in report.aggregate_metrics
    assert latency_stats.cold_start_ms >= 0.0
    assert latency_stats.warm_mean_ms >= 0.0
    assert latency_stats.total_duration_seconds > 0.0


def test_dense_index_artifact_fingerprints(sample_chunks: list[Chunk], tmp_path: Path) -> None:
    """Verify that dense index artifacts (vectors and metadata) produce non-null, valid SHA256 fingerprints matching disk bytes."""
    import hashlib
    from scripts.run_retrieval_eval_v2 import compute_dense_index_fingerprint, compute_file_sha256

    encoder = MockEmbeddingEncoder(dimension=16)
    retriever = DenseRetriever(encoder=encoder)
    retriever.build_index(sample_chunks)

    index_dir = tmp_path / "dense_index"
    retriever.save_index(index_dir)

    # 1. Verify expected files exist on disk
    vectors_file = index_dir / "dense_vectors.npz"
    meta_file = index_dir / "dense_meta.json"
    assert vectors_file.is_file()
    assert meta_file.is_file()

    # 2. Compute fingerprints using evaluation helper
    vectors_sha = compute_file_sha256(vectors_file)
    meta_sha = compute_file_sha256(meta_file)
    dir_fps = compute_dense_index_fingerprint(index_dir)

    # 3. Assert non-null and valid 64-character SHA256
    assert vectors_sha is not None
    assert meta_sha is not None
    assert len(vectors_sha) == 64
    assert len(meta_sha) == 64
    assert dir_fps["dense_vectors_sha256"] == vectors_sha
    assert dir_fps["dense_meta_sha256"] == meta_sha

    # 4. Verify exact byte-level match against direct hashlib
    expected_vectors_sha = hashlib.sha256(vectors_file.read_bytes()).hexdigest()
    expected_meta_sha = hashlib.sha256(meta_file.read_bytes()).hexdigest()
    assert vectors_sha == expected_vectors_sha
    assert meta_sha == expected_meta_sha

    # 5. Verify nonexistent artifact returns None (safeguard against ghost file dense_index.npy)
    assert compute_file_sha256(index_dir / "dense_index.npy") is None

    # 6. Verify real deployed benchmark index has valid non-null fingerprints matching disk bytes
    deployed_dense_dir = Path(__file__).resolve().parent.parent.parent / "datasets" / "indexes" / "dense"
    if deployed_dense_dir.exists():
        deployed_fps = compute_dense_index_fingerprint(deployed_dense_dir)
        assert deployed_fps["dense_vectors_sha256"] is not None
        assert deployed_fps["dense_meta_sha256"] is not None
        assert deployed_fps["dense_vectors_sha256"] == hashlib.sha256((deployed_dense_dir / "dense_vectors.npz").read_bytes()).hexdigest()
        assert deployed_fps["dense_meta_sha256"] == hashlib.sha256((deployed_dense_dir / "dense_meta.json").read_bytes()).hexdigest()


def test_retrieval_eval_v2_cli_args(tmp_path: Path) -> None:
    """Verify that run_retrieval_eval_v2 parse_args supports --query-ids and --diagnostic-output."""
    from scripts.run_retrieval_eval_v2 import parse_args

    # 1. Default invocation
    args_default = parse_args([])
    assert args_default.query_ids is None
    assert args_default.diagnostic_output is None
    assert args_default.split == "dev"
    assert args_default.provenance == "human_reviewed"
    assert args_default.top_k == 10
    assert args_default.candidate_k == 20
    assert args_default.blending_alpha == 0.0

    # 2. Targeted query filter and diagnostic output
    diag_file = tmp_path / "diagnostics.json"
    args_diag = parse_args([
        "--query-ids", "q2_dev_003,q2_dev_010",
        "--diagnostic-output", str(diag_file),
        "--blending-alpha", "0.5",
    ])
    assert args_diag.query_ids == "q2_dev_003,q2_dev_010"
    assert args_diag.diagnostic_output == diag_file
    assert args_diag.blending_alpha == 0.5


