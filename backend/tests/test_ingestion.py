"""Comprehensive Tests for FastRAG Document Ingestion, Normalization, Chunking, and Pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from app.ingestion.chunking import Chunker, generate_chunk_id
from app.ingestion.loaders import (
    DocumentLoadingError,
    load_directory,
    load_documents,
    load_jsonl,
    load_text_file,
)
from app.ingestion.models import Chunk, Document, IngestionSummary
from app.ingestion.normalization import normalize_text
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.validation import (
    IngestionValidationError,
    validate_chunk,
    validate_chunks_batch,
    validate_document,
    validate_documents_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


# ==============================================================================
# 1. Loading & Document Model Tests
# ==============================================================================


def test_document_model_valid() -> None:
    """Verify Document model instantiation and dictionary serialization."""
    doc = Document(
        document_id="doc_test_1",
        title="Test Title",
        text="Sample document body.",
        source="test/source.md",
        metadata={"category": "test", "priority": 1},
    )
    assert doc.document_id == "doc_test_1"
    assert doc.character_count == len("Sample document body.")
    assert not doc.is_empty

    doc_dict = doc.to_dict()
    assert doc_dict["document_id"] == "doc_test_1"
    assert doc_dict["metadata"]["category"] == "test"

    reconstituted = Document.from_dict(doc_dict)
    assert reconstituted.document_id == doc.document_id
    assert reconstituted.text == doc.text


def test_document_model_supports_doc_id_alias() -> None:
    """Verify Document.from_dict accepts 'doc_id' as used in corpus.jsonl."""
    data = {"doc_id": "alias_01", "text": "Body text", "title": "Alias Title"}
    doc = Document.from_dict(data)
    assert doc.document_id == "alias_01"


def test_document_model_invalid_fields() -> None:
    """Verify Document model rejects empty ID, non-string text, and non-serializable metadata."""
    with pytest.raises(ValueError, match="document_id must be a non-empty string"):
        Document(document_id="", text="valid text")

    with pytest.raises(ValueError, match="document_id must be a non-empty string"):
        Document(document_id="   ", text="valid text")

    with pytest.raises(TypeError, match="Document text must be a string"):
        Document(document_id="doc_1", text=123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Document metadata must be JSON-serializable"):
        Document(document_id="doc_1", text="valid", metadata={"unserializable": object()})


def test_load_text_file(tmp_path: Path) -> None:
    """Verify loading single .txt file creates a valid Document."""
    txt_file = tmp_path / "architecture_notes.txt"
    txt_file.write_text("FastRAG architecture overview notes.", encoding="utf-8")

    doc = load_text_file(txt_file)
    assert doc.document_id == "architecture_notes"
    assert doc.title == "architecture_notes"
    assert doc.text == "FastRAG architecture overview notes."
    assert doc.source == str(txt_file)


def test_load_jsonl_valid(tmp_path: Path) -> None:
    """Verify loading documents from a valid JSONL file."""
    jsonl_file = tmp_path / "test_corpus.jsonl"
    records = [
        {"doc_id": "d1", "title": "Doc 1", "text": "Content 1", "source": "src1"},
        {"document_id": "d2", "title": "Doc 2", "text": "Content 2", "source": "src2"},
    ]
    jsonl_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    docs = load_jsonl(jsonl_file)
    assert len(docs) == 2
    assert docs[0].document_id == "d1"
    assert docs[1].document_id == "d2"


def test_load_jsonl_invalid_syntax(tmp_path: Path) -> None:
    """Verify DocumentLoadingError on invalid JSON syntax in JSONL."""
    jsonl_file = tmp_path / "corrupt.jsonl"
    jsonl_file.write_text('{"doc_id": "d1", "text": "ok"}\n{bad json}\n', encoding="utf-8")

    with pytest.raises(DocumentLoadingError, match="invalid JSON syntax"):
        load_jsonl(jsonl_file)


def test_load_jsonl_missing_required_fields(tmp_path: Path) -> None:
    """Verify DocumentLoadingError when record lacks document_id."""
    jsonl_file = tmp_path / "missing_id.jsonl"
    jsonl_file.write_text('{"title": "No ID", "text": "Missing id"}\n', encoding="utf-8")

    with pytest.raises(DocumentLoadingError, match="missing required 'document_id'"):
        load_jsonl(jsonl_file)


def test_load_directory_mixed_files(tmp_path: Path) -> None:
    """Verify loading all supported files (.jsonl and .txt) from a directory."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "doc1.txt").write_text("Content TXT 1", encoding="utf-8")
    (tmp_path / "sub" / "doc2.txt").write_text("Content TXT 2", encoding="utf-8")
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps({"doc_id": "j1", "text": "Content JSONL"}) + "\n", encoding="utf-8"
    )

    docs = load_directory(tmp_path, recursive=True)
    assert len(docs) == 3
    doc_ids = {d.document_id for d in docs}
    assert doc_ids == {"doc1", "doc2", "j1"}


def test_load_documents_dispatch(tmp_path: Path) -> None:
    """Verify load_documents dispatches appropriately to TXT, JSONL, or Directory."""
    t_file = tmp_path / "single.txt"
    t_file.write_text("Single text content", encoding="utf-8")
    loaded = load_documents(t_file)
    assert len(loaded) == 1
    assert loaded[0].document_id == "single"

    unsupported_file = tmp_path / "unsupported.pdf"
    unsupported_file.write_bytes(b"%PDF-1.4 dummy")
    with pytest.raises(ValueError, match="Unsupported file format '.pdf'"):
        load_documents(unsupported_file)


# ==============================================================================
# 2. Text Normalization Tests
# ==============================================================================


def test_normalization_line_endings() -> None:
    """Verify CRLF and CR are standardized to Unix LF (\n)."""
    text_crlf = "Line 1\r\nLine 2\rLine 3\nLine 4"
    normalized = normalize_text(text_crlf, preserve_paragraphs=True)
    assert "\r" not in normalized
    assert normalized == "Line 1\nLine 2\nLine 3\nLine 4"


def test_normalization_null_bytes() -> None:
    """Verify null bytes are stripped cleanly."""
    text_with_null = "Hello\x00 World\x00\x00!"
    assert normalize_text(text_with_null) == "Hello World!"


def test_normalization_whitespace_and_paragraphs() -> None:
    """Verify tab/space runs are collapsed while preserving double-newline paragraphs."""
    raw = "Paragraph 1 with   extra    spaces.\n\n\n\nParagraph 2 with\t\ttabs."
    normalized = normalize_text(raw, preserve_paragraphs=True)
    assert normalized == "Paragraph 1 with extra spaces.\n\nParagraph 2 with tabs."


def test_normalization_preserves_unicode_nfc() -> None:
    """Verify NFC canonical composition without modifying accented characters or emojis."""
    raw_unicode = "Caf\u0065\u0301 and résumé with emoji 🚀 and non-latin текст."
    normalized = normalize_text(raw_unicode)
    assert "Café" in normalized
    assert "résumé" in normalized
    assert "🚀" in normalized
    assert "текст" in normalized


def test_normalization_preserves_technical_identifiers_and_code() -> None:
    """Verify mathematical notation, code syntax, URLs, and casing are strictly preserved."""
    code_text = (
        "In FastRAG, asyncio.gather(dense_search(), bm25_search()) executes in O(N * L^2) time. "
        "See https://example.com/api?param=1&b=0.75 for details. Constant k1=1.2 and L2 norm = 1.0."
    )
    normalized = normalize_text(code_text)
    assert "asyncio.gather(dense_search(), bm25_search())" in normalized
    assert "O(N * L^2)" in normalized
    assert "https://example.com/api?param=1&b=0.75" in normalized
    assert "k1=1.2" in normalized
    assert "L2 norm = 1.0" in normalized


def test_normalization_empty_and_whitespace() -> None:
    """Verify empty or pure whitespace text returns empty string."""
    assert normalize_text("") == ""
    assert normalize_text("   \n\t\r\n   ") == ""


# ==============================================================================
# 3. Chunking & Offset Alignment Tests
# ==============================================================================


def test_chunking_text_shorter_than_chunk_size() -> None:
    """Verify document shorter than chunk_size produces exactly 1 chunk covering entire text."""
    chunker = Chunker(chunk_size=500, chunk_overlap=100)
    doc = Document(document_id="doc_short", text="Short document text.")
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_index == 0
    assert c.text == "Short document text."
    assert c.start_offset == 0
    assert c.end_offset == len("Short document text.")
    assert doc.text[c.start_offset : c.end_offset] == c.text


def test_chunking_text_exactly_at_chunk_size() -> None:
    """Verify text of length exactly equal to chunk_size produces 1 chunk."""
    exact_text = "A" * 200
    chunker = Chunker(chunk_size=200, chunk_overlap=50)
    doc = Document(document_id="doc_exact", text=exact_text)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == 200
    assert chunks[0].text == exact_text


def test_chunking_text_longer_than_chunk_size_with_exact_offsets() -> None:
    """Verify multi-chunk splitting adheres strictly to character offsets and boundary seeking."""
    p1 = "Paragraph 1 explains the bi-encoder architecture and vector representations in RAG systems."
    p2 = "Paragraph 2 details the BM25 lexical inverted index and probabilistic relevance scoring."
    p3 = "Paragraph 3 discusses reciprocal rank fusion and multi-stage candidate reranking algorithms."
    full_text = f"{p1}\n\n{p2}\n\n{p3}"

    chunker = Chunker(chunk_size=120, chunk_overlap=30, preserve_paragraphs=True)
    doc = Document(document_id="doc_multi", text=full_text)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 3

    # CRITICAL: Verify offset alignment invariant for every generated chunk
    for chunk in chunks:
        assert chunk.document_id == "doc_multi"
        assert chunk.text == full_text[chunk.start_offset : chunk.end_offset]
        assert chunk.character_count == len(chunk.text)
        assert chunk.character_count <= chunker.chunk_size

    # Verify sequential indices
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunking_overlap_shares_content() -> None:
    """Verify subsequent chunks overlap by approximately the configured overlap size."""
    text = (
        "First sentence discussing dense retrieval methods. "
        "Second sentence discussing lexical search techniques. "
        "Third sentence discussing reciprocal rank fusion. "
        "Fourth sentence discussing cross-encoder reranking."
    )
    chunker = Chunker(chunk_size=110, chunk_overlap=40, preserve_paragraphs=False)
    doc = Document(document_id="doc_overlap", text=text)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        c1 = chunks[i]
        c2 = chunks[i + 1]
        # Next chunk should start before the previous chunk ends
        assert c2.start_offset < c1.end_offset


def test_chunker_invalid_configuration() -> None:
    """Verify Chunker constructor rejects invalid chunk sizes and overlaps."""
    with pytest.raises(ValueError, match="chunk_size must be strictly positive"):
        Chunker(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_size must be strictly positive"):
        Chunker(chunk_size=-100)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        Chunker(chunk_size=500, chunk_overlap=-10)

    with pytest.raises(ValueError, match="chunk_overlap .* must be strictly smaller than chunk_size"):
        Chunker(chunk_size=500, chunk_overlap=500)

    with pytest.raises(ValueError, match="chunk_overlap .* must be strictly smaller than chunk_size"):
        Chunker(chunk_size=500, chunk_overlap=600)

    with pytest.raises(TypeError):
        Chunker(chunk_size="500")  # type: ignore[arg-type]


def test_chunking_empty_document_produces_no_chunks() -> None:
    """Verify empty document produces an empty chunk list."""
    chunker = Chunker(chunk_size=500, chunk_overlap=100)
    doc = Document(document_id="doc_empty", text="")
    assert chunker.chunk_document(doc) == []

    doc_ws = Document(document_id="doc_ws", text="   \n\n\t   ")
    assert chunker.chunk_document(doc_ws) == []


# ==============================================================================
# 4. Chunk Identifiers & Metadata Preservation
# ==============================================================================


def test_chunk_identifier_determinism() -> None:
    """Verify chunk_id generation is deterministic across repeated runs."""
    id1 = generate_chunk_id("doc_01", 0, 1000, 150, "Sample text content")
    id2 = generate_chunk_id("doc_01", 0, 1000, 150, "Sample text content")
    assert id1 == id2
    assert id1.startswith("doc_01#c0000_")


def test_chunk_identifier_changes_with_content_or_config() -> None:
    """Verify chunk_id changes when text or chunking hyperparameters change."""
    base_id = generate_chunk_id("doc_01", 0, 1000, 150, "Original text")
    diff_text = generate_chunk_id("doc_01", 0, 1000, 150, "Modified text")
    diff_size = generate_chunk_id("doc_01", 0, 800, 150, "Original text")
    diff_overlap = generate_chunk_id("doc_01", 0, 1000, 200, "Original text")
    diff_idx = generate_chunk_id("doc_01", 1, 1000, 150, "Original text")

    assert base_id != diff_text
    assert base_id != diff_size
    assert base_id != diff_overlap
    assert base_id != diff_idx


def test_chunk_preserves_metadata_and_source() -> None:
    """Verify Chunks inherit the parent document's source and metadata."""
    doc = Document(
        document_id="doc_meta",
        text="Sample document text.",
        source="docs/retrieval.md",
        metadata={"domain": "nlp", "version": 2},
    )
    chunker = Chunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].source == "docs/retrieval.md"
    assert chunks[0].metadata == {"domain": "nlp", "version": 2}


# ==============================================================================
# 5. Validation Module Tests
# ==============================================================================


def test_validate_document_errors() -> None:
    """Verify validate_document detects malformed document records."""
    # Valid doc has no errors
    valid_doc = Document(document_id="d1", text="valid")
    assert validate_document(valid_doc) == []

    # Batch duplicate checking
    d1_a = Document(document_id="dup_id", text="text 1")
    d1_b = Document(document_id="dup_id", text="text 2")
    errors, dups = validate_documents_batch([d1_a, d1_b])
    assert "dup_id" in dups
    assert any("duplicate document_id 'dup_id'" in err for err in errors)


def test_validate_chunk_offset_mismatch() -> None:
    """Verify validate_chunk detects when text slice doesn't match offsets."""
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        chunk_index=0,
        text="Expected chunk text",
        source="src",
        start_offset=0,
        end_offset=10,  # Incorrect end_offset
    )
    errors = validate_chunk(chunk, original_text="Different long original text...")
    assert any("Offset mismatch" in err for err in errors)


def test_validate_chunks_batch_duplicate_chunk_id() -> None:
    """Verify validate_chunks_batch detects duplicate chunk_ids."""
    c1 = Chunk("dup_cid", "d1", 0, "Chunk text 1", "src", 0, 12)
    c2 = Chunk("dup_cid", "d1", 1, "Chunk text 2", "src", 15, 27)
    errors = validate_chunks_batch([c1, c2])
    assert any("duplicate chunk_id 'dup_cid'" in err for err in errors)


# ==============================================================================
# 6. Pipeline & Atomic Export Tests
# ==============================================================================


def test_pipeline_end_to_end_fixture(tmp_path: Path) -> None:
    """Verify end-to-end processing of a document fixture with JSONL export."""
    input_file = tmp_path / "fixture.jsonl"
    output_file = tmp_path / "processed" / "chunks.jsonl"

    fixture_docs = [
        {
            "doc_id": "fix_01",
            "title": "Document One",
            "text": "First test document discussing dense embeddings.\r\nSecond paragraph discussing BM25.",
            "source": "tests/fix_01",
            "metadata": {"topic": "search"},
        },
        {
            "doc_id": "fix_02",
            "title": "Document Two",
            "text": "Second document discussing reciprocal rank fusion.",
            "source": "tests/fix_02",
            "metadata": {"topic": "reranking"},
        },
    ]
    input_file.write_text("\n".join(json.dumps(d) for d in fixture_docs) + "\n", encoding="utf-8")

    pipeline = IngestionPipeline(chunk_size=100, chunk_overlap=20)
    chunks, summary = pipeline.process_file(input_file, output_path=output_file)

    assert summary.total_documents_input == 2
    assert summary.documents_processed == 2
    assert summary.documents_rejected == 0
    assert summary.total_chunks_output == len(chunks)
    assert output_file.is_file()

    # Verify JSONL lines on disk
    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(chunks)
    first_chunk = json.loads(lines[0])
    assert first_chunk["document_id"] == "fix_01"
    assert "chunk_id" in first_chunk


def test_pipeline_error_policy_collect_vs_fail_fast() -> None:
    """Verify 'collect' records rejected documents while 'fail_fast' raises."""
    valid_doc = Document(document_id="good_doc", text="Valid text")
    # Duplicate document in batch
    dup_doc = Document(document_id="good_doc", text="Duplicate text")

    # Policy: collect
    pipeline_collect = IngestionPipeline(on_error="collect")
    chunks, summary = pipeline_collect.process_documents([valid_doc, dup_doc])
    assert summary.documents_processed == 1
    assert summary.documents_rejected == 1
    assert len(summary.errors) >= 1

    # Policy: fail_fast
    pipeline_fail_fast = IngestionPipeline(on_error="fail_fast")
    with pytest.raises(IngestionValidationError, match="duplicate document_id"):
        pipeline_fail_fast.process_documents([valid_doc, dup_doc])


def test_pipeline_preserves_phase2_evaluation_dataset() -> None:
    """Verify running the ingestion pipeline on corpus.jsonl processes 100% of documents

    and does NOT modify or corrupt the original evaluation dataset.
    """
    corpus_path = DATASETS_DIR / "corpus.jsonl"
    manifest_path = DATASETS_DIR / "dataset_manifest.json"

    # Pre-run: read manifest expected sha
    manifest_pre = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_corpus_sha = manifest_pre["files"]["corpus.jsonl"]["sha256"]

    pipeline = IngestionPipeline(chunk_size=1000, chunk_overlap=150)
    chunks, summary = pipeline.process_file(corpus_path)

    assert summary.total_documents_input == 36
    assert summary.documents_processed == 36
    assert summary.documents_rejected == 0
    assert summary.empty_documents == 0
    assert summary.total_chunks_output == 36

    # Post-run: verify corpus.jsonl SHA256 has not changed
    hasher = hashlib.sha256()
    with open(corpus_path, "rb") as f:
        while b := f.read(65536):
            hasher.update(b)
    actual_corpus_sha = hasher.hexdigest()

    assert actual_corpus_sha == expected_corpus_sha, "Original corpus.jsonl was modified!"
