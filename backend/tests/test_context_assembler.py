"""Tests for Context Assembly and Token Budgeting Module."""

import pytest
from app.retrieval.base import RetrievalResult
from app.retrieval.context_assembler import (
    AssembledContext,
    ContextAssembler,
    approximate_token_counter,
)


def _make_result(
    chunk_id: str,
    document_id: str,
    rank: int,
    text: str,
    score: float = 5.0,
    title: str = "",
    start_offset: int | None = None,
    end_offset: int | None = None,
    source: str = "dataset.jsonl",
) -> RetrievalResult:
    meta = {
        "title": title,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "rerank_score": score,
        "original_score": 0.45,
        "original_score_type": "rrf",
    }
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        rank=rank,
        score=score,
        score_type="cross_encoder",
        text=text,
        source=source,
        metadata=meta,
    )


def test_approximate_token_counter():
    """Verify approximate token counter behaves reasonably and monotonically."""
    assert approximate_token_counter("") == 0
    short_text = "HNSW graph search"
    tokens_short = approximate_token_counter(short_text)
    assert tokens_short > 0

    long_text = "HNSW graph search " * 20
    tokens_long = approximate_token_counter(long_text)
    assert tokens_long > tokens_short


def test_context_assembler_ordering_and_citations():
    """Verify that assembler preserves candidate rank ordering and assigns [Doc-N] citations."""
    r1 = _make_result("c1", "doc_A", 1, "First chunk text about vector search.", title="Vector Search")
    r2 = _make_result("c2", "doc_B", 2, "Second chunk text about BM25 ranking.", title="BM25 Ranking")
    r3 = _make_result("c3", "doc_C", 3, "Third chunk text about RRF.", title="RRF Guide")

    assembler = ContextAssembler(max_chunks=5, max_tokens=1000)
    context = assembler.assemble([r1, r2, r3])

    assert context.total_chunks_selected == 3
    assert context.selected_chunk_ids == ["c1", "c2", "c3"]
    assert context.selected_document_ids == ["doc_A", "doc_B", "doc_C"]

    # Verify citation numbers
    assert context.selected_chunks[0].citation_id == "[Doc-1]"
    assert context.selected_chunks[1].citation_id == "[Doc-2]"
    assert context.selected_chunks[2].citation_id == "[Doc-3]"

    # Verify context text has citations and document headers
    assert "[Doc-1] Title: Vector Search (Doc: doc_A" in context.context_text
    assert "[Doc-2] Title: BM25 Ranking (Doc: doc_B" in context.context_text
    assert "[Doc-3] Title: RRF Guide (Doc: doc_C" in context.context_text


def test_context_assembler_token_budget_compliance():
    """Verify that candidate selection strictly respects maximum token limits."""
    # Create chunks each taking ~30 approximate tokens with distinct text
    results = [
        _make_result(f"c_{i}", f"doc_{i}", i + 1, f"Dense vector indexing section {i} with HNSW and Voronoi quantization cells. " * 3)
        for i in range(10)
    ]

    # Set budget that fits only ~2 chunks
    tokens_per_chunk = approximate_token_counter(f"[Doc-1] Doc: doc_0\n{results[0].text}")
    budget = tokens_per_chunk * 2 + 10

    assembler = ContextAssembler(max_tokens=budget, max_chunks=10)
    context = assembler.assemble(results)

    assert context.total_chunks_selected <= 3
    assert context.approximate_token_count <= budget
    assert context.budget_exceeded is True
    assert context.truncation_applied is True
    assert any("Token budget" in r for r in context.truncation_reasons)


def test_context_assembler_max_chunks_limit():
    """Verify that selection stops once max_chunks is reached."""
    results = [
        _make_result(f"c_{i}", f"doc_{i}", i + 1, f"Short chunk text {i}")
        for i in range(10)
    ]
    assembler = ContextAssembler(max_chunks=3, max_tokens=2000)
    context = assembler.assemble(results)

    assert context.total_chunks_selected == 3
    assert context.selected_chunk_ids == ["c_0", "c_1", "c_2"]
    assert any("Max chunks limit" in r for r in context.truncation_reasons)


def test_context_assembler_deduplicates_by_id_and_content():
    """Verify that identical chunk IDs and identical text passages are eliminated."""
    r1 = _make_result("c1", "doc_A", 1, "Unique passage one.")
    r2 = _make_result("c1", "doc_A", 2, "Duplicate chunk_id with slightly altered text.")
    r3 = _make_result("c2", "doc_B", 3, "Unique passage one.")  # Exact duplicate text of r1
    r4 = _make_result("c3", "doc_C", 4, "Unique passage three.")

    assembler = ContextAssembler(max_chunks=5, max_tokens=1000)
    context = assembler.assemble([r1, r2, r3, r4])

    assert context.total_chunks_selected == 2
    assert context.selected_chunk_ids == ["c1", "c3"]


def test_context_assembler_empty_input():
    """Verify that empty candidate list produces empty structured context without errors."""
    assembler = ContextAssembler()
    context = assembler.assemble([])

    assert context.context_text == ""
    assert context.total_chunks_selected == 0
    assert context.total_chunks_considered == 0
    assert context.approximate_token_count == 0
    assert context.selected_chunks == []
    assert context.truncation_reasons == []


def test_context_assembler_missing_optional_metadata():
    """Verify that missing metadata (no title, no offsets) does not crash or fabricate data."""
    # Chunk with bare minimum fields
    raw_cand = RetrievalResult(
        chunk_id="bare_c1",
        document_id="bare_d1",
        rank=1,
        score=0.95,
        score_type="cross_encoder",
        text="Minimal text without extra metadata.",
        source="dataset.jsonl",
        metadata={},  # Empty metadata
    )
    assembler = ContextAssembler()
    context = assembler.assemble([raw_cand])

    assert context.total_chunks_selected == 1
    chunk = context.selected_chunks[0]
    assert chunk.title == ""
    assert chunk.start_offset is None
    assert chunk.end_offset is None
    assert chunk.citation_id == "[Doc-1]"
    # Document and source preserved
    assert chunk.document_id == "bare_d1"
    assert chunk.source == "dataset.jsonl"


def test_context_assembler_oversized_first_chunk():
    """Verify handling when the very first chunk exceeds token budget."""
    huge_text = "Extremely long technical specification section. " * 50
    huge_cand = _make_result("huge_1", "doc_huge", 1, huge_text)

    # Very small budget (e.g. 50 tokens)
    assembler = ContextAssembler(max_tokens=50)
    context = assembler.assemble([huge_cand])

    # First chunk should be included alone with explicit budget_exceeded flag
    assert context.total_chunks_selected == 1
    assert context.budget_exceeded is True
    assert context.truncation_applied is True
    assert any("exceeds total budget" in r for r in context.truncation_reasons)


def test_context_assembler_source_document_diversity():
    """Verify source document diversity constraint (max_chunks_per_document)."""
    r1 = _make_result("c1", "doc_A", 1, "Doc A chunk 1 text.")
    r2 = _make_result("c2", "doc_A", 2, "Doc A chunk 2 text.")
    r3 = _make_result("c3", "doc_A", 3, "Doc A chunk 3 text.")
    r4 = _make_result("c4", "doc_B", 4, "Doc B chunk 1 text.")

    # Limit to 1 chunk per document
    assembler = ContextAssembler(max_chunks=4, max_chunks_per_document=1)
    context = assembler.assemble([r1, r2, r3, r4])

    assert context.total_chunks_selected == 2
    assert context.selected_chunk_ids == ["c1", "c4"]
    assert context.selected_document_ids == ["doc_A", "doc_B"]


def test_context_assembler_deterministic_output():
    """Verify that multiple assembly runs on identical input produce bit-for-bit identical output."""
    r1 = _make_result("c1", "doc_A", 1, "Passage A", title="Title A", score=9.1)
    r2 = _make_result("c2", "doc_B", 2, "Passage B", title="Title B", score=7.4)

    assembler = ContextAssembler(max_tokens=500, max_chunks=3)
    c_out1 = assembler.assemble([r1, r2])
    c_out2 = assembler.assemble([r1, r2])

    assert c_out1.context_text == c_out2.context_text
    assert c_out1.approximate_token_count == c_out2.approximate_token_count
    assert c_out1.selected_chunk_ids == c_out2.selected_chunk_ids


# -------------------------------------------------------------------------
# Phase 9.4 Tests: Relevance Thresholding & Boundary Ordering
# -------------------------------------------------------------------------

def test_relevance_threshold_filtering():
    """Verify that candidate chunks below min_relevance_score are filtered out."""
    r1 = _make_result("c1", "doc_A", 1, "Highly relevant passage about caching.", score=8.5)
    r2 = _make_result("c2", "doc_B", 2, "Marginally relevant passage.", score=0.5)
    r3 = _make_result("c3", "doc_C", 3, "Irrelevant noise passage.", score=-3.2)
    r4 = _make_result("c4", "doc_D", 4, "Deep negative noise passage.", score=-7.8)

    # Filter with threshold 0.0
    assembler = ContextAssembler(min_relevance_score=0.0)
    context = assembler.assemble([r1, r2, r3, r4])

    assert context.total_chunks_selected == 2
    assert context.selected_chunk_ids == ["c1", "c2"]
    assert context.context_insufficient is False
    assert any("below min_relevance_score" in r for r in context.truncation_reasons)


def test_relevance_threshold_boundary_exact_match():
    """Verify that candidates with score exactly equal to the threshold are retained."""
    r1 = _make_result("c1", "doc_A", 1, "Threshold boundary chunk.", score=0.0)
    r2 = _make_result("c2", "doc_B", 2, "Below threshold chunk.", score=-0.001)

    assembler = ContextAssembler()
    context = assembler.assemble([r1, r2], min_relevance_score=0.0)

    assert context.total_chunks_selected == 1
    assert context.selected_chunk_ids == ["c1"]
    assert context.selected_chunks[0].rerank_score == 0.0


def test_relevance_threshold_all_filtered_sets_insufficient():
    """Verify that when all candidates fall below threshold, context_insufficient is True."""
    r1 = _make_result("c1", "doc_A", 1, "Irrelevant negative candidate 1.", score=-6.5)
    r2 = _make_result("c2", "doc_B", 2, "Irrelevant negative candidate 2.", score=-8.2)

    assembler = ContextAssembler(min_relevance_score=-1.0)
    context = assembler.assemble([r1, r2])

    assert context.total_chunks_selected == 0
    assert context.selected_chunks == []
    assert context.context_text == ""
    assert context.context_insufficient is True
    assert any("fell below min_relevance_score" in r for r in context.truncation_reasons)


def test_relevance_threshold_none_preserves_backward_compatibility():
    """Verify that when threshold is None, all candidates are evaluated normally."""
    r1 = _make_result("c1", "doc_A", 1, "Candidate with negative logit.", score=-5.0)
    r2 = _make_result("c2", "doc_B", 2, "Candidate with positive logit.", score=4.0)

    assembler = ContextAssembler(min_relevance_score=None)
    context = assembler.assemble([r1, r2])

    assert context.total_chunks_selected == 2
    assert context.selected_chunk_ids == ["c1", "c2"]
    assert context.context_insufficient is False


def test_boundary_ordering_zero_and_single_candidate():
    """Verify boundary ordering for 0 and 1 chunk cases."""
    assembler = ContextAssembler(ordering_strategy="boundary")

    # Zero chunks
    ctx0 = assembler.assemble([])
    assert ctx0.selected_chunks == []
    assert ctx0.context_insufficient is True

    # Single chunk
    r1 = _make_result("c1", "doc_A", 1, "Single chunk text.", score=7.0)
    ctx1 = assembler.assemble([r1])
    assert ctx1.selected_chunk_ids == ["c1"]
    assert ctx1.selected_chunks[0].rank == 1
    assert ctx1.context_insufficient is False


def test_boundary_ordering_two_candidates():
    """Verify boundary ordering with 2 chunks keeps rank 1 first and rank 2 second."""
    r1 = _make_result("c1", "doc_A", 1, "Rank 1 chunk.", score=9.0)
    r2 = _make_result("c2", "doc_B", 2, "Rank 2 chunk.", score=8.0)

    assembler = ContextAssembler(ordering_strategy="boundary")
    context = assembler.assemble([r1, r2])

    assert context.selected_chunk_ids == ["c1", "c2"]
    assert context.selected_chunks[0].rank == 1
    assert context.selected_chunks[1].rank == 2


def test_boundary_ordering_three_and_multiple_candidates():
    """Verify boundary ordering: rank 1 first, rank 2 last, remaining in between."""
    r1 = _make_result("c1", "doc_A", 1, "Rank 1 text.", score=10.0)
    r2 = _make_result("c2", "doc_B", 2, "Rank 2 text.", score=8.0)
    r3 = _make_result("c3", "doc_C", 3, "Rank 3 text.", score=6.0)
    r4 = _make_result("c4", "doc_D", 4, "Rank 4 text.", score=4.0)
    r5 = _make_result("c5", "doc_E", 5, "Rank 5 text.", score=2.0)

    # 3 candidates: [c1, c2, c3] -> [c1, c3, c2]
    assembler_boundary = ContextAssembler(ordering_strategy="boundary")
    ctx3 = assembler_boundary.assemble([r1, r2, r3])
    assert ctx3.selected_chunk_ids == ["c1", "c3", "c2"]
    assert ctx3.selected_chunks[0].rank == 1
    assert ctx3.selected_chunks[1].rank == 3
    assert ctx3.selected_chunks[2].rank == 2

    # 5 candidates: [c1, c2, c3, c4, c5] -> [c1, c3, c4, c5, c2]
    ctx5 = assembler_boundary.assemble([r1, r2, r3, r4, r5])
    assert ctx5.selected_chunk_ids == ["c1", "c3", "c4", "c5", "c2"]
    assert ctx5.selected_chunks[0].rank == 1
    assert ctx5.selected_chunks[1].rank == 3
    assert ctx5.selected_chunks[2].rank == 4
    assert ctx5.selected_chunks[3].rank == 5
    assert ctx5.selected_chunks[4].rank == 2

    # Original scores preserved without modification
    assert ctx5.selected_chunks[0].rerank_score == 10.0
    assert ctx5.selected_chunks[-1].rerank_score == 8.0

    # Text contains rank 1 first, rank 2 last
    assert ctx5.context_text.startswith("[Doc-1]")
    assert "[Doc-2]" in ctx5.context_text.split("\n\n")[-1]


def test_invalid_ordering_strategy_validation():
    """Verify that an invalid ordering strategy raises ValueError."""
    with pytest.raises(ValueError, match="ordering_strategy"):
        ContextAssembler(ordering_strategy="random_shuffle")

    assembler = ContextAssembler()
    with pytest.raises(ValueError, match="ordering_strategy"):
        assembler.assemble([_make_result("c1", "doc1", 1, "test")], ordering_strategy="invalid")

