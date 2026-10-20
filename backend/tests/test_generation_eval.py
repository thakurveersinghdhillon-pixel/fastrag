"""Unit tests for FastRAG Generation and Retrieval Quality Evaluation Harness."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.generation.generator import RAGGenerator
from app.generation.providers import MockLLMProvider
from app.retrieval.context_assembler import AssembledContext, ContextAssembler, ContextChunk
from app.retrieval.pipeline import RAGPipeline
from evaluation.generation_eval import (
    GenerationEvaluationReport,
    GenerationEvaluator,
    compute_citation_metrics,
    compute_evidence_completeness,
    compute_groundedness,
    compute_token_f1,
    detect_abstention,
    extract_cited_document_ids,
    normalize_answer,
    split_into_sentences,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_V2_DIR = PROJECT_ROOT / "datasets" / "v2"


def test_normalize_answer() -> None:
    raw = "The quick, brown FOX jumps over a lazy dog!"
    normalized = normalize_answer(raw)
    assert normalized == "quick brown fox jumps over lazy dog"


def test_compute_token_f1() -> None:
    # 1. Exact match
    res_exact = compute_token_f1("recursive character chunking", "Recursive Character Chunking")
    assert res_exact["f1"] == 1.0
    assert res_exact["precision"] == 1.0
    assert res_exact["recall"] == 1.0

    # 2. Partial match
    res_partial = compute_token_f1("recursive chunking text", "recursive character chunking")
    assert 0.0 < res_partial["f1"] < 1.0

    # 3. Disjoint match
    res_disjoint = compute_token_f1("apple orange", "banana grape")
    assert res_disjoint["f1"] == 0.0


def test_compute_evidence_completeness() -> None:
    pred = "Recursive chunking uses double newlines, single newlines, and spaces as separators with 10% to 20% overlap."
    evidence = [
        "hierarchy of separators such as double newlines, single newlines, punctuation, and spaces",
        "Typical chunk sizes range from 256 to 512 tokens with a 10% to 20% token overlap",
    ]
    completeness = compute_evidence_completeness(pred, evidence)
    assert completeness > 0.0


def test_compute_groundedness() -> None:
    context = "Recursive character chunking splits documents using a hierarchy of separators such as double newlines."
    grounded_pred = "Recursive character chunking splits documents using double newlines."
    ungrounded_pred = "Quantum teleportation utilizes entangled photon pairs across optical fibers."

    score_grounded = compute_groundedness(grounded_pred, context)
    assert score_grounded >= 0.8

    score_ungrounded = compute_groundedness(ungrounded_pred, context)
    assert score_ungrounded <= 0.2


def test_groundedness_abbreviation_eg() -> None:
    """Verify sentences with abbreviations like 'e.g.' are not fragmented and score 1.0."""
    context = (
        "In BM25, parameter b (typically 0.75) penalizes long documents by normalizing term frequency. "
        "Setting b=0.5 dampens excessive length penalties, preventing brief function signatures from drowning out guides."
    )
    # Answer containing abbreviation 'e.g.,' and citation markers
    prediction = (
        "In the BM25 formulation, parameter b penalizes long documents by normalizing term frequency [Doc-1]. "
        "Adjusting b controls length penalties (e.g., lowering it dampens excessive length penalties) [Doc-2]."
    )
    score = compute_groundedness(prediction, context)
    assert score == 1.0


def test_groundedness_code_identifier_asyncio_gather() -> None:
    """Verify code identifiers like 'asyncio.gather' inside backticks/prose are not fragmented."""
    context = (
        "By dispatching both retrievers concurrently via Python's asyncio.gather, "
        "total wall-clock retrieval time is reduced to max(T_dense, T_lexical) plus negligible overhead."
    )
    prediction = (
        "Python's `asyncio.gather` allows concurrent dispatch of both dense and BM25 retrievers, "
        "reducing total retrieval time to `max(T_dense, T_lexical)` [Doc-1]."
    )
    score = compute_groundedness(prediction, context)
    assert score == 1.0


def test_completeness_concise_answer_omitting_supplementary_facts() -> None:
    """Verify a concise answer providing the core requested fact receives complete score.

    The expected evidence contains supplementary background (chunk sizes from 256 to 512 tokens),
    but the answer directly and accurately provides the requested overlap range (10% to 20%).
    """
    evidence = [
        "Typical chunk sizes range from 256 to 512 tokens with a 10% to 20% token overlap"
    ]
    # Direct answer answering the question concisely
    pred_concise = "For recursive character text chunking, the typical recommended token overlap range is 10% to 20% [Doc-1]."
    score_concise = compute_evidence_completeness(pred_concise, evidence)
    assert score_concise == 1.0

    # Ultra-concise direct answer
    pred_ultra = "The recommended token overlap range is 10% to 20%."
    score_ultra = compute_evidence_completeness(pred_ultra, evidence)
    assert score_ultra == 1.0


def test_completeness_genuine_omission_of_required_fact() -> None:
    """Verify an answer that genuinely omits a required evidence fact receives a lower score."""
    evidence = [
        "pre-normalizing all document embeddings to unit Euclidean length (L2 norm = 1.0)",
        "simplifies computationally to a fast vector dot product",
    ]

    # Incomplete answer genuinely omitting the unit Euclidean normalization requirement
    pred_incomplete = "Cosine similarity simplifies computationally to a fast vector dot product."
    score_incomplete = compute_evidence_completeness(pred_incomplete, evidence)
    assert score_incomplete == 0.5

    # Complete answer covering both required evidence facts
    pred_complete = (
        "When document embeddings are pre-normalized to unit Euclidean length (L2 norm = 1.0), "
        "cosine similarity simplifies computationally to a fast vector dot product."
    )
    score_complete = compute_evidence_completeness(pred_complete, evidence)
    assert score_complete == 1.0


def test_detect_abstention() -> None:
    assert detect_abstention("The provided context does not contain sufficient information.")
    assert detect_abstention("I cannot answer this question based on the provided documents.")
    assert detect_abstention("This is not mentioned in the provided context.")
    assert detect_abstention("", context_insufficient_flag=True)
    assert not detect_abstention("Recursive chunking uses a 10% to 20% token overlap to preserve boundaries.")


def test_compute_citation_metrics() -> None:
    cited = ["doc_001", "doc_002"]
    source = ["doc_001"]
    relevant = {"doc_001", "doc_002"}

    scores = compute_citation_metrics(
        cited_doc_ids=cited,
        source_doc_ids=source,
        relevant_doc_ids=relevant,
    )
    # Both cited docs are relevant -> precision = 2/2 = 1.0
    assert scores["citation_precision"] == 1.0
    # doc_001 is cited -> recall = 1/1 = 1.0
    assert scores["citation_recall"] == 1.0


def test_extract_cited_document_ids() -> None:
    chunk1 = ContextChunk(
        chunk_id="doc_001_chk_000",
        document_id="doc_001",
        citation_id="[Doc-1]",
        rank=1,
        text="Sample text",
        source="test",
        approximate_tokens=10,
    )
    context = AssembledContext(
        context_text="[Doc-1] Doc: doc_001\nSample text",
        selected_chunks=[chunk1],
        selected_chunk_ids=["doc_001_chk_000"],
        selected_document_ids=["doc_001"],
        total_chunks_selected=1,
        total_chunks_considered=1,
        approximate_token_count=10,
        token_budget=1000,
        max_chunks_budget=5,
        budget_exceeded=False,
        truncation_applied=False,
        truncation_reasons=[],
        assembly_duration_ms=0.5,
    )

    answer_with_tags = "According to [Doc-1], text is segmented recursively. Also see doc_002."
    cited = extract_cited_document_ids(answer_with_tags, context)
    assert "doc_001" in cited
    assert "doc_002" in cited


@pytest.mark.asyncio
async def test_generation_evaluator_end_to_end(tmp_path: Path) -> None:
    """Verify GenerationEvaluator runs end-to-end with MockLLMProvider."""
    from app.retrieval.bm25 import BM25Retriever
    from app.retrieval.dense import DenseRetriever, MockEmbeddingEncoder

    # Build lightweight pipeline
    bm25 = BM25Retriever()
    bm25.load_index(PROJECT_ROOT / "datasets" / "indexes" / "bm25")

    dense = DenseRetriever(encoder=MockEmbeddingEncoder())
    dense.load_index(PROJECT_ROOT / "datasets" / "indexes" / "dense")

    provider = MockLLMProvider(model_name="mock-test-llm")
    generator = RAGGenerator(provider=provider)
    assembler = ContextAssembler()

    pipeline = RAGPipeline(
        retriever=bm25,
        context_assembler=assembler,
        generator=generator,
    )

    evaluator = GenerationEvaluator(
        pipeline=pipeline,
        dataset_dir=DATASETS_V2_DIR,
        results_dir=tmp_path,
    )

    # Load dev queries filtered by human_reviewed
    queries = evaluator.load_queries(split="dev", provenance_filter="human_reviewed")
    assert len(queries) == 35

    # Run on small subset (first 2 queries)
    single_res = await evaluator.evaluate_single_query(queries[0])
    assert single_res.query_id == queries[0]["query_id"]
    assert len(single_res.generated_answer) > 0
    assert single_res.latency_ms > 0.0

    # Run full report on eligible queries
    report = await evaluator.run_evaluation(
        experiment_id="test_run",
        split="dev",
        provenance_filter="human_reviewed",
        candidate_k=5,
        save_results=True,
    )
    assert isinstance(report, GenerationEvaluationReport)
    assert report.total_queries_evaluated == 35
    assert "token_f1" in report.summary_metrics
    assert "groundedness" in report.summary_metrics
    assert "false_abstention_rate" in report.summary_metrics
    assert report.false_abstention_rate >= 0.0
    assert report.false_abstention_count >= 0
    assert "human_reviewed" in report.provenance_metrics
    assert report.output_path is not None
    assert Path(report.output_path).is_file()

    # Verify summary table formatting renders without exception
    table = report.summary_table()
    assert "false_abstention_rate" in table
    assert "FastRAG Generation Quality Evaluation Report" in table


@pytest.mark.asyncio
async def test_false_abstention_detection_and_rate(tmp_path: Path) -> None:
    """Verify false_abstention_rate correctly counts abstentions on answerable queries."""
    from app.retrieval.bm25 import BM25Retriever

    bm25 = BM25Retriever()
    bm25.load_index(PROJECT_ROOT / "datasets" / "indexes" / "bm25")

    # Mock provider that answers if query is answerable_ok, but abstains on answerable_abstained
    abstention_msg = "The provided context does not contain sufficient information."
    provider = MockLLMProvider(
        fixed_responses={
            "answerable_abstained query": abstention_msg,
            "unanswerable query": abstention_msg,
        }
    )
    generator = RAGGenerator(provider=provider)
    pipeline = RAGPipeline(
        retriever=bm25,
        context_assembler=ContextAssembler(),
        generator=generator,
    )
    evaluator = GenerationEvaluator(pipeline=pipeline, dataset_dir=tmp_path, results_dir=tmp_path)

    # 2 answerable queries (1 answers, 1 false abstention)
    q1 = {
        "query_id": "q1",
        "query": "What is recursive text chunking?",
        "category": "factual_direct",
        "split": "dev",
        "reference_answer": "Recursive character chunking splits documents using separators.",
        "relevance_judgments": {"doc_001": 2},
    }
    q2 = {
        "query_id": "q2",
        "query": "answerable_abstained query",
        "category": "factual_direct",
        "split": "dev",
        "reference_answer": "Expected valid answer.",
        "relevance_judgments": {"doc_001": 2},
    }
    # 1 unanswerable query (true abstention)
    q3 = {
        "query_id": "q3",
        "query": "unanswerable query",
        "category": "unanswerable_negative",
        "split": "dev",
        "reference_answer": None,
        "relevance_judgments": {},
    }

    # Write test queries file
    queries_file = tmp_path / "queries_dev.jsonl"
    with open(queries_file, "w") as f:
        for q in [q1, q2, q3]:
            f.write(json.dumps(q) + "\n")

    report = await evaluator.run_evaluation(
        experiment_id="test_fa",
        split="dev",
        provenance_filter="all",
        save_results=False,
    )

    # Total answerable = 2. q1 answered, q2 falsely abstained -> FA rate = 1/2 = 0.5
    assert report.summary_metrics["false_abstention_rate"]["false_abstention_count"] == 1
    assert report.summary_metrics["false_abstention_rate"]["rate"] == 0.5
    assert report.false_abstention_rate == 0.5
    assert report.false_abstention_count == 1

    # Unanswerable metrics: 1 unanswerable evaluated, correctly abstained -> 1.0 accuracy
    assert report.unanswerable_metrics["total"] == 1
    assert report.unanswerable_metrics["abstained_count"] == 1
    assert report.unanswerable_metrics["abstention_accuracy"] == 1.0


@pytest.mark.asyncio
async def test_false_abstention_zero_denominator(tmp_path: Path) -> None:
    """Verify zero-denominator handling when no answerable queries exist."""
    from app.retrieval.bm25 import BM25Retriever

    bm25 = BM25Retriever()
    bm25.load_index(PROJECT_ROOT / "datasets" / "indexes" / "bm25")

    pipeline = RAGPipeline(
        retriever=bm25,
        context_assembler=ContextAssembler(),
        generator=RAGGenerator(provider=MockLLMProvider()),
    )
    evaluator = GenerationEvaluator(pipeline=pipeline, dataset_dir=tmp_path, results_dir=tmp_path)

    # Only unanswerable query
    q_unans = {
        "query_id": "q_unans",
        "query": "What is the secret formula?",
        "category": "unanswerable_negative",
        "split": "dev",
        "reference_answer": None,
        "relevance_judgments": {},
    }
    with open(tmp_path / "queries_dev.jsonl", "w") as f:
        f.write(json.dumps(q_unans) + "\n")

    report = await evaluator.run_evaluation(
        experiment_id="test_zero_denom",
        split="dev",
        provenance_filter="all",
        save_results=False,
    )

    # Zero answerables -> FA rate and count must be 0, no exception
    assert report.summary_metrics["false_abstention_rate"]["total_answerable"] == 0
    assert report.summary_metrics["false_abstention_rate"]["rate"] == 0.0
    assert report.summary_metrics["false_abstention_rate"]["false_abstention_count"] == 0
    assert report.false_abstention_rate == 0.0
    assert report.false_abstention_count == 0


@pytest.mark.asyncio
async def test_generation_evaluator_limit(tmp_path: Path) -> None:
    """Verify limit parameter properly bounds evaluation query counts."""
    from app.retrieval.bm25 import BM25Retriever

    bm25 = BM25Retriever()
    bm25.load_index(PROJECT_ROOT / "datasets" / "indexes" / "bm25")

    pipeline = RAGPipeline(
        retriever=bm25,
        context_assembler=ContextAssembler(),
        generator=RAGGenerator(provider=MockLLMProvider()),
    )
    evaluator = GenerationEvaluator(
        pipeline=pipeline,
        dataset_dir=DATASETS_V2_DIR,
        results_dir=tmp_path,
    )

    # 1. limit=2 evaluates exactly 2 queries
    report_2 = await evaluator.run_evaluation(
        experiment_id="test_limit_2",
        split="dev",
        provenance_filter="human_reviewed",
        candidate_k=5,
        save_results=False,
        limit=2,
    )
    assert report_2.total_queries_evaluated == 2
    assert report_2.eligible_query_count == 2
    assert len(report_2.per_query_results) == 2

    # 2. limit=1 evaluates exactly 1 query
    report_1 = await evaluator.run_evaluation(
        experiment_id="test_limit_1",
        split="dev",
        provenance_filter="human_reviewed",
        candidate_k=5,
        save_results=False,
        limit=1,
    )
    assert report_1.total_queries_evaluated == 1
    assert report_1.eligible_query_count == 1
    assert len(report_1.per_query_results) == 1

    # 3. limit=None retains full evaluation (35 queries)
    report_none = await evaluator.run_evaluation(
        experiment_id="test_limit_none",
        split="dev",
        provenance_filter="human_reviewed",
        candidate_k=5,
        save_results=False,
        limit=None,
    )
    assert report_none.total_queries_evaluated == 35
    assert report_none.eligible_query_count == 35
    assert len(report_none.per_query_results) == 35

    # 4. limit=0 raises ValueError
    with pytest.raises(ValueError, match="Limit must be an integer >= 1"):
        await evaluator.run_evaluation(
            experiment_id="test_limit_zero",
            split="dev",
            provenance_filter="human_reviewed",
            save_results=False,
            limit=0,
        )

    # 5. limit < 0 raises ValueError
    with pytest.raises(ValueError, match="Limit must be an integer >= 1"):
        await evaluator.run_evaluation(
            experiment_id="test_limit_neg",
            split="dev",
            provenance_filter="human_reviewed",
            save_results=False,
            limit=-1,
        )

    # 6. boolean limit raises ValueError
    with pytest.raises(ValueError, match="Limit must be an integer >= 1"):
        await evaluator.run_evaluation(
            experiment_id="test_limit_bool",
            split="dev",
            provenance_filter="human_reviewed",
            save_results=False,
            limit=True,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_generation_evaluator_category_filter(tmp_path: Path) -> None:
    """Verify category_filter loads only specific query categories and runs unanswerable evaluation."""
    from app.retrieval.bm25 import BM25Retriever

    bm25 = BM25Retriever()
    bm25.load_index(PROJECT_ROOT / "datasets" / "indexes" / "bm25")

    pipeline = RAGPipeline(
        retriever=bm25,
        context_assembler=ContextAssembler(min_relevance_score=10.0),
        generator=RAGGenerator(provider=MockLLMProvider()),
    )
    evaluator = GenerationEvaluator(
        pipeline=pipeline,
        dataset_dir=DATASETS_V2_DIR,
        results_dir=tmp_path,
    )

    # 1. Filter human_reviewed unanswerable queries
    unans_hr = evaluator.load_queries(
        split="dev",
        provenance_filter="human_reviewed",
        category_filter="unanswerable_negative",
    )
    assert len(unans_hr) == 2
    assert [q["query_id"] for q in unans_hr] == ["q2_dev_031", "q2_dev_032"]

    # 2. Filter all unanswerable queries
    unans_all = evaluator.load_queries(
        split="dev",
        provenance_filter="all",
        category_filter="unanswerable_negative",
    )
    assert len(unans_all) == 10

    # 3. Run evaluation on unanswerables with min_relevance_score threshold
    report = await evaluator.run_evaluation(
        experiment_id="test_unans_eval",
        split="dev",
        provenance_filter="human_reviewed",
        category_filter="unanswerable_negative",
        save_results=False,
    )
    assert report.total_queries_evaluated == 2
    assert report.unanswerable_metrics["total"] == 2
    assert report.unanswerable_metrics["abstained_count"] == 2
    assert report.unanswerable_metrics["abstention_accuracy"] == 1.0


@pytest.mark.asyncio
async def test_regression_successful_model_abstention(tmp_path: Path) -> None:
    """Verify that an unanswerable query with provided context where the LLM explicitly abstains
    is classified as abstention_type='model', correct_abstention=True, and is_error=False.
    """
    from app.retrieval.bm25 import BM25Retriever

    bm25 = BM25Retriever()
    bm25.load_index(PROJECT_ROOT / "datasets" / "indexes" / "bm25")

    # Provider returns explicit model abstention string despite having context
    mock_provider = MockLLMProvider(
        fixed_responses={
            "What is the token-level late interaction scoring mechanism used in ColBERT?": (
                "The provided context does not contain sufficient information to answer this question."
            )
        }
    )
    pipeline = RAGPipeline(
        retriever=bm25,
        context_assembler=ContextAssembler(),
        generator=RAGGenerator(provider=mock_provider),
    )
    evaluator = GenerationEvaluator(
        pipeline=pipeline,
        dataset_dir=DATASETS_V2_DIR,
        results_dir=tmp_path,
    )

    q_record = {
        "query_id": "q2_dev_032",
        "query": "What is the token-level late interaction scoring mechanism used in ColBERT?",
        "category": "unanswerable_negative",
        "split": "dev",
        "annotation_provenance": {"status": "human_reviewed"},
        "reference_answer": None,
        "source_doc_ids": [],
        "expected_evidence": [],
        "relevance_judgments": {},
    }

    result = await evaluator.evaluate_single_query(q_record)
    assert not result.is_error
    assert result.raw_error is None
    assert result.abstention_type == "model"
    assert result.correct_abstention is True
    assert result.is_false_abstention is False
    assert result.context_chunks_count > 0
    assert "provided context does not contain sufficient information" in result.generated_answer


@pytest.mark.asyncio
async def test_regression_successful_substantive_answer(tmp_path: Path) -> None:
    """Verify that an answerable query receiving a substantive grounded answer
    is classified as abstention_type='none', correct_abstention=True, is_error=False,
    and has valid quantitative metrics.
    """
    from app.retrieval.bm25 import BM25Retriever

    bm25 = BM25Retriever()
    bm25.load_index(PROJECT_ROOT / "datasets" / "indexes" / "bm25")

    q_text = "What is the recommended token overlap range for recursive character text chunking?"
    mock_provider = MockLLMProvider(
        fixed_responses={
            q_text: (
                "According to [Doc-1], recursive character chunking typically uses a 10% to 20% token overlap."
            )
        }
    )
    pipeline = RAGPipeline(
        retriever=bm25,
        context_assembler=ContextAssembler(),
        generator=RAGGenerator(provider=mock_provider),
    )
    evaluator = GenerationEvaluator(
        pipeline=pipeline,
        dataset_dir=DATASETS_V2_DIR,
        results_dir=tmp_path,
    )

    q_record = {
        "query_id": "q2_dev_001",
        "query": q_text,
        "category": "factual_direct",
        "split": "dev",
        "annotation_provenance": {"status": "human_reviewed"},
        "reference_answer": "Recommended token overlap ranges from 10% to 20%.",
        "source_doc_ids": ["doc_001"],
        "expected_evidence": ["10% to 20% token overlap"],
        "relevance_judgments": {"doc_001": 2},
    }

    result = await evaluator.evaluate_single_query(q_record)
    assert not result.is_error
    assert result.raw_error is None
    assert result.abstention_type == "none"
    assert result.correct_abstention is True
    assert result.is_false_abstention is False
    assert result.token_f1 is not None and result.token_f1 > 0.0
    assert result.completeness is not None and result.completeness > 0.0
    assert result.groundedness is not None and result.groundedness > 0.0


@pytest.mark.asyncio
async def test_regression_local_context_filter_abstention(tmp_path: Path) -> None:
    """Verify that an unanswerable query filtered out locally by the context assembler
    is classified as abstention_type='local_context_filter', correct_abstention=True,
    and returns INSUFFICIENT_CONTEXT_ANSWER without provider errors.
    """
    from app.retrieval.bm25 import BM25Retriever

    bm25 = BM25Retriever()
    bm25.load_index(PROJECT_ROOT / "datasets" / "indexes" / "bm25")

    pipeline = RAGPipeline(
        retriever=bm25,
        context_assembler=ContextAssembler(min_relevance_score=100.0),  # Filter out all candidates
        generator=RAGGenerator(provider=MockLLMProvider()),
    )
    evaluator = GenerationEvaluator(
        pipeline=pipeline,
        dataset_dir=DATASETS_V2_DIR,
        results_dir=tmp_path,
    )

    q_record = {
        "query_id": "q2_dev_031",
        "query": "How does GraphRAG extract knowledge graph entity triplets from source documents?",
        "category": "unanswerable_negative",
        "split": "dev",
        "annotation_provenance": {"status": "human_reviewed"},
        "reference_answer": None,
        "source_doc_ids": [],
        "expected_evidence": [],
        "relevance_judgments": {},
    }

    result = await evaluator.evaluate_single_query(q_record)
    assert not result.is_error
    assert result.raw_error is None
    assert result.abstention_type == "local_context_filter"
    assert result.correct_abstention is True
    assert result.is_false_abstention is False
    assert result.context_chunks_count == 0
    assert result.generated_answer == "I do not have sufficient information in the provided context to answer this question."


@pytest.mark.asyncio
async def test_regression_provider_503_error(tmp_path: Path) -> None:
    """Verify that a provider 503 HTTP failure is explicitly classified as an error,
    does NOT count as an abstention, has all quality metrics set to None,
    and unanswerable_metrics records error_count=1 and abstention_accuracy=0.0.
    """
    from app.core.exceptions import ProviderError
    from app.retrieval.bm25 import BM25Retriever

    bm25 = BM25Retriever()
    bm25.load_index(PROJECT_ROOT / "datasets" / "indexes" / "bm25")

    err_503 = ProviderError(
        "Provider returned error status 503: [{'error': {'code': 503, 'message': 'The model is overloaded.', 'status': 'UNAVAILABLE'}}]"
    )
    provider_failing = MockLLMProvider(simulate_error=err_503)

    pipeline = RAGPipeline(
        retriever=bm25,
        context_assembler=ContextAssembler(),
        generator=RAGGenerator(provider=provider_failing),
    )
    evaluator = GenerationEvaluator(
        pipeline=pipeline,
        dataset_dir=DATASETS_V2_DIR,
        results_dir=tmp_path,
    )

    q_record = {
        "query_id": "q2_dev_031",
        "query": "How does GraphRAG extract knowledge graph entity triplets from source documents?",
        "category": "unanswerable_negative",
        "split": "dev",
        "annotation_provenance": {"status": "human_reviewed"},
        "reference_answer": None,
        "source_doc_ids": [],
        "expected_evidence": [],
        "relevance_judgments": {},
    }

    # 1. Test single query evaluation
    result = await evaluator.evaluate_single_query(q_record)
    assert result.is_error is True
    assert result.raw_error is not None
    assert "503" in result.raw_error
    assert result.generated_answer == ""
    assert result.abstention_type == "none"
    assert result.correct_abstention is False
    assert result.is_false_abstention is False
    # All metrics must be None, not 1.0
    assert result.token_f1 is None
    assert result.token_precision is None
    assert result.token_recall is None
    assert result.completeness is None
    assert result.groundedness is None
    assert result.citation_precision is None
    assert result.citation_recall is None

    # 2. Test full run_evaluation with failing provider on unanswerable query
    queries_file = tmp_path / "queries_dev.jsonl"
    with open(queries_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(q_record) + "\n")

    evaluator_tmp = GenerationEvaluator(
        pipeline=pipeline,
        dataset_dir=tmp_path,
        results_dir=tmp_path,
    )
    report = await evaluator_tmp.run_evaluation(
        experiment_id="test_provider_503",
        split="dev",
        provenance_filter="all",
        save_results=False,
    )

    assert report.total_queries_evaluated == 1
    # Check unanswerable metrics
    assert report.unanswerable_metrics["total"] == 1
    assert report.unanswerable_metrics["error_count"] == 1
    assert report.unanswerable_metrics["valid_count"] == 0
    assert report.unanswerable_metrics["abstained_count"] == 0
    assert report.unanswerable_metrics["abstention_accuracy"] == 0.0

    # Summary metrics should report error_count and excluded metrics should have count=0
    assert report.summary_metrics["error_count"]["count"] == 1
    assert report.summary_metrics["groundedness"]["count"] == 0
    assert report.summary_metrics["token_f1"]["count"] == 0

    # Summary table should display errors without throwing
    table = report.summary_table()
    assert "Errors=1" in table
    assert "error_count" in table



