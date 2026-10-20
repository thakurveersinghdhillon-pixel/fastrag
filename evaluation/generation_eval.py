"""FastRAG Generation and Retrieval Quality Evaluation Harness.

Provides reproducible, deterministic evaluation of grounded answer generation:
- Provenance eligibility filtering (e.g. human_reviewed only)
- Answer correctness against verified reference answers (token-level Precision, Recall, F1)
- Answer completeness against expected evidence spans
- Context groundedness / faithfulness (attestation against retrieved context)
- Citation precision and recall (traceability against source documents)
- Unanswerable query abstention detection
- Uncertainty quantification (standard deviation and 95% confidence intervals)
- Zero-leakage support for dev split experimentation and held-out test evaluation
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import string
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from app.core.logging_config import get_logger
from app.generation.generator import INSUFFICIENT_CONTEXT_ANSWER
from app.generation.models import GenerationResponse
from app.retrieval.context_assembler import AssembledContext
from app.retrieval.pipeline import PipelineResponse, RAGPipeline

logger = get_logger("fastrag.evaluation.generation")


# ---------------------------------------------------------------------------
# Text Normalization & Deterministic Metric Calculations
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Normalize text for deterministic token-level evaluation.

    Lowercases, removes punctuation, removes articles ('a', 'an', 'the'),
    and collapses redundant whitespace.
    """
    if not text:
        return ""

    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def remove_punctuation(t: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    return white_space_fix(remove_articles(remove_punctuation(text.lower())))


def compute_token_f1(prediction: str, ground_truth: str | None) -> dict[str, float]:
    """Compute token-level Precision, Recall, and F1 score against reference answer.

    Returns:
        dict with keys: "precision", "recall", "f1"
    """
    if ground_truth is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1

    for tok in pred_tokens:
        if gold_counts.get(tok, 0) > 0:
            common[tok] = common.get(tok, 0) + 1
            gold_counts[tok] -= 1

    num_same = sum(common.values())
    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


EVAL_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
})

_PROTECTED_DOT = "\uE000"


def _stem_token(token: str) -> str:
    """Normalize common English inflectional suffixes for consistent lexical matching."""
    t = token.lower()
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s", "ment", "tion", "al", "ly"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 3:
            return t[:-len(suffix)]
    return t


def extract_evaluation_content_tokens(text: str) -> set[str]:
    """Extract content tokens, preserving digits, technical identifiers, and stripping citation tags."""
    if not text:
        return set()

    # Remove citation markers like [Doc-1] or [doc_001]
    cleaned = re.sub(r"\[Doc-\d+\]", " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bdoc_\d{3}\b", " ", cleaned, flags=re.IGNORECASE)

    # Extract alphanumeric tokens, including dotted names (e.g. asyncio.gather) and numbers (e.g. 1.0)
    raw_tokens = re.findall(r"[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*", cleaned.lower())
    return set(
        _stem_token(tok)
        for tok in raw_tokens
        if tok not in EVAL_STOPWORDS and len(tok) >= 1
    )


def split_into_sentences(text: str) -> list[str]:
    """Segment text into individual sentences while preserving abbreviations, decimals, and code.

    Avoids false sentence boundaries on:
    - Code identifiers and method calls (e.g. 'asyncio.gather', 'model.predict')
    - Decimal numbers (e.g. '1.0', '0.75')
    - Common abbreviations (e.g. 'e.g.', 'i.e.', 'etc.', 'et al.', 'vs.')
    - Numbered list item markers (e.g. '1. ', '2. ')
    - Inline code expressions in backticks
    """
    if not text or not text.strip():
        return []

    # 1. Protect code spans enclosed in backticks
    def _mask_code(m: re.Match[str]) -> str:
        return m.group(0).replace(".", _PROTECTED_DOT)

    t = re.sub(r"`[^`]+`", _mask_code, text)

    # 2. Protect dots between word/digit characters (e.g. asyncio.gather, 1.0)
    t = re.sub(r"(?<=[a-zA-Z0-9_])\.(?=[a-zA-Z0-9_])", _PROTECTED_DOT, t)

    # 3. Protect common Latin and English abbreviations
    t = re.sub(r"\b([eE])\.([gG])\.", rf"\g<1>{_PROTECTED_DOT}\g<2>{_PROTECTED_DOT}", t)
    t = re.sub(r"\b([iI])\.([eE])\.", rf"\g<1>{_PROTECTED_DOT}\g<2>{_PROTECTED_DOT}", t)
    t = re.sub(
        r"\b(etc|al|vs|fig|approx|no|vol|dr|mr|mrs|ms|prof|dept)\.",
        rf"\g<1>{_PROTECTED_DOT}",
        t,
        flags=re.IGNORECASE,
    )

    # 4. Protect list markers at the beginning of a line or string (e.g. "1. ")
    t = re.sub(r"(^|[\r\n])(\s*\d+)\.", rf"\g<1>\g<2>{_PROTECTED_DOT}", t)

    # 5. Split on true sentence terminators (periods/exclamations/questions followed by whitespace or line breaks)
    raw = re.split(r"(?:[.!?]+(?:\s+|$)|[\r\n]+)", t)

    # 6. Restore protected dots and filter out empty / trivial fragments
    sentences = []
    for s in raw:
        restored = s.replace(_PROTECTED_DOT, ".").strip()
        if len(restored) > 10:
            sentences.append(restored)

    return sentences


def compute_evidence_completeness(
    prediction: str,
    expected_evidence: Sequence[str],
    threshold: float = 0.40,
) -> float:
    """Measure fraction of expected evidence concepts covered in generated answer.

    Evaluates token recall against expected evidence spans while preserving numerical
    quantities, technical identifiers, and handling parenthetical supplementary clauses.

    Args:
        prediction: The generated model answer.
        expected_evidence: List of expected evidence statements or spans.
        threshold: Minimum content-token recall required to consider an evidence span covered.
            Default is 0.40 (40%) to accommodate concise answers that provide target facts
            without repeating verbose sentence framing from source documents.
    """
    if not expected_evidence:
        return 1.0

    pred_tokens = extract_evaluation_content_tokens(prediction)
    if not pred_tokens:
        return 0.0

    covered = 0
    for span in expected_evidence:
        span_tokens = extract_evaluation_content_tokens(span)
        if not span_tokens:
            covered += 1
            continue

        # 1. Full span token recall
        hits = len(span_tokens & pred_tokens)
        if hits / len(span_tokens) >= threshold:
            covered += 1
            continue

        # 2. Check core span with parenthetical details removed (e.g. "(typically 0.75)")
        core_span = re.sub(r"\([^)]*\)", " ", span)
        core_tokens = extract_evaluation_content_tokens(core_span)
        if core_tokens and (len(core_tokens & pred_tokens) / len(core_tokens) >= threshold):
            covered += 1

    return round(covered / len(expected_evidence), 4)


def compute_groundedness(
    prediction: str,
    context_text: str,
    sentence_threshold: float = 0.50,
) -> float:
    """Measure fraction of generated answer sentences grounded in retrieved context.

    Uses robust sentence segmentation that preserves code identifiers (e.g. 'asyncio.gather'),
    decimals (e.g. '1.0'), and abbreviations (e.g. 'e.g.', 'i.e.'). A sentence is considered
    grounded if at least `sentence_threshold` (default 50%) of its content tokens appear in
    the retrieved context text.
    """
    if not prediction.strip():
        return 1.0
    if not context_text.strip():
        return 0.0

    context_tokens = extract_evaluation_content_tokens(context_text)
    sentences = split_into_sentences(prediction)
    if not sentences:
        return 1.0

    grounded_count = 0
    for sentence in sentences:
        s_tokens = extract_evaluation_content_tokens(sentence)
        if not s_tokens:
            grounded_count += 1
            continue
        hits = len(s_tokens & context_tokens)
        if hits / len(s_tokens) >= sentence_threshold:
            grounded_count += 1

    return round(grounded_count / len(sentences), 4)


ABSTENTION_PATTERNS = [
    r"not\s+enough\s+information",
    r"does\s+not\s+contain",
    r"insufficient\s+(information|context|evidence)",
    r"cannot\s+(be\s+)?answer",
    r"not\s+mentioned\s+in\s+the\s+(provided\s+)?context",
    r"corpus\s+does\s+not\s+contain",
    r"no\s+relevant\s+information",
    r"not\s+supported\s+by\s+the\s+context",
]


def detect_abstention(answer: str, context_insufficient_flag: bool = False) -> bool:
    """Determine if response correctly indicates lack of sufficient context."""
    if context_insufficient_flag:
        return True
    if not answer or not answer.strip():
        return False

    text_lower = answer.lower()
    for pat in ABSTENTION_PATTERNS:
        if re.search(pat, text_lower):
            return True
    return False


def extract_cited_document_ids(
    answer: str,
    context: AssembledContext | None = None,
    explicit_cited_docs: Sequence[str] | None = None,
) -> list[str]:
    """Extract and resolve document IDs cited in the answer text or response object."""
    resolved: list[str] = []

    # 1. Use explicit cited doc IDs from response if available
    if explicit_cited_docs:
        resolved.extend(explicit_cited_docs)

    # 2. Cross-reference [Doc-N] markers against assembled context
    if context is not None and context.selected_chunks:
        citation_map = {c.citation_id: c.document_id for c in context.selected_chunks}
        found_tags = re.findall(r"\[Doc-\d+\]", answer)
        for tag in found_tags:
            if tag in citation_map:
                resolved.append(citation_map[tag])

    # 3. Direct document references e.g. [doc_001] or doc_001
    direct_docs = re.findall(r"\bdoc_\d{3}\b", answer.lower())
    resolved.extend(direct_docs)

    # Deduplicate while preserving order
    return list(dict.fromkeys(resolved))


def compute_citation_metrics(
    cited_doc_ids: Sequence[str],
    source_doc_ids: Sequence[str],
    relevant_doc_ids: set[str],
) -> dict[str, float]:
    """Calculate citation precision and recall."""
    cited_set = set(cited_doc_ids)
    source_set = set(source_doc_ids)

    # Citation Recall: what fraction of ground truth source docs were cited?
    if not source_set:
        recall = 1.0 if not cited_set else 0.0
    else:
        hits = len(cited_set & source_set)
        recall = hits / len(source_set)

    # Citation Precision: what fraction of cited documents are genuinely relevant?
    if not cited_set:
        precision = 1.0 if not source_set else 0.0
    else:
        valid_citations = len(cited_set & relevant_doc_ids)
        precision = valid_citations / len(cited_set)

    return {
        "citation_precision": round(precision, 4),
        "citation_recall": round(recall, 4),
    }


# ---------------------------------------------------------------------------
# Data Models for Generation Evaluation
# ---------------------------------------------------------------------------

@dataclass
class QueryGenerationEvaluation:
    """Evaluation result for an individual evaluation query."""

    query_id: str
    query: str
    category: str
    split: str
    provenance_status: str
    reference_answer: str | None
    generated_answer: str
    source_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    cited_doc_ids: list[str]
    is_unanswerable: bool
    correct_abstention: bool
    token_f1: float | None
    token_precision: float | None
    token_recall: float | None
    completeness: float | None
    groundedness: float | None
    citation_precision: float | None
    citation_recall: float | None
    latency_ms: float
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    context_chunks_count: int = 0
    raw_error: str | None = None
    is_false_abstention: bool = False
    is_error: bool = False
    abstention_type: str = "none"  # "none" | "model" | "local_context_filter"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricSummaryStats:
    """Statistical summary including mean, std, count, and 95% confidence interval."""

    mean: float
    std: float
    min: float
    max: float
    count: int
    ci95_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "count": self.count,
            "ci95_margin": round(self.ci95_margin, 4),
            "ci95_range": [round(self.mean - self.ci95_margin, 4), round(self.mean + self.ci95_margin, 4)],
        }


def compute_metric_stats(values: Sequence[float | None]) -> MetricSummaryStats:
    """Compute aggregate statistics and 95% confidence intervals, filtering out None values."""
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return MetricSummaryStats(mean=0.0, std=0.0, min=0.0, max=0.0, count=0, ci95_margin=0.0)

    n = len(valid_values)
    mean_v = sum(valid_values) / n
    var = sum((x - mean_v) ** 2 for x in valid_values) / (n - 1) if n > 1 else 0.0
    std_v = math.sqrt(var)
    # 95% Confidence Interval z = 1.96
    ci95 = (1.96 * std_v / math.sqrt(n)) if n > 1 else 0.0

    return MetricSummaryStats(
        mean=mean_v,
        std=std_v,
        min=min(valid_values),
        max=max(valid_values),
        count=n,
        ci95_margin=ci95,
    )


@dataclass
class GenerationEvaluationReport:
    """Comprehensive generation evaluation report."""

    experiment_id: str
    timestamp_utc: str
    dataset_version: str
    split: str
    provenance_filter: str
    provider_type: str
    model_identifier: str
    retrieval_configuration: dict[str, Any]
    eligible_query_count: int
    total_queries_evaluated: int
    summary_metrics: dict[str, dict[str, Any]]
    category_metrics: dict[str, dict[str, Any]]
    unanswerable_metrics: dict[str, Any]
    per_query_results: list[dict[str, Any]]
    output_path: str | None = None
    provenance_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def false_abstention_rate(self) -> float:
        if "false_abstention_rate" in self.summary_metrics:
            m = self.summary_metrics["false_abstention_rate"]
            return m.get("rate", m.get("mean", 0.0))
        return 0.0

    @property
    def false_abstention_count(self) -> int:
        if "false_abstention_count" in self.summary_metrics:
            m = self.summary_metrics["false_abstention_count"]
            return m.get("count", 0)
        if "false_abstention_rate" in self.summary_metrics:
            m = self.summary_metrics["false_abstention_rate"]
            return m.get("false_abstention_count", 0)
        return 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary_table(self) -> str:
        """Produce clean ASCII summary table of generation metrics."""
        lines = [
            "=" * 70,
            f" FastRAG Generation Quality Evaluation Report: {self.experiment_id}",
            f" Timestamp: {self.timestamp_utc} | Split: {self.split.upper()} | Filter: {self.provenance_filter}",
            f" Provider: {self.provider_type} ({self.model_identifier})",
            f" Evaluated Queries: {self.total_queries_evaluated} (Eligible: {self.eligible_query_count})",
            "=" * 70,
            f" {'Metric':<24} | {'Mean':<8} | {'Std':<8} | {'95% CI':<16} | {'Count':<5}",
            "-" * 70,
        ]
        for m_name, stats in self.summary_metrics.items():
            if isinstance(stats, dict) and "ci95_margin" in stats and "mean" in stats:
                ci_str = f"±{stats['ci95_margin']:.4f}"
                std_str = f"{stats['std']:.4f}" if isinstance(stats['std'], (int, float)) else str(stats['std'])
                lines.append(
                    f" {m_name:<24} | {stats['mean']:<8.4f} | {std_str:<8} | {ci_str:<16} | {stats['count']:<5}"
                )
            elif isinstance(stats, dict) and "rate" in stats:
                lines.append(
                    f" {m_name:<24} | {stats['rate']:<8.4f} | {'N/A':<8} | {'N/A':<16} | {stats.get('total_answerable', stats.get('count', 0)):<5}"
                )
            elif isinstance(stats, dict) and "count" in stats:
                total_cnt = stats.get("total_queries", stats.get("total_answerable", 0))
                lines.append(
                    f" {m_name:<24} | {stats['count']:<8} | {'N/A':<8} | {'N/A':<16} | {total_cnt:<5}"
                )
            else:
                lines.append(f" {m_name:<24} | {str(stats):<8}")
        lines.append("=" * 70)
        lines.append(" Category Performance (F1 / Groundedness / Citation Precision / Citation Recall):")
        for cat, cstats in sorted(self.category_metrics.items()):
            f1 = cstats.get("token_f1", {}).get("mean", 0.0)
            ground = cstats.get("groundedness", {}).get("mean", 0.0)
            cp = cstats.get("citation_precision", {}).get("mean", 0.0)
            cr = cstats.get("citation_recall", {}).get("mean", 0.0)
            cnt = cstats.get("count", 0)
            errs = cstats.get("error_count", 0)
            err_str = f" | Errs={errs}" if errs > 0 else ""
            lines.append(f"   * {cat:<24} (N={cnt:<2}{err_str}): F1={f1:.4f} | Ground={ground:.4f} | CiteP={cp:.4f} | CiteR={cr:.4f}")
        if self.provenance_metrics:
            lines.append("=" * 70)
            lines.append(" Provenance Performance (Total / Ans / False Abstention Rate / F1):")
            for prov, pstats in sorted(self.provenance_metrics.items()):
                tot = pstats.get("total_count", 0)
                ans = pstats.get("answerable_count", 0)
                fa_rate = pstats.get("false_abstention_rate", 0.0)
                f1 = pstats.get("token_f1", {}).get("mean", 0.0)
                errs = pstats.get("error_count", 0)
                err_str = f", Errs={errs}" if errs > 0 else ""
                lines.append(f"   * {prov:<24} (Tot={tot:<2}, Ans={ans:<2}{err_str}): FalseAbstRate={fa_rate:.4f} | F1={f1:.4f}")
        lines.append("=" * 70)
        if self.unanswerable_metrics:
            u = self.unanswerable_metrics
            lines.append(
                f" Unanswerable Abstention: Evaluated={u.get('total', 0)} | "
                f"Abstained={u.get('abstained_count', 0)} (Model={u.get('model_abstained_count', 0)}, Local={u.get('local_abstained_count', 0)}) | "
                f"Errors={u.get('error_count', 0)} | "
                f"Abstention Accuracy={u.get('abstention_accuracy', 0.0):.4f}"
            )
            lines.append("=" * 70)
        if self.output_path:
            lines.append(f" Full results written to: {self.output_path}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generation Evaluator Runner
# ---------------------------------------------------------------------------

class GenerationEvaluator:
    """Executes FastRAG pipeline answer generation across an evaluation dataset."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        dataset_dir: Path | str,
        results_dir: Path | str | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.dataset_dir = Path(dataset_dir)
        self.results_dir = Path(results_dir) if results_dir else Path("benchmarks/results")

    def load_queries(
        self,
        split: str = "dev",
        provenance_filter: str = "human_reviewed",
        category_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load and filter queries by split, provenance eligibility, and optional category."""
        split_filename = f"queries_{split}.jsonl" if split in {"dev", "test"} else "queries.jsonl"
        queries_path = self.dataset_dir / split_filename

        if not queries_path.is_file():
            # Fallback to queries.jsonl and filter by split
            queries_path = self.dataset_dir / "queries.jsonl"

        with open(queries_path, "r", encoding="utf-8") as f:
            all_queries = [json.loads(line) for line in f if line.strip()]

        filtered: list[dict[str, Any]] = []
        for q in all_queries:
            if split in {"dev", "test"} and q.get("split") != split:
                continue
            prov_status = q.get("annotation_provenance", {}).get("status", "unknown")
            if provenance_filter != "all" and prov_status != provenance_filter:
                continue
            if category_filter is not None and category_filter != "all" and q.get("category") != category_filter:
                continue
            filtered.append(q)

        return filtered

    async def evaluate_single_query(
        self,
        query_record: dict[str, Any],
        candidate_k: int = 20,
        rerank_top_k: int = 10,
        generate_answer: bool = True,
        temperature: float = 0.0,
    ) -> QueryGenerationEvaluation:
        """Run single query through pipeline and score against reference annotations."""
        q_id = query_record["query_id"]
        q_text = query_record["query"]
        category = query_record.get("category", "unknown")
        split = query_record.get("split", "unknown")
        prov_status = query_record.get("annotation_provenance", {}).get("status", "unknown")
        ref_answer = query_record.get("reference_answer")
        source_doc_ids = query_record.get("source_doc_ids", [])
        expected_ev = query_record.get("expected_evidence", [])
        rel_judgments = query_record.get("relevance_judgments", {})
        relevant_doc_ids = {doc for doc, rel in rel_judgments.items() if rel > 0}

        is_unanswerable = category == "unanswerable_negative" or len(relevant_doc_ids) == 0

        t0 = time.perf_counter()
        raw_error: str | None = None
        pipe_resp: PipelineResponse | None = None

        try:
            pipe_resp = await self.pipeline.execute_async(
                query=q_text,
                candidate_k=candidate_k,
                rerank_top_k=rerank_top_k,
                generate_answer=generate_answer,
                generation_temperature=temperature,
            )
        except Exception as exc:
            raw_error = str(exc)
            logger.warning(f"Error during query execution {q_id}: {exc}")

        latency_ms = (time.perf_counter() - t0) * 1000.0

        is_error = raw_error is not None
        if pipe_resp is None or pipe_resp.generation is None:
            gen_answer = ""
            retrieved_doc_ids: list[str] = []
            cited_doc_ids: list[str] = []
            ttft_ms: float | None = None
            p_tokens: int | None = None
            c_tokens: int | None = None
            tot_tokens: int | None = None
            chunks_count = 0
            context_insufficient = False
            if raw_error is None:
                raw_error = "No generation response produced"
                is_error = True
        else:
            gen = pipe_resp.generation
            gen_answer = gen.answer
            retrieved_doc_ids = [c.document_id for c in pipe_resp.reranked_results]
            cited_doc_ids = extract_cited_document_ids(
                answer=gen.answer,
                context=pipe_resp.context,
                explicit_cited_docs=gen.cited_doc_ids,
            )
            ttft_ms = gen.diagnostics.ttft_ms if gen.diagnostics else None
            p_tokens = gen.usage.prompt_tokens if gen.usage else None
            c_tokens = gen.usage.completion_tokens if gen.usage else None
            tot_tokens = gen.usage.total_tokens if gen.usage else None
            chunks_count = pipe_resp.context.total_chunks_selected
            context_insufficient = gen.context_insufficient

        # Scoring metrics & abstention classification
        if is_error:
            abstained = False
            correct_abstention = False
            is_false_abstention = False
            abstention_type = "none"
            token_f1: float | None = None
            token_precision: float | None = None
            token_recall: float | None = None
            completeness: float | None = None
            groundedness: float | None = None
            citation_precision: float | None = None
            citation_recall: float | None = None
        else:
            if context_insufficient or gen_answer.strip() == INSUFFICIENT_CONTEXT_ANSWER:
                abstained = True
                abstention_type = "local_context_filter"
            elif detect_abstention(gen_answer, context_insufficient_flag=False):
                abstained = True
                abstention_type = "model"
            else:
                abstained = False
                abstention_type = "none"

            if is_unanswerable:
                correct_abstention = abstained
                is_false_abstention = False
                f1_dict = {
                    "precision": 1.0 if abstained else 0.0,
                    "recall": 1.0 if abstained else 0.0,
                    "f1": 1.0 if abstained else 0.0,
                }
                completeness = 1.0 if abstained else 0.0
            else:
                correct_abstention = not abstained
                is_false_abstention = abstained
                f1_dict = compute_token_f1(gen_answer, ref_answer)
                completeness = compute_evidence_completeness(gen_answer, expected_ev)

            context_str = pipe_resp.context.context_text if pipe_resp else ""
            groundedness = compute_groundedness(gen_answer, context_str)

            citation_scores = compute_citation_metrics(
                cited_doc_ids=cited_doc_ids,
                source_doc_ids=source_doc_ids,
                relevant_doc_ids=relevant_doc_ids,
            )
            token_f1 = f1_dict["f1"]
            token_precision = f1_dict["precision"]
            token_recall = f1_dict["recall"]
            citation_precision = citation_scores["citation_precision"]
            citation_recall = citation_scores["citation_recall"]

        return QueryGenerationEvaluation(
            query_id=q_id,
            query=q_text,
            category=category,
            split=split,
            provenance_status=prov_status,
            reference_answer=ref_answer,
            generated_answer=gen_answer,
            source_doc_ids=source_doc_ids,
            retrieved_doc_ids=retrieved_doc_ids,
            cited_doc_ids=cited_doc_ids,
            is_unanswerable=is_unanswerable,
            correct_abstention=correct_abstention,
            token_f1=token_f1,
            token_precision=token_precision,
            token_recall=token_recall,
            completeness=completeness,
            groundedness=groundedness,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            latency_ms=round(latency_ms, 3),
            ttft_ms=round(ttft_ms, 3) if ttft_ms is not None else None,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=tot_tokens,
            context_chunks_count=chunks_count,
            raw_error=raw_error,
            is_false_abstention=is_false_abstention,
            is_error=is_error,
            abstention_type=abstention_type,
        )

    async def run_evaluation(
        self,
        experiment_id: str,
        split: str = "dev",
        provenance_filter: str = "human_reviewed",
        candidate_k: int = 20,
        rerank_top_k: int = 10,
        save_results: bool = True,
        temperature: float = 0.0,
        limit: int | None = None,
        category_filter: str | None = None,
    ) -> GenerationEvaluationReport:
        """Run full generation evaluation suite across filtered queries."""
        queries = self.load_queries(
            split=split,
            provenance_filter=provenance_filter,
            category_filter=category_filter,
        )
        if not queries:
            raise ValueError(
                f"No eligible queries found for split='{split}' with provenance_filter='{provenance_filter}' and category_filter='{category_filter}'"
            )

        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError(f"Limit must be an integer >= 1, got {limit}")
            queries = queries[:limit]

        provider_name = getattr(self.pipeline.generator, "provider_name", "none") if self.pipeline.generator else "none"
        model_name = getattr(self.pipeline.generator, "model_name", "none") if self.pipeline.generator else "none"

        results: list[QueryGenerationEvaluation] = []
        for q in queries:
            res = await self.evaluate_single_query(
                query_record=q,
                candidate_k=candidate_k,
                rerank_top_k=rerank_top_k,
                generate_answer=True,
                temperature=temperature,
            )
            results.append(res)

        # Aggregate metrics over answerable queries vs unanswerable queries
        answerable_results = [r for r in results if not r.is_unanswerable]
        unanswerable_results = [r for r in results if r.is_unanswerable]

        # Valid answerables (excluding provider errors)
        valid_answerable = [r for r in answerable_results if not r.is_error]
        ans_count = len(valid_answerable)
        false_abstention_count = sum(1 for r in valid_answerable if r.is_false_abstention)
        false_abstention_rate = round(false_abstention_count / ans_count, 4) if ans_count > 0 else 0.0

        fa_values = [1.0 if r.is_false_abstention else 0.0 for r in valid_answerable]
        fa_stats = compute_metric_stats(fa_values).to_dict()
        fa_stats["rate"] = false_abstention_rate
        fa_stats["false_abstention_count"] = false_abstention_count
        fa_stats["total_answerable"] = ans_count

        total_errors = sum(1 for r in results if r.is_error)

        metric_keys = [
            ("token_f1", [r.token_f1 for r in valid_answerable]),
            ("token_precision", [r.token_precision for r in valid_answerable]),
            ("token_recall", [r.token_recall for r in valid_answerable]),
            ("completeness", [r.completeness for r in valid_answerable]),
            ("groundedness", [r.groundedness for r in results if not r.is_error]),
            ("citation_precision", [r.citation_precision for r in valid_answerable]),
            ("citation_recall", [r.citation_recall for r in valid_answerable]),
            ("latency_ms", [r.latency_ms for r in results]),
        ]

        summary_metrics: dict[str, dict[str, Any]] = {}
        for m_name, vals in metric_keys:
            summary_metrics[m_name] = compute_metric_stats(vals).to_dict()

        summary_metrics["false_abstention_rate"] = fa_stats
        summary_metrics["false_abstention_count"] = {
            "count": false_abstention_count,
            "rate": false_abstention_rate,
            "total_answerable": ans_count,
        }
        summary_metrics["error_count"] = {
            "count": total_errors,
            "total_queries": len(results),
        }

        # Category breakdowns
        categories = sorted(set(r.category for r in results))
        category_metrics: dict[str, dict[str, Any]] = {}
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            cat_ans = [r for r in cat_results if not r.is_unanswerable and not r.is_error]
            cat_valid = [r for r in cat_results if not r.is_error]
            cat_errors = sum(1 for r in cat_results if r.is_error)
            category_metrics[cat] = {
                "count": len(cat_results),
                "error_count": cat_errors,
                "token_f1": compute_metric_stats([r.token_f1 for r in cat_ans]).to_dict(),
                "groundedness": compute_metric_stats([r.groundedness for r in cat_valid]).to_dict(),
                "citation_precision": compute_metric_stats([r.citation_precision for r in cat_ans]).to_dict(),
                "citation_recall": compute_metric_stats([r.citation_recall for r in cat_ans]).to_dict(),
            }

        # Provenance breakdowns
        provenance_statuses = sorted(set(r.provenance_status for r in results))
        provenance_metrics: dict[str, dict[str, Any]] = {}
        for prov in provenance_statuses:
            prov_results = [r for r in results if r.provenance_status == prov]
            prov_ans = [r for r in prov_results if not r.is_unanswerable and not r.is_error]
            prov_unans = [r for r in prov_results if r.is_unanswerable and not r.is_error]
            prov_errors = sum(1 for r in prov_results if r.is_error)
            p_fa_count = sum(1 for r in prov_ans if r.is_false_abstention)
            p_fa_rate = round(p_fa_count / len(prov_ans), 4) if prov_ans else 0.0
            p_unans_abst = sum(1 for r in prov_unans if r.correct_abstention)
            p_unans_acc = round(p_unans_abst / len(prov_unans), 4) if prov_unans else 0.0
            provenance_metrics[prov] = {
                "total_count": len(prov_results),
                "error_count": prov_errors,
                "answerable_count": len(prov_ans),
                "unanswerable_count": len(prov_unans),
                "false_abstention_count": p_fa_count,
                "false_abstention_rate": p_fa_rate,
                "unanswerable_abstention_accuracy": p_unans_acc,
                "token_f1": compute_metric_stats([r.token_f1 for r in prov_ans]).to_dict(),
                "groundedness": compute_metric_stats([r.groundedness for r in prov_results if not r.is_error]).to_dict(),
                "citation_precision": compute_metric_stats([r.citation_precision for r in prov_ans]).to_dict(),
                "citation_recall": compute_metric_stats([r.citation_recall for r in prov_ans]).to_dict(),
            }

        # Unanswerable metrics
        unans_metrics: dict[str, Any] = {}
        if unanswerable_results:
            unans_errors = sum(1 for r in unanswerable_results if r.is_error)
            unans_valid = [r for r in unanswerable_results if not r.is_error]
            model_abstained = sum(1 for r in unans_valid if r.abstention_type == "model")
            local_abstained = sum(1 for r in unans_valid if r.abstention_type == "local_context_filter")
            abstained_count = model_abstained + local_abstained
            valid_count = len(unans_valid)
            abstention_accuracy = round(abstained_count / valid_count, 4) if valid_count > 0 else 0.0
            unans_metrics = {
                "total": len(unanswerable_results),
                "error_count": unans_errors,
                "valid_count": valid_count,
                "abstained_count": abstained_count,
                "model_abstained_count": model_abstained,
                "local_abstained_count": local_abstained,
                "abstention_accuracy": abstention_accuracy,
            }

        # File fingerprint helper
        def _get_sha256(p: Path) -> str | None:
            if not p.is_file():
                return None
            import hashlib
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()

        retrieval_config = {
            "candidate_k": candidate_k,
            "rerank_top_k": rerank_top_k,
            "blending_alpha": getattr(self.pipeline.reranker, "default_blending_alpha", 0.0) if self.pipeline.reranker else None,
            "retriever_type": type(self.pipeline.retriever).__name__,
            "reranker_type": type(self.pipeline.reranker).__name__ if self.pipeline.reranker else None,
            "context_assembler": {
                "min_relevance_score": getattr(self.pipeline.context_assembler, "min_relevance_score", None) if self.pipeline.context_assembler else None,
                "ordering_strategy": getattr(self.pipeline.context_assembler, "ordering_strategy", None) if self.pipeline.context_assembler else None,
                "max_tokens": getattr(self.pipeline.context_assembler, "max_tokens", None) if self.pipeline.context_assembler else None,
                "max_chunks": getattr(self.pipeline.context_assembler, "max_chunks", None) if self.pipeline.context_assembler else None,
            },
            "fingerprints": {
                "corpus_sha256": _get_sha256(self.dataset_dir / "corpus.jsonl"),
                "queries_sha256": _get_sha256(self.dataset_dir / f"queries_{split}.jsonl"),
                "qrels_sha256": _get_sha256(self.dataset_dir / f"qrels_{split}.tsv"),
            },
        }

        timestamp = datetime.now(timezone.utc).isoformat()
        output_filepath: str | None = None
        if save_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            safe_ts = timestamp.replace(":", "-").replace(".", "_")
            out_file = self.results_dir / f"generation_{experiment_id}_{safe_ts}.json"
            output_filepath = str(out_file)

        report = GenerationEvaluationReport(
            experiment_id=experiment_id,
            timestamp_utc=timestamp,
            dataset_version="2.0.0",
            split=split,
            provenance_filter=provenance_filter,
            provider_type=provider_name,
            model_identifier=model_name,
            retrieval_configuration=retrieval_config,
            eligible_query_count=len(queries),
            total_queries_evaluated=len(results),
            summary_metrics=summary_metrics,
            category_metrics=category_metrics,
            unanswerable_metrics=unans_metrics,
            per_query_results=[r.to_dict() for r in results],
            output_path=output_filepath,
            provenance_metrics=provenance_metrics,
        )

        if save_results and output_filepath:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(report.to_json())

        return report
