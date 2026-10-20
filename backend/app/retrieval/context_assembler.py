"""FastRAG Context Assembly and Token Budgeting Module.

Converts reranked candidates into a compact, deduplicated, citation-tagged context
package strictly constrained by token and chunk count budgets.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Sequence

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.retrieval.base import RetrievalResult, validate_top_k

logger = get_logger("fastrag.retrieval.context_assembler")


def approximate_token_counter(text: str) -> int:
    """Documented heuristic token counter for text passages.

    Approximates token volume using standard ~4 characters per token heuristic
    with whitespace token verification:
        tokens = max(ceil(len(text) / 4.0), ceil(len(text.split()) * 1.3))
    This is an approximation and does NOT claim exact BPE/WordPiece tokenization.
    """
    if not text:
        return 0
    char_estimate = math.ceil(len(text) / 4.0)
    words = text.split()
    word_estimate = math.ceil(len(words) * 1.3) if words else 0
    return max(char_estimate, word_estimate, 1)


@dataclass
class ContextChunk:
    """Represents a single selected passage in the assembled context."""

    chunk_id: str
    document_id: str
    citation_id: str
    rank: int
    text: str
    source: str
    approximate_tokens: int
    title: str = ""
    start_offset: int | None = None
    end_offset: int | None = None
    rerank_score: float | None = None
    retrieval_score: float | None = None
    retrieval_score_type: str | None = None
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AssembledContext:
    """Complete structured context package ready for downstream prompt injection."""

    context_text: str
    selected_chunks: list[ContextChunk]
    selected_chunk_ids: list[str]
    selected_document_ids: list[str]
    total_chunks_selected: int
    total_chunks_considered: int
    approximate_token_count: int
    token_budget: int
    max_chunks_budget: int
    budget_exceeded: bool
    truncation_applied: bool
    truncation_reasons: list[str]
    assembly_duration_ms: float
    is_approximate_token_count: bool = True
    context_insufficient: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_text": self.context_text,
            "selected_chunks": [c.to_dict() for c in self.selected_chunks],
            "selected_chunk_ids": self.selected_chunk_ids,
            "selected_document_ids": self.selected_document_ids,
            "total_chunks_selected": self.total_chunks_selected,
            "total_chunks_considered": self.total_chunks_considered,
            "approximate_token_count": self.approximate_token_count,
            "token_budget": self.token_budget,
            "max_chunks_budget": self.max_chunks_budget,
            "budget_exceeded": self.budget_exceeded,
            "truncation_applied": self.truncation_applied,
            "truncation_reasons": self.truncation_reasons,
            "assembly_duration_ms": round(self.assembly_duration_ms, 3),
            "is_approximate_token_count": self.is_approximate_token_count,
            "context_insufficient": self.context_insufficient,
        }


class ContextAssembler:
    """Builds structured, citation-grounded context within strict token budgets.

    Applies:
    - Rank-ordered candidate selection.
    - Exact chunk ID and text deduplication.
    - Optional source-document diversity constraints.
    - Token budget verification and explicit truncation tracking.
    - Deterministic output generation.
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        max_chunks: int | None = None,
        max_chunk_chars: int | None = None,
        max_chunks_per_document: int | None = None,
        citation_prefix: str = "Doc",
        token_counter: Callable[[str], int] | None = None,
        separator: str = "\n\n",
        min_relevance_score: float | None = None,
        ordering_strategy: Literal["descending", "boundary"] | str = "descending",
    ) -> None:
        settings = get_settings()
        self.max_tokens = max_tokens or settings.context_max_tokens
        self.max_chunks = max_chunks or settings.context_max_chunks
        self.max_chunk_chars = max_chunk_chars
        self.max_chunks_per_document = max_chunks_per_document
        self.citation_prefix = citation_prefix.strip() or "Doc"
        self.token_counter = token_counter or approximate_token_counter
        self.separator = separator
        self.min_relevance_score = (
            min_relevance_score
            if min_relevance_score is not None
            else settings.context_min_relevance_score
        )
        self.ordering_strategy = (
            ordering_strategy or settings.context_ordering_strategy or "descending"
        ).lower()
        if self.ordering_strategy not in ("descending", "boundary"):
            raise ValueError(
                f"ordering_strategy must be 'descending' or 'boundary', got '{self.ordering_strategy}'"
            )

        validate_top_k(self.max_tokens)
        validate_top_k(self.max_chunks)
        if self.max_chunk_chars is not None and self.max_chunk_chars <= 0:
            raise ValueError(f"max_chunk_chars must be positive, got {self.max_chunk_chars}")
        if self.max_chunks_per_document is not None and self.max_chunks_per_document <= 0:
            raise ValueError(
                f"max_chunks_per_document must be positive, got {self.max_chunks_per_document}"
            )

    def assemble(
        self,
        candidates: Sequence[RetrievalResult],
        max_tokens: int | None = None,
        max_chunks: int | None = None,
        min_relevance_score: float | None = None,
        ordering_strategy: Literal["descending", "boundary"] | str | None = None,
    ) -> AssembledContext:
        """Assemble candidate retrieval results into an AssembledContext package.

        Args:
            candidates: Sequence of RetrievalResult ordered by relevance rank.
            max_tokens: Override for maximum approximate tokens.
            max_chunks: Override for maximum chunks to include.
            min_relevance_score: Optional relevance score threshold to exclude poor candidates.
            ordering_strategy: Passage layout ordering ('descending' or 'boundary').

        Returns:
            Structured AssembledContext with formatted text and detailed provenance.
        """
        budget_tokens = max_tokens if max_tokens is not None else self.max_tokens
        budget_chunks = max_chunks if max_chunks is not None else self.max_chunks
        eff_min_score = (
            min_relevance_score
            if min_relevance_score is not None
            else self.min_relevance_score
        )
        eff_ordering = (
            ordering_strategy
            if ordering_strategy is not None
            else self.ordering_strategy
        ).lower()
        if eff_ordering not in ("descending", "boundary"):
            raise ValueError(
                f"ordering_strategy must be 'descending' or 'boundary', got '{eff_ordering}'"
            )

        validate_top_k(budget_tokens)
        validate_top_k(budget_chunks)

        start_t = time.perf_counter()

        if not candidates:
            return AssembledContext(
                context_text="",
                selected_chunks=[],
                selected_chunk_ids=[],
                selected_document_ids=[],
                total_chunks_selected=0,
                total_chunks_considered=0,
                approximate_token_count=0,
                token_budget=budget_tokens,
                max_chunks_budget=budget_chunks,
                budget_exceeded=False,
                truncation_applied=False,
                truncation_reasons=[],
                assembly_duration_ms=0.0,
                context_insufficient=True,
            )

        selected_chunks: list[ContextChunk] = []
        selected_chunk_ids: list[str] = []
        selected_doc_ids: list[str] = []
        seen_texts: set[str] = set()
        seen_chunk_ids: set[str] = set()
        doc_chunk_counts: dict[str, int] = {}
        truncation_reasons: list[str] = []
        running_token_count = 0
        budget_exceeded = False
        truncation_applied = False

        for rank_idx, cand in enumerate(candidates, start=1):
            # Guard: check chunk limit
            if len(selected_chunks) >= budget_chunks:
                truncation_reasons.append(
                    f"Max chunks limit ({budget_chunks}) reached; stopping candidate selection."
                )
                break

            # Guard: relevance score threshold (evaluate rerank_score logit from metadata if available)
            cand_score = (
                float(cand.metadata["rerank_score"])
                if cand.metadata and "rerank_score" in cand.metadata
                else cand.score
            )
            if eff_min_score is not None and cand_score < eff_min_score:
                truncation_reasons.append(
                    f"Candidate '{cand.chunk_id}' score {cand_score:.4f} below min_relevance_score ({eff_min_score:.4f}); omitted."
                )
                continue

            # Guard: duplicate chunk_id
            if cand.chunk_id in seen_chunk_ids:
                continue

            # Guard: duplicate text content
            normalized_text = " ".join(cand.text.strip().lower().split())
            if normalized_text in seen_texts:
                continue

            # Guard: per-document diversity limit
            if self.max_chunks_per_document is not None:
                current_doc_chunks = doc_chunk_counts.get(cand.document_id, 0)
                if current_doc_chunks >= self.max_chunks_per_document:
                    continue

            # Check individual chunk character cap
            cand_text = cand.text
            was_truncated = False
            if self.max_chunk_chars is not None and len(cand_text) > self.max_chunk_chars:
                cand_text = cand_text[: self.max_chunk_chars].rstrip() + "..."
                was_truncated = True
                truncation_applied = True
                truncation_reasons.append(
                    f"Chunk '{cand.chunk_id}' character count truncated to {self.max_chunk_chars} chars."
                )

            # Metadata extraction
            cand_meta = cand.metadata or {}
            title = str(cand_meta.get("title") or "")
            start_off = cand_meta.get("start_offset")
            end_off = cand_meta.get("end_offset")
            rerank_sc = cand_meta.get("rerank_score") or (
                cand.score if cand.score_type == "cross_encoder" else None
            )
            retrieval_sc = cand_meta.get("original_score") or (
                cand.score if cand.score_type != "cross_encoder" else None
            )
            retrieval_type = cand_meta.get("original_score_type") or cand.score_type

            # Citation label: 1-indexed count in selection order
            citation_num = len(selected_chunks) + 1
            citation_id = f"[{self.citation_prefix}-{citation_num}]"

            # Format the candidate text with header for context block
            header_parts = [citation_id]
            if title:
                header_parts.append(f"Title: {title}")
            header_parts.append(f"Doc: {cand.document_id}")
            header_line = " ".join(header_parts)
            formatted_passage = f"{header_line}\n{cand_text}"

            passage_tokens = self.token_counter(formatted_passage)

            # Token budget check
            if running_token_count + passage_tokens > budget_tokens:
                # If even the first chunk exceeds the total budget, include as much as possible safely
                if len(selected_chunks) == 0:
                    budget_exceeded = True
                    truncation_applied = True
                    truncation_reasons.append(
                        f"First chunk '{cand.chunk_id}' ({passage_tokens} approx tokens) exceeds total budget ({budget_tokens} tokens); including alone."
                    )
                    running_token_count += passage_tokens
                    orig_rank = cand.rank if getattr(cand, "rank", None) is not None else rank_idx
                    context_chunk = ContextChunk(
                        chunk_id=cand.chunk_id,
                        document_id=cand.document_id,
                        citation_id=citation_id,
                        rank=orig_rank,
                        text=cand_text,
                        source=cand.source,
                        approximate_tokens=passage_tokens,
                        title=title,
                        start_offset=start_off,
                        end_offset=end_off,
                        rerank_score=rerank_sc,
                        retrieval_score=retrieval_sc,
                        retrieval_score_type=retrieval_type,
                        truncated=was_truncated,
                        metadata=dict(cand_meta),
                    )
                    selected_chunks.append(context_chunk)
                    selected_chunk_ids.append(cand.chunk_id)
                    if cand.document_id not in selected_doc_ids:
                        selected_doc_ids.append(cand.document_id)
                else:
                    budget_exceeded = True
                    truncation_applied = True
                    truncation_reasons.append(
                        f"Token budget ({budget_tokens}) reached at chunk rank {rank_idx}; omitted remaining candidates."
                    )
                break

            running_token_count += passage_tokens
            seen_chunk_ids.add(cand.chunk_id)
            seen_texts.add(normalized_text)
            doc_chunk_counts[cand.document_id] = doc_chunk_counts.get(cand.document_id, 0) + 1

            orig_rank = cand.rank if getattr(cand, "rank", None) is not None else rank_idx
            context_chunk = ContextChunk(
                chunk_id=cand.chunk_id,
                document_id=cand.document_id,
                citation_id=citation_id,
                rank=orig_rank,
                text=cand_text,
                source=cand.source,
                approximate_tokens=passage_tokens,
                title=title,
                start_offset=start_off,
                end_offset=end_off,
                rerank_score=rerank_sc,
                retrieval_score=retrieval_sc,
                retrieval_score_type=retrieval_type,
                truncated=was_truncated,
                metadata=dict(cand_meta),
            )
            selected_chunks.append(context_chunk)
            selected_chunk_ids.append(cand.chunk_id)
            if cand.document_id not in selected_doc_ids:
                selected_doc_ids.append(cand.document_id)

        # Handle post-selection ordering and sufficiency determination
        if len(selected_chunks) == 0:
            context_insufficient = True
            context_text = ""
            if eff_min_score is not None and len(candidates) > 0:
                truncation_reasons.append(
                    f"All {len(candidates)} candidates fell below min_relevance_score {eff_min_score:.4f}; context marked insufficient."
                )
            ordered_chunks: list[ContextChunk] = []
        else:
            context_insufficient = False
            if eff_ordering == "boundary":
                if len(selected_chunks) <= 2:
                    ordered_chunks = list(selected_chunks)
                else:
                    # rank 1 first, rank 2 last, remaining candidates in between preserving relative order
                    ordered_chunks = [selected_chunks[0]] + selected_chunks[2:] + [selected_chunks[1]]
            else:
                ordered_chunks = list(selected_chunks)

            # Assemble full text with separators
            assembled_blocks: list[str] = []
            for c in ordered_chunks:
                hdr = f"{c.citation_id}"
                if c.title:
                    hdr += f" Title: {c.title}"
                hdr += f" (Doc: {c.document_id}, Source: {c.source or 'unknown'})"
                assembled_blocks.append(f"{hdr}\n{c.text}")

            context_text = self.separator.join(assembled_blocks)

        ordered_chunk_ids = [c.chunk_id for c in ordered_chunks]
        ordered_doc_ids = list(dict.fromkeys(c.document_id for c in ordered_chunks))
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return AssembledContext(
            context_text=context_text,
            selected_chunks=ordered_chunks,
            selected_chunk_ids=ordered_chunk_ids,
            selected_document_ids=ordered_doc_ids,
            total_chunks_selected=len(ordered_chunks),
            total_chunks_considered=len(candidates),
            approximate_token_count=running_token_count,
            token_budget=budget_tokens,
            max_chunks_budget=budget_chunks,
            budget_exceeded=budget_exceeded,
            truncation_applied=truncation_applied,
            truncation_reasons=truncation_reasons,
            assembly_duration_ms=elapsed_ms,
            context_insufficient=context_insufficient,
        )
