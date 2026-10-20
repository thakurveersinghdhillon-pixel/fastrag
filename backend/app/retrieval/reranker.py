"""FastRAG CrossEncoder Reranking Module.

Provides fine-grained semantic passage re-scoring on top of dense, BM25, or hybrid
retrieval candidates using cross-attention models (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2).
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

from app.core.config import get_settings
from app.core.exceptions import ConfigurationError, RetrievalError
from app.core.logging_config import get_logger
from app.retrieval.base import RetrievalResult, validate_top_k

logger = get_logger("fastrag.retrieval.reranker")

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankDiagnostics:
    """Telemetry capturing execution latency and candidate volume for reranking."""

    candidate_count: int
    reranked_count: int
    duration_ms: float
    model_name: str
    is_mock: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CrossEncoderScorer(ABC):
    """Abstract interface for scoring (query, passage) pairs."""

    @abstractmethod
    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Compute relevance scores for a sequence of (query, passage) text pairs."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the underlying model."""
        pass


class MockCrossEncoderScorer(CrossEncoderScorer):
    """Deterministic, zero-dependency scorer for offline testing and fast unit tests.

    Supports custom scoring callables or deterministic text-hash scoring to ensure
    tests never depend on network access or GPU/neural weights.
    """

    def __init__(
        self,
        score_fn: Callable[[str, str], float] | None = None,
        fixed_scores: dict[str, float] | None = None,
        model_name: str = "mock-cross-encoder-v1",
    ) -> None:
        self._score_fn = score_fn
        self._fixed_scores = fixed_scores or {}
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        scores: list[float] = []
        for query, passage in pairs:
            # Check fixed scores dictionary first
            if passage in self._fixed_scores:
                scores.append(float(self._fixed_scores[passage]))
            elif (query, passage) in self._fixed_scores:  # type: ignore[operator]
                scores.append(float(self._fixed_scores[(query, passage)]))  # type: ignore[index]
            elif self._score_fn is not None:
                scores.append(float(self._score_fn(query, passage)))
            else:
                # Deterministic pseudo-relevance using hash and token overlap
                q_tokens = set(query.lower().split())
                p_tokens = set(passage.lower().split())
                overlap = len(q_tokens.intersection(p_tokens))
                digest = hashlib.sha256(f"{query}::{passage}".encode("utf-8")).digest()
                hash_val = (digest[0] % 100) / 100.0
                score = float(overlap * 2.0 + hash_val)
                scores.append(score)
        return scores


class SentenceTransformerCrossEncoder(CrossEncoderScorer):
    """Production CrossEncoder powered by sentence-transformers with lazy loading.

    Loads model lazily on first usage and reuses the loaded instance across queries.
    Initial model download / warm-up introduces cold-start latency; subsequent warm
    inference runs substantially faster.
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        import threading
        settings = get_settings()
        self._model_name = model_name or settings.reranker_model_name or DEFAULT_RERANKER_MODEL
        self._batch_size = batch_size or settings.reranker_batch_size
        chosen = device or settings.reranker_device
        if chosen is not None:
            self._device = chosen
        else:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        self._device_lock = threading.Lock()
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_model(self) -> Any:
        """Lazily load the CrossEncoder model once."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed. Install it via "
                    "'pip install sentence-transformers' to use CrossEncoder reranking."
                ) from exc

            start_t = time.perf_counter()
            logger.info(f"Loading CrossEncoder model '{self._model_name}' (device: {self._device})...")
            try:
                # Try loading from local HuggingFace cache first
                self._model = CrossEncoder(
                    self._model_name,
                    device=self._device,
                    local_files_only=True,
                )
            except Exception:
                try:
                    self._model = CrossEncoder(
                        self._model_name,
                        device=self._device,
                    )
                except Exception as exc:
                    err_msg = f"Failed to load CrossEncoder model '{self._model_name}': {exc}"
                    logger.error(err_msg, exc_info=True)
                    raise RetrievalError(err_msg) from exc

            load_elapsed = time.perf_counter() - start_t
            logger.info(
                f"Loaded CrossEncoder model '{self._model_name}' in {load_elapsed:.2f}s",
                extra={"operation": "load_cross_encoder", "latency_ms": load_elapsed * 1000},
            )
        return self._model

    def warmup(self) -> None:
        """Pre-warm model weights and execution engine to eliminate runtime cold-start."""
        _ = self._ensure_model()
        _ = self.score_pairs([("warmup query", "warmup passage")])

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        model = self._ensure_model()

        def _do_predict() -> Any:
            try:
                import torch
                with torch.inference_mode():
                    return model.predict(
                        list(pairs),
                        batch_size=self._batch_size,
                        show_progress_bar=False,
                    )
            except (ImportError, AttributeError):
                return model.predict(
                    list(pairs),
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                )

        try:
            if getattr(self, "_device", "") == "mps":
                with self._device_lock:
                    raw_scores = _do_predict()
            else:
                raw_scores = _do_predict()

            # Normalize NumPy float array / scalars into standard Python floats
            if hasattr(raw_scores, "tolist"):
                return [float(s) for s in raw_scores.tolist()]
            return [float(s) for s in raw_scores]
        except Exception as exc:
            err_msg = f"CrossEncoder inference failed on {len(pairs)} pairs: {exc}"
            logger.error(err_msg, exc_info=True)
            raise RetrievalError(err_msg) from exc


class CrossEncoderReranker:
    """Orchestrates fine-grained passage reranking on candidate retrieval results.

    Applies a CrossEncoder to query-passage pairs, preserves original retrieval scores
    and metadata, eliminates duplicate scoring of identical chunks, and sorts candidates
    by descending CrossEncoder score with deterministic chunk_id tie-breaking.
    """

    def __init__(
        self,
        scorer: CrossEncoderScorer | None = None,
        model_name: str | None = None,
        batch_size: int = 32,
        device: str | None = None,
        default_top_k: int | None = None,
        default_blending_alpha: float = 0.0,
    ) -> None:
        settings = get_settings()
        if scorer is not None:
            if not isinstance(scorer, CrossEncoderScorer):
                raise TypeError(f"scorer must implement CrossEncoderScorer, got {type(scorer).__name__}")
            self.scorer = scorer
        else:
            self.scorer = SentenceTransformerCrossEncoder(
                model_name=model_name or settings.reranker_model_name,
                batch_size=batch_size or settings.reranker_batch_size,
                device=device or settings.reranker_device,
            )

        self.default_top_k = default_top_k or settings.default_top_k
        validate_top_k(self.default_top_k)
        if default_blending_alpha < 0.0:
            raise ValueError(f"default_blending_alpha must be non-negative (>= 0.0), got {default_blending_alpha}")
        self.default_blending_alpha = float(default_blending_alpha)
        self.last_diagnostics: RerankDiagnostics | None = None

    @property
    def model_name(self) -> str:
        return self.scorer.model_name

    def warmup(self) -> None:
        """Pre-warm underlying CrossEncoder model."""
        if hasattr(self.scorer, "warmup"):
            self.scorer.warmup()

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
        blending_alpha: float | None = None,
    ) -> list[RetrievalResult]:
        """Score and reorder candidate retrieval results using the CrossEncoder.

        Supports optional Reciprocal Rank Blending (RRF) between CrossEncoder rank
        and initial candidate rank when blending_alpha > 0.0.

        Args:
            query: User query string.
            candidates: Sequence of RetrievalResult candidates from prior retrieval.
            top_k: Maximum number of reranked results to return.
            blending_alpha: Optional reciprocal rank blending weight (alpha >= 0.0).
                If 0.0 (default), uses pure CrossEncoder rank and logit score.
                If > 0.0, blends 1/(60 + rank_ce) + alpha/(60 + rank_base).

        Returns:
            Ranked list of RetrievalResult sorted by descending score with chunk_id ascending tie-breaking.
        """
        effective_top_k = top_k if top_k is not None else self.default_top_k
        validate_top_k(effective_top_k)

        effective_alpha = (
            float(blending_alpha)
            if blending_alpha is not None
            else getattr(self, "default_blending_alpha", 0.0)
        )
        if effective_alpha < 0.0:
            raise ValueError(f"blending_alpha must be non-negative (>= 0.0), got {effective_alpha}")

        # Handle empty candidates or empty query
        if not candidates or not query or not query.strip():
            self.last_diagnostics = RerankDiagnostics(
                candidate_count=len(candidates) if candidates else 0,
                reranked_count=0,
                duration_ms=0.0,
                model_name=self.model_name,
                is_mock=isinstance(self.scorer, MockCrossEncoderScorer),
            )
            return []

        start_t = time.perf_counter()

        # Step 1: Deduplicate candidates by chunk_id, preserving the earliest candidate
        # to avoid redundant neural scoring of the same passage
        unique_candidates: list[RetrievalResult] = []
        seen_chunk_ids: set[str] = set()

        for cand in candidates:
            if cand.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(cand.chunk_id)
                unique_candidates.append(cand)

        if not unique_candidates:
            self.last_diagnostics = RerankDiagnostics(
                candidate_count=0,
                reranked_count=0,
                duration_ms=0.0,
                model_name=self.model_name,
                is_mock=isinstance(self.scorer, MockCrossEncoderScorer),
            )
            return []

        # Step 2: Prepare pairs for scoring
        pairs = [(query, cand.text) for cand in unique_candidates]

        # Step 3: Run inference via the scorer
        try:
            scores = self.scorer.score_pairs(pairs)
        except RetrievalError:
            raise
        except Exception as exc:
            err_msg = f"Unexpected failure in CrossEncoder scoring: {exc}"
            logger.error(err_msg, exc_info=True)
            raise RetrievalError(err_msg) from exc

        if len(scores) != len(unique_candidates):
            raise RetrievalError(
                f"Scorer returned {len(scores)} scores for {len(unique_candidates)} candidate pairs"
            )

        # Step 4: Rank candidates
        if effective_alpha == 0.0:
            # Pure CrossEncoder ranking (100% backward compatible)
            scored_entries: list[tuple[float, str, RetrievalResult]] = []
            for cand, score in zip(unique_candidates, scores):
                scored_entries.append((float(score), cand.chunk_id, cand))

            # Sort by descending score, breaking ties deterministically by chunk_id ascending
            scored_entries.sort(key=lambda item: (-item[0], item[1]))

            reranked_results: list[RetrievalResult] = []
            for rank_idx, (score, _, orig) in enumerate(scored_entries[:effective_top_k], start=1):
                merged_meta = dict(orig.metadata) if orig.metadata else {}
                # Preserve original retrieval details
                merged_meta["original_score"] = orig.score
                merged_meta["original_score_type"] = orig.score_type
                merged_meta["original_rank"] = orig.rank
                merged_meta["rerank_score"] = round(score, 6)
                merged_meta["rerank_model"] = self.model_name

                reranked_results.append(
                    RetrievalResult(
                        chunk_id=orig.chunk_id,
                        document_id=orig.document_id,
                        rank=rank_idx,
                        score=round(score, 6),
                        score_type="cross_encoder",
                        text=orig.text,
                        source=orig.source,
                        metadata=merged_meta,
                    )
                )
        else:
            # Reciprocal Rank Blending (RRF): blend CrossEncoder rank with initial candidate rank
            ce_sorted: list[tuple[float, str, RetrievalResult, int]] = []
            for idx, (cand, score) in enumerate(zip(unique_candidates, scores)):
                base_rank = cand.rank if (isinstance(cand.rank, int) and cand.rank >= 1) else (idx + 1)
                ce_sorted.append((float(score), cand.chunk_id, cand, base_rank))

            # Determine pure CrossEncoder ranks (1-indexed)
            ce_sorted.sort(key=lambda item: (-item[0], item[1]))

            # Compute blended score for each candidate
            blended_entries: list[tuple[float, str, float, int, int, RetrievalResult]] = []
            for ce_rank, (ce_score, chunk_id, orig, base_rank) in enumerate(ce_sorted, start=1):
                s_blended = (1.0 / (60.0 + ce_rank)) + (effective_alpha / (60.0 + base_rank))
                blended_entries.append((s_blended, chunk_id, ce_score, ce_rank, base_rank, orig))

            # Sort by descending blended score, breaking ties deterministically by chunk_id ascending
            blended_entries.sort(key=lambda item: (-round(item[0], 8), item[1]))

            reranked_results = []
            for rank_idx, (s_blended, _, ce_score, ce_rank, base_rank, orig) in enumerate(
                blended_entries[:effective_top_k], start=1
            ):
                merged_meta = dict(orig.metadata) if orig.metadata else {}
                merged_meta["original_score"] = orig.score
                merged_meta["original_score_type"] = orig.score_type
                merged_meta["original_rank"] = base_rank
                merged_meta["rerank_score"] = round(ce_score, 6)
                merged_meta["rerank_model"] = self.model_name
                merged_meta["cross_encoder_rank"] = ce_rank
                merged_meta["blended_score"] = round(s_blended, 6)
                merged_meta["blended_alpha"] = round(effective_alpha, 4)

                reranked_results.append(
                    RetrievalResult(
                        chunk_id=orig.chunk_id,
                        document_id=orig.document_id,
                        rank=rank_idx,
                        score=round(s_blended, 6),
                        score_type="blended_rrf",
                        text=orig.text,
                        source=orig.source,
                        metadata=merged_meta,
                    )
                )

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        self.last_diagnostics = RerankDiagnostics(
            candidate_count=len(candidates),
            reranked_count=len(reranked_results),
            duration_ms=elapsed_ms,
            model_name=self.model_name,
            is_mock=isinstance(self.scorer, MockCrossEncoderScorer),
        )

        logger.debug(
            f"Reranked {len(candidates)} candidates -> {len(reranked_results)} results in {elapsed_ms:.2f}ms "
            f"(model: {self.model_name})"
        )

        return reranked_results
