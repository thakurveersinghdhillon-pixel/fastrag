"""FastRAG LLM Generation Data Models and Schemas.

Defines validated data structures for generation requests, responses, streaming chunks,
diagnostics, and usage tracking.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class UsageStats:
    """Token consumption reported by provider or estimated when unavailable."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    is_estimated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationDiagnostics:
    """Execution telemetry and latency tracking for LLM response generation."""

    ttft_ms: float | None = None  # Time to First Token (milliseconds)
    generation_ms: float = 0.0  # Duration from first token to stream completion
    total_duration_ms: float = 0.0  # Total end-to-end generation stage duration
    chunk_count: int = 0  # Number of emitted stream chunks
    model_name: str = ""
    provider_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ttft_ms": round(self.ttft_ms, 3) if self.ttft_ms is not None else None,
            "generation_ms": round(self.generation_ms, 3),
            "total_duration_ms": round(self.total_duration_ms, 3),
            "chunk_count": self.chunk_count,
            "model_name": self.model_name,
            "provider_name": self.provider_name,
        }


@dataclass
class GenerationRequest:
    """Internal normalized request payload for LLM providers."""

    query: str
    context_text: str
    selected_chunk_ids: list[str] = field(default_factory=list)
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not isinstance(self.context_text, str):
            raise TypeError("context_text must be a string")
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError(f"temperature must be in [0.0, 2.0], got {self.temperature}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be strictly positive, got {self.max_tokens}")


@dataclass
class GenerationResponse:
    """Complete synthesized answer with provenance citations and diagnostic telemetry."""

    query: str
    answer: str
    model: str
    provider: str
    finish_reason: str = "stop"
    usage: UsageStats = field(default_factory=UsageStats)
    diagnostics: GenerationDiagnostics = field(default_factory=GenerationDiagnostics)
    cited_doc_ids: list[str] = field(default_factory=list)
    context_insufficient: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "model": self.model,
            "provider": self.provider,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "cited_doc_ids": self.cited_doc_ids,
            "context_insufficient": self.context_insufficient,
        }


@dataclass
class StreamChunk:
    """Incremental chunk emitted during real-time streaming."""

    delta: str = ""
    is_first: bool = False
    is_final: bool = False
    finish_reason: str | None = None
    usage: UsageStats | None = None
    diagnostics: GenerationDiagnostics | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "delta": self.delta,
            "is_first": self.is_first,
            "is_final": self.is_final,
        }
        if self.finish_reason is not None:
            data["finish_reason"] = self.finish_reason
        if self.usage is not None:
            data["usage"] = self.usage.to_dict()
        if self.diagnostics is not None:
            data["diagnostics"] = self.diagnostics.to_dict()
        return data


@dataclass
class StreamEvent:
    """Server-Sent Event (SSE) payload representation."""

    event: str  # "token", "context", "done", "error"
    data: dict[str, Any]

    def to_sse(self) -> str:
        """Serialize event to standard SSE text protocol."""
        return f"event: {self.event}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"
