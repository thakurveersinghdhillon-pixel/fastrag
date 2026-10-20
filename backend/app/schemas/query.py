"""FastRAG API Schemas for Query and Generation."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request payload for RAG query execution."""

    query: str = Field(
        ...,
        min_length=1,
        description="User question or query string",
        examples=["What is Reciprocal Rank Fusion RRF?"],
    )
    candidate_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of candidate chunks to retrieve from candidate retrieval",
    )
    rerank_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of top reranked chunks to select for context assembly",
    )
    bypass_reranking: bool = Field(
        default=False,
        description="Whether to bypass CrossEncoder reranking and use raw retriever rankings",
    )
    context_max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum token budget for context assembly (defaults to configured setting)",
    )
    context_max_chunks: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of chunks to include in context assembly (defaults to configured setting)",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM answer generation",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=16384,
        description="Maximum tokens to generate",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional custom system prompt override",
    )
    model: str | None = Field(
        default=None,
        description="Optional LLM model override",
    )
    min_relevance_score: float | None = Field(
        default=None,
        description="Optional minimum relevance score threshold for candidate context filtering",
    )
    ordering_strategy: str | None = Field(
        default=None,
        description="Context chunk ordering strategy ('descending' or 'boundary')",
    )


class QueryResponse(BaseModel):
    """Response payload for non-streaming RAG query execution."""

    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Synthesized grounded answer")
    model: str = Field(..., description="LLM model used for synthesis")
    provider: str = Field(..., description="LLM provider name (e.g. 'openai', 'mock')")
    finish_reason: str = Field(default="stop", description="Generation completion reason")
    cited_doc_ids: list[str] = Field(
        default_factory=list,
        description="Distinct document IDs cited in the answer",
    )
    context_insufficient: bool = Field(
        default=False,
        description="Flag indicating context was insufficient to answer query",
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Summary of assembled context passages",
    )
    diagnostics: dict[str, Any] = Field(
        default_factory=dict,
        description="Latency breakdown and execution telemetry across pipeline stages",
    )
    usage: dict[str, Any] = Field(
        default_factory=dict,
        description="Token usage statistics (prompt, completion, total, is_estimated)",
    )
