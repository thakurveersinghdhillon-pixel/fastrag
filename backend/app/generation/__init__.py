"""FastRAG LLM Generation and Streaming Package.

Provides pluggable LLM provider abstractions (OpenAI-compatible and Mock),
grounded prompt engineering, citation verification, and real-time streaming.
"""

from app.generation.generator import RAGGenerator
from app.generation.models import (
    GenerationDiagnostics,
    GenerationRequest,
    GenerationResponse,
    StreamChunk,
    StreamEvent,
    UsageStats,
)
from app.generation.prompts import DEFAULT_SYSTEM_PROMPT, build_grounded_prompt
from app.generation.providers import (
    LLMProvider,
    MockLLMProvider,
    OpenAICompatibleProvider,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "GenerationDiagnostics",
    "GenerationRequest",
    "GenerationResponse",
    "LLMProvider",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "RAGGenerator",
    "StreamChunk",
    "StreamEvent",
    "UsageStats",
    "build_grounded_prompt",
]
