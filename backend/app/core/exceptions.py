"""FastRAG Core Exceptions.

Provides the foundational exception hierarchy for the FastRAG application.
"""


class FastRAGError(Exception):
    """Base exception for all FastRAG domain errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(FastRAGError):
    """Raised when application or subsystem configuration is invalid."""
    pass


class PipelineError(FastRAGError):
    """Base exception for retrieval or generation pipeline execution failures."""
    pass


class RetrievalError(PipelineError):
    """Raised when candidate retrieval fails across one or more branches."""
    pass


class BranchTimeoutError(RetrievalError):
    """Raised when an asynchronous retrieval branch exceeds its allocated timeout."""
    pass


class RerankingError(PipelineError):
    """Raised when neural or mock reranking fails."""
    pass


class GenerationError(PipelineError):
    """Base exception for LLM generation failures."""
    pass


class ProviderError(GenerationError):
    """Raised when an upstream LLM provider returns an error."""
    pass


class ProviderAuthenticationError(ProviderError):
    """Raised when LLM provider authentication or API credentials fail."""
    pass


class ProviderQuotaExhaustedError(ProviderError):
    """Raised when an LLM provider hard quota (daily/project/free-tier) is exhausted."""
    pass


class ProviderTimeoutError(GenerationError):
    """Raised when an LLM provider request exceeds its allocated timeout."""
    pass


class StreamInterruptedError(GenerationError):
    """Raised when an active token stream is prematurely terminated or corrupted."""
    pass

