"""FastRAG Application Configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_CONFIG_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _CONFIG_DIR.parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent

EnvironmentType = Literal["development", "staging", "production", "test"]
LogLevelType = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Application settings loaded from environment variables with safe defaults."""

    app_name: str = Field(
        default="FastRAG",
        description="Name of the application",
    )
    environment: EnvironmentType = Field(
        default="development",
        description="Deployment environment (development, staging, production, test)",
    )
    log_level: LogLevelType = Field(
        default="INFO",
        description="Logging level for standard Python logging",
    )
    api_v1_prefix: str = Field(
        default="/api/v1",
        description="Base routing prefix for API version 1",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Server listening host",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server listening port",
    )
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="CrossEncoder model name for passage reranking",
    )
    reranker_batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for CrossEncoder inference",
    )
    reranker_device: str | None = Field(
        default=None,
        description="Device for CrossEncoder execution (e.g. cpu, cuda, mps)",
    )
    rerank_blending_alpha: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "FASTRAG_RERANK_BLENDING_ALPHA",
            "RERANK_BLENDING_ALPHA",
            "rerank_blending_alpha",
        ),
        description="Reciprocal rank blending weight (alpha >= 0.0) between CrossEncoder rank and base retriever rank",
    )
    default_top_k: int = Field(
        default=10,
        ge=1,
        description="Default top-K results to return after reranking",
    )
    context_max_tokens: int = Field(
        default=1000,
        ge=1,
        description="Maximum approximate tokens for assembled context",
    )
    context_max_chunks: int = Field(
        default=5,
        ge=1,
        description="Maximum number of chunks to include in assembled context",
    )
    context_min_relevance_score: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "FASTRAG_CONTEXT_MIN_RELEVANCE_SCORE",
            "CONTEXT_MIN_RELEVANCE_SCORE",
            "context_min_relevance_score",
        ),
        description="Optional minimum relevance score threshold for candidate context filtering",
    )
    context_ordering_strategy: Literal["descending", "boundary"] = Field(
        default="descending",
        validation_alias=AliasChoices(
            "FASTRAG_CONTEXT_ORDERING_STRATEGY",
            "CONTEXT_ORDERING_STRATEGY",
            "context_ordering_strategy",
        ),
        description="Context ordering strategy: 'descending' or 'boundary'",
    )
    llm_provider: str = Field(
        default="openai",
        validation_alias=AliasChoices(
            "FASTRAG_LLM_PROVIDER", "LLM_PROVIDER", "llm_provider"
        ),
        description="LLM provider name: 'openai' or 'mock'",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices(
            "FASTRAG_LLM_MODEL", "LLM_MODEL", "llm_model"
        ),
        description="LLM model identifier (e.g. 'gemini-3.8-flash', 'gpt-4o-mini')",
    )
    llm_api_key: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices(
            "FASTRAG_LLM_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "llm_api_key"
        ),
        description="API key for hosted LLM provider (or set FASTRAG_LLM_API_KEY / OPENAI_API_KEY)",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices(
            "FASTRAG_LLM_BASE_URL", "LLM_BASE_URL", "llm_base_url"
        ),
        description="Base URL for OpenAI-compatible LLM endpoint",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        validation_alias=AliasChoices(
            "FASTRAG_LLM_TEMPERATURE", "LLM_TEMPERATURE", "llm_temperature"
        ),
        description="Sampling temperature for grounded generation",
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=1,
        le=16384,
        validation_alias=AliasChoices(
            "FASTRAG_LLM_MAX_TOKENS", "LLM_MAX_TOKENS", "llm_max_tokens"
        ),
        description="Maximum completion tokens to generate",
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        validation_alias=AliasChoices(
            "FASTRAG_LLM_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS", "llm_timeout_seconds"
        ),
        description="Timeout in seconds for provider requests",
    )
    llm_streaming: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "FASTRAG_LLM_STREAMING", "LLM_STREAMING", "llm_streaming"
        ),
        description="Whether streaming is enabled by default for generation",
    )
    enable_embedding_cache: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "FASTRAG_ENABLE_EMBEDDING_CACHE", "ENABLE_EMBEDDING_CACHE", "enable_embedding_cache"
        ),
        description="Whether to enable Tier 2 query embedding vector cache",
    )
    embedding_cache_size: int = Field(
        default=10000,
        ge=1,
        validation_alias=AliasChoices(
            "FASTRAG_EMBEDDING_CACHE_SIZE", "EMBEDDING_CACHE_SIZE", "embedding_cache_size"
        ),
        description="Maximum entries in Tier 2 query embedding cache",
    )
    enable_query_cache: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "FASTRAG_ENABLE_QUERY_CACHE", "ENABLE_QUERY_CACHE", "enable_query_cache"
        ),
        description="Whether to enable Tier 1 query response cache",
    )
    query_cache_size: int = Field(
        default=2000,
        ge=1,
        validation_alias=AliasChoices(
            "FASTRAG_QUERY_CACHE_SIZE", "QUERY_CACHE_SIZE", "query_cache_size"
        ),
        description="Maximum entries in Tier 1 query response cache",
    )
    query_cache_ttl_seconds: float = Field(
        default=3600.0,
        gt=0.0,
        validation_alias=AliasChoices(
            "FASTRAG_QUERY_CACHE_TTL_SECONDS", "QUERY_CACHE_TTL_SECONDS", "query_cache_ttl_seconds"
        ),
        description="Time-to-live in seconds for Tier 1 query response cache entries",
    )

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, v: str) -> str:
        if isinstance(v, str):
            normalized = v.strip().lower()
            allowed = {"development", "staging", "production", "test"}
            if normalized not in allowed:
                raise ValueError(
                    f"Invalid environment '{v}'. Must be one of: {', '.join(sorted(allowed))}"
                )
            return normalized
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        if isinstance(v, str):
            normalized = v.strip().upper()
            allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if normalized not in allowed:
                raise ValueError(
                    f"Invalid log_level '{v}'. Must be one of: {', '.join(sorted(allowed))}"
                )
            return normalized
        return v

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def validate_and_clean_base_url(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError(f"llm_base_url must be a string, got {type(v).__name__}")
        cleaned = v.strip()
        # Reject markdown links or formatting brackets/parentheses
        for forbidden in ("[", "]", "(", ")", "<", ">", "`", '"', "'"):
            if forbidden in cleaned:
                raise ValueError(
                    f"Invalid base URL '{cleaned}': base URL must be a plain URL without markdown formatting, brackets, or parentheses."
                )
        parsed = urlparse(cleaned)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"Invalid base URL '{cleaned}': base URL must be a valid HTTP or HTTPS URL with a network location."
            )
        return cleaned.rstrip("/")

    @field_validator("llm_api_key", mode="before")
    @classmethod
    def resolve_api_key(cls, v: str | None) -> str | None:
        if v is not None and isinstance(v, str) and v.strip():
            return v.strip()
        import os
        return os.getenv("FASTRAG_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or None

    def to_safe_dict(self) -> dict[str, Any]:
        """Return configuration dictionary with sensitive secrets masked."""
        data = self.model_dump()
        if data.get("llm_api_key"):
            data["llm_api_key"] = "***REDACTED***"
        return data

    model_config = SettingsConfigDict(
        env_file=(
            _PROJECT_ROOT / ".env",
            _BACKEND_DIR / ".env",
            ".env",
        ),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
