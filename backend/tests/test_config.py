"""Tests for FastRAG Configuration and validation."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def test_default_configuration() -> None:
    """Verify default configuration has expected production-safe development values."""
    settings = Settings()
    assert settings.app_name == "FastRAG"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000


def test_environment_variable_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify environment variables properly override default settings."""
    monkeypatch.setenv("APP_NAME", "FastRAG-Cluster")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("API_V1_PREFIX", "/api/v2")
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "9090")

    settings = Settings()
    assert settings.app_name == "FastRAG-Cluster"
    assert settings.environment == "production"
    assert settings.log_level == "WARNING"
    assert settings.api_v1_prefix == "/api/v2"
    assert settings.host == "127.0.0.1"
    assert settings.port == 9090


def test_case_insensitive_and_whitespace_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify environment variables handle case variance and whitespace gracefully."""
    monkeypatch.setenv("ENVIRONMENT", "  Staging  ")
    monkeypatch.setenv("LOG_LEVEL", "debug")

    settings = Settings()
    assert settings.environment == "staging"
    assert settings.log_level == "DEBUG"


def test_invalid_environment_validation() -> None:
    """Verify invalid environment raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(environment="invalid_env")  # type: ignore[arg-type]
    assert "Invalid environment 'invalid_env'" in str(exc_info.value)


def test_invalid_log_level_validation() -> None:
    """Verify invalid log level raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(log_level="VERBOSE")  # type: ignore[arg-type]
    assert "Invalid log_level 'VERBOSE'" in str(exc_info.value)


def test_invalid_port_validation() -> None:
    """Verify port numbers outside standard range (1-65535) raise validation error."""
    with pytest.raises(ValidationError):
        Settings(port=70000)

    with pytest.raises(ValidationError):
        Settings(port=0)


def test_get_settings_caching() -> None:
    """Verify get_settings returns a cached instance and can be cleared."""
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()


def test_logger_setup() -> None:
    """Verify setup_logging configures root and module loggers."""
    from app.core.logging_config import setup_logging, get_logger

    setup_logging("DEBUG")
    logger = get_logger("fastrag.test")
    assert logger is not None
    import logging
    assert logging.getLogger().level == logging.DEBUG


def test_measure_latency_context_manager(caplog: pytest.LogCaptureFixture) -> None:
    """Verify measure_latency logs operation duration with proper tags."""
    import logging
    from app.core.logging_config import get_logger, measure_latency

    logger = get_logger("fastrag.latency_test")
    with caplog.at_level(logging.INFO):
        with measure_latency(logger, "test_operation") as ctx:
            ctx["custom_meta"] = "test_val"

    assert len(caplog.records) >= 1
    record = caplog.records[-1]
    assert "Operation 'test_operation' completed in" in record.message
    assert getattr(record, "operation", None) == "test_operation"
    assert hasattr(record, "latency_ms")
    assert record.latency_ms >= 0.0


def test_llm_environment_variable_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify FASTRAG_LLM_* environment variables properly override LLM settings."""
    monkeypatch.setenv("FASTRAG_LLM_PROVIDER", "openai")
    monkeypatch.setenv("FASTRAG_LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv(
        "FASTRAG_LLM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )
    monkeypatch.setenv("FASTRAG_LLM_API_KEY", "dummy-gemini-key-12345")
    monkeypatch.setenv("FASTRAG_LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("FASTRAG_LLM_MAX_TOKENS", "2048")
    monkeypatch.setenv("FASTRAG_LLM_TIMEOUT_SECONDS", "45.0")
    monkeypatch.setenv("FASTRAG_LLM_STREAMING", "false")

    settings = Settings()
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gemini-3.8-flash"
    assert settings.llm_base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert settings.llm_api_key == "dummy-gemini-key-12345"
    assert settings.llm_temperature == 0.7
    assert settings.llm_max_tokens == 2048
    assert settings.llm_timeout_seconds == 45.0
    assert settings.llm_streaming is False


def test_openai_api_key_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fallback to OPENAI_API_KEY and precedence of FASTRAG_LLM_API_KEY."""
    monkeypatch.delenv("FASTRAG_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-dummy-openai-key")

    settings = Settings()
    assert settings.llm_api_key == "fallback-dummy-openai-key"

    # Precedence test: FASTRAG_LLM_API_KEY overrides OPENAI_API_KEY
    monkeypatch.setenv("FASTRAG_LLM_API_KEY", "primary-dummy-fastrag-key")
    settings2 = Settings()
    assert settings2.llm_api_key == "primary-dummy-fastrag-key"


def test_api_key_secret_masking() -> None:
    """Verify API keys are never exposed in repr or safe dictionary serialization."""
    settings = Settings(llm_api_key="secret-sensitive-token-999")
    # repr must not contain the raw key
    assert "secret-sensitive-token-999" not in repr(settings)
    assert "secret-sensitive-token-999" not in str(settings)

    # safe dictionary masks the key
    safe_data = settings.to_safe_dict()
    assert safe_data["llm_api_key"] == "***REDACTED***"


def test_invalid_base_url_validation() -> None:
    """Verify incorrect or markdown-formatted base URLs are rejected and not silently converted."""
    invalid_urls = [
        "[https://example.com](https://example.com)",
        "(https://example.com)",
        "[https://example.com]",
        "<https://example.com>",
        "`https://example.com`",
        "ftp://generativelanguage.googleapis.com",
        "not-a-valid-url",
        "https://",
    ]

    for bad_url in invalid_urls:
        with pytest.raises(ValidationError):
            Settings(llm_base_url=bad_url)


def test_mock_provider_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify mock provider configuration continues to work."""
    monkeypatch.setenv("FASTRAG_LLM_PROVIDER", "mock")
    monkeypatch.setenv("FASTRAG_LLM_MODEL", "mock-custom-model")

    settings = Settings()
    assert settings.llm_provider == "mock"
    assert settings.llm_model == "mock-custom-model"


