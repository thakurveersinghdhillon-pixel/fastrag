"""Tests for FastRAG Health endpoint and application lifecycle."""

from fastapi.testclient import TestClient
import pytest

from app.main import app, create_app
from app.core.config import Settings


def test_application_import() -> None:
    """Verify that the FastAPI application instance and factory can be imported without error."""
    assert app is not None
    assert app.title == "FastRAG"


def test_health_endpoint_default() -> None:
    """Verify GET /health returns HTTP 200 with expected default payload."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "FastRAG"
    assert data["environment"] == "development"
    assert data["version"] == "0.1.0"


def test_health_endpoint_custom_settings() -> None:
    """Verify GET /health reflects custom settings when created via app factory."""
    custom_settings = Settings(
        app_name="FastRAG-Staging",
        environment="staging",
        log_level="DEBUG",
    )
    custom_app = create_app(settings=custom_settings)
    client = TestClient(custom_app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "FastRAG-Staging"
    assert data["environment"] == "staging"


def test_health_endpoint_not_found_on_invalid_route() -> None:
    """Verify 404 for unmapped endpoints to ensure clean routing."""
    client = TestClient(app)
    response = client.get("/nonexistent-endpoint")
    assert response.status_code == 404
