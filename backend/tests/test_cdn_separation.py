"""PARALLAX — Frontend CDN Separation, Caching & Header Tests."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from app.core.config import settings
from app.main import app


def test_api_responses_strictly_no_store() -> None:
    """Verify that all API responses enforce strict NO-STORE to prevent CDN caching."""
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        cache_control = response.headers.get("Cache-Control", "")
        assert "no-store" in cache_control
        assert "no-cache" in cache_control
        assert "private" in cache_control
        assert response.headers.get("Pragma") == "no-cache"
        assert response.headers.get("Expires") == "0"


def test_request_id_generation_and_propagation() -> None:
    with TestClient(app) as client:
        # 1. Automatic generation when absent
        resp1 = client.get("/health/liveness")
        req_id1 = resp1.headers.get("X-Request-ID")
        assert req_id1 is not None
        assert len(req_id1) >= 16

        # 2. Propagation when supplied by client or edge CDN
        custom_id = "trace-edge-cdn-xyz-888"
        resp2 = client.get("/health/liveness", headers={"X-Request-ID": custom_id})
        assert resp2.headers.get("X-Request-ID") == custom_id


def test_global_security_headers_present() -> None:
    with TestClient(app) as client:
        response = client.get("/health/liveness")
        headers = response.headers

        assert headers.get("X-Content-Type-Options") == "nosniff"
        assert headers.get("X-Frame-Options") == "DENY"
        assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert "camera=()" in headers.get("Permissions-Policy", "")
        assert "default-src 'self'" in headers.get("Content-Security-Policy", "")


def test_fastapi_does_not_serve_spa_when_serve_frontend_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When SERVE_FRONTEND is false or in production, should_serve_frontend is False."""
    # Explicitly disabled
    monkeypatch.setattr(settings, "SERVE_FRONTEND", False)
    assert settings.should_serve_frontend is False

    # Default in production environment
    monkeypatch.setattr(settings, "SERVE_FRONTEND", None)
    monkeypatch.setattr(settings, "APP_ENV", "production")
    assert settings.should_serve_frontend is False

    # Default in development environment (supports local dev without CDN)
    monkeypatch.setattr(settings, "APP_ENV", "development")
    assert settings.should_serve_frontend is True
