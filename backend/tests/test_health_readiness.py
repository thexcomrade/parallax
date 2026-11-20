"""PARALLAX — Liveness, Readiness, and Dependency Health Tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from starlette.testclient import TestClient

from app.main import app


def test_liveness_probe_returns_alive() -> None:
    with TestClient(app) as client:
        response = client.get("/health/liveness")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

        # /health alias
        response_alias = client.get("/health")
        assert response_alias.status_code == 200
        assert response_alias.json() == {"status": "alive"}


def test_readiness_probe_healthy_dependencies() -> None:
    with TestClient(app) as client:
        response = client.get("/health/readiness")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["checks"]["database"] == "ok"
        assert data["checks"]["redis"] == "ok"
        assert data["checks"]["storage"] == "ok"


def test_readiness_probe_degraded_on_redis_failure() -> None:
    with patch(
        "app.core.redis.check_redis_health",
        new_callable=AsyncMock,
        return_value=False,
    ), TestClient(app) as client:
        response = client.get("/health/readiness")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unready"
        assert data["checks"]["redis"] == "unreachable"


def test_readiness_probe_does_not_leak_secrets() -> None:
    with TestClient(app) as client:
        response = client.get("/health/readiness")
        text = response.text

        # Verify no credentials or secret keys are exposed
        assert "password" not in text.lower()
        assert "secret" not in text.lower()
        assert "postgres:" not in text
        assert "redis:" not in text
