"""PARALLAX — Prometheus Metrics & Observability Tests."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from app.core.config import settings
from app.core.metrics import (
    AUTH_FAILURES_TOTAL,
    HTTP_REQUESTS_TOTAL,
    collect_runtime_metrics,
    get_latest_metrics,
)
from app.main import app


def test_prometheus_metrics_endpoint_returns_exposition() -> None:
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        body = response.text

        # Verify key application metrics exist in exposition
        assert "parallax_http_requests_total" in body
        assert "parallax_db_connected" in body
        assert "parallax_redis_connected" in body
        assert "parallax_outbox_backlog_count" in body
        assert "parallax_gpu_available" in body


def test_metrics_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "METRICS_ENABLED", False)
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 404


def test_http_request_metric_increment() -> None:
    initial_count = HTTP_REQUESTS_TOTAL.labels(
        method="GET", endpoint="/health/liveness", status="200"
    )._value.get()

    with TestClient(app) as client:
        resp = client.get("/health/liveness")
        assert resp.status_code == 200

    new_count = HTTP_REQUESTS_TOTAL.labels(
        method="GET", endpoint="/health/liveness", status="200"
    )._value.get()
    assert new_count == initial_count + 1


def test_auth_failure_metric_increment() -> None:
    initial_fails = AUTH_FAILURES_TOTAL.labels(reason="invalid_credentials")._value.get()

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "nonexistent@parallax.internal", "password": "WrongPassword123!"},
        )
        assert resp.status_code in (400, 401)

    new_fails = AUTH_FAILURES_TOTAL.labels(reason="invalid_credentials")._value.get()
    assert new_fails == initial_fails + 1


@pytest.mark.asyncio
async def test_dynamic_runtime_metrics_collector() -> None:
    await collect_runtime_metrics()
    metrics_bytes = await get_latest_metrics()
    exposition = metrics_bytes.decode("utf-8")

    assert "# TYPE parallax_outbox_backlog_count gauge" in exposition
    assert "# TYPE parallax_db_pool_size gauge" in exposition
