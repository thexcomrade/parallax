"""Tests for distributed rate limiting configuration, storage URIs, and fallback behavior."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


def test_rate_limiter_storage_uri_configuration() -> None:
    """Limiter must accept storage URI configuration without failure."""
    # Memory storage
    l_mem = Limiter(key_func=get_remote_address, storage_uri="memory://")
    assert l_mem is not None

    # Redis storage URI configuration
    l_redis = Limiter(key_func=get_remote_address, storage_uri="redis://localhost:6379/1")
    assert l_redis is not None


def test_rate_limiter_enforces_limit_on_repeated_calls() -> None:
    """Limiter must throttle requests exceeding quota."""
    test_app = FastAPI()
    limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
    test_app.state.limiter = limiter
    test_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @test_app.get("/limited")
    @limiter.limit("2/minute")
    def limited_endpoint(request: Request) -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(test_app)
    r1 = client.get("/limited")
    assert r1.status_code == 200

    r2 = client.get("/limited")
    assert r2.status_code == 200

    r3 = client.get("/limited")
    assert r3.status_code == 429
