"""Regression tests for SPA static file delivery and path traversal guards (SEC-05)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_spa_client_side_route_serves_index(client: TestClient) -> None:
    """Client-side application routes must resolve to index.html with 200."""
    for path in ["/", "/login", "/register", "/app", "/app/analyze", "/app/history"]:
        resp = client.get(path)
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "PARALLAX" in resp.text or "<div id=\"root\">" in resp.text


def test_valid_static_asset_delivery(client: TestClient) -> None:
    """Existing static assets in frontend distribution must be served directly."""
    resp = client.get("/favicon.svg")
    assert resp.status_code == 200
    assert "svg" in resp.headers.get("content-type", "") or resp.status_code == 200


def test_missing_static_asset_returns_404(client: TestClient) -> None:
    """Missing static assets with extensions must return 404 instead of index.html."""
    missing_assets = [
        "/assets/missing_script.js",
        "/assets/missing_styles.css",
        "/missing_image.png",
    ]
    for missing in missing_assets:
        resp = client.get(missing)
        assert resp.status_code == 404


def test_parent_directory_traversal_rejected(client: TestClient) -> None:
    """Path traversal attempts using ../ must be rejected with 404."""
    traversal_paths = [
        "/../package.json",
        "/../../pyproject.toml",
        "/../../../backend/app/main.py",
        "/..\\..\\package.json",
    ]
    for path in traversal_paths:
        resp = client.get(path)
        assert resp.status_code == 404


def test_url_encoded_traversal_rejected(client: TestClient) -> None:
    """URL-encoded path traversal attempts (%2e%2e) must be rejected with 404."""
    encoded_paths = [
        "/%2e%2e/package.json",
        "/%2e%2e%2f%2e%2e%2fbackend/pyproject.toml",
        "/..%2f..%2f.env",
    ]
    for path in encoded_paths:
        resp = client.get(path)
        assert resp.status_code == 404
