"""API Security & Access Control Tests for PARALLAX.

Verifies role-based separation between public users and authorized forensic operators:
1. Public users receive strictly binary verdict ('REAL' or 'FAKE') and non-diagnostic metadata.
2. Internal probabilities (p_fake), weights (w_audio, w_video), modality scores,
   synchronization metrics (sync_score), and timelines are NEVER exposed to unprivileged users.
3. Forensic artifacts (heatmaps, spectrograms) are restricted to operators (403 Forbidden).
4. Unauthenticated public endpoint (/analyses/{id}/public) strictly exposes binary verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import create_access_token
from app.db.models import Analysis, AnalysisArtifact, User
from app.services.auth_service import AuthService


@pytest.fixture
def operator_user(db_session: Session) -> User:
    """Create an authorized operator user."""
    return AuthService.register_user(
        db=db_session,
        email="operator_secure@parallax.ai",
        password="ValidPassword123!",
        role="operator",
    )


@pytest.fixture
def public_user(db_session: Session) -> User:
    """Create an unprivileged public user."""
    return AuthService.register_user(
        db=db_session,
        email="public_guest@parallax.ai",
        password="ValidPassword123!",
        role="public",
    )


@pytest.fixture
def operator_token(operator_user: User) -> str:
    return create_access_token({"sub": str(operator_user.id), "role": "operator"})


@pytest.fixture
def public_token(public_user: User) -> str:
    return create_access_token({"sub": str(public_user.id), "role": "public"})


@pytest.fixture
def mock_operator_analysis(db_session: Session, operator_user: User, tmp_path: Path) -> Analysis:
    """Create a completed mock analysis for operator user."""
    analysis = Analysis(
        id="mock-op-analysis-001",
        user_id=operator_user.id,
        original_name="interview_fake.mp4",
        sha256="abc123def456" * 5 + "abcd",
        size_bytes=1048576,
        duration_s=4.0,
        status="completed",
        stage="done",
        verdict="FAKE",
        p_fake=0.9421,
        w_audio=0.231,
        w_video=0.769,
        sync_score=0.884,
        face_coverage=0.98,
        audio_present=True,
        model_version="1.0.0",
        model_mode="trained",
        timeline_json=json.dumps(
            [
                {"t": 1.0, "p_face": 0.92, "p_audio": 0.85, "p_fused": 0.94},
                {"t": 2.0, "p_face": 0.95, "p_audio": 0.88, "p_fused": 0.96},
            ]
        ),
        reasons_json=json.dumps(["Visual facial artifacts detected", "Audio-visual desync"]),
    )
    db_session.add(analysis)
    db_session.commit()
    db_session.refresh(analysis)

    # Add mock artifact
    art = AnalysisArtifact(
        analysis_id=analysis.id,
        kind="heatmap",
        name="cam_overlay_000.png",
        path="artifacts/mock-op-analysis-001/cam_overlay_000.png",
    )
    db_session.add(art)
    db_session.commit()
    return analysis


@pytest.fixture
def mock_public_analysis(db_session: Session, public_user: User) -> Analysis:
    """Create a completed mock analysis for public user with AUTHENTIC internal verdict."""
    analysis = Analysis(
        id="mock-pub-analysis-002",
        user_id=public_user.id,
        original_name="verified_real.mp4",
        sha256="123456abcdef" * 5 + "1234",
        size_bytes=2048576,
        duration_s=6.0,
        status="completed",
        stage="done",
        verdict="AUTHENTIC",
        p_fake=0.0382,
        w_audio=0.012,
        w_video=0.988,
        sync_score=0.991,
        face_coverage=1.0,
        audio_present=True,
        model_version="1.0.0",
        model_mode="trained",
        timeline_json=json.dumps(
            [
                {"t": 1.0, "p_face": 0.04, "p_audio": 0.02, "p_fused": 0.03},
                {"t": 2.0, "p_face": 0.05, "p_audio": 0.03, "p_fused": 0.04},
            ]
        ),
        reasons_json=json.dumps([]),
    )
    db_session.add(analysis)
    db_session.commit()
    db_session.refresh(analysis)
    return analysis


def test_operator_receives_full_analysis_detail(
    client: TestClient,
    operator_token: str,
    mock_operator_analysis: Analysis,
) -> None:
    """Authorized operator calling GET /analyses/{id} receives complete forensic telemetry."""
    resp = client.get(
        f"/api/v1/analyses/{mock_operator_analysis.id}",
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Operator gets all diagnostic telemetry
    assert data["id"] == mock_operator_analysis.id
    assert data["verdict"] == "FAKE"
    assert data["p_fake"] == pytest.approx(0.9421, rel=1e-3)
    assert data["w_audio"] == pytest.approx(0.231, rel=1e-3)
    assert data["w_video"] == pytest.approx(0.769, rel=1e-3)
    assert data["sync_score"] == pytest.approx(0.884, rel=1e-3)
    assert "timeline" in data and len(data["timeline"]) == 2
    assert "artifacts" in data and len(data["artifacts"]) == 1
    assert "reasons" in data and len(data["reasons"]) == 2


def test_public_user_receives_only_binary_verdict(
    client: TestClient,
    public_token: str,
    mock_public_analysis: Analysis,
) -> None:
    """Public user calling GET /analyses/{id} receives binary verdict without internal metrics."""
    resp = client.get(
        f"/api/v1/analyses/{mock_public_analysis.id}",
        headers={"Authorization": f"Bearer {public_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Public user receives sanitized binary verdict: 'AUTHENTIC' -> 'REAL'
    assert data["id"] == mock_public_analysis.id
    assert data["verdict"] == "REAL"
    assert data["status"] == "completed"
    assert data["stage"] == "done"
    assert data["duration_s"] == 6.0
    assert data["audio_present"] is True

    # Internal telemetry MUST NOT be exposed
    assert data.get("p_fake") is None
    assert data.get("w_audio") is None
    assert data.get("w_video") is None
    assert data.get("sync_score") is None
    assert data.get("timeline") is None
    assert data.get("artifacts") is None
    assert data.get("reasons") is None


def test_public_endpoint_unauthenticated_binary_verdict(
    client: TestClient,
    mock_public_analysis: Analysis,
) -> None:
    """Unauthenticated public endpoint /analyses/{id}/public returns binary verdict."""
    resp = client.get(f"/api/v1/analyses/{mock_public_analysis.id}/public")
    assert resp.status_code == 200
    data = resp.json()

    # Strictly binary verdict
    assert data["verdict"] == "REAL"
    assert data["id"] == mock_public_analysis.id
    assert data.get("p_fake") is None
    assert data.get("w_audio") is None
    assert data.get("w_video") is None
    assert data.get("sync_score") is None
    assert data.get("timeline") is None
    assert data.get("artifacts") is None


def test_public_endpoint_fake_binary_verdict(
    client: TestClient,
    mock_operator_analysis: Analysis,
) -> None:
    """Unauthenticated public endpoint on FAKE video returns strictly 'FAKE'."""
    resp = client.get(f"/api/v1/analyses/{mock_operator_analysis.id}/public")
    assert resp.status_code == 200
    data = resp.json()

    assert data["verdict"] == "FAKE"
    assert data.get("p_fake") is None
    assert data.get("w_audio") is None
    assert data.get("w_video") is None


def test_public_user_forbidden_from_artifacts(
    client: TestClient,
    public_token: str,
    mock_public_analysis: Analysis,
) -> None:
    """Public user attempting to download forensic artifact receives 403 Forbidden."""
    resp = client.get(
        f"/api/v1/analyses/{mock_public_analysis.id}/artifacts/cam_overlay_000.png",
        headers={"Authorization": f"Bearer {public_token}"},
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "FORBIDDEN"


def test_unauthenticated_caller_unauthorized_for_artifacts(
    client: TestClient,
    mock_operator_analysis: Analysis,
) -> None:
    """Unauthenticated caller attempting to download artifact receives 401 Unauthorized."""
    resp = client.get(
        f"/api/v1/analyses/{mock_operator_analysis.id}/artifacts/cam_overlay_000.png",
    )
    assert resp.status_code == 401
    assert resp.json()["code"] == "UNAUTHORIZED"


def test_list_analyses_redacts_p_fake_for_public_users(
    client: TestClient,
    public_token: str,
    mock_public_analysis: Analysis,
) -> None:
    """GET /analyses list endpoint redacts p_fake and normalizes verdict for public users."""
    resp = client.get(
        "/api/v1/analyses",
        headers={"Authorization": f"Bearer {public_token}"},
    )
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) >= 1
    target = next((item for item in items if item["id"] == mock_public_analysis.id), None)
    assert target is not None
    assert target["verdict"] == "REAL"
    assert target.get("p_fake") is None


def test_list_analyses_preserves_p_fake_for_operators(
    client: TestClient,
    operator_token: str,
    mock_operator_analysis: Analysis,
) -> None:
    """GET /analyses list endpoint retains p_fake for authorized operators."""
    resp = client.get(
        "/api/v1/analyses",
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) >= 1
    target = next((item for item in items if item["id"] == mock_operator_analysis.id), None)
    assert target is not None
    assert target["verdict"] == "FAKE"
    assert target["p_fake"] == pytest.approx(0.9421, rel=1e-3)
