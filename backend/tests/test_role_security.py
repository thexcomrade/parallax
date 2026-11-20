"""Role & Authorization Security Tests for PARALLAX.

Audits and verifies:
1. Newly registered users receive strictly the least-privileged 'public' role.
2. Ordinary users cannot escalate to 'operator' or 'admin' through registration payloads.
3. Ordinary users cannot escalate their role through profile routes or unprivileged role endpoints.
4. Only explicitly authorized administrators can grant 'operator' or 'admin' roles.
5. Existing operator accounts retain full access to forensic telemetry and artifacts.
6. Public users and anonymous callers cannot access internal analysis metrics or artifacts.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import create_access_token
from app.db.models import Analysis, AnalysisArtifact, User
from app.services.auth_service import AuthService


@pytest.fixture
def admin_user(db_session: Session) -> User:
    """Create a verified administrator user."""
    return AuthService.register_user(
        db=db_session,
        email="admin_security@parallax.ai",
        password="ValidPassword123!",
        role="admin",
    )


@pytest.fixture
def admin_token(admin_user: User) -> str:
    return create_access_token({"sub": str(admin_user.id), "role": "admin"})


@pytest.fixture
def operator_user(db_session: Session) -> User:
    """Create a verified operator user."""
    return AuthService.register_user(
        db=db_session,
        email="operator_analyst@parallax.ai",
        password="ValidPassword123!",
        role="operator",
    )


@pytest.fixture
def operator_token(operator_user: User) -> str:
    return create_access_token({"sub": str(operator_user.id), "role": "operator"})


@pytest.fixture
def public_user(db_session: Session) -> User:
    """Create a standard public user."""
    return AuthService.register_user(
        db=db_session,
        email="public_citizen@parallax.ai",
        password="ValidPassword123!",
        role="public",
    )


@pytest.fixture
def public_token(public_user: User) -> str:
    return create_access_token({"sub": str(public_user.id), "role": "public"})


@pytest.fixture
def operator_analysis(db_session: Session, operator_user: User) -> Analysis:
    """Create a sample analysis owned by operator."""
    analysis = Analysis(
        id="op-forensic-sample-01",
        user_id=operator_user.id,
        original_name="target_clip.mp4",
        sha256="1111222233334444" * 4,
        size_bytes=1048576,
        duration_s=4.0,
        status="completed",
        stage="done",
        verdict="FAKE",
        p_fake=0.9123,
        w_audio=0.150,
        w_video=0.850,
        sync_score=0.789,
        face_coverage=1.0,
        audio_present=True,
        model_version="1.0.0",
        model_mode="trained",
        timeline_json=json.dumps([{"t": 1.0, "p_face": 0.9, "p_audio": 0.8, "p_fused": 0.91}]),
        reasons_json=json.dumps(["Facial manipulation detected"]),
    )
    db_session.add(analysis)
    db_session.commit()
    db_session.refresh(analysis)

    art = AnalysisArtifact(
        analysis_id=analysis.id,
        kind="heatmap",
        name="cam_overlay_000.png",
        path="artifacts/op-forensic-sample-01/cam_overlay_000.png",
    )
    db_session.add(art)
    db_session.commit()
    return analysis


def test_new_registration_receives_public_role(client: TestClient) -> None:
    """Newly registered users receive strictly the least-privileged 'public' role."""
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": "newbie@parallax.ai", "password": "StrongPassword123!"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["role"] == "public", f"Expected role 'public', got {data['role']}"


def test_registration_role_escalation_attempt_neutralized(client: TestClient) -> None:
    """Attempting to self-register with 'operator' or 'admin' role is rejected or neutralized."""
    # Attempt 1: send role=operator
    resp1 = client.post(
        "/api/v1/auth/register",
        json={
            "email": "hacker1@parallax.ai",
            "password": "StrongPassword123!",
            "role": "operator",
        },
    )
    assert resp1.status_code in (201, 422)
    if resp1.status_code == 201:
        assert resp1.json()["role"] == "public", "Escalation to operator must be denied"

    # Attempt 2: send role=admin
    resp2 = client.post(
        "/api/v1/auth/register",
        json={
            "email": "hacker2@parallax.ai",
            "password": "StrongPassword123!",
            "role": "admin",
        },
    )
    assert resp2.status_code in (201, 422)
    if resp2.status_code == 201:
        assert resp2.json()["role"] == "public", "Escalation to admin must be denied"


def test_ordinary_user_cannot_update_own_role(
    client: TestClient,
    public_token: str,
    public_user: User,
) -> None:
    """Ordinary user calling admin role endpoint receives 403 Forbidden."""
    resp = client.patch(
        f"/api/v1/auth/users/{public_user.id}/role",
        headers={"Authorization": f"Bearer {public_token}"},
        json={"role": "operator"},
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "FORBIDDEN"


def test_unauthenticated_cannot_update_role(
    client: TestClient,
    public_user: User,
) -> None:
    """Unauthenticated caller attempting to update role receives 401 Unauthorized."""
    resp = client.patch(
        f"/api/v1/auth/users/{public_user.id}/role",
        json={"role": "operator"},
    )
    assert resp.status_code == 401
    assert resp.json()["code"] == "UNAUTHORIZED"


def test_admin_can_grant_operator_role(
    client: TestClient,
    admin_token: str,
    public_user: User,
) -> None:
    """Explicitly authorized administrator can grant operator role."""
    resp = client.patch(
        f"/api/v1/auth/users/{public_user.id}/role",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"role": "operator"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == public_user.id
    assert data["role"] == "operator"


def test_admin_rejects_invalid_role(
    client: TestClient,
    admin_token: str,
    public_user: User,
) -> None:
    """Administrator passing invalid role string receives 422 Validation Error."""
    resp = client.patch(
        f"/api/v1/auth/users/{public_user.id}/role",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"role": "supergod"},
    )
    assert resp.status_code == 422


def test_existing_operator_retains_required_access(
    client: TestClient,
    operator_token: str,
    operator_analysis: Analysis,
) -> None:
    """Existing operator account retains full access to internal telemetry and artifacts."""
    resp = client.get(
        f"/api/v1/analyses/{operator_analysis.id}",
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "FAKE"
    assert data["p_fake"] == pytest.approx(0.9123, rel=1e-3)
    assert data["w_audio"] == pytest.approx(0.150, rel=1e-3)
    assert data["w_video"] == pytest.approx(0.850, rel=1e-3)
    assert data["sync_score"] == pytest.approx(0.789, rel=1e-3)
    assert "timeline" in data and len(data["timeline"]) == 1
    assert "artifacts" in data and len(data["artifacts"]) == 1


def test_public_user_cannot_access_internal_metrics_or_artifacts(
    client: TestClient,
    public_token: str,
    public_user: User,
    db_session: Session,
) -> None:
    """Public user receives only binary verdict and cannot access artifacts."""
    # Create analysis for public user
    analysis = Analysis(
        id="pub-eval-sample-02",
        user_id=public_user.id,
        original_name="test_pub.mp4",
        sha256="5555666677778888" * 4,
        size_bytes=1048576,
        duration_s=5.0,
        status="completed",
        stage="done",
        verdict="AUTHENTIC",
        p_fake=0.045,
        w_audio=0.02,
        w_video=0.98,
        sync_score=0.95,
        audio_present=True,
        model_version="1.0.0",
        model_mode="trained",
    )
    db_session.add(analysis)
    db_session.commit()

    resp = client.get(
        f"/api/v1/analyses/{analysis.id}",
        headers={"Authorization": f"Bearer {public_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Binary verdict only
    assert data["verdict"] == "REAL"
    assert data.get("p_fake") is None
    assert data.get("w_audio") is None
    assert data.get("w_video") is None
    assert data.get("sync_score") is None
    assert data.get("timeline") is None
    assert data.get("artifacts") is None

    # Artifact download must be 403 Forbidden
    art_resp = client.get(
        f"/api/v1/analyses/{analysis.id}/artifacts/cam_overlay_000.png",
        headers={"Authorization": f"Bearer {public_token}"},
    )
    assert art_resp.status_code == 403
    assert art_resp.json()["code"] == "FORBIDDEN"


def test_anonymous_user_receives_only_intended_public_response(
    client: TestClient,
    operator_analysis: Analysis,
) -> None:
    """Anonymous user accessing /analyses/{id}/public receives only binary verdict."""
    resp = client.get(f"/api/v1/analyses/{operator_analysis.id}/public")
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "FAKE"
    assert data["id"] == operator_analysis.id
    assert data.get("p_fake") is None
    assert data.get("w_audio") is None
    assert data.get("w_video") is None
    assert data.get("timeline") is None
    assert data.get("artifacts") is None
