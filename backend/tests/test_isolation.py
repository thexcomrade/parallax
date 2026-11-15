"""Cross-user data isolation tests.

Security requirements (README §8.8, §12 check 07):
- User B cannot list, read, stream or delete User A's analyses or artifacts.
- Every endpoint returns 404 (NOT 403) for foreign analysis IDs to prevent
  cross-user resource enumeration.
- Traversal attempts on artifact names (e.g., ../../secret) are rejected.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import create_access_token
from app.db.models import Analysis, AnalysisArtifact, User
from app.services.auth_service import AuthService


def _auth_headers(user: User) -> dict[str, str]:
    token = create_access_token(data={"sub": str(user.id)})
    return {"Authorization": f"Bearer {token}"}


def _create_user(db: Session, email: str) -> User:
    return AuthService.register_user(
        db=db,
        email=email,
        password="SecurePassword2026!Long",
    )


def _seed_analysis(
    db: Session,
    user: User,
    analysis_id: str | None = None,
    original_name: str = "sample_clip.mp4",
) -> Analysis:
    aid = analysis_id or str(uuid.uuid4())
    analysis = Analysis(
        id=aid,
        user_id=user.id,
        original_name=original_name,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        size_bytes=1024 * 1024,
        duration_s=3.0,
        status="completed",
        stage="done",
        verdict="AUTHENTIC",
        p_fake=0.15,
        w_audio=0.45,
        w_video=0.55,
        sync_score=0.82,
        face_coverage=0.95,
        audio_present=True,
        model_version="0.0.0",
        model_mode="untrained",
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def _seed_artifact(
    db: Session,
    analysis: Analysis,
    name: str = "heatmap_frame_001.png",
    tmp_path: Path | None = None,
) -> AnalysisArtifact:
    if tmp_path:
        file_path = tmp_path / name
        file_path.write_bytes(b"\x89PNG\r\n\x1a\nfake_png_data")
        saved_path = str(file_path)
    else:
        saved_path = f"/tmp/{name}"

    artifact = AnalysisArtifact(
        analysis_id=analysis.id,
        kind="heatmap",
        name=name,
        path=saved_path,
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


def test_user_b_cannot_get_user_a_analysis_detail(
    client: TestClient,
    db_session: Session,
) -> None:
    """User B requesting User A's analysis ID receives 404 (not 403)."""
    user_a = _create_user(db_session, "user_a@test.io")
    user_b = _create_user(db_session, "user_b@test.io")
    analysis_a = _seed_analysis(db_session, user_a)

    headers_b = _auth_headers(user_b)
    resp = client.get(f"/api/v1/analyses/{analysis_a.id}", headers=headers_b)

    assert resp.status_code == 404
    data = resp.json()
    assert data["code"] == "NOT_FOUND"


def test_user_b_cannot_stream_user_a_events(
    client: TestClient,
    db_session: Session,
) -> None:
    """User B opening SSE events stream for User A's analysis receives 404."""
    user_a = _create_user(db_session, "user_a_sse@test.io")
    user_b = _create_user(db_session, "user_b_sse@test.io")
    analysis_a = _seed_analysis(db_session, user_a)

    headers_b = _auth_headers(user_b)
    resp = client.get(f"/api/v1/analyses/{analysis_a.id}/events", headers=headers_b)

    assert resp.status_code == 404
    assert resp.json()["code"] == "NOT_FOUND"


def test_user_b_cannot_get_user_a_artifact(
    client: TestClient,
    db_session: Session,
    tmp_path: Path,
) -> None:
    """User B requesting an artifact of User A's analysis receives 404."""
    user_a = _create_user(db_session, "user_a_art@test.io")
    user_b = _create_user(db_session, "user_b_art@test.io")
    analysis_a = _seed_analysis(db_session, user_a)
    artifact = _seed_artifact(db_session, analysis_a, tmp_path=tmp_path)

    headers_b = _auth_headers(user_b)
    resp = client.get(
        f"/api/v1/analyses/{analysis_a.id}/artifacts/{artifact.name}",
        headers=headers_b,
    )

    assert resp.status_code == 404
    assert resp.json()["code"] == "NOT_FOUND"


def test_user_b_cannot_delete_user_a_analysis(
    client: TestClient,
    db_session: Session,
) -> None:
    """User B attempting to delete User A's analysis receives 404."""
    user_a = _create_user(db_session, "user_a_del@test.io")
    user_b = _create_user(db_session, "user_b_del@test.io")
    analysis_a = _seed_analysis(db_session, user_a)

    headers_b = _auth_headers(user_b)
    resp = client.delete(f"/api/v1/analyses/{analysis_a.id}", headers=headers_b)

    assert resp.status_code == 404
    assert resp.json()["code"] == "NOT_FOUND"

    # Confirm record was not deleted
    refreshed = db_session.get(Analysis, analysis_a.id)
    assert refreshed is not None


def test_list_analyses_only_returns_own_records(
    client: TestClient,
    db_session: Session,
) -> None:
    """GET /analyses only returns records owned by the requesting user."""
    user_a = _create_user(db_session, "user_a_list@test.io")
    user_b = _create_user(db_session, "user_b_list@test.io")

    _seed_analysis(db_session, user_a, original_name="clip_a1.mp4")
    _seed_analysis(db_session, user_a, original_name="clip_a2.mp4")
    analysis_b = _seed_analysis(db_session, user_b, original_name="clip_b1.mp4")

    # User B lists
    headers_b = _auth_headers(user_b)
    resp_b = client.get("/api/v1/analyses", headers=headers_b)
    assert resp_b.status_code == 200
    data_b = resp_b.json()
    assert data_b["total"] == 1
    assert len(data_b["items"]) == 1
    assert data_b["items"][0]["id"] == analysis_b.id
    assert data_b["items"][0]["original_name"] == "clip_b1.mp4"

    # User A lists
    headers_a = _auth_headers(user_a)
    resp_a = client.get("/api/v1/analyses", headers=headers_a)
    assert resp_a.status_code == 200
    data_a = resp_a.json()
    assert data_a["total"] == 2
    assert {it["original_name"] for it in data_a["items"]} == {"clip_a1.mp4", "clip_a2.mp4"}


def test_artifact_traversal_guard(
    client: TestClient,
    db_session: Session,
    tmp_path: Path,
) -> None:
    """Path traversal sequences in artifact name must be rejected."""
    user_a = _create_user(db_session, "user_trav@test.io")
    analysis_a = _seed_analysis(db_session, user_a)

    headers_a = _auth_headers(user_a)

    # Various traversal attempts
    bad_names = [
        "../../etc/passwd",
        "..%2F..%2Fetc%2Fpasswd",
        "..\\..\\windows\\system32",
        "sub/../../secret.txt",
    ]
    for bad_name in bad_names:
        resp = client.get(
            f"/api/v1/analyses/{analysis_a.id}/artifacts/{bad_name}",
            headers=headers_a,
        )
        # Must be rejected (400, 404, or 422), never allowed (200)
        assert resp.status_code in (400, 404, 422)
        assert resp.status_code != 200
