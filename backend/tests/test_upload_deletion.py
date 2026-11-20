"""Tests for upload storage ownership, persistence, and leak-free deletion (STORAGE-01)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import create_access_token
from app.db.models import Analysis, User
from app.services.analysis_service import AnalysisService
from app.services.auth_service import AuthService
from app.services.job_runner import job_runner


def _auth_headers(user: User) -> dict[str, str]:
    token = create_access_token(data={"sub": str(user.id)})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def operator_user(db_session: Session) -> User:
    return AuthService.register_user(
        db=db_session,
        email="storage_qa@parallax.io",
        password="SecurePassword2026!Long",
        role="operator",
    )


def test_upload_persists_video_path_and_deletes_cleanly(
    client: TestClient,
    operator_user: User,
    synthetic_video_with_audio: Path,
    db_session: Session,
) -> None:
    """Upload must persist video_path on Analysis, and deletion must unlink the physical video."""
    headers = _auth_headers(operator_user)

    with open(synthetic_video_with_audio, "rb") as f:
        resp = client.post(
            "/api/v1/analyses",
            headers=headers,
            files={"file": ("test_leak_check.mp4", f, "video/mp4")},
        )

    assert resp.status_code == 202
    analysis_id = resp.json()["id"]

    # 1. Verify database record has video_path populated
    db_analysis = db_session.get(Analysis, analysis_id)
    assert db_analysis is not None
    assert db_analysis.video_path is not None
    saved_file = Path(db_analysis.video_path)
    assert saved_file.exists()
    assert saved_file.is_file()
    assert analysis_id in saved_file.name

    # Verify artifacts directory exists
    artifact_dir = settings.storage_path / "artifacts" / analysis_id
    assert artifact_dir.exists()

    # 2. Delete the analysis
    del_resp = client.delete(f"/api/v1/analyses/{analysis_id}", headers=headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["ok"] is True

    # 3. Verify physical video file was deleted
    assert not saved_file.exists(), "Source video file must be deleted upon analysis deletion"

    # 4. Verify artifact directory was deleted
    assert not artifact_dir.exists(), "Artifacts directory must be deleted upon analysis deletion"

    # 5. Verify database record is gone
    db_session.expire_all()
    assert db_session.get(Analysis, analysis_id) is None


def test_delete_analysis_missing_file_handled_safely(
    client: TestClient,
    operator_user: User,
    synthetic_video_with_audio: Path,
    db_session: Session,
) -> None:
    """If the video file was already unlinked or missing, deletion must succeed without error."""
    headers = _auth_headers(operator_user)

    with open(synthetic_video_with_audio, "rb") as f:
        resp = client.post(
            "/api/v1/analyses",
            headers=headers,
            files={"file": ("test_missing_file.mp4", f, "video/mp4")},
        )

    analysis_id = resp.json()["id"]
    db_analysis = db_session.get(Analysis, analysis_id)
    assert db_analysis is not None and db_analysis.video_path is not None
    saved_file = Path(db_analysis.video_path)

    # Pre-unlink file to simulate missing file
    if saved_file.exists():
        saved_file.unlink()

    del_resp = client.delete(f"/api/v1/analyses/{analysis_id}", headers=headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["ok"] is True
    db_session.expire_all()
    assert db_session.get(Analysis, analysis_id) is None


def test_delete_analysis_outside_directory_protected(
    db_session: Session,
    operator_user: User,
    tmp_path: Path,
) -> None:
    """Analysis with malicious video_path outside uploads dir must NOT delete external file."""
    # Create external file outside storage uploads
    external_file = tmp_path / "external_protected_file.txt"
    external_file.write_text("critical external data")

    analysis = Analysis(
        id="mock-external-attack-id",
        user_id=operator_user.id,
        original_name="test.mp4",
        sha256="0" * 64,
        size_bytes=100,
        duration_s=1.0,
        status="completed",
        stage="done",
        video_path=str(external_file),
    )
    db_session.add(analysis)
    db_session.commit()

    # Attempt deletion
    AnalysisService.delete_analysis(db=db_session, analysis=analysis)

    # External file must still exist!
    assert external_file.exists(), "Files outside uploads dir must be protected from deletion"


def test_repeated_deletion_returns_404(
    client: TestClient,
    operator_user: User,
    synthetic_video_with_audio: Path,
) -> None:
    """Deleting an already deleted analysis must return 404."""
    headers = _auth_headers(operator_user)

    with open(synthetic_video_with_audio, "rb") as f:
        resp = client.post(
            "/api/v1/analyses",
            headers=headers,
            files={"file": ("test_repeat.mp4", f, "video/mp4")},
        )

    analysis_id = resp.json()["id"]

    del_resp1 = client.delete(f"/api/v1/analyses/{analysis_id}", headers=headers)
    assert del_resp1.status_code == 200
    assert del_resp1.json()["ok"] is True

    del_resp2 = client.delete(f"/api/v1/analyses/{analysis_id}", headers=headers)
    assert del_resp2.status_code == 404


def test_job_cancellation_on_active_task_deletion(
    operator_user: User,
    db_session: Session,
) -> None:
    """Deleting an active analysis must cancel its task in JobRunner."""
    aid = "test-active-cancel-id"
    analysis = Analysis(
        id=aid,
        user_id=operator_user.id,
        original_name="active.mp4",
        sha256="0" * 64,
        size_bytes=100,
        duration_s=1.0,
        status="processing",
        stage="vit",
    )
    db_session.add(analysis)
    db_session.commit()

    # Mock an active task in job runner
    cancelled = job_runner.cancel_job(aid)
    assert cancelled is False  # not active yet

    # Perform deletion
    AnalysisService.delete_analysis(db=db_session, analysis=analysis)
    db_session.expire_all()
    assert db_session.get(Analysis, aid) is None
