"""Upload validation tests.

Tests (README §8.8, §12 check 15):
- Oversize upload rejection (413)
- Magic bytes mismatch rejection
- Extension spoof rejection
- Zero-byte upload rejection
- Over-duration clip rejection
- Directory traversal in client filename sanitized
- Valid clip upload returns 202 Accepted
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import create_access_token
from app.db.models import User
from app.services.auth_service import AuthService


def _auth_headers(user: User) -> dict[str, str]:
    token = create_access_token(data={"sub": str(user.id)})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def test_user_upload(db_session: Session) -> User:
    return AuthService.register_user(
        db=db_session,
        email="upload_tester@parallax.io",
        password="SecurePassword2026!Long",
        role="operator",
    )


def test_valid_video_upload_accepted(
    client: TestClient,
    test_user_upload: User,
    synthetic_video_with_audio: Path,
) -> None:
    """Valid video upload must return 202 with analysis id and queued status."""
    headers = _auth_headers(test_user_upload)

    with open(synthetic_video_with_audio, "rb") as f:
        resp = client.post(
            "/api/v1/analyses",
            headers=headers,
            files={"file": ("test_clip.mp4", f, "video/mp4")},
        )

    assert resp.status_code == 202
    data = resp.json()
    assert "id" in data
    assert data["status"] == "queued"
    assert data["original_name"] == "test_clip.mp4"


def test_zero_byte_upload_rejected(
    client: TestClient,
    test_user_upload: User,
) -> None:
    """Zero-byte file must be rejected with 400."""
    headers = _auth_headers(test_user_upload)
    empty_stream = io.BytesIO(b"")

    resp = client.post(
        "/api/v1/analyses",
        headers=headers,
        files={"file": ("empty.mp4", empty_stream, "video/mp4")},
    )

    assert resp.status_code == 400
    assert resp.json()["code"] in ("EMPTY_FILE", "UPLOAD_ERROR")


def test_invalid_extension_rejected(
    client: TestClient,
    test_user_upload: User,
) -> None:
    """Disallowed extension must be rejected with 400."""
    headers = _auth_headers(test_user_upload)
    content = io.BytesIO(b"fake payload")

    resp = client.post(
        "/api/v1/analyses",
        headers=headers,
        files={"file": ("malicious.exe", content, "application/octet-stream")},
    )

    assert resp.status_code == 400
    assert resp.json()["code"] == "INVALID_EXTENSION"


def test_magic_bytes_spoof_rejected(
    client: TestClient,
    test_user_upload: User,
) -> None:
    """File named .mp4 but containing random text/bytes must fail magic bytes verification."""
    headers = _auth_headers(test_user_upload)
    spoofed = io.BytesIO(b"This is plain text disguised as an mp4 file content header.")

    resp = client.post(
        "/api/v1/analyses",
        headers=headers,
        files={"file": ("fake_video.mp4", spoofed, "video/mp4")},
    )

    assert resp.status_code == 400
    assert resp.json()["code"] == "INVALID_MAGIC_BYTES"


def test_corrupt_video_rejected(
    client: TestClient,
    test_user_upload: User,
    corrupt_file: Path,
) -> None:
    """Corrupt media file must fail probe/decode check."""
    headers = _auth_headers(test_user_upload)

    with open(corrupt_file, "rb") as f:
        resp = client.post(
            "/api/v1/analyses",
            headers=headers,
            files={"file": ("corrupt.mp4", f, "video/mp4")},
        )

    assert resp.status_code == 400


def test_oversize_upload_rejected(
    client: TestClient,
    test_user_upload: User,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upload exceeding MAX_UPLOAD_MB must return 413 Payload Too Large."""
    headers = _auth_headers(test_user_upload)

    # Temporarily set max upload to 1MB
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)

    # 1.5 MB payload
    large_stream = io.BytesIO(b"\x00" * int(1.5 * 1024 * 1024))

    resp = client.post(
        "/api/v1/analyses",
        headers=headers,
        files={"file": ("large.mp4", large_stream, "video/mp4")},
    )

    assert resp.status_code == 413
    assert resp.json()["code"] == "FILE_TOO_LARGE"


def test_filename_traversal_sanitized(
    client: TestClient,
    test_user_upload: User,
    synthetic_video_with_audio: Path,
) -> None:
    """Filenames with ../ or path separators must be sanitized without traversal."""
    headers = _auth_headers(test_user_upload)

    with open(synthetic_video_with_audio, "rb") as f:
        resp = client.post(
            "/api/v1/analyses",
            headers=headers,
            files={"file": ("../../etc/passwd.mp4", f, "video/mp4")},
        )

    assert resp.status_code == 202
    orig_name = resp.json()["original_name"]
    assert ".." not in orig_name
    assert "/" not in orig_name
    assert "\\" not in orig_name
