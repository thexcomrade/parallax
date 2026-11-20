"""Tests for S3-compatible and Local Object Storage Abstraction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import create_access_token
from app.db.models import Analysis, User
from app.services.storage_s3 import LocalStorage, S3Storage, StorageError


@pytest.fixture
def temp_storage(tmp_path: Path):
    return LocalStorage(base_dir=tmp_path)


def test_local_storage_upload_download_lifecycle(
    temp_storage: LocalStorage, tmp_path: Path
) -> None:
    """Local storage must upload, verify existence, read metadata, and download cleanly."""
    src_file = tmp_path / "sample.mp4"
    content = b"TEST_VIDEO_BYTES_CONTENT_12345"
    src_file.write_bytes(content)

    remote_key = "uploads/sample_upload.mp4"
    dest_path = temp_storage.upload_file(src_file, remote_key)
    assert Path(dest_path).exists()
    assert temp_storage.object_exists(remote_key) is True

    # Check metadata
    meta = temp_storage.get_object_metadata(remote_key)
    assert meta["size_bytes"] == len(content)

    # Download back
    download_dest = tmp_path / "downloaded.mp4"
    temp_storage.download_file(remote_key, download_dest)
    assert download_dest.read_bytes() == content


def test_local_storage_checksum_verification(temp_storage: LocalStorage, tmp_path: Path) -> None:
    """verify_checksum must return True on matching SHA256 and False on mismatch."""
    src_file = tmp_path / "checksum_test.mp4"
    data = b"HASH_TEST_DATA_98765"
    src_file.write_bytes(data)
    expected_hash = hashlib.sha256(data).hexdigest()

    temp_storage.upload_file(src_file, "uploads/hash_test.mp4")
    assert temp_storage.verify_checksum("uploads/hash_test.mp4", expected_hash) is True
    assert temp_storage.verify_checksum("uploads/hash_test.mp4", "0" * 64) is False


def test_local_storage_path_traversal_guard(temp_storage: LocalStorage) -> None:
    """Storage must reject keys attempting directory traversal outside base directory."""
    with pytest.raises(StorageError):
        temp_storage.upload_file(Path("foo"), "../../etc/passwd")


def test_local_storage_delete_prefix(temp_storage: LocalStorage, tmp_path: Path) -> None:
    """delete_prefix must delete all files under an analysis folder."""
    f1 = tmp_path / "f1.png"
    f2 = tmp_path / "f2.png"
    f1.write_bytes(b"1")
    f2.write_bytes(b"2")

    temp_storage.upload_file(f1, "artifacts/aid-100/heatmap1.png")
    temp_storage.upload_file(f2, "artifacts/aid-100/heatmap2.png")

    assert temp_storage.object_exists("artifacts/aid-100/heatmap1.png") is True
    deleted_count = temp_storage.delete_prefix("artifacts/aid-100/")
    assert deleted_count == 2
    assert temp_storage.object_exists("artifacts/aid-100/heatmap1.png") is False


def test_s3_storage_mocked_operations(tmp_path: Path) -> None:
    """S3Storage correctly invokes boto3 client methods."""
    mock_boto = MagicMock()
    mock_s3 = MagicMock()
    mock_boto.client.return_value = mock_s3

    with (
        patch("boto3.client", return_value=mock_s3),
        patch.object(settings, "STORAGE_BACKEND", "s3"),
    ):
        storage = S3Storage()

        # Upload
        local_f = tmp_path / "dummy.mp4"
        local_f.write_bytes(b"s3_data")
        storage.upload_file(local_f, "uploads/s3_test.mp4", content_type="video/mp4")
        mock_s3.upload_file.assert_called_once()

        # Presigned download URL
        mock_s3.generate_presigned_url.return_value = "https://s3.amazonaws.com/test?sig=xyz"
        url = storage.get_presigned_download_url("uploads/s3_test.mp4")
        assert "https://s3.amazonaws.com" in url

        # Presigned upload policy
        mock_s3.generate_presigned_post.return_value = {
            "url": "https://s3.amazonaws.com/bucket",
            "fields": {"key": "uploads/s3_test.mp4"},
        }
        policy = storage.get_presigned_upload_policy(
            "uploads/s3_test.mp4",
            max_size_bytes=100000,
        )
        assert policy["url"] == "https://s3.amazonaws.com/bucket"

        # Object exists
        mock_s3.head_object.return_value = {"ContentLength": 100}
        assert storage.object_exists("uploads/s3_test.mp4") is True


def test_api_upload_policy_endpoint(client: TestClient, db_session: Session) -> None:
    """POST /api/v1/analyses/upload-policy returns pre-signed credentials."""
    user = User(
        email="policy_user@parallax.io",
        password_hash="fake_hash",
        role="public",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    token = create_access_token({"sub": str(user.id)})
    response = client.post(
        "/api/v1/analyses/upload-policy?filename=clip.mp4",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "analysis_id" in data
    assert "key" in data
    assert data["key"].startswith("uploads/")
    assert "url" in data


def test_api_artifact_presigned_url_operator_access(
    client: TestClient, db_session: Session
) -> None:
    """GET /api/v1/analyses/{id}/artifacts/{name}/url restricts access to operators."""
    op_user = User(
        email="art_operator@parallax.io",
        password_hash="fake_hash",
        role="operator",
    )
    pub_user = User(
        email="art_public@parallax.io",
        password_hash="fake_hash",
        role="public",
    )
    db_session.add_all([op_user, pub_user])
    db_session.commit()
    db_session.refresh(op_user)
    db_session.refresh(pub_user)

    analysis = Analysis(
        id="art-url-aid-505",
        user_id=op_user.id,
        original_name="art_test.mp4",
        sha256="c" * 64,
        size_bytes=1024,
        duration_s=1.0,
        status="completed",
        stage="done",
        verdict="REAL",
        audio_present=True,
        model_version="1.0.0",
        model_mode="trained",
    )
    db_session.add(analysis)
    db_session.commit()

    # Public user forbidden from operator URL
    pub_token = create_access_token({"sub": str(pub_user.id)})
    res_forbidden = client.get(
        f"/api/v1/analyses/{analysis.id}/artifacts/heatmap.png/url",
        headers={"Authorization": f"Bearer {pub_token}"},
    )
    assert res_forbidden.status_code in (403, 404)

    # Operator gets signed download URL
    op_token = create_access_token({"sub": str(op_user.id)})
    res_ok = client.get(
        f"/api/v1/analyses/{analysis.id}/artifacts/heatmap.png/url",
        headers={"Authorization": f"Bearer {op_token}"},
    )
    assert res_ok.status_code == 200
    assert "url" in res_ok.json()
