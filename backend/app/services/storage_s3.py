"""S3-compatible and Local Object Storage Abstraction for PARALLAX.

Supports:
- Local filesystem storage (default for development and tests).
- MinIO for local Docker staging (path-style addressing, custom endpoint).
- AWS S3 for production deployment.
- Pre-signed upload policy generation with size and content-type constraints.
- Pre-signed download URL generation with expiration.
- Object existence, size, and checksum verification.
- Cleanup of source videos and generated artifacts on analysis deletion.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.errors import AppException, NotFoundException
from app.core.logging import get_logger
from app.core.metrics import (
    STORAGE_ERRORS_TOTAL,
    STORAGE_UPLOAD_DURATION_SECONDS,
    STORAGE_UPLOADS_TOTAL,
)

logger = get_logger("storage")


class StorageError(AppException):
    def __init__(
        self,
        message: str,
        code: str = "STORAGE_ERROR",
        status_code: int = 500,
    ) -> None:
        super().__init__(code=code, message=message, status_code=status_code)


class StorageBackend(ABC):
    """Abstract interface for object storage operations."""

    @abstractmethod
    def upload_file(
        self,
        local_path: Path,
        remote_key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload a local file to storage at remote_key."""
        ...

    @abstractmethod
    def download_file(self, remote_key: str, local_dest: Path) -> Path:
        """Download remote_key to local_dest."""
        ...

    @abstractmethod
    def object_exists(self, remote_key: str) -> bool:
        """Check if an object exists in storage."""
        ...

    @abstractmethod
    def get_object_metadata(self, remote_key: str) -> dict[str, Any]:
        """Retrieve metadata including size and checksum."""
        ...

    @abstractmethod
    def get_presigned_download_url(
        self,
        remote_key: str,
        filename: str | None = None,
        expires_in: int = 3600,
    ) -> str:
        """Generate a short-lived signed URL for downloading an object."""
        ...

    @abstractmethod
    def get_presigned_upload_policy(
        self,
        remote_key: str,
        max_size_bytes: int,
        expires_in: int = 3600,
        allowed_content_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate a restricted pre-signed POST/PUT upload policy."""
        ...

    @abstractmethod
    def delete_object(self, remote_key: str) -> bool:
        """Delete an individual object."""
        ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        """Delete all objects matching a key prefix."""
        ...

    @abstractmethod
    def check_health(self) -> bool:
        """Verify storage backend connectivity and accessibility."""
        ...

    def verify_checksum(self, remote_key: str, expected_sha256: str) -> bool:
        """Verify whether an object matches an expected SHA-256 hash."""
        meta = self.get_object_metadata(remote_key)
        actual_sha = meta.get("sha256")
        if actual_sha:
            return str(actual_sha).lower() == expected_sha256.lower()

        # If remote metadata lacks sha256, download to temp and calculate
        temp_dir = settings.storage_path / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"chk_{Path(remote_key).name}"
        try:
            self.download_file(remote_key, temp_path)
            hasher = hashlib.sha256()
            with open(temp_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return hasher.hexdigest().lower() == expected_sha256.lower()
        finally:
            if temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()


class LocalStorage(StorageBackend):
    """Local filesystem storage implementation."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = (base_dir or settings.storage_path).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_key(self, remote_key: str) -> Path:
        clean_key = remote_key.lstrip("/\\")
        path = (self.base_dir / clean_key).resolve()
        if not str(path).startswith(str(self.base_dir)):
            raise StorageError("Path traversal forbidden in local storage", code="FORBIDDEN_PATH")
        return path

    def upload_file(
        self,
        local_path: Path,
        remote_key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        t0 = time.perf_counter()
        try:
            dest = self._resolve_key(remote_key)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if local_path.resolve() != dest.resolve():
                shutil.copy2(local_path, dest)
            STORAGE_UPLOADS_TOTAL.labels(backend="local", status="success").inc()
            STORAGE_UPLOAD_DURATION_SECONDS.labels(backend="local").observe(
                time.perf_counter() - t0
            )
            return str(dest)
        except Exception:
            STORAGE_UPLOADS_TOTAL.labels(backend="local", status="failure").inc()
            STORAGE_ERRORS_TOTAL.labels(backend="local", operation="upload").inc()
            raise

    def download_file(self, remote_key: str, local_dest: Path) -> Path:
        try:
            src = self._resolve_key(remote_key)
            if not src.exists() or not src.is_file():
                raise NotFoundException(f"Object not found in local storage: {remote_key}")
            local_dest.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != local_dest.resolve():
                shutil.copy2(src, local_dest)
            return local_dest
        except Exception:
            STORAGE_ERRORS_TOTAL.labels(backend="local", operation="download").inc()
            raise

    def object_exists(self, remote_key: str) -> bool:
        try:
            path = self._resolve_key(remote_key)
            return path.exists() and path.is_file()
        except StorageError:
            return False

    def get_object_metadata(self, remote_key: str) -> dict[str, Any]:
        path = self._resolve_key(remote_key)
        if not path.exists() or not path.is_file():
            raise NotFoundException(f"Object not found: {remote_key}")
        stat = path.stat()
        return {
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
            "key": remote_key,
        }

    def get_presigned_download_url(
        self,
        remote_key: str,
        filename: str | None = None,
        expires_in: int = 3600,
    ) -> str:
        # In local storage, return standard internal asset route or direct relative path
        clean_key = remote_key.lstrip("/\\").replace("\\", "/")
        return f"/api/v1/storage/{clean_key}"

    def get_presigned_upload_policy(
        self,
        remote_key: str,
        max_size_bytes: int,
        expires_in: int = 3600,
        allowed_content_types: list[str] | None = None,
    ) -> dict[str, Any]:
        clean_key = remote_key.lstrip("/\\").replace("\\", "/")
        return {
            "url": "/api/v1/analyses/upload",
            "fields": {
                "key": clean_key,
                "max_size_bytes": str(max_size_bytes),
            },
            "expires_in": expires_in,
        }

    def delete_object(self, remote_key: str) -> bool:
        try:
            path = self._resolve_key(remote_key)
            if path.exists() and path.is_file():
                path.unlink()
                return True
            return False
        except Exception:
            return False

    def delete_prefix(self, prefix: str) -> int:
        clean_prefix = prefix.lstrip("/\\")
        target_dir = (self.base_dir / clean_prefix).resolve()
        count = 0
        if target_dir.exists():
            if target_dir.is_dir():
                for _root, _, files in os.walk(target_dir):
                    count += len(files)
                shutil.rmtree(target_dir, ignore_errors=True)
            elif target_dir.is_file():
                target_dir.unlink()
                count = 1
        return count

    def check_health(self) -> bool:
        """Verify local storage directory existence and write accessibility."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            test_file = self.base_dir / ".healthcheck"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            return True
        except Exception:
            return False


class S3Storage(StorageBackend):
    """S3-compatible object storage implementation for MinIO and AWS S3."""

    def __init__(self) -> None:
        import boto3  # type: ignore[import-untyped]
        from botocore.client import Config  # type: ignore[import-untyped]

        self.bucket = settings.S3_BUCKET_NAME
        client_kwargs: dict[str, Any] = {
            "region_name": settings.S3_REGION_NAME,
        }
        if settings.S3_ENDPOINT_URL:
            client_kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL
        if settings.S3_ACCESS_KEY_ID:
            client_kwargs["aws_access_key_id"] = settings.S3_ACCESS_KEY_ID
        if settings.S3_SECRET_ACCESS_KEY:
            client_kwargs["aws_secret_access_key"] = settings.S3_SECRET_ACCESS_KEY

        # MinIO and custom S3 endpoints typically require path-style addressing
        config_kwargs: dict[str, Any] = {"signature_version": "s3v4"}
        if settings.S3_FORCE_PATH_STYLE:
            config_kwargs["s3"] = {"addressing_style": "path"}

        client_kwargs["config"] = Config(**config_kwargs)
        self.s3_client = boto3.client("s3", **client_kwargs)

    def upload_file(
        self,
        local_path: Path,
        remote_key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        clean_key = remote_key.lstrip("/\\").replace("\\", "/")
        extra_args: dict[str, Any] = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            extra_args["Metadata"] = metadata

        t0 = time.perf_counter()
        try:
            self.s3_client.upload_file(
                str(local_path),
                self.bucket,
                clean_key,
                ExtraArgs=extra_args if extra_args else None,
            )
            STORAGE_UPLOADS_TOTAL.labels(backend="s3", status="success").inc()
            STORAGE_UPLOAD_DURATION_SECONDS.labels(backend="s3").observe(
                time.perf_counter() - t0
            )
            return clean_key
        except Exception as e:
            STORAGE_UPLOADS_TOTAL.labels(backend="s3", status="failure").inc()
            STORAGE_ERRORS_TOTAL.labels(backend="s3", operation="upload").inc()
            logger.exception(
                "Failed to upload object %s to bucket %s: %s",
                clean_key,
                self.bucket,
                e,
            )
            raise StorageError(f"S3 upload failed: {e}") from e

    def download_file(self, remote_key: str, local_dest: Path) -> Path:
        clean_key = remote_key.lstrip("/\\").replace("\\", "/")
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.s3_client.download_file(self.bucket, clean_key, str(local_dest))
            return local_dest
        except Exception as e:
            STORAGE_ERRORS_TOTAL.labels(backend="s3", operation="download").inc()
            if "404" in str(e) or "NoSuchKey" in str(e):
                raise NotFoundException(f"Object not found in S3 bucket: {clean_key}") from e
            logger.exception(
                "Failed to download object %s from bucket %s: %s",
                clean_key,
                self.bucket,
                e,
            )
            raise StorageError(f"S3 download failed: {e}") from e

    def object_exists(self, remote_key: str) -> bool:
        clean_key = remote_key.lstrip("/\\").replace("\\", "/")
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=clean_key)
            return True
        except Exception:
            return False

    def get_object_metadata(self, remote_key: str) -> dict[str, Any]:
        clean_key = remote_key.lstrip("/\\").replace("\\", "/")
        try:
            resp = self.s3_client.head_object(Bucket=self.bucket, Key=clean_key)
            return {
                "size_bytes": resp.get("ContentLength", 0),
                "content_type": resp.get("ContentType"),
                "etag": resp.get("ETag", "").strip('"'),
                "metadata": resp.get("Metadata", {}),
                "modified_at": resp.get("LastModified"),
                "key": clean_key,
            }
        except Exception as e:
            if "404" in str(e) or "NoSuchKey" in str(e):
                raise NotFoundException(f"Object not found: {clean_key}") from e
            raise StorageError(f"S3 metadata check failed: {e}") from e

    def get_presigned_download_url(
        self,
        remote_key: str,
        filename: str | None = None,
        expires_in: int = 3600,
    ) -> str:
        clean_key = remote_key.lstrip("/\\").replace("\\", "/")
        params: dict[str, Any] = {"Bucket": self.bucket, "Key": clean_key}
        if filename:
            params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'

        try:
            url: str = self.s3_client.generate_presigned_url(
                "get_object",
                Params=params,
                ExpiresIn=expires_in,
            )
            return url
        except Exception as e:
            logger.exception("Failed to generate presigned download URL for %s: %s", clean_key, e)
            raise StorageError(f"Failed to generate download URL: {e}") from e

    def get_presigned_upload_policy(
        self,
        remote_key: str,
        max_size_bytes: int,
        expires_in: int = 3600,
        allowed_content_types: list[str] | None = None,
    ) -> dict[str, Any]:
        clean_key = remote_key.lstrip("/\\").replace("\\", "/")
        conditions: list[Any] = [
            ["content-length-range", 1, max_size_bytes],
        ]
        if allowed_content_types:
            # S3 condition: Content-Type starts with video/
            conditions.append(["starts-with", "$Content-Type", "video/"])

        try:
            post_data: dict[str, Any] = self.s3_client.generate_presigned_post(
                Bucket=self.bucket,
                Key=clean_key,
                Fields={"Content-Type": "video/mp4"},
                Conditions=conditions,
                ExpiresIn=expires_in,
            )
            return post_data
        except Exception as e:
            logger.exception("Failed to generate presigned upload policy for %s: %s", clean_key, e)
            raise StorageError(f"Failed to generate upload policy: {e}") from e

    def delete_object(self, remote_key: str) -> bool:
        clean_key = remote_key.lstrip("/\\").replace("\\", "/")
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=clean_key)
            return True
        except Exception as e:
            logger.warning("Failed to delete S3 object %s: %s", clean_key, e)
            return False

    def delete_prefix(self, prefix: str) -> int:
        clean_prefix = prefix.lstrip("/\\").replace("\\", "/")
        deleted_count = 0
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=clean_prefix):
                items = page.get("Contents", [])
                if items:
                    delete_keys = [{"Key": item["Key"]} for item in items]
                    self.s3_client.delete_objects(
                        Bucket=self.bucket,
                        Delete={"Objects": delete_keys},
                    )
                    deleted_count += len(delete_keys)
            return deleted_count
        except Exception as e:
            logger.warning("Failed to delete S3 prefix %s: %s", clean_prefix, e)
            return deleted_count

    def check_health(self) -> bool:
        """Verify S3 bucket existence and read accessibility."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False


_storage_instance: StorageBackend | None = None


def get_storage_service() -> StorageBackend:
    """Return configured storage backend (local filesystem or S3/MinIO)."""
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance

    backend_type = settings.STORAGE_BACKEND.lower()
    if backend_type in ("s3", "minio"):
        if settings.is_production or settings.APP_ENV == "staging":
            _storage_instance = S3Storage()
            logger.info(
                "Initialized S3/MinIO storage backend on bucket '%s'",
                settings.S3_BUCKET_NAME,
            )
        else:
            try:
                _storage_instance = S3Storage()
                logger.info(
                    "Initialized S3/MinIO storage backend on bucket '%s'",
                    settings.S3_BUCKET_NAME,
                )
            except Exception as e:
                logger.warning(
                    "Failed to initialize S3 backend (%s); falling back to LocalStorage in dev",
                    e,
                )
                _storage_instance = LocalStorage()
    else:
        _storage_instance = LocalStorage()
        logger.info("Initialized LocalStorage backend at '%s'", settings.storage_path)

    return _storage_instance
