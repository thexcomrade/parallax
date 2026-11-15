"""Upload service: streaming upload with byte cap, magic bytes & ffprobe verification.

Security rules (README §8.5, §12 check 15):
- Running byte cap: abort immediately if bytes exceed MAX_UPLOAD_MB.
- File extension validation.
- Magic bytes inspection (never trust Content-Type).
- ffprobe decodability + duration + resolution checks via probe_media.
- UUID-named storage in storage/uploads/ (never client filename).
- Cleanup on failure.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from app.core.config import settings
from app.core.errors import AppException
from app.ml.preprocess.probe import (
    CorruptMediaError,
    MediaLimitsExceededError,
    ProbeResult,
    probe_media,
)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
CHUNK_SIZE = 64 * 1024  # 64 KB streaming chunks


class UploadError(AppException):
    def __init__(self, message: str, code: str = "UPLOAD_ERROR", status_code: int = 400) -> None:
        super().__init__(code=code, message=message, status_code=status_code)


@dataclass(frozen=True)
class UploadMetadata:
    saved_path: Path
    original_name: str
    sha256: str
    size_bytes: int
    duration_s: float
    audio_present: bool
    width: int
    height: int


def _verify_magic_bytes(header: bytes, ext: str) -> bool:
    """Verify container magic bytes match expected signatures."""
    if len(header) < 12:
        return False

    if ext in (".mp4", ".mov"):
        # ISO Base Media File Format: bytes 4..8 usually b"ftyp" or b"moov"
        return header[4:8] in (b"ftyp", b"moov", b"wide", b"skip")
    elif ext == ".avi":
        # RIFF....AVI
        return header[:4] == b"RIFF" and header[8:12] == b"AVI "
    elif ext in (".webm", ".mkv"):
        # Matroska / EBML header
        return header[:4] == b"\x1a\x45\xdf\xa3"
    return False


def _sanitize_original_name(name: str) -> str:
    """Strip path characters from original filename."""
    base = os.path.basename(name.replace("\\", "/"))
    # Replace dangerous or control characters
    sanitized = re.sub(r"[^\w\-. ]", "_", base).strip()
    return sanitized or "upload.mp4"


def process_upload_stream(
    stream: BinaryIO,
    filename: str,
    content_length: int | None = None,
) -> UploadMetadata:
    """Stream incoming upload to a temporary file, validate, and move to permanent storage.

    Args:
        stream: Binary file-like stream from request.
        filename: Client-supplied filename.
        content_length: Optional Content-Length header for pre-check.

    Returns:
        UploadMetadata

    Raises:
        UploadError if size, magic bytes, extension or ffprobe decode fails.
    """
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    if content_length is not None and content_length > max_bytes:
        raise UploadError(
            f"File size exceeds maximum allowed upload size of {settings.MAX_UPLOAD_MB}MB",
            code="FILE_TOO_LARGE",
            status_code=413,
        )

    # 1. Extension check
    orig_name = _sanitize_original_name(filename)
    ext = Path(orig_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UploadError(
            f"Unsupported file format '{ext}'. Allowed formats: {allowed_list}",
            code="INVALID_EXTENSION",
            status_code=400,
        )

    # 2. Stream to temporary file while calculating SHA256 and enforcing byte cap
    uploads_dir = settings.storage_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    temp_path = uploads_dir / f"tmp_{uuid.uuid4().hex}{ext}"

    hasher = hashlib.sha256()
    total_bytes = 0
    header_bytes = b""

    try:
        with open(temp_path, "wb") as f_out:
            while True:
                chunk = stream.read(CHUNK_SIZE)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise UploadError(
                        f"File size exceeds limit of {settings.MAX_UPLOAD_MB}MB",
                        code="FILE_TOO_LARGE",
                        status_code=413,
                    )

                if len(header_bytes) < 32:
                    header_bytes += chunk[: 32 - len(header_bytes)]

                hasher.update(chunk)
                f_out.write(chunk)

        if total_bytes == 0:
            raise UploadError(
                "Uploaded file is empty (0 bytes)", code="EMPTY_FILE", status_code=400
            )

        # 3. Magic bytes validation
        if not _verify_magic_bytes(header_bytes, ext):
            raise UploadError(
                "File contents do not match the expected video container format",
                code="INVALID_MAGIC_BYTES",
                status_code=400,
            )

        # 4. ffprobe media decode verification (duration, resolution, streams)
        try:
            probe_info: ProbeResult = probe_media(temp_path)
        except MediaLimitsExceededError as e:
            raise UploadError(str(e), code="MEDIA_LIMITS_EXCEEDED", status_code=400) from e
        except CorruptMediaError as e:
            raise UploadError(
                "Uploaded media file is corrupt or unreadable",
                code="CORRUPT_MEDIA",
                status_code=400,
            ) from e

        # 5. Move validated file to permanent UUID path
        final_id = uuid.uuid4().hex
        final_path = uploads_dir / f"{final_id}{ext}"
        os.replace(temp_path, final_path)

        return UploadMetadata(
            saved_path=final_path,
            original_name=orig_name,
            sha256=hasher.hexdigest(),
            size_bytes=total_bytes,
            duration_s=probe_info.duration_s,
            audio_present=probe_info.audio_present,
            width=probe_info.width,
            height=probe_info.height,
        )

    except Exception:
        # Clean up temporary file on failure
        if temp_path.exists():
            with contextlib.suppress(OSError):
                temp_path.unlink()
        raise
