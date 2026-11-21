"""Analyses API routes: uploads, detail, SSE event streaming, artifacts and deletion.

All endpoints verify owner access through deps.get_owned_analysis (README §8.4, §9).
"""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import anyio.to_thread
from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.deps import get_current_operator, get_current_user, get_db, get_owned_analysis
from app.core.errors import AppException, ConflictException, NotFoundException
from app.core.rate_limit import limiter
from app.db.models import Analysis, User
from app.ml.preprocess.probe import CorruptMediaError, MediaLimitsExceededError, probe_media
from app.schemas.analysis import (
    AnalysisCreated,
    AnalysisDetail,
    AnalysisListResponse,
    AnalysisPublicDetail,
    FinalizeUploadRequest,
    UploadPolicyResponse,
)
from app.services.analysis_service import AnalysisService
from app.services.job_runner import job_runner
from app.services.storage_s3 import get_storage_service
from app.services.upload_service import (
    ALLOWED_EXTENSIONS,
    UploadMetadata,
    _sanitize_original_name,
    _verify_magic_bytes,
    process_upload_stream,
)

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post(
    "",
    response_model=AnalysisCreated,
    status_code=status.HTTP_202_ACCEPTED,
)
@limiter.limit("10/minute")
async def create_analysis(
    request: Request,
    file: UploadFile = File(...),
    content_length: int | None = Header(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisCreated:
    """Upload media file, validate container and decodability, and start analysis."""
    # 1. Enforce queue capacity & per-tenant backpressure before processing file (Workstream C)
    await job_runner.check_admission_capacity(user_id=current_user.id)

    analysis_id = str(uuid.uuid4())
    # 2. Process and validate the upload stream off the async event loop (Workstream A)
    upload_meta = await anyio.to_thread.run_sync(
        process_upload_stream,
        file.file,
        file.filename or "upload.mp4",
        content_length,
        analysis_id,
    )

    # 3. Prevent duplicate submission of identical SHA-256 within 60 seconds (Workstream C)
    lock_ok = await job_runner.check_and_acquire_upload_lock(current_user.id, upload_meta.sha256)
    if not lock_ok:
        raise ConflictException(
            code="UPLOAD_IN_PROGRESS",
            message=(
                "An identical file analysis is already in progress. "
                "Please wait for completion."
            ),
        )

    # Persist database record and enqueue job
    analysis = await AnalysisService.create_analysis(
        db=db,
        user_id=current_user.id,
        upload=upload_meta,
        analysis_id=analysis_id,
    )

    return AnalysisCreated(
        id=analysis.id,
        status=analysis.status,
        original_name=analysis.original_name,
    )


@router.post(
    "/upload-policy",
    response_model=UploadPolicyResponse,
    status_code=status.HTTP_200_OK,
)
@limiter.limit("20/minute")
def get_upload_policy(
    request: Request,
    filename: str = Query(..., max_length=255),
    current_user: User = Depends(get_current_user),
) -> UploadPolicyResponse:
    """Generate pre-signed upload credentials/policy for direct S3 or staging upload."""
    orig_name = _sanitize_original_name(filename)
    ext = Path(orig_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise AppException(
            code="INVALID_EXTENSION",
            message=f"Unsupported file format '{ext}'",
            status_code=400,
        )

    analysis_id = str(uuid.uuid4())
    remote_key = f"uploads/{analysis_id}{ext}"
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    storage = get_storage_service()

    policy_data = storage.get_presigned_upload_policy(
        remote_key=remote_key,
        max_size_bytes=max_bytes,
        expires_in=settings.S3_PRESIGNED_EXPIRATION_S,
        allowed_content_types=["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"],
    )

    return UploadPolicyResponse(
        analysis_id=analysis_id,
        key=remote_key,
        url=policy_data["url"],
        fields=policy_data.get("fields", {}),
        expires_in=policy_data.get("expires_in", settings.S3_PRESIGNED_EXPIRATION_S),
    )


@router.post(
    "/finalize",
    response_model=AnalysisCreated,
    status_code=status.HTTP_202_ACCEPTED,
)
@limiter.limit("10/minute")
async def finalize_upload(
    request: Request,
    payload: FinalizeUploadRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisCreated:
    """Verify uploaded object in storage, inspect decodability, and enqueue analysis."""
    # 0. Enforce queue capacity & per-tenant backpressure (Workstream C)
    await job_runner.check_admission_capacity(user_id=current_user.id)

    storage = get_storage_service()

    # 1. Verify object exists in storage
    if not storage.object_exists(payload.key):
        raise NotFoundException(f"Uploaded media object not found in storage: {payload.key}")

    # 2. Check duplicate submission lock
    lock_ok = await job_runner.check_and_acquire_upload_lock(current_user.id, payload.sha256)
    if not lock_ok:
        raise ConflictException("Duplicate analysis for this file is already in progress")

    # 3. Verify size constraints
    meta = storage.get_object_metadata(payload.key)
    size_bytes = int(meta.get("size_bytes", payload.size_bytes))
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if size_bytes > max_bytes:
        storage.delete_object(payload.key)
        raise AppException(
            code="FILE_TOO_LARGE",
            message="File exceeds maximum upload size",
            status_code=413,
        )

    # 4. Verify media decodability & magic bytes via temporary probe
    temp_dir = settings.storage_path / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(payload.key).suffix.lower()
    temp_path = temp_dir / f"chk_{payload.analysis_id}{ext}"
    try:
        storage.download_file(payload.key, temp_path)
        with open(temp_path, "rb") as f:
            header = f.read(32)
        if not _verify_magic_bytes(header, ext):
            storage.delete_object(payload.key)
            raise AppException(
                code="INVALID_MAGIC_BYTES",
                message="Invalid media magic bytes",
                status_code=400,
            )

        probe_info = probe_media(temp_path)
    except (MediaLimitsExceededError, CorruptMediaError) as e:
        storage.delete_object(payload.key)
        raise AppException(code="CORRUPT_MEDIA", message=str(e), status_code=400) from e
    finally:
        if temp_path.exists():
            with contextlib.suppress(OSError):
                temp_path.unlink()

    # 5. Persist database record and enqueue job
    upload_meta = UploadMetadata(
        saved_path=Path(payload.key),
        original_name=_sanitize_original_name(payload.original_name),
        sha256=payload.sha256,
        size_bytes=size_bytes,
        duration_s=probe_info.duration_s,
        audio_present=probe_info.audio_present,
        width=probe_info.width,
        height=probe_info.height,
    )

    analysis = await AnalysisService.create_analysis(
        db=db,
        user_id=current_user.id,
        upload=upload_meta,
        analysis_id=payload.analysis_id,
    )

    return AnalysisCreated(
        id=analysis.id,
        status=analysis.status,
        original_name=analysis.original_name,
    )


@router.get("", response_model=AnalysisListResponse)
def list_analyses(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    q: str | None = Query(None, max_length=100),
    verdict: str | None = Query(None, max_length=20),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisListResponse:
    """List authenticated user's analyses with pagination and optional search/verdict filter."""
    items, total = AnalysisService.list_analyses(
        db=db,
        user_id=current_user.id,
        page=page,
        size=size,
        q=q,
        verdict=verdict,
    )
    # Sanitize for public/unprivileged users: redact p_fake and normalize verdict
    if current_user.role not in ("operator", "admin"):
        for item in items:
            item.p_fake = None
            if item.verdict:
                v_up = item.verdict.strip().upper()
                item.verdict = "REAL" if v_up in ("AUTHENTIC", "REAL") else "FAKE"
    return AnalysisListResponse(items=items, total=total, page=page, size=size)


@router.get(
    "/{id}/public",
    response_model=AnalysisPublicDetail,
    summary="Public verification endpoint with binary verdict",
)
def get_public_analysis(
    id: str,
    db: Session = Depends(get_db),
) -> AnalysisPublicDetail:
    """Public verification endpoint returning strictly binary verdict REAL/FAKE.

    Zero internal probabilities, DWF weights, modality scores, or timeline telemetry.
    """
    analysis = db.execute(select(Analysis).where(Analysis.id == id)).scalar_one_or_none()
    if not analysis:
        raise NotFoundException(f"Analysis '{id}' not found")
    return AnalysisService.get_public_analysis_detail(analysis=analysis)


@router.get(
    "/{id}",
    response_model=AnalysisDetail | AnalysisPublicDetail,
    summary="Retrieve analysis detail",
)
def get_analysis(
    analysis: Analysis = Depends(get_owned_analysis),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisDetail | AnalysisPublicDetail:
    """Retrieve analysis report.

    Authorized operators receive full AnalysisDetail with forensics, timeline,
    and weights. Public users receive AnalysisPublicDetail with strictly binary
    verdict (REAL/FAKE).
    """
    if current_user.role in ("operator", "admin"):
        return AnalysisService.get_analysis_detail(db=db, analysis=analysis)
    return AnalysisService.get_public_analysis_detail(analysis=analysis)


@router.get("/{id}/events")
async def stream_analysis_events(
    request: Request,
    id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    last_event_id_header: str | None = Header(None, alias="Last-Event-ID"),
    last_event_id: str | None = None,
) -> StreamingResponse:
    """Stream SSE stage events live during analysis execution.

    Releases database connection immediately after ownership verification
    to prevent holding pool connections during long-lived SSE streams (Workstream B).
    """
    try:
        analysis = db.execute(
            select(Analysis).where(
                Analysis.id == id,
                Analysis.user_id == current_user.id,
            )
        ).scalar_one_or_none()
        if not analysis:
            raise NotFoundException("Analysis not found")
        init_status = analysis.status
        init_error = analysis.error
    finally:
        db.close()

    effective_last_id = last_event_id_header or last_event_id

    async def event_generator() -> AsyncGenerator[str, None]:
        # If already completed or failed, emit terminal event immediately
        if init_status == "completed":
            terminal = {
                "stage": "done",
                "progress": 1.0,
                "message": "Analysis complete",
                "elapsed_s": 0.0,
            }
            yield f"id: terminal\ndata: {json.dumps(terminal)}\n\n"
            return
        elif init_status == "failed":
            terminal = {
                "stage": "error",
                "progress": 1.0,
                "message": init_error or "Analysis failed",
                "elapsed_s": 0.0,
            }
            yield f"id: terminal\ndata: {json.dumps(terminal)}\n\n"
            return

        # Stream from durable Redis event stream
        has_yielded = False
        async for msg_id, payload in job_runner.stream_events(
            id, last_event_id=effective_last_id
        ):
            has_yielded = True
            yield f"id: {msg_id}\ndata: {payload.model_dump_json()}\n\n"

        # Fallback to local in-memory subscriber if Redis yielded no events
        # and analysis is still processing
        if not has_yielded:
            queue = job_runner.subscribe(id)
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield f"data: {event.model_dump_json()}\n\n"
            finally:
                job_runner.unsubscribe(id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{id}/artifacts/{name}")
def get_artifact(
    name: str,
    analysis: Analysis = Depends(get_owned_analysis),
    _operator: User = Depends(get_current_operator),
) -> FileResponse:
    """Stream forensic artifact file (heatmaps, spectrogram) with traversal guards.

    Strictly restricted to authorized operators.
    """
    file_path = AnalysisService.get_artifact_path(analysis=analysis, artifact_name=name)

    media_type = "image/png"
    if name.endswith(".json"):
        media_type = "application/json"

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=name,
    )


@router.get("/{id}/artifacts/{name}/url")
def get_artifact_presigned_url(
    name: str,
    analysis: Analysis = Depends(get_owned_analysis),
    _operator: User = Depends(get_current_operator),
) -> dict[str, str]:
    """Generate pre-signed S3/MinIO or local download URL for forensic artifact.

    Strictly restricted to authorized operators.
    """
    storage = get_storage_service()
    remote_key = f"artifacts/{analysis.id}/{name}"
    url = storage.get_presigned_download_url(remote_key=remote_key, filename=name)
    return {"url": url, "name": name}


@router.delete("/{id}")
def delete_analysis(
    analysis: Analysis = Depends(get_owned_analysis),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Delete analysis, all artifact files, and uploaded video."""
    AnalysisService.delete_analysis(db=db, analysis=analysis)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"ok": True, "message": "Analysis deleted"},
    )
