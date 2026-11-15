"""Analyses API routes: uploads, detail, SSE event streaming, artifacts and deletion.

All endpoints verify owner access through deps.get_owned_analysis (README §8.4, §9).
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

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
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, get_db, get_owned_analysis
from app.core.rate_limit import limiter
from app.db.models import Analysis, User
from app.schemas.analysis import (
    AnalysisCreated,
    AnalysisDetail,
    AnalysisListResponse,
)
from app.services.analysis_service import AnalysisService
from app.services.job_runner import job_runner
from app.services.upload_service import process_upload_stream

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
    # Process and validate the upload stream
    upload_meta = process_upload_stream(
        stream=file.file,
        filename=file.filename or "upload.mp4",
        content_length=content_length,
    )

    # Persist database record and enqueue job
    analysis = await AnalysisService.create_analysis(
        db=db,
        user_id=current_user.id,
        upload=upload_meta,
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
    return AnalysisListResponse(items=items, total=total, page=page, size=size)


@router.get("/{id}", response_model=AnalysisDetail)
def get_analysis(
    analysis: Analysis = Depends(get_owned_analysis),
    db: Session = Depends(get_db),
) -> AnalysisDetail:
    """Retrieve full analysis report and timeline."""
    return AnalysisService.get_analysis_detail(db=db, analysis=analysis)


@router.get("/{id}/events")
async def stream_analysis_events(
    analysis: Analysis = Depends(get_owned_analysis),
) -> StreamingResponse:
    """Stream SSE stage events live during analysis execution."""

    async def event_generator() -> AsyncGenerator[str, None]:
        # If already completed or failed, emit terminal event immediately
        if analysis.status == "completed":
            terminal = {
                "stage": "done",
                "progress": 1.0,
                "message": "Analysis complete",
                "elapsed_s": 0.0,
            }
            yield f"data: {json.dumps(terminal)}\n\n"
            return
        elif analysis.status == "failed":
            terminal = {
                "stage": "error",
                "progress": 1.0,
                "message": analysis.error or "Analysis failed",
                "elapsed_s": 0.0,
            }
            yield f"data: {json.dumps(terminal)}\n\n"
            return

        # Subscribe to live updates from job runner
        queue = job_runner.subscribe(analysis.id)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {event.model_dump_json()}\n\n"
        finally:
            job_runner.unsubscribe(analysis.id, queue)

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
) -> FileResponse:
    """Stream forensic artifact file (heatmaps, spectrogram) with traversal guards."""
    file_path = AnalysisService.get_artifact_path(analysis=analysis, artifact_name=name)

    media_type = "image/png"
    if name.endswith(".json"):
        media_type = "application/json"

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=name,
    )


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
