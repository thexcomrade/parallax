"""Analysis service: database persistence, artifact retrieval & job glue.

Only layer that coordinates DB entities, job scheduling, and file system cleanup.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import uuid
from pathlib import Path

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import AppException, NotFoundException
from app.db.models import Analysis, AnalysisArtifact
from app.schemas.analysis import (
    AnalysisDetail,
    AnalysisSummary,
    ArtifactItem,
    TimelineItem,
)
from app.services.job_runner import job_runner
from app.services.upload_service import UploadMetadata

logger = logging.getLogger("parallax.analysis")

# Safe artifact filename pattern: alphanumeric, dots, hyphens, underscores only
SAFE_ARTIFACT_RE = re.compile(r"^[A-Za-z0-9_\-\.]+$")


class AnalysisService:
    @staticmethod
    async def create_analysis(
        db: Session,
        user_id: int,
        upload: UploadMetadata,
    ) -> Analysis:
        """Create new Analysis record and enqueue background processing."""
        analysis_id = str(uuid.uuid4())
        artifact_dir = settings.storage_path / "artifacts" / analysis_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        analysis = Analysis(
            id=analysis_id,
            user_id=user_id,
            original_name=upload.original_name,
            sha256=upload.sha256,
            size_bytes=upload.size_bytes,
            duration_s=upload.duration_s,
            status="queued",
            stage="probe",
            audio_present=upload.audio_present,
            model_version="0.0.0",
            model_mode="untrained",
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        # Enqueue ML job
        await job_runner.enqueue(
            analysis_id=analysis_id,
            video_path=upload.saved_path,
            artifact_dir=artifact_dir,
        )

        return analysis

    @staticmethod
    def list_analyses(
        db: Session,
        user_id: int,
        page: int = 1,
        size: int = 20,
    ) -> tuple[list[AnalysisSummary], int]:
        """List user's analyses with pagination, sorted by created_at desc."""
        offset = (page - 1) * size

        total_stmt = select(func.count(Analysis.id)).where(Analysis.user_id == user_id)
        total = db.execute(total_stmt).scalar() or 0

        stmt = (
            select(Analysis)
            .where(Analysis.user_id == user_id)
            .order_by(desc(Analysis.created_at))
            .offset(offset)
            .limit(size)
        )
        analyses = db.execute(stmt).scalars().all()

        items = [AnalysisSummary.model_validate(a) for a in analyses]
        return items, total

    @staticmethod
    def get_analysis_detail(db: Session, analysis: Analysis) -> AnalysisDetail:
        """Format an Analysis ORM model into a rich AnalysisDetail schema."""
        timeline_list: list[TimelineItem] | None = None
        if analysis.timeline_json:
            try:
                raw_tl = json.loads(analysis.timeline_json)
                timeline_list = [TimelineItem(**pt) for pt in raw_tl]
            except Exception:
                pass

        reasons_list: list[str] | None = None
        if analysis.reasons_json:
            with contextlib.suppress(Exception):
                reasons_list = json.loads(analysis.reasons_json)

        # Load artifacts
        art_stmt = select(AnalysisArtifact).where(AnalysisArtifact.analysis_id == analysis.id)
        artifacts = db.execute(art_stmt).scalars().all()
        artifact_items = [ArtifactItem(id=a.id, kind=a.kind, name=a.name) for a in artifacts]

        return AnalysisDetail(
            id=analysis.id,
            original_name=analysis.original_name,
            status=analysis.status,
            stage=analysis.stage,
            verdict=analysis.verdict,
            p_fake=analysis.p_fake,
            duration_s=analysis.duration_s,
            created_at=analysis.created_at,
            completed_at=analysis.completed_at,
            w_audio=analysis.w_audio,
            w_video=analysis.w_video,
            sync_score=analysis.sync_score,
            face_coverage=analysis.face_coverage,
            audio_present=analysis.audio_present,
            model_version=analysis.model_version,
            model_mode=analysis.model_mode,
            timeline=timeline_list,
            reasons=reasons_list,
            error=analysis.error,
            artifacts=artifact_items,
        )

    @staticmethod
    def get_artifact_path(analysis: Analysis, artifact_name: str) -> Path:
        """Resolve and validate an artifact file path with traversal guards."""
        if not SAFE_ARTIFACT_RE.match(artifact_name) or ".." in artifact_name:
            raise AppException(
                code="INVALID_ARTIFACT_NAME",
                message="Invalid artifact file name requested",
                status_code=400,
            )

        artifact_dir = (settings.storage_path / "artifacts" / analysis.id).resolve()
        requested_path = (artifact_dir / artifact_name).resolve()

        # Strict containment check: requested path must be directly inside artifact_dir
        if not str(requested_path).startswith(str(artifact_dir)):
            raise AppException(
                code="FORBIDDEN_TRAVERSAL",
                message="Path traversal forbidden",
                status_code=400,
            )

        if not requested_path.exists() or not requested_path.is_file():
            raise NotFoundException("Artifact not found")

        return requested_path

    @staticmethod
    def delete_analysis(db: Session, analysis: Analysis) -> None:
        """Delete an analysis, its artifacts directory, and uploaded video."""
        # 1. Remove artifacts folder
        artifact_dir = settings.storage_path / "artifacts" / analysis.id
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)

        # 2. Remove uploaded source video if present
        uploads_dir = settings.storage_path / "uploads"
        # Search by SHA or remove matching files
        for p in uploads_dir.glob(f"*{analysis.id}*"):
            with contextlib.suppress(OSError):
                p.unlink()

        # 3. Delete database row (cascades to analysis_artifacts)
        db.delete(analysis)
        db.commit()
