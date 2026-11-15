"""Job runner: bounded async worker pool for CPU/GPU ML pipeline execution.

Features (README §8.5, §12):
- Concurrency bounded by asyncio.Semaphore(settings.MAX_CONCURRENT_JOBS).
- Per-job execution timeout.
- SSE stage event broadcaster with per-analysis subscriber queues.
- Asynchronous database updates at each milestone.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Analysis, AnalysisArtifact
from app.db.session import SessionLocal
from app.ml.pipeline import PipelineEvent, run_pipeline
from app.schemas.analysis import StageEventPayload

logger = logging.getLogger("parallax.jobs")

JOB_TIMEOUT_S = 180  # 3 minutes maximum per analysis


class JobRunner:
    """Manages concurrent analysis job execution and SSE event fan-out."""

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_JOBS)
        self._subscribers: dict[str, list[asyncio.Queue[StageEventPayload | None]]] = defaultdict(
            list
        )
        self._active_tasks: dict[str, asyncio.Task[None]] = {}

    def subscribe(self, analysis_id: str) -> asyncio.Queue[StageEventPayload | None]:
        """Create a new event queue for an SSE client."""
        queue: asyncio.Queue[StageEventPayload | None] = asyncio.Queue(maxsize=100)
        self._subscribers[analysis_id].append(queue)
        return queue

    def unsubscribe(self, analysis_id: str, queue: asyncio.Queue[StageEventPayload | None]) -> None:
        """Remove an event queue when an SSE client disconnects."""
        if analysis_id in self._subscribers:
            with contextlib.suppress(ValueError):
                self._subscribers[analysis_id].remove(queue)
            if not self._subscribers[analysis_id]:
                del self._subscribers[analysis_id]

    def _broadcast(self, analysis_id: str, event: StageEventPayload | None) -> None:
        """Broadcast event to all subscribed SSE queues."""
        if analysis_id in self._subscribers:
            for q in list(self._subscribers[analysis_id]):
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(event)

    async def enqueue(
        self,
        analysis_id: str,
        video_path: Path,
        artifact_dir: Path,
    ) -> None:
        """Schedule an analysis job in the background."""
        task = asyncio.create_task(self._run_job_guarded(analysis_id, video_path, artifact_dir))
        self._active_tasks[analysis_id] = task

        def _cleanup(t: asyncio.Task[None]) -> None:
            self._active_tasks.pop(analysis_id, None)

        task.add_done_callback(_cleanup)

    async def _run_job_guarded(
        self,
        analysis_id: str,
        video_path: Path,
        artifact_dir: Path,
    ) -> None:
        """Acquire semaphore and execute job with timeout protection."""
        async with self._semaphore:
            try:
                await asyncio.wait_for(
                    self._run_job(analysis_id, video_path, artifact_dir),
                    timeout=JOB_TIMEOUT_S,
                )
            except TimeoutError:
                logger.error("Job %s timed out after %ds", analysis_id, JOB_TIMEOUT_S)
                self._record_failure(analysis_id, f"Analysis timed out after {JOB_TIMEOUT_S}s")
            except Exception as e:
                logger.exception("Unexpected failure in job %s: %s", analysis_id, e)
                self._record_failure(analysis_id, f"Pipeline error: {e}")
            finally:
                self._broadcast(analysis_id, None)  # Signal SSE streams to end

    async def _run_job(
        self,
        analysis_id: str,
        video_path: Path,
        artifact_dir: Path,
    ) -> None:
        """Run pipeline steps synchronously in a worker thread, yielding events."""
        loop = asyncio.get_running_loop()

        # Update initial status to processing
        self._update_db_stage(analysis_id, status="processing", stage="probe")

        # Run generator in thread pool step by step
        generator = run_pipeline(video_path, artifact_dir)

        def _next_event() -> PipelineEvent | None:
            try:
                return next(generator)
            except StopIteration:
                return None

        while True:
            event = await loop.run_in_executor(None, _next_event)
            if event is None:
                break

            payload = StageEventPayload(
                stage=event.stage,
                progress=event.progress,
                message=event.message,
                elapsed_s=event.elapsed_s,
            )
            self._broadcast(analysis_id, payload)
            self._update_db_stage(analysis_id, status="processing", stage=event.stage)

            if event.stage == "done" and event.result is not None:
                self._record_success(analysis_id, event, artifact_dir)
                break

    def _update_db_stage(self, analysis_id: str, status: str, stage: str) -> None:
        db: Session = SessionLocal()
        try:
            analysis = db.execute(
                select(Analysis).where(Analysis.id == analysis_id)
            ).scalar_one_or_none()
            if analysis:
                analysis.status = status
                analysis.stage = stage
                db.commit()
        except Exception:
            logger.exception("Failed to update status for %s", analysis_id)
        finally:
            db.close()

    def _record_success(
        self,
        analysis_id: str,
        event: PipelineEvent,
        artifact_dir: Path,
    ) -> None:
        db: Session = SessionLocal()
        try:
            analysis = db.execute(
                select(Analysis).where(Analysis.id == analysis_id)
            ).scalar_one_or_none()
            if not analysis or not event.result:
                return

            res = event.result
            analysis.status = "completed"
            analysis.stage = "done"
            analysis.completed_at = datetime.now(UTC)
            analysis.verdict = res.inference.verdict
            analysis.p_fake = res.inference.p_fake
            analysis.w_audio = res.inference.w_audio
            analysis.w_video = res.inference.w_video
            analysis.sync_score = res.inference.sync_score
            analysis.face_coverage = res.video.face_coverage
            analysis.audio_present = res.audio.audio_present
            analysis.model_version = res.inference.model_version
            analysis.model_mode = res.inference.model_mode
            analysis.reasons_json = json.dumps(res.reasons)

            # Store timeline JSON
            timeline_data = [
                {"t": pt.t, "p_face": pt.p_face, "p_audio": pt.p_audio, "p_fused": pt.p_fused}
                for pt in res.explain.timeline
            ]
            analysis.timeline_json = json.dumps(timeline_data)

            # Store artifact records
            for hpath in res.explain.heatmap_paths:
                db.add(
                    AnalysisArtifact(
                        analysis_id=analysis_id,
                        kind="heatmap",
                        name=hpath,
                        path=str(artifact_dir / hpath),
                    )
                )

            db.add(
                AnalysisArtifact(
                    analysis_id=analysis_id,
                    kind="spectrogram",
                    name=res.explain.spectrogram_path,
                    path=str(artifact_dir / res.explain.spectrogram_path),
                )
            )

            db.commit()
            logger.info("Analysis %s completed successfully: %s", analysis_id, analysis.verdict)
        except Exception:
            logger.exception("Failed to record completion for %s", analysis_id)
        finally:
            db.close()

    def _record_failure(self, analysis_id: str, error_message: str) -> None:
        db: Session = SessionLocal()
        try:
            analysis = db.execute(
                select(Analysis).where(Analysis.id == analysis_id)
            ).scalar_one_or_none()
            if analysis:
                analysis.status = "failed"
                analysis.error = error_message
                analysis.completed_at = datetime.now(UTC)
                db.commit()
            payload = StageEventPayload(
                stage="error",
                progress=1.0,
                message=error_message,
                elapsed_s=0.0,
            )
            self._broadcast(analysis_id, payload)
        except Exception:
            logger.exception("Failed to record failure for %s", analysis_id)
        finally:
            db.close()


# Global runner instance
job_runner = JobRunner()
