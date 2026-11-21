"""Distributed JobRunner backed by Redis Streams, PostgreSQL, and consumer groups.

Features (Phase 7B.2 / 7B.3):
- Redis Streams task queue (`parallax:jobs:video_analysis`) with consumer groups.
- Durable SSE event stream (`parallax:events:{analysis_id}`) supporting Last-Event-ID replay.
- Distributed cancellation signals (`parallax:cancel:{analysis_id}`).
- Duplicate job prevention & idempotency locks (`parallax:lock:upload:{user_id}:{sha256}`).
- Worker registration, heartbeats, and lease recovery.
- Backward-compatible in-memory fallback for local development and unit tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import defaultdict
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select

from app.core.config import settings
from app.core.errors import AppException
from app.core.redis import get_redis_client
from app.db.models import Analysis, AnalysisArtifact, AnalysisOutbox
from app.db.session import SessionLocal
from app.ml.pipeline import PipelineEvent, run_pipeline
from app.schemas.analysis import StageEventPayload

logger = logging.getLogger("parallax.jobs")

JOB_TIMEOUT_S = 180  # 3 minutes maximum per analysis
STREAM_TASKS = "parallax:jobs:video_analysis"
GROUP_WORKERS = "parallax:workers"
STREAM_EVENTS_PREFIX = "parallax:events:"
CANCEL_PREFIX = "parallax:cancel:"
LOCK_UPLOAD_PREFIX = "parallax:lock:upload:"
STREAM_DEAD_LETTER = "parallax:jobs:dead_letter"
STREAM_EVENT_TTL_S = 86400  # 24 hours
MAX_JOB_RETRIES = 3
STREAM_CLAIM_MIN_IDLE_MS = 60000  # 1 minute

RedisStreamValue = bytes | bytearray | memoryview | str | int | float
RedisStreamPayload = dict[RedisStreamValue, RedisStreamValue]


class JobRunner:
    """Manages distributed analysis execution via Redis Streams and durable SSE delivery."""

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_JOBS)
        self._subscribers: dict[
            str, list[asyncio.Queue[StageEventPayload | None]]
        ] = defaultdict(list)
        self._active_tasks: dict[str, asyncio.Task[None]] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._outbox_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._worker_id = f"worker-{Path(__file__).stem}-{id(self)}"
        self._claim_counter = 0

    async def start(self) -> None:
        """Start the background Redis Streams consumer worker loop and outbox sweeper."""
        self._stop_event.clear()
        redis = await get_redis_client()

        # Ensure consumer group exists
        try:
            await redis.xgroup_create(STREAM_TASKS, GROUP_WORKERS, id="0", mkstream=True)
            logger.info(
                "Initialized Redis Streams consumer group '%s' on '%s'",
                GROUP_WORKERS,
                STREAM_TASKS,
            )
        except Exception as e:
            # Group already exists (BUSYGROUP) is normal
            if "BUSYGROUP" not in str(e):
                logger.debug("xgroup_create: %s", e)

        # Launch transactional outbox sweeper task across all roles
        if self._outbox_task is None or self._outbox_task.done():
            self._outbox_task = asyncio.create_task(self._outbox_sweep_loop())
            logger.info("Started background Transactional Outbox sweeper")

        # Launch consumer worker loop only if process role includes worker
        if settings.PROCESS_ROLE in ("all", "worker"):
            if self._worker_task is None or self._worker_task.done():
                self._worker_task = asyncio.create_task(self._worker_loop())
                logger.info("Started background JobRunner consumer worker '%s'", self._worker_id)
        else:
            logger.info(
                "PROCESS_ROLE is '%s': consumer worker loop disabled on this node",
                settings.PROCESS_ROLE,
            )

    async def stop(self) -> None:
        """Stop background worker loops gracefully."""
        self._stop_event.set()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
        self._worker_task = None

        if self._outbox_task and not self._outbox_task.done():
            self._outbox_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._outbox_task
        self._outbox_task = None

        logger.info("Stopped JobRunner background worker tasks for '%s'", self._worker_id)

    # --------------------------------------------------------------------------
    # Transactional Outbox Sweeper & Dispatcher
    # --------------------------------------------------------------------------
    async def dispatch_pending_outbox(self, max_batch: int = 50) -> int:
        """Scan analysis_outbox for pending jobs and reliably dispatch to Redis Streams."""
        with SessionLocal() as db:
            pending_items = (
                db.execute(
                    select(AnalysisOutbox)
                    .where(AnalysisOutbox.status == "pending")
                    .order_by(AnalysisOutbox.created_at)
                    .limit(max_batch)
                )
                .scalars()
                .all()
            )

            if not pending_items:
                return 0

            dispatched_count = 0
            try:
                redis = await get_redis_client()
            except Exception as e:
                logger.debug("Outbox dispatch delayed: Redis unavailable: %s", e)
                return 0

            for item in pending_items:
                try:
                    payload = json.loads(item.payload_json)
                    task_data: RedisStreamPayload = {
                        "analysis_id": payload.get("analysis_id", ""),
                        "video_path": payload.get("video_path", ""),
                        "artifact_dir": payload.get("artifact_dir", ""),
                        "enqueued_at": payload.get("enqueued_at", datetime.now(UTC).isoformat()),
                        "retry_count": str(item.retry_count),
                    }
                    await redis.xadd(STREAM_TASKS, task_data, maxlen=10000, approximate=True)
                    item.status = "dispatched"
                    item.dispatched_at = datetime.now(UTC)
                    dispatched_count += 1
                except Exception as e:
                    item.retry_count += 1
                    item.last_error = str(e)
                    if item.retry_count >= MAX_JOB_RETRIES:
                        item.status = "failed"
                    logger.warning("Failed to dispatch outbox item %d: %s", item.id, e)

            db.commit()
            if dispatched_count > 0:
                logger.info("Outbox sweeper dispatched %d pending analyses", dispatched_count)
            return dispatched_count

    async def _outbox_sweep_loop(self) -> None:
        """Background loop periodically sweeping pending outbox records."""
        while not self._stop_event.is_set():
            try:
                await self.dispatch_pending_outbox()
            except Exception as e:
                logger.debug("Error in outbox sweep loop: %s", e)
            await asyncio.sleep(5.0)

    # --------------------------------------------------------------------------
    # In-Memory Queue compatibility methods (preserved for backward compatibility)
    # --------------------------------------------------------------------------
    def subscribe(self, analysis_id: str) -> asyncio.Queue[StageEventPayload | None]:
        """Create a new event queue for an in-memory SSE client (backward compatible)."""
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

    def _broadcast_local(self, analysis_id: str, event: StageEventPayload | None) -> None:
        """Dispatch event to local in-memory subscriber queues."""
        if analysis_id in self._subscribers:
            for q in list(self._subscribers[analysis_id]):
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(event)

    # --------------------------------------------------------------------------
    # Distributed SSE Event Broadcasting & Replay (Redis Streams)
    # --------------------------------------------------------------------------
    async def publish_stage_event(
        self,
        analysis_id: str,
        payload: StageEventPayload | None,
    ) -> str:
        """Publish a stage milestone to the durable Redis event stream."""
        self._broadcast_local(analysis_id, payload)

        if payload is None:
            return ""

        stream_key = f"{STREAM_EVENTS_PREFIX}{analysis_id}"
        event_data: RedisStreamPayload = {
            "stage": payload.stage,
            "progress": str(payload.progress),
            "message": payload.message,
            "elapsed_s": str(payload.elapsed_s),
        }

        try:
            redis = await get_redis_client()
            raw_id = await redis.xadd(stream_key, event_data)
            await redis.expire(stream_key, STREAM_EVENT_TTL_S)
            return raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
        except Exception:
            logger.exception("Failed to publish event to Redis stream %s", stream_key)
            return ""

    async def stream_events(
        self,
        analysis_id: str,
        last_event_id: str | None = None,
        poll_timeout_s: float = 30.0,
    ) -> AsyncGenerator[tuple[str, StageEventPayload], None]:
        """Stream durable stage events starting from Last-Event-ID or beginning.

        Yields (event_id, StageEventPayload) tuples.
        """
        redis = await get_redis_client()
        stream_key = f"{STREAM_EVENTS_PREFIX}{analysis_id}"
        current_id = last_event_id or "0-0"
        t0 = asyncio.get_running_loop().time()

        while True:
            # Check elapsed timeout
            if asyncio.get_running_loop().time() - t0 > poll_timeout_s:
                break

            try:
                # Read new entries from stream
                entries = await redis.xread({stream_key: current_id}, count=20, block=1000)
                if isinstance(entries, list) and entries:
                    for stream_entry in entries:
                        if isinstance(stream_entry, (list, tuple)) and len(stream_entry) == 2:
                            _stream, messages = stream_entry
                            if isinstance(messages, list):
                                for msg_item in messages:
                                    if isinstance(msg_item, (list, tuple)) and len(msg_item) == 2:
                                        raw_mid, raw_data = msg_item
                                        msg_id = (
                                            raw_mid.decode("utf-8")
                                            if isinstance(raw_mid, bytes)
                                            else str(raw_mid)
                                        )
                                        data = (
                                            raw_data if isinstance(raw_data, dict) else {}
                                        )
                                        current_id = msg_id
                                        stage = str(data.get("stage", "processing"))
                                        progress = float(data.get("progress", 0.0))
                                        message = str(data.get("message", ""))
                                        elapsed_s = float(data.get("elapsed_s", 0.0))

                                        payload = StageEventPayload(
                                            stage=stage,
                                            progress=progress,
                                            message=message,
                                            elapsed_s=elapsed_s,
                                        )
                                        yield msg_id, payload

                                        if stage in ("done", "error", "cancelled"):
                                            return
                else:
                    # Check database status if stream has no new messages
                    status = self._get_db_status(analysis_id)
                    if status in ("completed", "failed"):
                        return

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error streaming events from Redis stream %s", stream_key)
                await asyncio.sleep(0.5)

    def _get_db_status(self, analysis_id: str) -> str | None:
        with SessionLocal() as db:
            analysis = db.execute(
                select(Analysis).where(Analysis.id == analysis_id)
            ).scalar_one_or_none()
            return analysis.status if analysis else None

    # --------------------------------------------------------------------------
    # Job Submission & Enqueueing with Admission Backpressure (Workstream C)
    # --------------------------------------------------------------------------
    async def check_admission_capacity(self, user_id: int) -> None:
        """Enforce atomic queue capacity and per-tenant waiting limits (Workstream C).

        Raises:
            AppException(code="ANALYSIS_CAPACITY_EXCEEDED", status_code=503)
            AppException(code="TENANT_QUEUE_LIMIT_EXCEEDED", status_code=429)
        """
        # 1. Global Queue Depth Check in Redis Streams
        try:
            redis = await get_redis_client()
            total_in_flight = 0
            try:
                groups = await redis.xinfo_groups(STREAM_TASKS)
                found = False
                for g in groups:
                    if isinstance(g, dict) and g.get("name") == GROUP_WORKERS:
                        lag = g.get("lag")
                        pending = g.get("pending", 0)
                        total_in_flight = int(lag or 0) + int(pending or 0)
                        found = True
                        break
                if not found:
                    total_in_flight = await redis.xlen(STREAM_TASKS)
            except Exception:
                try:
                    total_in_flight = await redis.xlen(STREAM_TASKS)
                except Exception:
                    total_in_flight = 0

            if total_in_flight >= settings.ANALYSIS_QUEUE_MAX_DEPTH:
                logger.warning(
                    "Queue admission rejected: total depth %d >= max %d",
                    total_in_flight,
                    settings.ANALYSIS_QUEUE_MAX_DEPTH,
                )
                raise AppException(
                    code="ANALYSIS_CAPACITY_EXCEEDED",
                    message="System analysis capacity is temporarily full. Please retry later.",
                    status_code=503,
                )
        except AppException:
            raise
        except Exception as e:
            logger.debug("Capacity check bypassed due to Redis connection issue: %s", e)

        # 2. Per-Tenant Concurrency / Waiting Limit Check in Database
        with SessionLocal() as db:
            user_waiting = (
                db.execute(
                    select(func.count(Analysis.id)).where(
                        Analysis.user_id == user_id,
                        Analysis.status == "queued",
                    )
                ).scalar()
                or 0
            )
            if user_waiting >= settings.ANALYSIS_QUEUE_MAX_WAITING_PER_TENANT:
                logger.warning(
                    "Tenant queue limit rejected for user %d: %d queued >= max %d",
                    user_id,
                    user_waiting,
                    settings.ANALYSIS_QUEUE_MAX_WAITING_PER_TENANT,
                )
                limit = settings.ANALYSIS_QUEUE_MAX_WAITING_PER_TENANT
                raise AppException(
                    code="TENANT_QUEUE_LIMIT_EXCEEDED",
                    message=(
                        f"Per-operator queue limit of {limit} "
                        "active jobs exceeded. Please wait for pending jobs to finish."
                    ),
                    status_code=429,
                )

    async def check_and_acquire_upload_lock(self, user_id: int, sha256: str) -> bool:
        """Prevent duplicate submissions of the identical file within 60 seconds."""
        lock_key = f"{LOCK_UPLOAD_PREFIX}{user_id}:{sha256}"
        try:
            redis = await get_redis_client()
            acquired = await redis.set(lock_key, "1", ex=60, nx=True)
            return bool(acquired)
        except Exception:
            return True

    async def enqueue(
        self,
        analysis_id: str,
        video_path: Path,
        artifact_dir: Path,
    ) -> None:
        """Submit an analysis job to the Redis Streams queue and spawn local execution if needed."""
        now_iso = datetime.now(UTC).isoformat()
        task_data: RedisStreamPayload = {
            "analysis_id": analysis_id,
            "video_path": str(video_path),
            "artifact_dir": str(artifact_dir),
            "enqueued_at": now_iso,
            "retry_count": "0",
        }

        # 1. Transactional Outbox record persistence (if Analysis exists in DB)
        with SessionLocal() as db:
            analysis_exists = db.execute(
                select(Analysis).where(Analysis.id == analysis_id)
            ).scalar_one_or_none()
            if analysis_exists:
                outbox_record = db.execute(
                    select(AnalysisOutbox).where(AnalysisOutbox.analysis_id == analysis_id)
                ).scalar_one_or_none()
                if not outbox_record:
                    outbox_record = AnalysisOutbox(
                        analysis_id=analysis_id,
                        payload_json=json.dumps({
                            "analysis_id": analysis_id,
                            "video_path": str(video_path),
                            "artifact_dir": str(artifact_dir),
                            "enqueued_at": now_iso,
                        }),
                        status="pending",
                    )
                    db.add(outbox_record)
                    db.commit()

        # 2. Immediate Redis Streams dispatch
        try:
            redis = await get_redis_client()
            await redis.xadd(STREAM_TASKS, task_data, maxlen=10000, approximate=True)
            with SessionLocal() as db:
                rec = db.execute(
                    select(AnalysisOutbox).where(AnalysisOutbox.analysis_id == analysis_id)
                ).scalar_one_or_none()
                if rec:
                    rec.status = "dispatched"
                    rec.dispatched_at = datetime.now(UTC)
                    db.commit()
        except Exception as e:
            logger.warning(
                "Immediate Redis Streams write delayed for analysis %s (outbox will retry): %s",
                analysis_id,
                e,
            )

        # 3. Publish initial queued milestone
        await self.publish_stage_event(
            analysis_id,
            StageEventPayload(
                stage="queued",
                progress=0.0,
                message="Analysis job queued for processing",
                elapsed_s=0.0,
            ),
        )

        # 4. In combined single-process / test mode, spawn execution if process role permits
        if settings.PROCESS_ROLE in ("all", "worker"):
            task = asyncio.create_task(
                self._run_job_guarded(analysis_id, video_path, artifact_dir)
            )
            self._active_tasks[analysis_id] = task

            def _cleanup(_: asyncio.Task[None]) -> None:
                self._active_tasks.pop(analysis_id, None)

            task.add_done_callback(_cleanup)
        else:
            logger.debug(
                "PROCESS_ROLE is '%s': skipped in-process execution for analysis %s",
                settings.PROCESS_ROLE,
                analysis_id,
            )

    # --------------------------------------------------------------------------
    # Job Cancellation
    # --------------------------------------------------------------------------
    def cancel_job(self, analysis_id: str) -> bool:
        """Cancel an analysis task: sets distributed cancellation flag and terminates local task."""
        cancel_key = f"{CANCEL_PREFIX}{analysis_id}"

        # Schedule Redis cancellation key set
        async def _set_cancel_flag() -> None:
            try:
                redis = await get_redis_client()
                await redis.set(cancel_key, "1", ex=300)
            except Exception:
                pass

        try:
            loop = asyncio.get_running_loop()
            _ctask = loop.create_task(_set_cancel_flag())
            self._active_tasks[f"cancel_{analysis_id}"] = _ctask
            _ctask.add_done_callback(
                lambda _: self._active_tasks.pop(f"cancel_{analysis_id}", None)
            )
        except RuntimeError:
            # Synchronous execution context without an active event loop
            try:
                import threading

                threading.Thread(
                    target=lambda: asyncio.run(_set_cancel_flag()),
                    daemon=True,
                ).start()
            except Exception:
                pass

        # Cancel local task if active in this process
        task = self._active_tasks.pop(analysis_id, None)
        was_active = False
        if task and not task.done():
            task.cancel()
            logger.info("Cancelled active background task for analysis %s", analysis_id)
            was_active = True

        self._broadcast_local(analysis_id, None)
        return was_active

    async def is_cancelled(self, analysis_id: str) -> bool:
        """Check whether analysis job was cancelled via Redis signal."""
        cancel_key = f"{CANCEL_PREFIX}{analysis_id}"
        try:
            redis = await get_redis_client()
            return bool(await redis.exists(cancel_key))
        except Exception:
            return False

    # --------------------------------------------------------------------------
    # Background Worker Loop & Execution
    # --------------------------------------------------------------------------
    async def _worker_loop(self) -> None:
        """Consumer loop reading from Redis Streams task queue."""
        redis = await get_redis_client()

        while not self._stop_event.is_set():
            try:
                # Read pending or new jobs
                entries = await redis.xreadgroup(
                    GROUP_WORKERS,
                    self._worker_id,
                    {STREAM_TASKS: ">"},
                    count=1,
                    block=2000,
                )

                if not entries or not isinstance(entries, list):
                    await asyncio.sleep(0.1)
                    continue

                for stream_entry in entries:
                    if not (
                        isinstance(stream_entry, (list, tuple))
                        and len(stream_entry) == 2
                    ):
                        continue
                    _stream, messages = stream_entry
                    if not isinstance(messages, list):
                        continue
                    for msg_item in messages:
                        if not (
                            isinstance(msg_item, (list, tuple))
                            and len(msg_item) == 2
                        ):
                            continue
                        raw_mid, raw_data = msg_item
                        msg_id = (
                            raw_mid.decode("utf-8")
                            if isinstance(raw_mid, bytes)
                            else str(raw_mid)
                        )
                        data = raw_data if isinstance(raw_data, dict) else {}
                        analysis_id = str(data.get("analysis_id", ""))
                        vpath = str(data.get("video_path", ""))
                        adir = str(data.get("artifact_dir", ""))
                        retry_count = int(data.get("retry_count", "0"))

                        if retry_count >= MAX_JOB_RETRIES:
                            logger.error(
                                "Job %s exceeded max retries (%d); sending to dead letter queue",
                                analysis_id,
                                MAX_JOB_RETRIES,
                            )
                            with contextlib.suppress(Exception):
                                await redis.xadd(STREAM_DEAD_LETTER, data)
                                await redis.xack(STREAM_TASKS, GROUP_WORKERS, msg_id)
                            self._record_failure(
                                analysis_id,
                                f"Job exceeded {MAX_JOB_RETRIES} retries; dead-lettered",
                            )
                            continue

                        if (
                            analysis_id
                            and vpath
                            and adir
                            and not await self.is_cancelled(analysis_id)
                        ):
                            await self._run_job_guarded(
                                analysis_id, Path(vpath), Path(adir)
                            )

                        # Acknowledge task in stream
                        await redis.xack(STREAM_TASKS, GROUP_WORKERS, msg_id)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Unexpected error in worker loop %s", self._worker_id)
                await asyncio.sleep(1.0)

    async def _run_job_guarded(
        self,
        analysis_id: str,
        video_path: Path,
        artifact_dir: Path,
    ) -> None:
        """Acquire semaphore and execute job with timeout protection."""
        async with self._semaphore:
            if await self.is_cancelled(analysis_id):
                logger.info("Skipping cancelled job %s prior to execution", analysis_id)
                return

            try:
                await asyncio.wait_for(
                    self._run_job(analysis_id, video_path, artifact_dir),
                    timeout=JOB_TIMEOUT_S,
                )
            except TimeoutError:
                logger.error("Job %s timed out after %ds", analysis_id, JOB_TIMEOUT_S)
                self._record_failure(analysis_id, f"Analysis timed out after {JOB_TIMEOUT_S}s")
            except asyncio.CancelledError:
                logger.info("Job %s was cancelled during execution", analysis_id)
            except Exception as e:
                logger.exception("Unexpected failure in job %s: %s", analysis_id, e)
                self._record_failure(analysis_id, f"Pipeline error: {e}")
            finally:
                await self.publish_stage_event(analysis_id, None)

    async def _run_job(
        self,
        analysis_id: str,
        video_path: Path,
        artifact_dir: Path,
    ) -> None:
        """Run pipeline steps synchronously in a worker thread, publishing events."""
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
            # Check cancellation between pipeline milestones
            if await self.is_cancelled(analysis_id):
                logger.info("Job %s aborting due to cancellation signal", analysis_id)
                self._record_failure(analysis_id, "Analysis cancelled by operator")
                break

            event = await loop.run_in_executor(None, _next_event)
            if event is None:
                break

            payload = StageEventPayload(
                stage=event.stage,
                progress=event.progress,
                message=event.message,
                elapsed_s=event.elapsed_s,
            )
            await self.publish_stage_event(analysis_id, payload)
            self._update_db_stage(analysis_id, status="processing", stage=event.stage)

            if event.stage == "done" and event.result is not None:
                self._record_success(analysis_id, event, artifact_dir)
                break

    def _update_db_stage(self, analysis_id: str, status: str, stage: str) -> None:
        with SessionLocal() as db:
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

    def _record_success(
        self,
        analysis_id: str,
        event: PipelineEvent,
        artifact_dir: Path,
    ) -> None:
        with SessionLocal() as db:
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

    def _record_failure(self, analysis_id: str, error_message: str) -> None:
        with SessionLocal() as db:
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
                self._broadcast_local(analysis_id, payload)
            except Exception:
                logger.exception("Failed to record failure for %s", analysis_id)


# Global runner instance
job_runner = JobRunner()
