"""Standalone GPU Inference Worker for PARALLAX.

Decouples deep learning inference from FastAPI web instances.
Key features:
- Consumes tasks from Redis Streams (`parallax:jobs:video_analysis`).
- Loads AVT²-DWF model once per worker process at startup.
- Manages GPU memory conservatively with strict concurrency limits (default 1 for 4GB VRAM).
- Handles S3/MinIO remote video retrieval and artifact storage.
- Recovers unacknowledged tasks via XAUTOCLAIM.
- Routes failing poison-pill tasks to the dead-letter stream after 3 retries.
- Publishes real-time milestones through the durable SSE event bus.
- Updates PostgreSQL analysis records transactionally.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from sqlalchemy import select

from app.core.config import settings
from app.core.logging import get_logger, set_job_context, setup_logging
from app.core.metrics import (
    INFERENCE_DURATION_SECONDS,
    INFERENCE_FAILED_TOTAL,
    WORKER_ACTIVE_JOBS,
    WORKER_HEARTBEAT_TIMESTAMP,
)
from app.core.redis import close_redis, get_redis_client
from app.db.models import Analysis, AnalysisArtifact
from app.db.session import SessionLocal
from app.ml.device import get_device_info
from app.ml.inference import get_model, load_model
from app.ml.pipeline import PipelineEvent, run_pipeline
from app.schemas.analysis import StageEventPayload
from app.services.job_runner import (
    GROUP_WORKERS,
    MAX_JOB_RETRIES,
    STREAM_CLAIM_MIN_IDLE_MS,
    STREAM_DEAD_LETTER,
    STREAM_TASKS,
    job_runner,
)
from app.services.storage_s3 import get_storage_service

setup_logging()
logger = get_logger("worker")


class InferenceWorker:
    """Dedicated background worker executing AVT²-DWF forensic pipelines."""

    def __init__(
        self,
        worker_id: str | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        pid = os.getpid()
        self.worker_id = worker_id or f"worker-{pid}"
        concurrency = (
            max_concurrency
            if max_concurrency is not None
            else settings.WORKER_MAX_CONCURRENCY
        )
        self.max_concurrency = max(1, concurrency)
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._stop_event = asyncio.Event()
        self._active_tasks: set[asyncio.Task[None]] = set()
        self._claim_counter = 0

        # Centralized inference device resolution
        self.device_info = get_device_info()
        self.device = self.device_info.device
        self.device_name = self.device_info.device_name
        self.device_type = self.device.type

        if self.device_info.is_cuda:
            logger.info(
                "InferenceWorker initialized on GPU: %s (%.0f MB VRAM, max concurrency=%d)",
                self.device_name,
                self.device_info.total_memory_mb,
                self.max_concurrency,
            )
        else:
            logger.info(
                "InferenceWorker initialized on CPU (max concurrency=%d)",
                self.max_concurrency,
            )

    async def startup(self) -> None:
        """Initialize worker dependencies, Redis group, and preload model."""
        logger.info("Starting worker '%s'...", self.worker_id)

        # Enforce GPU requirement if configured
        if settings.REQUIRE_GPU and not self.device_info.is_cuda:
            msg = (
                f"Worker '{self.worker_id}' requires GPU execution (REQUIRE_GPU=True), "
                f"but initialized with device '{self.device_type}'."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info(
            "[GPU] PyTorch version: %s | CUDA build: %s",
            torch.__version__,
            getattr(torch.version, "cuda", "none"),
        )
        logger.info(
            "[GPU] CUDA available: %s | Device count: %d",
            torch.cuda.is_available(),
            torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )
        logger.info(
            "[GPU] Selected inference device: %s (%s)",
            self.device,
            self.device_name,
        )
        logger.info("[GPU] Worker max concurrency: %d", self.max_concurrency)

        # 1. Connect to Redis and ensure consumer group exists
        redis = await get_redis_client()
        try:
            await redis.xgroup_create(STREAM_TASKS, GROUP_WORKERS, id="0", mkstream=True)
            logger.info("Ensured consumer group '%s' on '%s'", GROUP_WORKERS, STREAM_TASKS)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.debug("xgroup_create: %s", e)

        # 2. Preload AVT²-DWF model into memory once on target device
        t0 = asyncio.get_running_loop().time()
        load_model(device=self.device)
        model = get_model()
        model_device = next(model.parameters()).device
        dt = asyncio.get_running_loop().time() - t0
        logger.info("[GPU] Model loaded on device: %s in %.2fs", model_device, dt)

        if self.device_info.is_cuda and model_device.type != "cuda":
            raise RuntimeError(
                f"Model placement failure: expected cuda, got {model_device}"
            )

    async def run(self) -> None:
        """Main worker consumption and task execution loop."""
        redis = await get_redis_client()
        logger.info("Worker '%s' listening for tasks on '%s'...", self.worker_id, STREAM_TASKS)

        while not self._stop_event.is_set():
            WORKER_HEARTBEAT_TIMESTAMP.labels(worker_id=self.worker_id).set(time.time())
            # Periodic recovery of stalled/orphaned tasks from crashed workers
            self._claim_counter += 1
            if self._claim_counter % 10 == 0:
                await self._recover_stalled_tasks(redis)

            try:
                # Read new tasks from the consumer group
                entries = await redis.xreadgroup(
                    GROUP_WORKERS,
                    self.worker_id,
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
                        data = (
                            {
                                (k.decode("utf-8") if isinstance(k, bytes) else str(k)): (
                                    v.decode("utf-8") if isinstance(v, bytes) else str(v)
                                )
                                for k, v in raw_data.items()
                            }
                            if isinstance(raw_data, dict)
                            else {}
                        )

                        task = asyncio.create_task(self._process_guarded(msg_id, data))
                        self._active_tasks.add(task)
                        task.add_done_callback(self._active_tasks.discard)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Worker loop error: %s", e)
                await asyncio.sleep(1.0)

    async def _recover_stalled_tasks(self, redis: Any) -> None:
        """Reclaim pending tasks abandoned by dead/crashed workers."""
        try:
            claim_res = await redis.xautoclaim(
                STREAM_TASKS,
                GROUP_WORKERS,
                self.worker_id,
                min_idle_time=STREAM_CLAIM_MIN_IDLE_MS,
                start_id="0-0",
                count=5,
            )
            if isinstance(claim_res, (list, tuple)) and len(claim_res) >= 2:
                claimed_messages = claim_res[1]
                if isinstance(claimed_messages, list) and claimed_messages:
                    logger.info("Claimed %d stalled tasks from dead workers", len(claimed_messages))
                    for msg_item in claimed_messages:
                        if isinstance(msg_item, (list, tuple)) and len(msg_item) == 2:
                            raw_mid, raw_data = msg_item
                            msg_id = (
                                raw_mid.decode("utf-8")
                                if isinstance(raw_mid, bytes)
                                else str(raw_mid)
                            )
                            data = (
                                {
                                    (k.decode("utf-8") if isinstance(k, bytes) else str(k)): (
                                        v.decode("utf-8") if isinstance(v, bytes) else str(v)
                                    )
                                    for k, v in raw_data.items()
                                }
                                if isinstance(raw_data, dict)
                                else {}
                            )
                            task = asyncio.create_task(self._process_guarded(msg_id, data))
                            self._active_tasks.add(task)
                            task.add_done_callback(self._active_tasks.discard)
        except Exception as e:
            logger.debug("xautoclaim exception (normal if no pending entries): %s", e)

    async def _process_guarded(self, msg_id: str, data: dict[str, str]) -> None:
        """Execute inference task with concurrency semaphore and retry limits."""
        analysis_id = data.get("analysis_id", "")
        set_job_context(analysis_id)
        WORKER_ACTIVE_JOBS.labels(worker_id=self.worker_id).inc()

        try:
            async with self._semaphore:
                video_ref = data.get("video_path", "")
                artifact_dir_ref = data.get("artifact_dir", "")
                retry_count = int(data.get("retry_count", "0"))

                redis = await get_redis_client()

                # Dead-letter routing on exceeded retries
                if retry_count >= MAX_JOB_RETRIES:
                    logger.error(
                        "Job %s exceeded retry limit (%d); moving to dead-letter stream",
                        analysis_id,
                        MAX_JOB_RETRIES,
                        extra={"analysis_id": analysis_id, "worker_id": self.worker_id},
                    )
                    INFERENCE_FAILED_TOTAL.labels(stage="max_retries_exceeded").inc()
                    try:
                        dl_payload: dict[
                            bytes | bytearray | memoryview | str | int | float,
                            bytes | bytearray | memoryview | str | int | float,
                        ] = {}
                        for k, v in data.items():
                            dl_payload[k] = v
                        await redis.xadd(STREAM_DEAD_LETTER, dl_payload)
                        await redis.xack(STREAM_TASKS, GROUP_WORKERS, msg_id)
                    except Exception as e:
                        logger.error("Failed to move job %s to dead letter: %s", analysis_id, e)

                    self._record_db_failure(
                        analysis_id,
                        (
                            f"Job failed after {MAX_JOB_RETRIES} attempts "
                            "and was routed to dead letter"
                        ),
                    )
                    return

                if not analysis_id or not video_ref:
                    logger.error("Malformed task data: %s", data)
                    await redis.xack(STREAM_TASKS, GROUP_WORKERS, msg_id)
                    return

                # Check if job was cancelled
                if await job_runner.is_cancelled(analysis_id):
                    logger.info(
                        "Job %s was cancelled before worker execution; skipping",
                        analysis_id,
                    )
                    await redis.xack(STREAM_TASKS, GROUP_WORKERS, msg_id)
                    return

                # GPU-Aware VRAM Admission Check (Workstream D)
                if self.device_info.is_cuda:
                    free_b, _ = torch.cuda.mem_get_info(self.device)
                    free_mb = free_b / (1024 * 1024)
                    if free_mb < settings.INFERENCE_GPU_MEMORY_RESERVE_MB:
                        logger.warning(
                            "VRAM pressure: %.1f MB free < %d MB reserve; clearing CUDA cache...",
                            free_mb,
                            settings.INFERENCE_GPU_MEMORY_RESERVE_MB,
                        )
                        torch.cuda.empty_cache()
                        free_b, _ = torch.cuda.mem_get_info(self.device)
                        free_mb = free_b / (1024 * 1024)
                        if free_mb < (settings.INFERENCE_GPU_MEMORY_RESERVE_MB / 2):
                            raise RuntimeError(
                                f"Insufficient GPU VRAM ({free_mb:.1f} MB); yielding retry"
                            )

                t0 = time.perf_counter()
                try:
                    await asyncio.wait_for(
                        self._execute_pipeline(analysis_id, video_ref, artifact_dir_ref),
                        timeout=float(settings.INFERENCE_JOB_TIMEOUT_SECONDS),
                    )
                    dt = time.perf_counter() - t0
                    INFERENCE_DURATION_SECONDS.labels(status="completed").observe(dt)
                    # Successful completion: Acknowledge task in stream
                    await redis.xack(STREAM_TASKS, GROUP_WORKERS, msg_id)
                    logger.info(
                        "Successfully completed and acknowledged job %s in %.2fs",
                        analysis_id,
                        dt,
                        extra={
                            "analysis_id": analysis_id,
                            "worker_id": self.worker_id,
                            "duration_ms": dt * 1000,
                        },
                    )
                except Exception as e:
                    dt = time.perf_counter() - t0
                    INFERENCE_DURATION_SECONDS.labels(status="failed").observe(dt)
                    INFERENCE_FAILED_TOTAL.labels(stage="pipeline").inc()
                    logger.exception(
                        "Inference execution failed for job %s: %s",
                        analysis_id,
                        e,
                        extra={"analysis_id": analysis_id, "worker_id": self.worker_id},
                    )
                    # Increment retry count
                    data["retry_count"] = str(retry_count + 1)
                    self._record_db_failure(analysis_id, f"Pipeline error: {e}")
                    # Acknowledge failed attempt in stream; outbox or retry will re-enqueue
                    await redis.xack(STREAM_TASKS, GROUP_WORKERS, msg_id)
        finally:
            WORKER_ACTIVE_JOBS.labels(worker_id=self.worker_id).dec()

    async def _execute_pipeline(
        self,
        analysis_id: str,
        video_ref: str,
        artifact_dir_ref: str,
    ) -> None:
        """Retrieve video, execute pipeline steps, store artifacts, and update DB."""
        storage = get_storage_service()
        scratch_dir = settings.storage_path / "scratch" / analysis_id
        scratch_dir.mkdir(parents=True, exist_ok=True)
        local_video_path = Path(video_ref)

        # Download from S3/MinIO if remote key or missing locally
        if not local_video_path.exists():
            remote_key = video_ref.replace("\\", "/").lstrip("/")
            local_dest = scratch_dir / "source.mp4"
            logger.info("Downloading remote video %s to %s", remote_key, local_dest)
            storage.download_file(remote_key, local_dest)
            local_video_path = local_dest

        local_artifact_dir = (
            Path(artifact_dir_ref)
            if Path(artifact_dir_ref).is_absolute()
            else settings.storage_path / "artifacts" / analysis_id
        )
        local_artifact_dir.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_running_loop()
        generator = run_pipeline(local_video_path, local_artifact_dir)

        def _next_event() -> PipelineEvent | None:
            try:
                return next(generator)
            except StopIteration:
                return None

        while True:
            # Check cancellation between stages
            if await job_runner.is_cancelled(analysis_id):
                logger.info("Job %s aborted due to cancellation signal", analysis_id)
                self._record_db_failure(analysis_id, "Analysis cancelled by operator")
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
            await job_runner.publish_stage_event(analysis_id, payload)
            self._update_db_stage(analysis_id, status="processing", stage=event.stage)

            if event.stage == "done" and event.result is not None:
                # Store artifacts in object storage if configured
                self._persist_artifacts_and_success(analysis_id, event, local_artifact_dir)

        # Publish terminal stream event
        await job_runner.publish_stage_event(analysis_id, None)

        # Cleanup scratch files
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def _persist_artifacts_and_success(
        self,
        analysis_id: str,
        event: PipelineEvent,
        artifact_dir: Path,
    ) -> None:
        """Upload artifacts to storage and commit completed status to PostgreSQL."""
        result = event.result
        if result is None:
            return

        storage = get_storage_service()

        with SessionLocal() as db:
            analysis = db.execute(
                select(Analysis).where(Analysis.id == analysis_id)
            ).scalar_one_or_none()
            if not analysis:
                return

            analysis.status = "completed"
            analysis.stage = "done"
            analysis.verdict = result.inference.verdict
            analysis.p_fake = result.inference.p_fake
            analysis.w_audio = result.inference.w_audio
            analysis.w_video = result.inference.w_video
            analysis.sync_score = result.inference.sync_score
            analysis.face_coverage = result.video.face_coverage
            analysis.model_version = result.inference.model_version
            analysis.model_mode = result.inference.model_mode
            analysis.timeline_json = json.dumps(
                [
                    {"timestamp_s": round(t, 2), "p_fake": round(p, 4)}
                    for t, p in zip(
                        result.video.timestamps,
                        result.inference.per_window_scores,
                        strict=False,
                    )
                ]
            )
            analysis.reasons_json = json.dumps(result.reasons)
            analysis.completed_at = datetime.now(UTC)

            # Upload generated artifacts to storage
            if artifact_dir.exists():
                for art_file in artifact_dir.glob("*.png"):
                    rel_key = f"artifacts/{analysis_id}/{art_file.name}"
                    storage.upload_file(
                        art_file,
                        rel_key,
                        content_type="image/png",
                    )
                    db.add(
                        AnalysisArtifact(
                            analysis_id=analysis_id,
                            kind="heatmap",
                            name=art_file.name,
                            path=rel_key,
                        )
                    )

            db.commit()
            logger.info(
                "Persisted analysis %s: verdict=%s (p_fake=%.4f)",
                analysis_id,
                analysis.verdict,
                analysis.p_fake or 0.0,
            )

    def _update_db_stage(self, analysis_id: str, status: str, stage: str) -> None:
        with SessionLocal() as db:
            analysis = db.execute(
                select(Analysis).where(Analysis.id == analysis_id)
            ).scalar_one_or_none()
            if analysis and analysis.status != "cancelled":
                analysis.status = status
                analysis.stage = stage
                db.commit()

    def _record_db_failure(self, analysis_id: str, error_message: str) -> None:
        with SessionLocal() as db:
            analysis = db.execute(
                select(Analysis).where(Analysis.id == analysis_id)
            ).scalar_one_or_none()
            if analysis:
                analysis.status = "error"
                analysis.stage = "error"
                analysis.error = error_message
                analysis.completed_at = datetime.now(UTC)
                db.commit()

    async def shutdown(self) -> None:
        """Graceful shutdown of worker loops and active tasks."""
        logger.info("Initiating graceful shutdown for worker '%s'...", self.worker_id)
        self._stop_event.set()

        if self._active_tasks:
            logger.info("Cancelling %d active tasks...", len(self._active_tasks))
            for t in list(self._active_tasks):
                t.cancel()
            await asyncio.gather(*self._active_tasks, return_exceptions=True)

        await close_redis()
        logger.info("Worker '%s' cleanly terminated.", self.worker_id)


async def main() -> None:
    """Entrypoint for standalone worker command."""
    worker = InferenceWorker()

    loop = asyncio.get_running_loop()
    # Register termination signals
    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(worker.shutdown()))

    try:
        await worker.startup()
        await worker.run()
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        await worker.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
