"""Tests for Standalone GPU Inference Worker."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.redis import close_redis, get_redis_client
from app.db.models import Analysis, Base, User
from app.services.job_runner import GROUP_WORKERS, STREAM_TASKS
from app.workers.inference_worker import InferenceWorker


@pytest.fixture
def worker_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    with TestingSession() as session:
        user = User(
            email="worker_tester@parallax.io",
            password_hash="hash_placeholder",
            role="operator",
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        analysis = Analysis(
            id="worker-aid-202",
            user_id=user.id,
            original_name="test_worker.mp4",
            sha256="b" * 64,
            size_bytes=4096,
            duration_s=2.0,
            status="queued",
            stage="probe",
            audio_present=True,
            model_version="1.0.0",
            model_mode="trained",
        )
        session.add(analysis)
        session.commit()
        session.refresh(analysis)
        yield session


def test_worker_initialization_and_concurrency_guard() -> None:
    """Worker must initialize with configurable concurrency and correct device detection."""
    worker = InferenceWorker(worker_id="test-worker-1", max_concurrency=1)
    assert worker.worker_id == "test-worker-1"
    assert worker.max_concurrency == 1
    assert worker.device_type in ("cuda", "cpu")
    assert worker._semaphore._value == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_worker_startup_and_model_loading() -> None:
    """Worker startup must ensure consumer group on Redis Streams and preload model."""
    await close_redis()
    worker = InferenceWorker(worker_id="startup-worker", max_concurrency=1)

    with patch("app.workers.inference_worker.load_model") as mock_load:
        await worker.startup()
        mock_load.assert_called_once()

    redis = await get_redis_client()
    groups = await redis.xinfo_groups(STREAM_TASKS)
    found = any(g.get("name") == GROUP_WORKERS for g in groups)
    assert found, "Consumer group must exist after worker startup"
    await worker.shutdown()


@pytest.mark.asyncio
async def test_worker_cancellation_signal_handling(worker_db) -> None:
    """When a job cancellation key is set in Redis, worker must abort before pipeline execution."""
    await close_redis()
    worker = InferenceWorker(worker_id="cancel-worker", max_concurrency=1)
    redis = await get_redis_client()

    aid = "worker-aid-202"
    cancel_key = f"parallax:cancel:{aid}"
    await redis.set(cancel_key, "1", ex=300)

    task_data = {
        "analysis_id": aid,
        "video_path": "uploads/fake.mp4",
        "artifact_dir": "artifacts/worker-aid-202",
        "retry_count": "0",
    }

    with patch.object(worker, "_execute_pipeline") as mock_exec:
        await worker._process_guarded("msg-id-123", task_data)
        mock_exec.assert_not_called()

    await worker.shutdown()


@pytest.mark.asyncio
async def test_worker_stalled_task_recovery() -> None:
    """_recover_stalled_tasks must execute xautoclaim without unhandled errors."""
    await close_redis()
    worker = InferenceWorker(worker_id="recovery-worker", max_concurrency=1)
    redis = await get_redis_client()

    # Ensure group exists
    with contextlib.suppress(Exception):
        await redis.xgroup_create(STREAM_TASKS, GROUP_WORKERS, id="0", mkstream=True)

    # Should run smoothly even when PEL is empty
    await worker._recover_stalled_tasks(redis)
    await worker.shutdown()


@pytest.mark.asyncio
async def test_worker_graceful_shutdown() -> None:
    """Worker shutdown must set stop event, cancel pending tasks, and close redis client."""
    worker = InferenceWorker(worker_id="shutdown-worker", max_concurrency=1)
    dummy_task = asyncio.create_task(asyncio.sleep(10.0))
    worker._active_tasks.add(dummy_task)

    await worker.shutdown()
    assert worker._stop_event.is_set()
    assert dummy_task.cancelled()
