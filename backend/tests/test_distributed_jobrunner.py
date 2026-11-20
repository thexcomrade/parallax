"""Tests for Redis-backed distributed JobRunner, consumer groups, and task lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.core.redis import close_redis, get_redis_client
from app.services.job_runner import STREAM_TASKS, JobRunner


@pytest.fixture(autouse=True)
async def cleanup_redis() -> None:
    yield
    await close_redis()


@pytest.mark.asyncio
async def test_redis_client_connection_and_fallback() -> None:
    """get_redis_client must return an active client (live or in-memory fake)."""
    redis = await get_redis_client()
    assert redis is not None
    pong = await redis.ping()
    assert pong is True


@pytest.mark.asyncio
async def test_duplicate_upload_lock_acquisition() -> None:
    """Duplicate submissions of identical file within 60s must be prevented."""
    runner = JobRunner()
    user_id = 9999
    sha256 = "a" * 64

    # First acquisition succeeds
    acq1 = await runner.check_and_acquire_upload_lock(user_id=user_id, sha256=sha256)
    assert acq1 is True

    # Immediate second acquisition fails (locked)
    acq2 = await runner.check_and_acquire_upload_lock(user_id=user_id, sha256=sha256)
    assert acq2 is False


@pytest.mark.asyncio
async def test_job_submission_and_redis_stream_persistence(tmp_path: Path) -> None:
    """Submitting a job must push task parameters into Redis Streams."""
    runner = JobRunner()
    aid = "stream-test-aid-001"
    vpath = tmp_path / "test.mp4"
    vpath.write_text("fake video bytes")
    adir = tmp_path / "artifacts"
    adir.mkdir()

    redis = await get_redis_client()
    await runner.enqueue(aid, vpath, adir)

    # Verify task entry was written to Redis stream
    entries = await redis.xread({STREAM_TASKS: "0-0"}, count=10)
    assert len(entries) > 0
    found = False
    for _stream, msgs in entries:
        for _mid, data in msgs:
            if data.get("analysis_id") == aid:
                found = True
                assert data.get("video_path") == str(vpath)
                assert data.get("artifact_dir") == str(adir)
    assert found, "Enqueued analysis ID must exist in Redis stream"


@pytest.mark.asyncio
async def test_distributed_cancellation_signal() -> None:
    """Cancelling a job sets a distributed cancellation key in Redis."""
    runner = JobRunner()
    aid = "cancel-test-aid-002"

    assert await runner.is_cancelled(aid) is False
    runner.cancel_job(aid)

    # Allow async task to persist cancellation key
    await asyncio.sleep(0.05)
    assert await runner.is_cancelled(aid) is True
