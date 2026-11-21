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


@pytest.mark.asyncio
async def test_admission_capacity_tenant_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a tenant has reached max waiting jobs, admission must be rejected with 429."""
    from app.core.config import settings
    from app.core.errors import AppException
    from app.db.models import Analysis, User
    from app.db.session import SessionLocal

    runner = JobRunner()
    monkeypatch.setattr(settings, "ANALYSIS_QUEUE_MAX_WAITING_PER_TENANT", 3)

    with SessionLocal() as db:
        # Create test user
        test_user = db.query(User).filter(User.email == "tenant_limit_test@example.com").first()
        if not test_user:
            test_user = User(
                email="tenant_limit_test@example.com",
                password_hash="fakehash",
                role="operator",
            )
            db.add(test_user)
            db.commit()
            db.refresh(test_user)

        # Clear any prior queued analyses for this user
        db.query(Analysis).filter(
            Analysis.user_id == test_user.id, Analysis.status == "queued"
        ).delete()
        db.commit()

        # Add 3 queued analyses
        for i in range(3):
            db.add(
                Analysis(
                    id=f"tenant-test-job-{i}",
                    user_id=test_user.id,
                    status="queued",
                    original_name=f"test_{i}.mp4",
                    sha256=f"hash_{i}",
                    size_bytes=1024,
                    duration_s=2.0,
                    video_path="/tmp/fake.mp4",
                )
            )
        db.commit()

        # 4th check should raise 429 TENANT_QUEUE_LIMIT_EXCEEDED
        with pytest.raises(AppException) as exc_info:
            await runner.check_admission_capacity(user_id=test_user.id)
        assert exc_info.value.code == "TENANT_QUEUE_LIMIT_EXCEEDED"
        assert exc_info.value.status_code == 429

        # Cleanup
        db.query(Analysis).filter(Analysis.user_id == test_user.id).delete()
        db.delete(test_user)
        db.commit()


@pytest.mark.asyncio
async def test_admission_capacity_global_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    """When global stream depth reaches max, admission must be rejected with 503."""
    from app.core.config import settings
    from app.core.errors import AppException

    runner = JobRunner()
    monkeypatch.setattr(settings, "ANALYSIS_QUEUE_MAX_DEPTH", 2)

    redis = await get_redis_client()
    # Add dummy stream items to exceed depth
    await redis.xadd(STREAM_TASKS, {"analysis_id": "dummy-1"})
    await redis.xadd(STREAM_TASKS, {"analysis_id": "dummy-2"})

    with pytest.raises(AppException) as exc_info:
        await runner.check_admission_capacity(user_id=999)
    assert exc_info.value.code == "ANALYSIS_CAPACITY_EXCEEDED"
    assert exc_info.value.status_code == 503

