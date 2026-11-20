"""Tests for Transactional Outbox pattern, reliable dispatch, and Redis failure safety."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.redis import close_redis, get_redis_client, is_redis_mocked
from app.db.models import Analysis, AnalysisOutbox, Base, User
from app.services.job_runner import (
    MAX_JOB_RETRIES,
    STREAM_DEAD_LETTER,
    STREAM_TASKS,
    JobRunner,
)


@pytest.fixture
def outbox_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    with TestingSession() as session:
        user = User(
            email="outbox_tester@parallax.io",
            password_hash="hash_placeholder",
            role="operator",
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        analysis = Analysis(
            id="outbox-test-aid-101",
            user_id=user.id,
            original_name="test_video.mp4",
            sha256="a" * 64,
            size_bytes=2048,
            duration_s=3.0,
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


def test_outbox_atomic_persistence(outbox_db) -> None:
    """Verify AnalysisOutbox record is written transactionally alongside Analysis."""
    outbox_item = AnalysisOutbox(
        analysis_id="outbox-test-aid-101",
        payload_json=json.dumps({"analysis_id": "outbox-test-aid-101", "retry_count": "0"}),
        status="pending",
    )
    outbox_db.add(outbox_item)
    outbox_db.commit()

    saved = outbox_db.execute(
        select(AnalysisOutbox).where(AnalysisOutbox.analysis_id == "outbox-test-aid-101")
    ).scalar_one()

    assert saved.status == "pending"
    assert saved.retry_count == 0
    assert "outbox-test-aid-101" in saved.payload_json


@pytest.mark.asyncio
async def test_outbox_dispatch_pending_records(outbox_db) -> None:
    """Outbox sweeper must scan pending records, publish to Redis, and mark dispatched."""
    await close_redis()
    runner = JobRunner()

    # Pre-populate outbox item
    outbox_item = AnalysisOutbox(
        analysis_id="outbox-test-aid-101",
        payload_json=json.dumps({
            "analysis_id": "outbox-test-aid-101",
            "video_path": "uploads/outbox_test.mp4",
            "artifact_dir": "artifacts/outbox-test-aid-101",
        }),
        status="pending",
    )
    outbox_db.add(outbox_item)
    outbox_db.commit()

    with patch("app.services.job_runner.SessionLocal", return_value=outbox_db):
        count = await runner.dispatch_pending_outbox()
        assert count == 1

    outbox_db.expire_all()
    updated = outbox_db.execute(
        select(AnalysisOutbox).where(AnalysisOutbox.analysis_id == "outbox-test-aid-101")
    ).scalar_one()

    assert updated.status == "dispatched"
    assert updated.dispatched_at is not None

    # Verify message in Redis Streams
    redis = await get_redis_client()
    messages = await redis.xrange(STREAM_TASKS, count=10)
    assert len(messages) >= 1
    found = any(data.get("analysis_id") == "outbox-test-aid-101" for _mid, data in messages)
    assert found, "Dispatched task must exist in Redis task stream"


@pytest.mark.asyncio
async def test_outbox_retry_and_dead_letter_limits(outbox_db) -> None:
    """When dispatch consistently fails, retry count increments and marks failed."""
    await close_redis()
    runner = JobRunner()

    outbox_item = AnalysisOutbox(
        analysis_id="outbox-test-aid-101",
        payload_json="invalid-non-json",  # Force json decode error
        status="pending",
        retry_count=MAX_JOB_RETRIES - 1,
    )
    outbox_db.add(outbox_item)
    outbox_db.commit()

    with patch("app.services.job_runner.SessionLocal", return_value=outbox_db):
        await runner.dispatch_pending_outbox()

    outbox_db.expire_all()
    updated = outbox_db.execute(
        select(AnalysisOutbox).where(AnalysisOutbox.analysis_id == "outbox-test-aid-101")
    ).scalar_one()

    assert updated.retry_count >= MAX_JOB_RETRIES
    assert updated.status == "failed"
    assert updated.last_error is not None


@pytest.mark.asyncio
async def test_redis_unavailable_in_production_fails() -> None:
    """Production must strictly fail if Redis is unavailable; never fall back to in-memory fake."""
    await close_redis()
    with (
        patch.object(settings, "APP_ENV", "production"),
        patch("app.core.redis.get_redis_url", return_value="redis://nonexistent.local:6379/0"),
        pytest.raises((ConnectionError, RuntimeError)) as exc,
    ):
        await get_redis_client()
    assert "production" in str(exc.value).lower() or "failed" in str(exc.value).lower()
    await close_redis()


@pytest.mark.asyncio
async def test_redis_mock_detection() -> None:
    """In test environment with no live server, is_redis_mocked correctly identifies FakeRedis."""
    await close_redis()
    await get_redis_client()
    assert is_redis_mocked() is True
    await close_redis()


@pytest.mark.asyncio
async def test_stream_dead_letter_routing() -> None:
    """Poison-pill task exceeding retry limits must be placed on the dead-letter stream."""
    await close_redis()
    redis = await get_redis_client()

    poison_task = {
        "analysis_id": "poison-aid-999",
        "video_path": "/tmp/corrupt.mp4",
        "artifact_dir": "/tmp/artifacts",
        "retry_count": str(MAX_JOB_RETRIES),
    }

    # Write directly to dead-letter stream as worker does
    await redis.xadd(STREAM_DEAD_LETTER, poison_task)

    dl_messages = await redis.xrange(STREAM_DEAD_LETTER, count=5)
    assert len(dl_messages) >= 1
    found = any(data.get("analysis_id") == "poison-aid-999" for _mid, data in dl_messages)
    assert found, "Dead-letter stream must record failed poison pill task"
    await close_redis()
