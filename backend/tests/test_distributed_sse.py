"""Tests for Redis-backed distributed SSE event streaming, ordering, and Last-Event-ID replay."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.redis import close_redis
from app.core.security import create_access_token
from app.db.models import Analysis
from app.schemas.analysis import StageEventPayload
from app.services.auth_service import AuthService
from app.services.job_runner import JobRunner


@pytest.fixture(autouse=True)
async def cleanup_redis() -> None:
    yield
    await close_redis()


@pytest.mark.asyncio
async def test_durable_sse_milestone_publishing_and_ordering() -> None:
    """Stage milestones must be written to Redis Streams with monotonic event IDs."""
    runner = JobRunner()
    aid = "sse-order-aid-101"

    id1 = await runner.publish_stage_event(
        aid, StageEventPayload(stage="probe", progress=0.05, message="Probing", elapsed_s=0.1)
    )
    id2 = await runner.publish_stage_event(
        aid, StageEventPayload(stage="frames", progress=0.15, message="Frames", elapsed_s=0.3)
    )
    id3 = await runner.publish_stage_event(
        aid, StageEventPayload(stage="faces", progress=0.30, message="Faces", elapsed_s=0.8)
    )

    assert id1 and id2 and id3
    assert id1 < id2 < id3, "Redis stream IDs must be strictly monotonically increasing"


@pytest.mark.asyncio
async def test_last_event_id_reconnection_replay() -> None:
    """Reconnecting with Last-Event-ID must replay only missed events."""
    runner = JobRunner()
    aid = "sse-replay-aid-202"

    id1 = await runner.publish_stage_event(
        aid, StageEventPayload(stage="probe", progress=0.05, message="Probing", elapsed_s=0.1)
    )
    await runner.publish_stage_event(
        aid, StageEventPayload(stage="frames", progress=0.15, message="Frames", elapsed_s=0.3)
    )
    await runner.publish_stage_event(
        aid, StageEventPayload(stage="done", progress=1.0, message="Done", elapsed_s=1.2)
    )

    # Reconnection starting from id1 should replay id2 and id3
    replayed_stages: list[str] = []
    async for _mid, payload in runner.stream_events(aid, last_event_id=id1, poll_timeout_s=2.0):
        replayed_stages.append(payload.stage)

    assert replayed_stages == [
        "frames",
        "done",
    ], "Replay must return only events subsequent to Last-Event-ID"


def test_sse_endpoint_terminal_status_delivery(
    client: TestClient,
    db_session: Session,
) -> None:
    """Opening SSE endpoint on already completed analysis immediately returns terminal payload."""
    user = AuthService.register_user(
        db=db_session,
        email="sse_tester@parallax.io",
        password="SecurePassword2026!Long",
        role="operator",
    )
    analysis = Analysis(
        id="terminal-sse-analysis-uuid",
        user_id=user.id,
        original_name="test.mp4",
        sha256="0" * 64,
        size_bytes=1000,
        duration_s=3.5,
        status="completed",
        stage="done",
        verdict="REAL",
        audio_present=True,
        model_version="1.0.0",
        model_mode="trained",
    )
    db_session.add(analysis)
    db_session.commit()

    token = create_access_token({"sub": str(user.id), "role": user.role})
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.get(f"/api/v1/analyses/{analysis.id}/events", headers=headers)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert "done" in resp.text
    assert "Analysis complete" in resp.text
