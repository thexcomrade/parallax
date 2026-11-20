"""Tests for API and Worker process role separation (Task 1.C)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.core.redis import close_redis
from app.services.job_runner import JobRunner


@pytest.mark.asyncio
async def test_api_role_does_not_launch_worker_consumer_loop() -> None:
    """When PROCESS_ROLE='api', JobRunner.start() must NOT spawn _worker_task."""
    await close_redis()
    runner = JobRunner()

    with patch.object(settings, "PROCESS_ROLE", "api"):
        await runner.start()
        assert runner._worker_task is None, "API process role must NOT launch worker consumer task"
        assert runner._outbox_task is not None, "API process role must still run outbox sweeper"
        await runner.stop()
    await close_redis()


@pytest.mark.asyncio
async def test_api_role_enqueue_does_not_execute_locally() -> None:
    """When PROCESS_ROLE='api', enqueue must NOT spawn in-process job execution."""
    await close_redis()
    runner = JobRunner()

    with (
        patch.object(settings, "PROCESS_ROLE", "api"),
        patch.object(runner, "_run_job_guarded") as mock_run,
    ):
        await runner.enqueue(
            analysis_id="api-role-aid-1",
            video_path=Path("dummy.mp4"),
            artifact_dir=Path("artifacts/api-role-aid-1"),
        )
        mock_run.assert_not_called()
    await close_redis()


@pytest.mark.asyncio
async def test_worker_role_launches_consumer_loop() -> None:
    """When PROCESS_ROLE='worker', JobRunner.start() launches _worker_task."""
    await close_redis()
    runner = JobRunner()

    with patch.object(settings, "PROCESS_ROLE", "worker"):
        await runner.start()
        assert runner._worker_task is not None, "Worker process role must launch consumer task"
        await runner.stop()
    await close_redis()
