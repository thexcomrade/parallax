"""PARALLAX — Prometheus Metrics & Observability Collector.

Collects and exposes low-overhead Prometheus metrics covering:
- HTTP API request volume, latency, status codes, active requests, auth failures, and rate limits
- PostgreSQL connection pool utilization, query errors, and connectivity status
- Redis queue depth, pending jobs, consumer lag, outbox backlog, and dead-letter size
- GPU worker VRAM allocation, active inference jobs, inference latency, and heartbeats
- Storage upload throughput, duration, and error counts
"""

from __future__ import annotations

import logging
from typing import Any

import prometheus_client
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from sqlalchemy import func, select

logger = logging.getLogger("parallax.metrics")

PROMETHEUS_CONTENT_TYPE: str = CONTENT_TYPE_LATEST

# API Metrics
HTTP_REQUESTS_TOTAL = Counter(
    "parallax_http_requests_total",
    "Total number of HTTP requests processed",
    ["method", "endpoint", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "parallax_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

HTTP_ACTIVE_REQUESTS = Gauge(
    "parallax_http_active_requests",
    "Number of HTTP requests currently being handled",
    ["method"],
)

AUTH_FAILURES_TOTAL = Counter(
    "parallax_auth_failures_total",
    "Total number of failed authentication attempts",
    ["reason"],
)

RATELIMIT_REJECTIONS_TOTAL = Counter(
    "parallax_ratelimit_rejections_total",
    "Total number of requests rejected due to rate limiting",
    ["endpoint"],
)

# Database Metrics (PostgreSQL / SQLite)
DB_CONNECTED = Gauge(
    "parallax_db_connected",
    "Database connectivity status (1 = connected, 0 = unreachable)",
)

DB_POOL_SIZE = Gauge(
    "parallax_db_pool_size",
    "Configured maximum database connection pool size",
)

DB_POOL_CHECKED_OUT = Gauge(
    "parallax_db_pool_checked_out",
    "Number of database connections currently checked out by workers",
)

DB_ERRORS_TOTAL = Counter(
    "parallax_db_errors_total",
    "Total number of database errors encountered",
    ["operation"],
)

# Redis & Queue Metrics
REDIS_CONNECTED = Gauge(
    "parallax_redis_connected",
    "Redis connectivity status (1 = connected, 0 = unreachable)",
)

REDIS_QUEUE_LENGTH = Gauge(
    "parallax_redis_queue_length",
    "Number of unconsumed tasks in Redis Stream",
    ["stream"],
)

REDIS_PENDING_JOBS = Gauge(
    "parallax_redis_pending_jobs",
    "Number of unacknowledged jobs in consumer group PEL",
    ["stream", "group"],
)

REDIS_DEAD_LETTER_SIZE = Gauge(
    "parallax_redis_dead_letter_size",
    "Number of unrecoverable jobs routed to dead-letter stream",
)

OUTBOX_BACKLOG_COUNT = Gauge(
    "parallax_outbox_backlog_count",
    "Number of pending analysis events in Transactional Outbox awaiting dispatch",
)

# GPU & Worker Metrics
GPU_AVAILABLE = Gauge(
    "parallax_gpu_available",
    "GPU hardware availability status (1 = available, 0 = CPU fallback)",
    ["device"],
)

GPU_VRAM_ALLOCATED_BYTES = Gauge(
    "parallax_gpu_vram_allocated_bytes",
    "Current GPU VRAM allocated in bytes",
    ["device"],
)

GPU_VRAM_RESERVED_BYTES = Gauge(
    "parallax_gpu_vram_reserved_bytes",
    "Current GPU VRAM reserved in bytes",
    ["device"],
)

WORKER_ACTIVE_JOBS = Gauge(
    "parallax_worker_active_jobs",
    "Number of inference tasks currently active on worker",
    ["worker_id"],
)

INFERENCE_DURATION_SECONDS = Histogram(
    "parallax_inference_duration_seconds",
    "Duration of end-to-end ML inference execution in seconds",
    ["status"],
    buckets=(0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 60.0),
)

INFERENCE_FAILED_TOTAL = Counter(
    "parallax_inference_failed_total",
    "Total number of failed inference executions",
    ["stage"],
)

WORKER_HEARTBEAT_TIMESTAMP = Gauge(
    "parallax_worker_heartbeat_timestamp",
    "Unix timestamp of the latest worker heartbeat",
    ["worker_id"],
)

# Storage Metrics
STORAGE_UPLOADS_TOTAL = Counter(
    "parallax_storage_uploads_total",
    "Total number of file uploads processed",
    ["backend", "status"],
)

STORAGE_UPLOAD_DURATION_SECONDS = Histogram(
    "parallax_storage_upload_duration_seconds",
    "Duration of storage upload operations in seconds",
    ["backend"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

STORAGE_ERRORS_TOTAL = Counter(
    "parallax_storage_errors_total",
    "Total number of object storage errors encountered",
    ["backend", "operation"],
)


# ==============================================================================
# Dynamic Metrics Collection
# ==============================================================================

async def collect_runtime_metrics() -> None:
    """Sample live runtime gauges from database, Redis, and GPU."""
    # 1. Sample Database pool & outbox backlog
    try:
        from app.db.models import AnalysisOutbox
        from app.db.session import SessionLocal, engine

        # Connectivity & Pool status
        if hasattr(engine.pool, "size"):
            DB_POOL_SIZE.set(engine.pool.size())
        if hasattr(engine.pool, "checkedout"):
            DB_POOL_CHECKED_OUT.set(engine.pool.checkedout())
        DB_CONNECTED.set(1)

        with SessionLocal() as db:
            pending_outbox = db.execute(
                select(func.count(AnalysisOutbox.id)).where(
                    AnalysisOutbox.status == "pending"
                )
            ).scalar() or 0
            OUTBOX_BACKLOG_COUNT.set(pending_outbox)
    except Exception as e:
        logger.debug("Database metrics collection failed: %s", e)
        DB_CONNECTED.set(0)

    # 2. Sample Redis Stream depth & DLQ
    try:
        from app.core.redis import get_redis_client
        from app.services.job_runner import (
            GROUP_WORKERS,
            STREAM_DEAD_LETTER,
            STREAM_TASKS,
        )

        redis = await get_redis_client()
        is_alive = await redis.ping()
        REDIS_CONNECTED.set(1 if is_alive else 0)

        # Main task stream length
        q_len = await redis.xlen(STREAM_TASKS)
        REDIS_QUEUE_LENGTH.labels(stream=STREAM_TASKS).set(q_len)

        # Dead letter stream length
        dlq_len = await redis.xlen(STREAM_DEAD_LETTER)
        REDIS_DEAD_LETTER_SIZE.set(dlq_len)

        # Consumer group pending count
        try:
            pending_info: dict[str, Any] = await redis.xpending(STREAM_TASKS, GROUP_WORKERS)
            pending_count = pending_info.get("pending", 0) if isinstance(pending_info, dict) else 0
            REDIS_PENDING_JOBS.labels(stream=STREAM_TASKS, group=GROUP_WORKERS).set(pending_count)
        except Exception:
            REDIS_PENDING_JOBS.labels(stream=STREAM_TASKS, group=GROUP_WORKERS).set(0)
    except Exception as e:
        logger.debug("Redis metrics collection failed: %s", e)
        REDIS_CONNECTED.set(0)

    # 3. Sample GPU hardware metrics (PyTorch CUDA)
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            GPU_AVAILABLE.labels(device=device_name).set(1)
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            GPU_VRAM_ALLOCATED_BYTES.labels(device=device_name).set(allocated)
            GPU_VRAM_RESERVED_BYTES.labels(device=device_name).set(reserved)
        else:
            GPU_AVAILABLE.labels(device="cpu").set(0)
            GPU_VRAM_ALLOCATED_BYTES.labels(device="cpu").set(0)
            GPU_VRAM_RESERVED_BYTES.labels(device="cpu").set(0)
    except Exception as e:
        logger.debug("GPU metrics collection failed: %s", e)
        GPU_AVAILABLE.labels(device="unknown").set(0)


async def get_latest_metrics() -> bytes:
    """Gather dynamic state and format as Prometheus exposition text."""
    await collect_runtime_metrics()
    return generate_latest(prometheus_client.REGISTRY)
