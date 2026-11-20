"""Redis client and connection pool management for distributed operations.

Supports:
- Environment-configured REDIS_URL with connection pooling.
- Graceful reconnection and error handling.
- Transparent in-memory FakeRedis fallback for testing and local dev when no Redis is running.
- Graceful startup and shutdown lifecycle.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import redis.asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger("parallax.redis")

_redis_client: aioredis.Redis | None = None
_fake_redis_instance: Any = None


def get_redis_url() -> str | None:
    """Return configured REDIS_URL or None."""
    if settings.REDIS_URL and settings.REDIS_URL.strip():
        return settings.REDIS_URL.strip()
    return None


def is_redis_mocked() -> bool:
    """Return True if active client is using in-memory fake, False if live Redis."""
    global _redis_client, _fake_redis_instance
    return _redis_client is not None and _redis_client is _fake_redis_instance


async def check_redis_health() -> bool:
    """Liveness probe: verifies Redis client connection is responsive."""
    try:
        client = await get_redis_client()
        return bool(await client.ping())
    except Exception as e:
        logger.error("Redis health check failed: %s", e)
        return False


async def get_redis_client() -> aioredis.Redis:
    """Return an active async Redis client with connection pooling.

    Safety Rules:
    - Production: Must fail immediately if REDIS_URL is missing or unreachable.
      Never silently fall back to in-memory FakeRedis in production.
    - Development: If REDIS_URL is explicitly set and unreachable, raises ConnectionError.
    - Testing / Local fallback: Only allowed when REDIS_URL is omitted or in test mode.
    """
    global _redis_client, _fake_redis_instance

    if _redis_client is not None:
        return _redis_client

    url = get_redis_url()

    # Rule 1: Production and Staging require live Redis
    if settings.is_production or settings.APP_ENV == "staging":
        if not url:
            raise RuntimeError(f"REDIS_URL is required in {settings.APP_ENV} environment.")
        try:
            pool = aioredis.ConnectionPool.from_url(
                url,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                decode_responses=True,
            )
            client = aioredis.Redis(connection_pool=pool)
            await client.ping()
            _redis_client = client
            logger.info("Connected to %s Redis at %s", settings.APP_ENV, url)
            return _redis_client
        except Exception as e:
            logger.critical("Failed to connect to %s Redis at %s: %s", settings.APP_ENV, url, e)
            raise ConnectionError(
                f"{settings.APP_ENV.capitalize()} Redis connection failed at {url}: {e}"
            ) from e

    # Rule 2: Live Redis connection in non-production when URL provided
    if url and not url.startswith("fakeredis"):
        try:
            pool = aioredis.ConnectionPool.from_url(
                url,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                decode_responses=True,
            )
            client = aioredis.Redis(connection_pool=pool)
            await client.ping()
            _redis_client = client
            logger.info("Connected to Redis at %s", url)
            return _redis_client
        except Exception as e:
            if settings.APP_ENV != "test":
                logger.error("Configured REDIS_URL %s is unreachable: %s", url, e)
                raise ConnectionError(
                    f"Configured Redis at {url} unreachable: {e}"
                ) from e
            logger.warning(
                "Test mode: live Redis %s failed (%s); falling back to fakeredis.",
                url,
                e,
            )

    # Rule 3: In-memory FakeRedis allowed ONLY in test or unconfigured dev mode
    if _fake_redis_instance is None:
        import fakeredis.aioredis

        _fake_redis_instance = fakeredis.aioredis.FakeRedis(decode_responses=True)
        logger.info("Initialized in-memory FakeRedis instance for test/dev environment")

    _redis_client = _fake_redis_instance
    return _redis_client


async def close_redis() -> None:
    """Close active Redis connections on application shutdown."""
    global _redis_client, _fake_redis_instance
    if _redis_client is not None:
        with contextlib.suppress(Exception):
            await _redis_client.aclose()
        _redis_client = None
    _fake_redis_instance = None
