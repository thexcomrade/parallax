import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings

_testing = os.getenv("TESTING") == "1" or os.getenv("APP_ENV") == "testing"

storage_uri = "memory://"
if settings.REDIS_URL and not settings.REDIS_URL.startswith("fakeredis"):
    storage_uri = settings.REDIS_URL

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=storage_uri,
    enabled=not _testing,
)


def get_login_rate_limit_key(request: Request) -> str:
    """Rate limit by client IP and email parameter if present."""
    ip = get_remote_address(request)
    return f"login:{ip}"
