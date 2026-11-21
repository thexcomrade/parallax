import os

from fastapi import Request
from slowapi import Limiter

from app.core.config import settings

_testing = os.getenv("TESTING") == "1" or os.getenv("APP_ENV") == "testing"

storage_uri = "memory://"
if settings.REDIS_URL and not settings.REDIS_URL.startswith("fakeredis"):
    storage_uri = settings.REDIS_URL

def get_client_ip(request: Request) -> str:
    """Extract client IP, honoring X-Forwarded-For if behind a reverse proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


limiter = Limiter(
    key_func=get_client_ip,
    storage_uri=storage_uri,
    enabled=not _testing,
)


def get_login_rate_limit_key(request: Request) -> str:
    """Rate limit by client IP and email parameter if present."""
    ip = get_client_ip(request)
    return f"login:{ip}"
