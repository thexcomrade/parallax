from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)


def get_login_rate_limit_key(request: Request) -> str:
    """Rate limit by client IP and email parameter if present."""
    ip = get_remote_address(request)
    return f"login:{ip}"
