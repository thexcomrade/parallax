import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from app.core.config import settings

# Argon2id hasher with calibrated production parameters
_ph = PasswordHasher(
    time_cost=2,
    memory_cost=102400,  # 100 MiB
    parallelism=8,
    hash_len=32,
    salt_len=16,
)

# Pre-computed dummy hash to equalize timing on non-existent accounts
DUMMY_PASSWORD_HASH: str = _ph.hash("dummy-non-existent-user-password-salt")


def hash_password(password: str) -> str:
    """Hash password using Argon2id."""
    return _ph.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against Argon2id hash with constant-time protection."""
    try:
        return _ph.verify(hashed_password, plain_password)
    except (VerifyMismatchError, Exception):
        return False


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create short-lived JWT access token."""
    to_encode = data.copy()
    now = datetime.now(UTC)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TTL_MIN)
    to_encode.update({"iat": int(now.timestamp()), "exp": int(expire.timestamp())})
    encoded_jwt: str = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate JWT access token."""
    try:
        payload: dict[str, Any] = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError as e:
        raise ValueError(f"Invalid access token: {e}") from e


def generate_refresh_token() -> str:
    """Generate high-entropy cryptographically secure refresh token string."""
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    """Hash refresh token with SHA-256 for persistent database storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
