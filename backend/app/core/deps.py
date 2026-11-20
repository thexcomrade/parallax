from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ForbiddenException, NotFoundException, UnauthorizedException
from app.core.metrics import AUTH_FAILURES_TOTAL
from app.core.security import decode_access_token
from app.db.models import Analysis, User
from app.db.session import get_db

__all__ = [
    "get_current_admin",
    "get_current_operator",
    "get_current_user",
    "get_db",
    "get_optional_current_user",
    "get_owned_analysis",
]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Validate Bearer access token and load authenticated user."""
    if not token:
        AUTH_FAILURES_TOTAL.labels(reason="missing_token").inc()
        raise UnauthorizedException("Authentication token required")

    try:
        payload = decode_access_token(token)
    except Exception as e:
        AUTH_FAILURES_TOTAL.labels(reason="invalid_token").inc()
        raise UnauthorizedException("Invalid or expired authentication token") from e

    user_id = payload.get("sub")
    if not user_id:
        raise UnauthorizedException("Malformed authentication token")

    try:
        user_id_int = int(user_id)
    except (ValueError, TypeError) as e:
        raise UnauthorizedException("Malformed user identifier in token") from e

    user = db.execute(select(User).where(User.id == user_id_int)).scalar_one_or_none()
    if not user:
        raise UnauthorizedException("User account no longer exists")

    return user


def get_optional_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User | None:
    """Load authenticated user if Bearer token is provided, otherwise return None."""
    if not token:
        return None

    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return None
        user_id_int = int(user_id)
        return db.execute(select(User).where(User.id == user_id_int)).scalar_one_or_none()
    except Exception:
        return None


def get_current_operator(
    current_user: User = Depends(get_current_user),
) -> User:
    """Ensure authenticated user has operator or admin privileges for internal forensics."""
    if current_user.role not in ("operator", "admin"):
        raise ForbiddenException(
            "Operator privileges required to access internal forensic metrics and artifacts"
        )
    return current_user


def get_current_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """Ensure authenticated user has administrator privileges."""
    if current_user.role != "admin":
        raise ForbiddenException("Administrator privileges required")
    return current_user


def get_owned_analysis(
    id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Analysis:
    """Ensure analysis is owned by current user.

    Returns 404 on foreign ID to prevent cross-user probing.
    """
    analysis = db.execute(
        select(Analysis).where(
            Analysis.id == id,
            Analysis.user_id == user.id,
        )
    ).scalar_one_or_none()

    if not analysis:
        # Standard 404 prevents enumerating valid IDs belonging to other users
        raise NotFoundException("Analysis not found")

    return analysis
