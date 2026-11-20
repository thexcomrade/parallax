from datetime import UTC, datetime, timedelta

from fastapi import Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import (
    AppException,
    InvalidCredentialsException,
    NotFoundException,
    UnauthorizedException,
)
from app.core.metrics import AUTH_FAILURES_TOTAL
from app.core.security import (
    DUMMY_PASSWORD_HASH,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from app.db.models import RefreshToken, User


class AuthService:
    @staticmethod
    def register_user(db: Session, email: str, password: str, role: str = "public") -> User:
        """Register a new user with normalized email, role and Argon2 hash."""
        normalized_email = email.strip().lower()

        existing = db.execute(
            select(User).where(User.email == normalized_email)
        ).scalar_one_or_none()
        if existing:
            raise AppException(
                code="EMAIL_TAKEN",
                message="An account with this email address already exists",
                status_code=400,
            )

        hashed = hash_password(password)
        user = User(email=normalized_email, password_hash=hashed, role=role)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def update_user_role(db: Session, user_id: int, role: str) -> User:
        """Update a user's role (admin only)."""
        if role not in ("public", "operator", "admin"):
            raise AppException(
                code="INVALID_ROLE",
                message="Role must be one of: public, operator, admin",
                status_code=400,
            )
        user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
        if not user:
            raise NotFoundException(f"User with id {user_id} not found")
        user.role = role
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def authenticate_user(db: Session, email: str, password: str) -> User:
        """Authenticate with constant-time dummy verify to prevent account enumeration."""
        normalized_email = email.strip().lower()
        user = db.execute(select(User).where(User.email == normalized_email)).scalar_one_or_none()

        if not user:
            # Equalize timing by executing an Argon2 verification against pre-computed dummy hash
            verify_password("dummy-password", DUMMY_PASSWORD_HASH)
            AUTH_FAILURES_TOTAL.labels(reason="invalid_credentials").inc()
            raise InvalidCredentialsException()

        if not verify_password(password, user.password_hash):
            AUTH_FAILURES_TOTAL.labels(reason="invalid_credentials").inc()
            raise InvalidCredentialsException()

        return user

    @staticmethod
    def issue_refresh_token(db: Session, user_id: int) -> str:
        """Issue high-entropy refresh token and store SHA-256 hash."""
        raw_token = generate_refresh_token()
        token_hash = hash_refresh_token(raw_token)
        expires_at = datetime.now(UTC) + timedelta(days=settings.REFRESH_TTL_DAYS)

        record = RefreshToken(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        db.add(record)
        db.commit()
        return raw_token

    @staticmethod
    def rotate_refresh_token(db: Session, raw_token: str) -> tuple[str, User]:
        """Rotate refresh token. Automatically detects token reuse and revokes entire chain."""
        token_hash = hash_refresh_token(raw_token)
        now = datetime.now(UTC)

        token_record = db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        ).scalar_one_or_none()

        if not token_record:
            raise UnauthorizedException("Invalid or non-existent refresh token")

        # REUSE DETECTION: If this token was already revoked, revoke the ENTIRE chain!
        if token_record.revoked_at is not None:
            # Compromise detected: someone is reusing an already-rotated token!
            all_user_tokens = (
                db.execute(
                    select(RefreshToken).where(
                        RefreshToken.user_id == token_record.user_id,
                        RefreshToken.revoked_at.is_(None),
                    )
                )
                .scalars()
                .all()
            )
            for t in all_user_tokens:
                t.revoked_at = now
            db.commit()
            raise UnauthorizedException(
                "Session compromise detected; token reuse prevented. Sign in again."
            )

        # Expiration check
        # Normalize naive datetime from sqlite if needed
        exp = token_record.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=UTC)
        if exp < now:
            token_record.revoked_at = now
            db.commit()
            raise UnauthorizedException("Session expired. Please log in again.")

        # Generate new token
        new_raw_token = generate_refresh_token()
        new_token_hash = hash_refresh_token(new_raw_token)
        new_expires_at = now + timedelta(days=settings.REFRESH_TTL_DAYS)

        # Mark current token replaced
        token_record.revoked_at = now
        token_record.replaced_by = new_token_hash

        # Add replacement
        new_record = RefreshToken(
            user_id=token_record.user_id,
            token_hash=new_token_hash,
            expires_at=new_expires_at,
        )
        db.add(new_record)
        db.commit()

        user = token_record.user
        return new_raw_token, user

    @staticmethod
    def revoke_refresh_token(db: Session, raw_token: str) -> None:
        """Revoke a specific refresh token upon logout."""
        token_hash = hash_refresh_token(raw_token)
        token_record = db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        ).scalar_one_or_none()

        if token_record and token_record.revoked_at is None:
            token_record.revoked_at = datetime.now(UTC)
            db.commit()

    @staticmethod
    def revoke_all_tokens_for_user(db: Session, user_id: int) -> None:
        """Revoke all active sessions for a user (e.g. sign out everywhere)."""
        now = datetime.now(UTC)
        active_tokens = (
            db.execute(
                select(RefreshToken).where(
                    RefreshToken.user_id == user_id,
                    RefreshToken.revoked_at.is_(None),
                )
            )
            .scalars()
            .all()
        )
        for t in active_tokens:
            t.revoked_at = now
        db.commit()

    @staticmethod
    def change_password(db: Session, user: User, old_pass: str, new_pass: str) -> None:
        """Verify old password and set Argon2 hashed new password."""
        if not verify_password(old_pass, user.password_hash):
            raise UnauthorizedException("Current password verification failed")
        if len(new_pass) < 12:
            raise AppException(
                "WEAK_PASSWORD", "Password must be at least 12 characters", status_code=400
            )
        user.password_hash = hash_password(new_pass)
        db.commit()
        AuthService.revoke_all_tokens_for_user(db, user.id)

    @staticmethod
    def delete_account(db: Session, user: User) -> None:
        """Permanently delete user account and cascade delete all owned records."""
        db.delete(user)
        db.commit()

    @staticmethod
    def validate_csrf_origin(request: "Request") -> None:
        """Validate Origin/Referer header on cookie-authenticated requests to prevent CSRF."""
        origin = request.headers.get("origin")
        referer = request.headers.get("referer")

        # If neither Origin nor Referer is present on browser mutation, reject in prod
        target = origin or referer
        if not target:
            # Native curl / backend test clients without origin are permitted in non-production
            if settings.is_production:
                raise AppException(
                    code="CSRF_FAILED",
                    message="Missing Origin or Referer header on cookie mutation",
                    status_code=403,
                )
            return

        from urllib.parse import urlparse

        parsed = urlparse(target)
        origin_host = parsed.netloc.split(":")[0]
        request_host = request.base_url.hostname or "localhost"

        allowed_hosts = {request_host, "localhost", "127.0.0.1"}
        if settings.is_production:
            allowed_hosts = {request_host}

        if origin_host not in allowed_hosts:
            raise AppException(
                code="CSRF_REJECTED",
                message=f"Untrusted origin '{origin_host}' rejected by CSRF guard",
                status_code=403,
            )

    @staticmethod
    def set_cookie(response: Response, raw_token: str) -> None:
        """Set rotating refresh token in HttpOnly SameSite=Strict cookie."""
        response.set_cookie(
            key="refresh_token",
            value=raw_token,
            httponly=True,
            secure=settings.COOKIE_SECURE or settings.is_production,
            samesite="strict",
            max_age=settings.REFRESH_TTL_DAYS * 86400,
            path="/api/v1/auth",
        )

    @staticmethod
    def clear_cookie(response: Response) -> None:
        """Clear refresh token cookie."""
        response.delete_cookie(
            key="refresh_token",
            path="/api/v1/auth",
            httponly=True,
            secure=settings.COOKIE_SECURE or settings.is_production,
            samesite="strict",
        )
