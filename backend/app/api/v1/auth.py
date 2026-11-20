from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.deps import get_current_admin, get_current_user, get_db
from app.core.errors import UnauthorizedException
from app.core.rate_limit import get_login_rate_limit_key, limiter
from app.core.security import create_access_token
from app.db.models import User
from app.schemas.auth import (
    ChangePasswordRequest,
    MessageResponse,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
    UserRoleUpdate,
)
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])

AUTH_LIMIT = "120/minute" if not settings.is_production else "10/minute"
REFRESH_LIMIT = "300/minute" if not settings.is_production else "30/minute"


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user account",
)
@limiter.limit(AUTH_LIMIT)
def register(
    request: Request,
    payload: UserRegister,
    db: Session = Depends(get_db),
) -> UserResponse:
    user = AuthService.register_user(db, payload.email, payload.password, role="public")
    return UserResponse.model_validate(user)


@router.patch(
    "/users/{user_id}/role",
    response_model=UserResponse,
    summary="Update user role (Administrator only)",
)
def update_user_role(
    user_id: int,
    payload: UserRoleUpdate,
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> UserResponse:
    """Allow authorized administrators to promote or demote user roles."""
    user = AuthService.update_user_role(db=db, user_id=user_id, role=payload.role)
    return UserResponse.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate and receive access token + rotating refresh cookie",
)
@limiter.limit(AUTH_LIMIT, key_func=get_login_rate_limit_key)
def login(
    request: Request,
    response: Response,
    payload: UserLogin,
    db: Session = Depends(get_db),
) -> TokenResponse:
    user = AuthService.authenticate_user(db, payload.email, payload.password)
    access_token = create_access_token({"sub": str(user.id), "email": user.email})
    refresh_token = AuthService.issue_refresh_token(db, user.id)
    AuthService.set_cookie(response, refresh_token)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate refresh token cookie and receive fresh access token",
)
@limiter.limit(REFRESH_LIMIT)
def refresh(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> TokenResponse:
    AuthService.validate_csrf_origin(request)
    raw_token = request.cookies.get("refresh_token")
    if not raw_token:
        raise UnauthorizedException("No refresh session cookie found")

    new_token, user = AuthService.rotate_refresh_token(db, raw_token)
    AuthService.set_cookie(response, new_token)
    access_token = create_access_token({"sub": str(user.id), "email": user.email})

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Revoke current session and clear cookie",
)
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> MessageResponse:
    AuthService.validate_csrf_origin(request)
    raw_token = request.cookies.get("refresh_token")
    if raw_token:
        AuthService.revoke_refresh_token(db, raw_token)
    AuthService.clear_cookie(response)
    return MessageResponse(message="Session terminated successfully")


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Retrieve authenticated user profile",
)
def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.model_validate(current_user)


@router.post(
    "/change-password",
    response_model=MessageResponse,
    summary="Update account password and revoke existing sessions",
)
def change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageResponse:
    AuthService.change_password(
        db=db,
        user=current_user,
        old_pass=payload.current_password,
        new_pass=payload.new_password,
    )
    return MessageResponse(message="Password successfully updated; previous sessions revoked")


@router.post(
    "/logout-all",
    response_model=MessageResponse,
    summary="Sign out everywhere by revoking all refresh tokens for this user",
)
@router.post(
    "/revoke-all",
    response_model=MessageResponse,
    summary="Sign out everywhere by revoking all refresh tokens for this user",
)
def revoke_all(
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageResponse:
    AuthService.revoke_all_tokens_for_user(db=db, user_id=current_user.id)
    AuthService.clear_cookie(response)
    return MessageResponse(message="All active sessions revoked")


@router.delete(
    "/account",
    response_model=MessageResponse,
    summary="Permanently delete user account and associated analyses",
)
def delete_account(
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageResponse:
    AuthService.delete_account(db=db, user=current_user)
    AuthService.clear_cookie(response)
    return MessageResponse(message="Account and forensic records permanently purged")
