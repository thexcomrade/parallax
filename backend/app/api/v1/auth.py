from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, get_db
from app.core.errors import UnauthorizedException
from app.core.rate_limit import limiter
from app.core.security import create_access_token
from app.db.models import User
from app.schemas.auth import MessageResponse, TokenResponse, UserLogin, UserRegister, UserResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user account",
)
@limiter.limit("10/minute")
def register(
    request: Request,
    payload: UserRegister,
    db: Session = Depends(get_db),
) -> UserResponse:
    user = AuthService.register_user(db, payload.email, payload.password)
    return UserResponse.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate and receive access token + rotating refresh cookie",
)
@limiter.limit("10/minute")
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
@limiter.limit("30/minute")
def refresh(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> TokenResponse:
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
