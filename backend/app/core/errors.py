
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger("errors")


class AppException(Exception):
    """Base domain exception with consistent client codes and HTTP statuses."""

    def __init__(
        self, code: str, message: str, status_code: int = status.HTTP_400_BAD_REQUEST
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class InvalidCredentialsException(AppException):
    def __init__(self, message: str = "Invalid email or password") -> None:
        super().__init__(
            code="INVALID_CREDENTIALS",
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


class UnauthorizedException(AppException):
    def __init__(self, message: str = "Authentication required") -> None:
        super().__init__(
            code="UNAUTHORIZED",
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


class TokenExpiredException(AppException):
    def __init__(self, message: str = "Token has expired") -> None:
        super().__init__(
            code="TOKEN_EXPIRED",
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


class ForbiddenException(AppException):
    def __init__(self, message: str = "Access forbidden") -> None:
        super().__init__(
            code="FORBIDDEN",
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
        )


class NotFoundException(AppException):
    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(
            code="NOT_FOUND",
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
        )


class ConflictException(AppException):
    def __init__(self, message: str = "Resource conflict", code: str = "CONFLICT") -> None:
        super().__init__(
            code=code,
            message=message,
            status_code=status.HTTP_409_CONFLICT,
        )


class RateLimitedException(AppException):
    def __init__(self, message: str = "Too many requests. Please slow down.") -> None:
        super().__init__(
            code="RATE_LIMITED",
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )


async def app_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Format domain exceptions to client envelope."""
    app_exc = exc if isinstance(exc, AppException) else AppException("INTERNAL_ERROR", str(exc))
    return JSONResponse(
        status_code=app_exc.status_code,
        content={"code": app_exc.code, "message": app_exc.message},
    )


async def validation_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Format Pydantic schema validation errors."""
    if isinstance(exc, RequestValidationError):
        first_error = exc.errors()[0] if exc.errors() else {}
        msg = first_error.get("msg", "Invalid request parameter")
        loc = ".".join(str(x) for x in first_error.get("loc", []))
        formatted_msg = f"{loc}: {msg}" if loc else msg
    else:
        formatted_msg = "Invalid request parameter"
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"code": "VALIDATION_ERROR", "message": formatted_msg},
    )


async def generic_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Catch-all unhandled exception handler that never leaks stack traces."""
    logger.error("Unhandled internal server error: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"code": "INTERNAL_ERROR", "message": "An unexpected error occurred"},
    )
