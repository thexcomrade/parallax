import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import inspect, text

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.errors import (
    AppException,
    app_exception_handler,
    generic_exception_handler,
    validation_exception_handler,
)
from app.core.logging import get_logger, set_request_context, setup_logging
from app.core.metrics import (
    HTTP_ACTIVE_REQUESTS,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    PROMETHEUS_CONTENT_TYPE,
    RATELIMIT_REJECTIONS_TOTAL,
    get_latest_metrics,
)
from app.core.rate_limit import limiter
from app.db.session import Base, SessionLocal, engine
from app.services.storage_s3 import get_storage_service

# Initialize centralized structured logging
setup_logging()
logger = get_logger("api")


async def _rate_limit_handler(request: Request, exc: Exception) -> Response:
    """Typed adapter for slowapi's rate-limit handler (no stubs available)."""
    endpoint = request.url.path
    RATELIMIT_REJECTIONS_TOTAL.labels(endpoint=endpoint).inc()
    if isinstance(exc, RateLimitExceeded):
        result: Response = _rate_limit_exceeded_handler(request, exc)
        return result
    return JSONResponse(
        status_code=429, content={"code": "RATE_LIMITED", "message": "Too many requests"}
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown orchestration."""
    # Ensure storage paths exist
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    # Ensure role column exists if running against pre-existing database
    with engine.connect() as conn:
        inspector = inspect(conn)
        if "users" in inspector.get_table_names():
            columns = {col["name"] for col in inspector.get_columns("users")}
            if "role" not in columns:
                conn.execute(
                    text(
                        "ALTER TABLE users ADD COLUMN role VARCHAR(32) NOT NULL DEFAULT 'public'"
                    )
                )
                conn.commit()
        if "analyses" in inspector.get_table_names():
            analysis_columns = {col["name"] for col in inspector.get_columns("analyses")}
            if "video_path" not in analysis_columns:
                conn.execute(
                    text(
                        "ALTER TABLE analyses ADD COLUMN video_path VARCHAR(512)"
                    )
                )
                conn.commit()

    # Start distributed JobRunner consumer loop
    from app.core.redis import close_redis
    from app.services.job_runner import job_runner

    await job_runner.start()

    yield

    # Graceful shutdown
    await job_runner.stop()
    await close_redis()


app = FastAPI(
    title="PARALLAX Deepfake Detection System",
    version="1.0.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    openapi_url="/openapi.json" if not settings.is_production else None,
    lifespan=lifespan,
)

# Connect rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

# Register uniform error handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


# Observability, Request ID & Metrics Middleware
@app.middleware("http")
async def observability_and_metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    # 1. Request ID propagation and correlation
    req_id = request.headers.get("X-Request-ID")
    if not req_id or len(req_id) > 64:
        req_id = uuid.uuid4().hex
    set_request_context(req_id)

    method = request.method
    endpoint = request.url.path
    t0 = time.perf_counter()

    HTTP_ACTIVE_REQUESTS.labels(method=method).inc()
    try:
        response = await call_next(request)
    finally:
        HTTP_ACTIVE_REQUESTS.labels(method=method).dec()

    duration_s = time.perf_counter() - t0
    status_code = response.status_code

    # 2. Record Prometheus request metrics
    # Normalize endpoint path for low-cardinality metric dimensions
    normalized_endpoint = endpoint
    if endpoint.startswith("/api/v1/analyses/") and len(endpoint.split("/")) > 4:
        parts = endpoint.split("/")
        # /api/v1/analyses/{id}/events -> /api/v1/analyses/:id/events
        normalized_endpoint = f"/api/v1/analyses/:id/{parts[4]}"
    elif endpoint.startswith("/api/v1/analyses/"):
        normalized_endpoint = "/api/v1/analyses/:id"

    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=method, endpoint=normalized_endpoint
    ).observe(duration_s)
    HTTP_REQUESTS_TOTAL.labels(
        method=method, endpoint=normalized_endpoint, status=str(status_code)
    ).inc()

    # 3. Security & CDN Cache-Control Headers
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https://api.dicebear.com; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    if settings.is_production or settings.COOKIE_SECURE:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    # Strictly prevent CDN caching of API responses and private analysis payloads
    if endpoint.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    elif endpoint.startswith("/assets/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"

    # Structured request completion log
    logger.info(
        "HTTP %s %s -> %d (%.2fms)",
        method,
        endpoint,
        status_code,
        duration_s * 1000,
        extra={"duration_ms": duration_s * 1000, "status_code": status_code},
    )

    return response


# Configurable CORS (README §8.1, Phase 6)
cors_origins = (
    settings.cors_origins_list
    if settings.cors_origins_list
    else (["http://localhost:5173", "http://127.0.0.1:5173"] if not settings.is_production else [])
)
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ==============================================================================
# Health Checks & Readiness Probes
# ==============================================================================

@app.get("/health", include_in_schema=False)
@app.get("/health/live", summary="Liveness Probe")
@app.get("/health/liveness", summary="Liveness Probe")
async def liveness_probe() -> dict[str, str]:
    """Lightweight process liveness probe for Kubernetes and Docker."""
    return {"status": "alive"}


@app.get("/health/ready", summary="Readiness Probe")
@app.get("/health/readiness", summary="Readiness Probe")
async def readiness_probe() -> JSONResponse:
    """Readiness probe evaluating PostgreSQL, Redis, and storage dependencies.

    Note: API readiness strictly does NOT require GPU model loading.
    """
    checks: dict[str, str] = {
        "database": "unknown",
        "redis": "unknown",
        "storage": "unknown",
    }
    all_ready = True

    # 1. Database check
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
            checks["database"] = "ok"
    except Exception as e:
        logger.warning("Readiness probe database check failed: %s", e)
        checks["database"] = f"unhealthy: {type(e).__name__}"
        all_ready = False

    # 2. Redis check
    try:
        from app.core.redis import check_redis_health

        redis_ok = await check_redis_health()
        checks["redis"] = "ok" if redis_ok else "unreachable"
        if not redis_ok:
            all_ready = False
    except Exception as e:
        logger.warning("Readiness probe Redis check failed: %s", e)
        checks["redis"] = f"unhealthy: {type(e).__name__}"
        all_ready = False

    # 3. Storage check
    try:
        storage = get_storage_service()
        storage_ok = storage.check_health()
        checks["storage"] = "ok" if storage_ok else "unhealthy"
        if not storage_ok:
            all_ready = False
    except Exception as e:
        logger.warning("Readiness probe storage check failed: %s", e)
        checks["storage"] = f"unhealthy: {type(e).__name__}"
        all_ready = False

    status_code = 200 if all_ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_ready else "unready",
            "checks": checks,
        },
    )


# ==============================================================================
# Prometheus Metrics Exposition
# ==============================================================================

@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Expose application and system Prometheus metrics."""
    if not settings.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics collection disabled")

    metrics_payload = await get_latest_metrics()
    return Response(content=metrics_payload, media_type=PROMETHEUS_CONTENT_TYPE)


# Mount primary API prefix
app.include_router(api_router, prefix="/api/v1")


# ==============================================================================
# Frontend Static Asset Delivery (Conditional CDN Fallback)
# ==============================================================================

frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if not frontend_dist.exists():
    for candidate in [
        Path("/app/frontend/dist"),
        Path(__file__).resolve().parent.parent / "dist",
    ]:
        if candidate.exists():
            frontend_dist = candidate
            break

# Only mount SPA static delivery if should_serve_frontend is enabled (e.g. local development)
if settings.should_serve_frontend and frontend_dist.exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> FileResponse:
        cleaned_path = unquote(full_path).lstrip("/\\")
        base_dir = frontend_dist.resolve()

        # Any unhandled path within /api must return 404, never the SPA HTML page
        if cleaned_path == "api" or cleaned_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")

        if cleaned_path:
            target_file = (base_dir / cleaned_path).resolve()

            # Strict path traversal containment guard
            try:
                if not target_file.is_relative_to(base_dir):
                    raise HTTPException(status_code=404, detail="Not Found")
            except (ValueError, TypeError):
                raise HTTPException(status_code=404, detail="Not Found") from None

            if target_file.is_file():
                response = FileResponse(target_file)
                if cleaned_path.startswith("assets/"):
                    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
                return response

            target_name = target_file.name
            if "." in target_name and not target_name.endswith(".html"):
                raise HTTPException(status_code=404, detail="Static asset not found")

        index_file = base_dir / "index.html"
        if index_file.is_file():
            response = FileResponse(index_file)
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
            return response
        raise HTTPException(status_code=404, detail="SPA distribution not found")
