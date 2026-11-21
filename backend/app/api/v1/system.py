import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import settings
from app.db.session import SessionLocal
from app.schemas.system import HealthResponse, ModelCardResponse, ReadyResponse

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
def health() -> HealthResponse:
    """Fast healthcheck for container orchestration and uptime monitors."""
    return HealthResponse(status="ok")


_last_ready_time: float = 0.0
_last_ready_response: ReadyResponse | None = None
_ready_lock = threading.Lock()


@router.get("/ready", response_model=ReadyResponse, summary="Readiness check")
def ready() -> ReadyResponse:
    """Verifies database connectivity and ML model readiness.

    Applies double-checked locking with a 1.0s TTL cache under high concurrency
    to avoid DB connection pool exhaustion (thundering herd protection).
    """
    global _last_ready_time, _last_ready_response

    now = time.monotonic()
    is_cached = (now - _last_ready_time < 1.0) and (_last_ready_response is not None)
    if os.environ.get("TESTING") != "1" and is_cached and _last_ready_response is not None:
        return _last_ready_response

    with _ready_lock:
        now = time.monotonic()
        is_cached = (now - _last_ready_time < 1.0) and (_last_ready_response is not None)
        if os.environ.get("TESTING") != "1" and is_cached and _last_ready_response is not None:
            return _last_ready_response

        db_ok = False
        try:
            with SessionLocal() as db:
                db.execute(text("SELECT 1"))
                db_ok = True
        except Exception:
            db_ok = False

        checkpoint_exists = Path(settings.MODEL_PATH).exists()
        mode = "trained" if checkpoint_exists else "untrained"

        resp = ReadyResponse(
            status="ready" if db_ok else "degraded",
            db_connected=db_ok,
            model_loaded=checkpoint_exists,
            model_mode=mode,
        )
        _last_ready_time = now
        _last_ready_response = resp
        return resp


@router.get("/model", response_model=ModelCardResponse, summary="Model Card metadata")
def model_card() -> ModelCardResponse:
    """Serve model card metrics from disk; report untrained state when no checkpoint found."""
    weights_dir = Path(settings.MODEL_PATH).parent
    card_file = weights_dir / "model_card.json"
    checkpoint_exists = Path(settings.MODEL_PATH).exists()

    live_metrics: dict[str, Any] | None = None
    checkpoint_hash: str | None = None

    if card_file.exists():
        try:
            with open(card_file, encoding="utf-8") as f:
                card_data = json.load(f)
                live_metrics = card_data.get("live_metrics")
                checkpoint_hash = card_data.get("checkpoint_hash")
        except Exception:
            live_metrics = None

    # Baseline numbers from project report as mandated in README §1
    reported_metrics = {
        "accuracy_controlled": 0.91,
        "accuracy_real_world": 0.78,
        "auc_roc": 0.92,
        "source": "Reported in project report for AVT2-DWF",
    }

    return ModelCardResponse(
        model_name="AVT2-DWF",
        model_version="1.0.0",
        model_mode="trained" if checkpoint_exists else "untrained",
        checkpoint_hash=checkpoint_hash,
        reported_metrics=reported_metrics,
        live_metrics=live_metrics,
    )
