import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.schemas.system import HealthResponse, ModelCardResponse, ReadyResponse

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
def health() -> HealthResponse:
    """Fast healthcheck for container orchestration and uptime monitors."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse, summary="Readiness check")
def ready(db: Session = Depends(get_db)) -> ReadyResponse:
    """Verifies database connectivity and ML model readiness."""
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    checkpoint_exists = Path(settings.MODEL_PATH).exists()
    mode = "trained" if checkpoint_exists else "untrained"

    return ReadyResponse(
        status="ready" if db_ok else "degraded",
        db_connected=db_ok,
        model_loaded=checkpoint_exists,
        model_mode=mode,
    )


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
