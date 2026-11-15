from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    status: str = "ready"
    db_connected: bool
    model_loaded: bool
    model_mode: str  # "trained" or "untrained"


class ModelCardResponse(BaseModel):
    model_name: str = "AVT2-DWF"
    model_version: str = "1.0.0"
    model_mode: str  # "trained" or "untrained"
    checkpoint_hash: str | None = None
    reported_metrics: dict[str, Any]
    live_metrics: dict[str, Any] | None = None
    disclaimer: str = "Probability reflects model confidence on this clip, not forensic proof."
