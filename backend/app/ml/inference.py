"""Model inference: load-once, trained/untrained mode, verdict thresholds.

Loaded at application startup via `load_model()`. The module-level `MODEL_MODE`
reflects whether a real checkpoint was found. When no checkpoint exists the model
runs with random weights and MODEL_MODE = "untrained" — the caller must surface
the persistent banner.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from app.core.config import settings
from app.ml.models.avt2_dwf import AVT2DWF, AVT2Result

logger = logging.getLogger(__name__)

Verdict = Literal["FAKE", "AUTHENTIC", "INCONCLUSIVE"]
ModelMode = Literal["trained", "untrained"]

# Module-level singletons — set by load_model()
_model: AVT2DWF | None = None
MODEL_MODE: ModelMode = "untrained"
CHECKPOINT_HASH: str | None = None


@dataclass(frozen=True)
class InferenceResult:
    verdict: Verdict
    p_fake: float  # calibrated probability of FAKE
    p_authentic: float  # calibrated probability of AUTHENTIC
    w_audio: float  # audio fusion weight (0 when audio absent)
    w_video: float  # video fusion weight
    sync_score: float  # lip-sync proxy cosine score
    per_window_scores: list[float]  # per-window fake probability
    model_mode: ModelMode
    model_version: str


def load_model() -> None:
    """Load (or initialise random) AVT2DWF model. Called once at startup."""
    global _model, MODEL_MODE, CHECKPOINT_HASH

    model = AVT2DWF(pretrained_vit=False)  # always False on inference host
    model.eval()

    checkpoint_path = Path(settings.MODEL_PATH)
    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        try:
            state = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state)
            MODEL_MODE = "trained"
            # SHA-256 of checkpoint for provenance
            sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            CHECKPOINT_HASH = sha[:16]
            logger.info("Loaded checkpoint %s (sha256:%s…)", checkpoint_path, CHECKPOINT_HASH)
        except Exception:
            logger.exception("Failed to load checkpoint %s; running untrained.", checkpoint_path)
            MODEL_MODE = "untrained"
            CHECKPOINT_HASH = None
    else:
        MODEL_MODE = "untrained"
        CHECKPOINT_HASH = None
        logger.warning(
            "No checkpoint found at %s — MODEL_MODE=untrained. Scores are not meaningful.",
            checkpoint_path,
        )

    _model = model


def get_model() -> AVT2DWF:
    """Return the loaded model, loading it if necessary."""
    if _model is None:
        load_model()
    assert _model is not None
    return _model


def run_inference(
    video_windows: Tensor,
    audio_segments: Tensor,
    audio_mask: Tensor,
    model_version: str = "0.0.0",
) -> InferenceResult:
    """Run AVT²-DWF inference on preprocessed tensors.

    Args:
        video_windows:  Tensor[B, K, 16, 3, 224, 224]  — from align.py
        audio_segments: Tensor[B, K, 104, 100]
        audio_mask:     Tensor[K]
        model_version:  str from model_card.json

    Returns:
        InferenceResult — typed result with verdict, weights and scores.
    """
    model = get_model()

    with torch.inference_mode():
        result: AVT2Result = model(video_windows, audio_segments, audio_mask)

    # Convert logits → calibrated probabilities
    probs: Tensor = F.softmax(result.logits, dim=-1)  # [B, 2]
    p_authentic_val = float(probs[0, 0].item())
    p_fake_val = float(probs[0, 1].item())

    # Verdict thresholds from config (never hard-coded in UI)
    if p_fake_val >= settings.P_FAKE_HI:
        verdict: Verdict = "FAKE"
    elif p_fake_val <= settings.P_FAKE_LO:
        verdict = "AUTHENTIC"
    else:
        verdict = "INCONCLUSIVE"

    w_a = float(result.w_a[0, 0].item())
    w_v = float(result.w_v[0, 0].item())

    per_window = [float(s.item()) for s in result.per_window_scores[0]]

    return InferenceResult(
        verdict=verdict,
        p_fake=p_fake_val,
        p_authentic=p_authentic_val,
        w_audio=w_a,
        w_video=w_v,
        sync_score=float(result.sync_score.item()),
        per_window_scores=per_window,
        model_mode=MODEL_MODE,
        model_version=model_version,
    )
