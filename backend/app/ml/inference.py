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

import torch  # pyright: ignore[reportMissingImports]
import torch.nn.functional as F  # pyright: ignore[reportMissingImports]
from torch import Tensor  # pyright: ignore[reportMissingImports]

from app.core.config import settings
from app.ml.device import get_inference_device
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


def load_model(
    checkpoint_path: Path | str | None = None,
    device: torch.device | None = None,
) -> None:
    """Load (or initialise random) AVT2DWF model. Called once at startup."""
    global _model, MODEL_MODE, CHECKPOINT_HASH

    target_device = device if device is not None else get_inference_device()

    raw_path = Path(checkpoint_path) if checkpoint_path else Path(settings.MODEL_PATH)
    path_to_use = raw_path
    if not path_to_use.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        backend_dir = Path(__file__).resolve().parent.parent.parent
        if (backend_dir / path_to_use).exists():
            path_to_use = backend_dir / path_to_use
        elif (repo_root / path_to_use).exists():
            path_to_use = repo_root / path_to_use

    if path_to_use.exists() and path_to_use.stat().st_size > 0:
        try:
            state = torch.load(
                path_to_use,
                map_location="cpu",
                weights_only=True,
            )
            sd = (
                state["model_state_dict"]
                if isinstance(state, dict) and "model_state_dict" in state
                else state
            )
            has_audio_norm = any(
                k.startswith(("audio_encoder.proj_norm", "audio_encoder.norm"))
                for k in sd
            )
            has_input_norm = any(k.startswith("audio_encoder.input_norm") for k in sd)
            has_aux = any(
                k.startswith(("audio_classifier.", "video_classifier."))
                for k in sd
            )

            model = AVT2DWF(
                pretrained_vit=False,
                audio_layernorm=has_audio_norm,
                audio_input_norm=has_input_norm,
                auxiliary_heads=has_aux,
            )
            model.load_state_dict(sd, strict=False)
            model.to(target_device)
            model.eval()

            MODEL_MODE = "trained"
            sha = hashlib.sha256(path_to_use.read_bytes()).hexdigest()
            CHECKPOINT_HASH = sha[:16]
            logger.info(
                "Loaded checkpoint %s on %s (sha256:%s…)",
                path_to_use,
                target_device,
                CHECKPOINT_HASH,
            )
            _model = model
            return
        except Exception:
            logger.exception("Failed to load checkpoint %s; running untrained.", path_to_use)
            MODEL_MODE = "untrained"
            CHECKPOINT_HASH = None

    model = AVT2DWF(pretrained_vit=False)
    model.to(target_device)
    model.eval()
    MODEL_MODE = "untrained"
    CHECKPOINT_HASH = None
    logger.warning(
        "No checkpoint found at %s — MODEL_MODE=untrained on %s. Scores are not meaningful.",
        path_to_use,
        target_device,
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
    device: torch.device | None = None,
) -> InferenceResult:
    """Run AVT²-DWF inference on preprocessed tensors.

    Args:
        video_windows:  Tensor[B, K, 16, 3, 224, 224]  — from align.py
        audio_segments: Tensor[B, K, 104, 100]
        audio_mask:     Tensor[K]
        model_version:  str from model_card.json
        device:         Optional target torch.device override

    Returns:
        InferenceResult — typed result with verdict, weights and scores.
    """
    model = get_model()
    target_device = device if device is not None else next(model.parameters()).device

    # Ensure all input tensors are on target_device with matching float32 dtypes
    v_win = video_windows.to(device=target_device, dtype=torch.float32)
    a_seg = audio_segments.to(device=target_device, dtype=torch.float32)
    a_msk = audio_mask.to(device=target_device, dtype=torch.float32)

    with torch.inference_mode():
        try:
            result: AVT2Result = model(v_win, a_seg, a_msk)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                "CUDA OutOfMemoryError during AVT2-DWF inference on %s: %s",
                target_device,
                e,
            )
            raise

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
