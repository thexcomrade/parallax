"""Explainability: attention rollout heatmaps, spectrogram overlay, timeline JSON.

Writes artefacts to storage/artifacts/<analysis_id>/  (created if absent).
No live network calls; all inputs are tensors from the inference pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

TOP_K_FRAMES: int = 5  # most suspicious frames to annotate
HEATMAP_ALPHA: float = 0.55  # overlay blend factor


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelinePoint:
    t: float  # midpoint timestamp (seconds)
    p_face: float  # fake probability from video-only windows
    p_audio: float  # fake probability from audio-only windows
    p_fused: float  # fused fake probability


@dataclass(frozen=True)
class ExplainResult:
    heatmap_paths: list[str]  # relative artifact names, e.g. "heatmap_frame_02.png"
    spectrogram_path: str  # relative artifact name
    timeline: list[TimelinePoint]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _attention_rollout(
    attentions: list[Tensor],
) -> Tensor:
    """Compute attention rollout map from a list of per-layer attention tensors.

    Args:
        attentions: list of Tensor[num_heads, N, N]  — one per layer

    Returns:
        rollout: Tensor[N, N]  — fused attention map
    """
    result = torch.eye(attentions[0].shape[-1], device=attentions[0].device)
    for attn in attentions:
        # Average over heads
        attn_mean = attn.mean(dim=0)  # [N, N]
        # Add identity (residual connection) and renormalise
        attn_res = attn_mean + torch.eye(attn_mean.shape[-1], device=attn_mean.device)
        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        result = torch.matmul(attn_res, result)
    return result


def _get_vit_attentions(
    model_vit: torch.nn.Module,
    frame_tensor: Tensor,
) -> list[Tensor]:
    """Extract per-layer attention matrices from ViT-B/16 for a single frame.

    Uses forward hooks on the transformer blocks' attention modules.
    """
    captured: list[Tensor] = []
    blocks = getattr(model_vit, "blocks", None)
    if not isinstance(blocks, (torch.nn.ModuleList, list, tuple)):
        return captured

    hooks = []
    for block in blocks:
        if isinstance(block, torch.nn.Module):
            attn = getattr(block, "attn", None)
            if isinstance(attn, torch.nn.Module):
                h = attn.register_forward_hook(lambda m, i, o: captured.append(_extract_attn(m, i)))
                hooks.append(h)

    with torch.no_grad():
        model_vit(frame_tensor)

    for h in hooks:
        h.remove()

    return captured


def _extract_attn(module: torch.nn.Module, inp: tuple[Tensor, ...]) -> Tensor:
    """Compute attention weights from a timm Attention module's forward input."""
    x_in = inp[0]  # [1, N, D]
    B_h, N_h, D_h = x_in.shape
    qkv_fn = getattr(module, "qkv", None)
    num_heads_val = getattr(module, "num_heads", None)
    if not callable(qkv_fn) or not isinstance(num_heads_val, int) or num_heads_val <= 0:
        return torch.eye(N_h)

    qkv_out = qkv_fn(x_in)
    if not isinstance(qkv_out, Tensor):
        return torch.eye(N_h)
    qkv: Tensor = qkv_out  # [1, N, 3*D]
    num_heads = num_heads_val
    head_dim = D_h // num_heads
    qkv_r = qkv.reshape(B_h, N_h, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q_h, k_h = qkv_r[0], qkv_r[1]  # [1, heads, N, head_dim]
    scale = head_dim**-0.5
    attn_w = (q_h @ k_h.transpose(-2, -1)) * scale
    attn_w = F.softmax(attn_w, dim=-1)
    res: Tensor = attn_w.squeeze(0).detach().cpu()
    return res  # [heads, N, N]


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------


def _frame_heatmap_png(
    frame_bgr: np.ndarray,
    rollout: Tensor,
    out_path: Path,
) -> None:
    """Overlay attention rollout on a frame and write a PNG."""
    # rollout: [N+1, N+1] (CLS + patches); take CLS row [1:]
    n_patches = rollout.shape[0] - 1
    patch_side = int(n_patches**0.5)  # 14 for ViT-B/16 (196 patches)
    cls_attn = rollout[0, 1:].reshape(patch_side, patch_side).numpy()

    # Normalise
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

    h, w = frame_bgr.shape[:2]
    heat_resized = cv2.resize(cls_attn, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_uint8 = (heat_resized * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_INFERNO)

    overlay = cv2.addWeighted(frame_bgr, 1.0 - HEATMAP_ALPHA, colormap, HEATMAP_ALPHA, 0)
    cv2.imwrite(str(out_path), overlay)


# ---------------------------------------------------------------------------
# Spectrogram overlay PNG
# ---------------------------------------------------------------------------


def _spectrogram_overlay_png(
    log_mel: Tensor,
    per_window_scores: list[float],
    window_timestamps: list[float],
    out_path: Path,
) -> None:
    """Write a spectrogram PNG with per-segment anomaly band overlay.

    log_mel: Tensor[64, T_frames]  — raw log-mel from audio preprocessing
    per_window_scores: anomaly scores per window
    window_timestamps: midpoint timestamps for each window
    """
    if log_mel.numel() == 0:
        cv2.imwrite(str(out_path), np.zeros((64, 100, 3), dtype=np.uint8))
        return

    mel_np = log_mel.numpy()
    # Normalise to [0, 255]
    mel_norm = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-8)
    mel_uint8 = (mel_norm * 255).astype(np.uint8)

    # Convert to BGR image (flip freq axis so low-freq is at bottom)
    mel_flipped = np.flipud(mel_uint8)
    n_mels, n_time = mel_flipped.shape
    # Upscale for visibility: target ~400px height, width proportional
    scale_h = max(1, 400 // n_mels)
    scale_w = max(1, 4)
    img = cv2.resize(
        mel_flipped,
        (n_time * scale_w, n_mels * scale_h),
        interpolation=cv2.INTER_NEAREST,
    )
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw anomaly bands
    if window_timestamps and per_window_scores:
        total_duration = window_timestamps[-1] + 1.0  # rough estimate
        img_w = img_bgr.shape[1]
        for ts, score in zip(window_timestamps, per_window_scores, strict=False):
            if score > 0.5:
                x_start = int((ts / total_duration) * img_w)
                x_end = int(((ts + 1.0) / total_duration) * img_w)
                alpha_band = min(1.0, score)
                # Red alert band overlay
                band = img_bgr[:, x_start:x_end].copy()
                red_overlay = np.zeros_like(band)
                red_overlay[:, :, 2] = 255  # BGR red
                img_bgr[:, x_start:x_end] = cv2.addWeighted(
                    band,
                    1.0 - alpha_band * 0.4,
                    red_overlay,
                    alpha_band * 0.4,
                    0,
                )

    cv2.imwrite(str(out_path), img_bgr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_explanations(
    model_vit: torch.nn.Module,
    raw_crops: list[np.ndarray],
    frame_timestamps: list[float],
    log_mel: Tensor,
    per_window_scores: list[float],
    window_timestamps: list[float],
    audio_per_window_scores: list[float],
    fused_per_window_scores: list[float],
    artifact_dir: Path,
) -> ExplainResult:
    """Generate heatmaps, spectrogram overlay and timeline; write to artifact_dir.

    Args:
        model_vit:               The ViT backbone (timm model) for attention extraction.
        raw_crops:               List of BGR numpy arrays (H, W, 3) — per-frame crops.
        frame_timestamps:        Timestamps for each frame.
        log_mel:                 Tensor[64, T_frames] from audio preprocessing.
        per_window_scores:       Per-window fake probability (video).
        window_timestamps:       Midpoint timestamps for each window.
        audio_per_window_scores: Per-window fake probability (audio proxy).
        fused_per_window_scores: Fused per-window fake probability.
        artifact_dir:            Directory to write PNG/JSON files.

    Returns:
        ExplainResult with artifact names (not full paths).
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # --- Attention heatmaps for top-K most suspicious frames ---
    heatmap_paths: list[str] = []

    if raw_crops and per_window_scores:
        # Map per-window scores to per-frame by nearest window
        n_frames = len(raw_crops)
        n_windows = len(per_window_scores)
        frame_scores = []
        for fi in range(n_frames):
            win_idx = min(int(fi * n_windows / max(n_frames, 1)), n_windows - 1)
            frame_scores.append(per_window_scores[win_idx])

        sorted_frame_idxs = sorted(range(n_frames), key=lambda i: frame_scores[i], reverse=True)
        top_k_idxs = sorted_frame_idxs[:TOP_K_FRAMES]

        for fi in top_k_idxs:
            frame_bgr = raw_crops[fi]
            fname = f"heatmap_frame_{fi:03d}.png"
            out_path = artifact_dir / fname

            try:
                # Preprocess frame tensor for ViT
                from torchvision import transforms  # type: ignore[import-untyped]

                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                )
                # Determine device of model_vit
                try:
                    vit_device = next(model_vit.parameters()).device
                except StopIteration:
                    vit_device = torch.device("cpu")

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_tensor = (
                    transform(frame_rgb).unsqueeze(0).to(device=vit_device)
                )  # [1, 3, 224, 224]

                attentions = _get_vit_attentions(model_vit, frame_tensor)
                if attentions:
                    rollout = _attention_rollout(attentions)
                    _frame_heatmap_png(frame_bgr, rollout, out_path)
                    heatmap_paths.append(fname)
                else:
                    # Fallback: write original frame without overlay
                    cv2.imwrite(str(out_path), frame_bgr)
                    heatmap_paths.append(fname)
            except Exception:
                logger.exception("Heatmap generation failed for frame %d", fi)
                # Write original frame as fallback so artifact always exists
                cv2.imwrite(str(out_path), frame_bgr)
                heatmap_paths.append(fname)

    # --- Spectrogram overlay ---
    spectrogram_name = "spectrogram.png"
    spectrogram_path = artifact_dir / spectrogram_name
    try:
        _spectrogram_overlay_png(log_mel, per_window_scores, window_timestamps, spectrogram_path)
    except Exception:
        logger.exception("Spectrogram overlay generation failed")
        # Write empty placeholder so artifact record is consistent
        cv2.imwrite(str(spectrogram_path), np.zeros((64, 100, 3), dtype=np.uint8))

    # --- Timeline JSON ---
    timeline: list[TimelinePoint] = []
    for idx, ts in enumerate(window_timestamps):
        p_face = per_window_scores[idx] if idx < len(per_window_scores) else 0.0
        p_audio = audio_per_window_scores[idx] if idx < len(audio_per_window_scores) else 0.0
        p_fused = fused_per_window_scores[idx] if idx < len(fused_per_window_scores) else 0.0
        timeline.append(TimelinePoint(t=ts, p_face=p_face, p_audio=p_audio, p_fused=p_fused))

    timeline_name = "timeline.json"
    timeline_path = artifact_dir / timeline_name
    timeline_data = [
        {"t": pt.t, "p_face": pt.p_face, "p_audio": pt.p_audio, "p_fused": pt.p_fused}
        for pt in timeline
    ]
    timeline_path.write_text(json.dumps(timeline_data, indent=2), encoding="utf-8")

    return ExplainResult(
        heatmap_paths=heatmap_paths,
        spectrogram_path=spectrogram_name,
        timeline=timeline,
    )
