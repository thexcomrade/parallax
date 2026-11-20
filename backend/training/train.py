"""Offline training script for the AVT²-DWF multimodal deepfake detection model.

Supports full training with AdamW, CosineAnnealingLR, class-balanced sampling,
early stopping on validation AUC, and a --smoke mode for fast verification (README §5, §8.7).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from app.ml.models.avt2_dwf import AVT2DWF, AVT2Result
from training.augment import MultimodalAugmenter
from training.dataset import MultimodalDataset, create_balanced_sampler, multimodal_collate_fn
from training.manifest import build_smoke_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("parallax.training.train")


def safe_torch_save(
    state: Any, target_path: Path, max_retries: int = 5, retry_delay: float = 0.5
) -> None:
    """Safely save PyTorch checkpoint with atomic write and retry for Windows file locks."""
    target_path = Path(target_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(
        f"{target_path.stem}_tmp_{os.getpid()}_{int(time.time() * 1000)}.pt"
    )

    try:
        torch.save(state, tmp_path)
        for attempt in range(max_retries):
            try:
                if target_path.exists():
                    try:
                        os.replace(tmp_path, target_path)
                    except OSError:
                        target_path.unlink(missing_ok=True)
                        shutil.move(str(tmp_path), str(target_path))
                else:
                    shutil.move(str(tmp_path), str(target_path))
                return
            except (PermissionError, OSError, RuntimeError) as e:
                if attempt == max_retries - 1:
                    logger.warning(
                        "Atomic replace failed (%s); trying direct torch.save: %s", e, target_path
                    )
                    torch.save(state, target_path)
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
                    return
                time.sleep(retry_delay)
    finally:
        if tmp_path.exists():
            with contextlib.suppress(Exception):
                tmp_path.unlink(missing_ok=True)


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML config at {config_path}")
        return data


def compute_auc(y_true: list[int], y_scores: list[float]) -> float:
    """Compute Area Under ROC Curve safely with fallback."""
    if len(set(y_true)) < 2:
        return 0.5  # degenerate single-class split
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_scores))
    except Exception:
        # Fallback manual calculation if sklearn unavailable or error
        pos_scores = [s for y, s in zip(y_true, y_scores, strict=False) if y == 1]
        neg_scores = [s for y, s in zip(y_true, y_scores, strict=False) if y == 0]
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        if n_pos == 0 or n_neg == 0:
            return 0.5
        correct = sum(1.0 for p in pos_scores for n in neg_scores if p > n)
        ties = sum(0.5 for p in pos_scores for n in neg_scores if p == n)
        return float((correct + ties) / (n_pos * n_neg))


def train_epoch(
    model: AVT2DWF,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_accum_steps: int = 1,
    aux_audio_weight: float = 0.0,
    aux_video_weight: float = 0.0,
    max_steps: int | None = None,
    modality_dropout_prob: float = 0.0,
) -> dict[str, float]:
    """Train for one epoch, returning loss, metrics, and gradient statistics."""
    model.train()
    total_loss = 0.0
    total_loss_fused = 0.0
    total_loss_audio = 0.0
    total_loss_video = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_scores: list[float] = []

    # Unimodal tracking
    y_audio_true: list[int] = []
    y_audio_pred: list[int] = []
    y_audio_scores: list[float] = []
    y_video_true: list[int] = []
    y_video_pred: list[int] = []
    y_video_scores: list[float] = []

    # Gradient tracking
    total_grad_norm_audio = 0.0
    total_grad_norm_video = 0.0
    total_grad_norm_dwf = 0.0
    opt_steps = 0

    optimizer.zero_grad()
    actual_steps = 0

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        video_windows = batch["video_windows"].to(device)
        audio_segments = batch["audio_segments"].to(device)
        audio_mask = batch["audio_mask"].to(device)
        labels = batch["labels"].to(device)

        if modality_dropout_prob > 0.0 and random.random() < modality_dropout_prob:
            audio_mask = torch.zeros_like(audio_mask)

        out: AVT2Result = model(video_windows, audio_segments, audio_mask)
        loss_fused = criterion(out.logits, labels)

        loss_audio = (
            criterion(out.audio_logits, labels)
            if (out.audio_logits is not None and aux_audio_weight > 0.0)
            else torch.tensor(0.0, device=device)
        )
        loss_video = (
            criterion(out.video_logits, labels)
            if (out.video_logits is not None and aux_video_weight > 0.0)
            else torch.tensor(0.0, device=device)
        )

        raw_loss = loss_fused + aux_audio_weight * loss_audio + aux_video_weight * loss_video

        # Check for NaN / Inf
        if not torch.isfinite(raw_loss):
            raise ValueError(f"Non-finite loss detected at step {step}: {raw_loss.item()}")

        loss = raw_loss / max(1, grad_accum_steps)
        loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            # Gradient norm tracking before clipping
            audio_grads = [
                p.grad.detach() for p in model.audio_encoder.parameters() if p.grad is not None
            ]
            video_grads = [
                p.grad.detach() for p in model.face_encoder.parameters() if p.grad is not None
            ]
            dwf_grads = [
                p.grad.detach() for p in model.dwf.parameters() if p.grad is not None
            ]
            if audio_grads:
                total_grad_norm_audio += float(
                    torch.norm(torch.stack([torch.norm(g, 2) for g in audio_grads]), 2).item()
                )
            if video_grads:
                total_grad_norm_video += float(
                    torch.norm(torch.stack([torch.norm(g, 2) for g in video_grads]), 2).item()
                )
            if dwf_grads:
                total_grad_norm_dwf += float(
                    torch.norm(torch.stack([torch.norm(g, 2) for g in dwf_grads]), 2).item()
                )
            opt_steps += 1

            # Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

        total_loss += float(raw_loss.item())
        total_loss_fused += float(loss_fused.item())
        total_loss_audio += float(loss_audio.item())
        total_loss_video += float(loss_video.item())

        preds = torch.argmax(out.logits, dim=-1)
        probs = torch.softmax(out.logits, dim=-1)[:, 1]
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_scores.extend(probs.detach().cpu().tolist())

        if out.audio_logits is not None:
            a_preds = torch.argmax(out.audio_logits, dim=-1)
            a_probs = torch.softmax(out.audio_logits, dim=-1)[:, 1]
            y_audio_true.extend(labels.detach().cpu().tolist())
            y_audio_pred.extend(a_preds.detach().cpu().tolist())
            y_audio_scores.extend(a_probs.detach().cpu().tolist())

        if out.video_logits is not None:
            v_preds = torch.argmax(out.video_logits, dim=-1)
            v_probs = torch.softmax(out.video_logits, dim=-1)[:, 1]
            y_video_true.extend(labels.detach().cpu().tolist())
            y_video_pred.extend(v_preds.detach().cpu().tolist())
            y_video_scores.extend(v_probs.detach().cpu().tolist())

        actual_steps += 1

    avg_loss = total_loss / max(1, actual_steps)
    avg_loss_fused = total_loss_fused / max(1, actual_steps)
    avg_loss_audio = total_loss_audio / max(1, actual_steps)
    avg_loss_video = total_loss_video / max(1, actual_steps)
    accuracy = correct / max(1, total)

    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

    precision = float(precision_score(y_true, y_pred, zero_division=0.0))
    recall = float(recall_score(y_true, y_pred, zero_division=0.0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0.0))
    auc = compute_auc(y_true, y_scores)
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    res: dict[str, float] = {
        "loss": avg_loss,
        "loss_fused": avg_loss_fused,
        "loss_audio": avg_loss_audio,
        "loss_video": avg_loss_video,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "balanced_acc": balanced_acc,
        "grad_norm_audio": total_grad_norm_audio / max(1, opt_steps),
        "grad_norm_video": total_grad_norm_video / max(1, opt_steps),
        "grad_norm_dwf": total_grad_norm_dwf / max(1, opt_steps),
    }

    if y_audio_true:
        res["audio_acc"] = float(
            sum(p == t for p, t in zip(y_audio_pred, y_audio_true, strict=False))
            / len(y_audio_true)
        )
        res["audio_f1"] = float(f1_score(y_audio_true, y_audio_pred, zero_division=0.0))
        res["audio_auc"] = compute_auc(y_audio_true, y_audio_scores)

    if y_video_true:
        res["video_acc"] = float(
            sum(p == t for p, t in zip(y_video_pred, y_video_true, strict=False))
            / len(y_video_true)
        )
        res["video_f1"] = float(f1_score(y_video_true, y_video_pred, zero_division=0.0))
        res["video_auc"] = compute_auc(y_video_true, y_video_scores)

    return res


def evaluate_epoch(
    model: AVT2DWF,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    aux_audio_weight: float = 0.0,
    aux_video_weight: float = 0.0,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Evaluate on validation set, returning dictionary of fused and unimodal metrics."""
    model.eval()
    total_loss = 0.0
    total_loss_fused = 0.0
    total_loss_audio = 0.0
    total_loss_video = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_scores: list[float] = []
    y_pred: list[int] = []

    # Unimodal tracking
    y_audio_true: list[int] = []
    y_audio_pred: list[int] = []
    y_audio_scores: list[float] = []
    y_video_true: list[int] = []
    y_video_pred: list[int] = []
    y_video_scores: list[float] = []

    # Fusion weights tracking
    all_w_a: list[float] = []
    all_w_v: list[float] = []

    actual_steps = 0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break

            video_windows = batch["video_windows"].to(device)
            audio_segments = batch["audio_segments"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            labels = batch["labels"].to(device)

            out: AVT2Result = model(video_windows, audio_segments, audio_mask)
            loss_fused = criterion(out.logits, labels)

            loss_audio = (
                criterion(out.audio_logits, labels)
                if (out.audio_logits is not None and aux_audio_weight > 0.0)
                else torch.tensor(0.0, device=device)
            )
            loss_video = (
                criterion(out.video_logits, labels)
                if (out.video_logits is not None and aux_video_weight > 0.0)
                else torch.tensor(0.0, device=device)
            )

            raw_loss = loss_fused + aux_audio_weight * loss_audio + aux_video_weight * loss_video

            total_loss += float(raw_loss.item())
            total_loss_fused += float(loss_fused.item())
            total_loss_audio += float(loss_audio.item())
            total_loss_video += float(loss_video.item())

            preds = torch.argmax(out.logits, dim=-1)
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().tolist()
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

            y_scores.extend(probs)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

            all_w_a.extend(out.w_a.squeeze(-1).cpu().tolist())
            all_w_v.extend(out.w_v.squeeze(-1).cpu().tolist())

            if out.audio_logits is not None:
                a_preds = torch.argmax(out.audio_logits, dim=-1)
                a_probs = torch.softmax(out.audio_logits, dim=-1)[:, 1]
                y_audio_true.extend(labels.cpu().tolist())
                y_audio_pred.extend(a_preds.cpu().tolist())
                y_audio_scores.extend(a_probs.cpu().tolist())

            if out.video_logits is not None:
                v_preds = torch.argmax(out.video_logits, dim=-1)
                v_probs = torch.softmax(out.video_logits, dim=-1)[:, 1]
                y_video_true.extend(labels.cpu().tolist())
                y_video_pred.extend(v_preds.cpu().tolist())
                y_video_scores.extend(v_probs.cpu().tolist())

            actual_steps += 1

    avg_loss = total_loss / max(1, actual_steps)
    avg_loss_fused = total_loss_fused / max(1, actual_steps)
    avg_loss_audio = total_loss_audio / max(1, actual_steps)
    avg_loss_video = total_loss_video / max(1, actual_steps)
    accuracy = correct / max(1, total)
    auc = compute_auc(y_true, y_scores)

    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

    precision = float(precision_score(y_true, y_pred, zero_division=0.0))
    recall = float(recall_score(y_true, y_pred, zero_division=0.0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0.0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    mean_w_a = float(np.mean(all_w_a)) if all_w_a else 0.5
    mean_w_v = float(np.mean(all_w_v)) if all_w_v else 0.5
    std_w_a = float(np.std(all_w_a)) if all_w_a else 0.0
    std_w_v = float(np.std(all_w_v)) if all_w_v else 0.0

    min_w_a = float(np.min(all_w_a)) if all_w_a else 0.0
    max_w_a = float(np.max(all_w_a)) if all_w_a else 1.0
    min_w_v = float(np.min(all_w_v)) if all_w_v else 0.0
    max_w_v = float(np.max(all_w_v)) if all_w_v else 1.0

    w_a_real = [w for w, y in zip(all_w_a, y_true, strict=False) if y == 0]
    w_a_fake = [w for w, y in zip(all_w_a, y_true, strict=False) if y == 1]
    w_v_real = [w for w, y in zip(all_w_v, y_true, strict=False) if y == 0]
    w_v_fake = [w for w, y in zip(all_w_v, y_true, strict=False) if y == 1]

    mean_w_a_real = float(np.mean(w_a_real)) if w_a_real else 0.5
    std_w_a_real = float(np.std(w_a_real)) if w_a_real else 0.0
    mean_w_v_real = float(np.mean(w_v_real)) if w_v_real else 0.5
    std_w_v_real = float(np.std(w_v_real)) if w_v_real else 0.0

    mean_w_a_fake = float(np.mean(w_a_fake)) if w_a_fake else 0.5
    std_w_a_fake = float(np.std(w_a_fake)) if w_a_fake else 0.0
    mean_w_v_fake = float(np.mean(w_v_fake)) if w_v_fake else 0.5
    std_w_v_fake = float(np.std(w_v_fake)) if w_v_fake else 0.0

    neg_total = sum(1 for yt in y_true if yt == 0)
    tn_count = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == 0 and yp == 0)
    specificity = float(tn_count / max(1, neg_total))

    res: dict[str, float] = {
        "loss": avg_loss,
        "loss_fused": avg_loss_fused,
        "loss_audio": avg_loss_audio,
        "loss_video": avg_loss_video,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "balanced_acc": balanced_acc,
        "mean_w_a": mean_w_a,
        "mean_w_v": mean_w_v,
        "std_w_a": std_w_a,
        "std_w_v": std_w_v,
        "min_w_a": min_w_a,
        "max_w_a": max_w_a,
        "min_w_v": min_w_v,
        "max_w_v": max_w_v,
        "mean_w_a_real": mean_w_a_real,
        "std_w_a_real": std_w_a_real,
        "mean_w_v_real": mean_w_v_real,
        "std_w_v_real": std_w_v_real,
        "mean_w_a_fake": mean_w_a_fake,
        "std_w_a_fake": std_w_a_fake,
        "mean_w_v_fake": mean_w_v_fake,
        "std_w_v_fake": std_w_v_fake,
    }

    if y_audio_true:
        res["audio_acc"] = float(
            sum(p == t for p, t in zip(y_audio_pred, y_audio_true, strict=False))
            / len(y_audio_true)
        )
        res["audio_prec"] = float(precision_score(y_audio_true, y_audio_pred, zero_division=0.0))
        res["audio_rec"] = float(recall_score(y_audio_true, y_audio_pred, zero_division=0.0))
        res["audio_f1"] = float(f1_score(y_audio_true, y_audio_pred, zero_division=0.0))
        res["audio_auc"] = compute_auc(y_audio_true, y_audio_scores)
        res["audio_balanced_acc"] = float(balanced_accuracy_score(y_audio_true, y_audio_pred))

    if y_video_true:
        res["video_acc"] = float(
            sum(p == t for p, t in zip(y_video_pred, y_video_true, strict=False))
            / len(y_video_true)
        )
        res["video_prec"] = float(precision_score(y_video_true, y_video_pred, zero_division=0.0))
        res["video_rec"] = float(recall_score(y_video_true, y_video_pred, zero_division=0.0))
        res["video_f1"] = float(f1_score(y_video_true, y_video_pred, zero_division=0.0))
        res["video_auc"] = compute_auc(y_video_true, y_video_scores)
        res["video_balanced_acc"] = float(balanced_accuracy_score(y_video_true, y_video_pred))

    return res


def run_smoke_training(output_checkpoint: Path, device: torch.device) -> None:
    """Fast smoke verification: train 2 steps, 1 val step, save checkpoint."""
    logger.info("Initializing AVT²-DWF smoke training pipeline...")
    smoke_dir = Path("storage")
    manifest_path = build_smoke_manifest(smoke_dir)
    logger.info("Smoke manifest ready at %s", manifest_path)

    augmenter = MultimodalAugmenter(mismatch_prob=0.0)
    train_ds = MultimodalDataset(manifest_path, split="train", augmenter=augmenter)
    val_ds = MultimodalDataset(manifest_path, split="val", augmenter=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        collate_fn=multimodal_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
    )

    model = AVT2DWF(pretrained_vit=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    logger.info("Running 2 smoke training steps...")
    train_res = train_epoch(
        model, train_loader, optimizer, criterion, device, max_steps=2
    )
    logger.info(
        "Smoke train step complete: loss=%.4f, acc=%.2f",
        train_res["loss"],
        train_res["accuracy"],
    )

    logger.info("Running 1 smoke validation step...")
    val_res = evaluate_epoch(
        model, val_loader, criterion, device, max_steps=1
    )
    logger.info(
        "Smoke validation complete: loss=%.4f, acc=%.2f, auc=%.2f",
        val_res["loss"],
        val_res["accuracy"],
        val_res["auc"],
    )

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_checkpoint)
    logger.info("Smoke checkpoint saved successfully to %s", output_checkpoint)


def compute_and_save_audio_norm_stats(
    manifest_path: Path,
    output_path: Path,
    manifest_sha256: str = "",
    preprocessing_version: str = "2.1.0",
) -> dict[str, Any]:
    """Compute channel-wise mean and standard deviation from audio features of manifest samples."""
    import datetime

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    repo_root = Path(__file__).resolve().parent.parent.parent
    processed_audio_dir = repo_root / "datasets" / "processed" / "audio"

    feat_dim = 104
    total_frames = 0
    sum_feat = torch.zeros(feat_dim, dtype=torch.float64)
    sum_sq_feat = torch.zeros(feat_dim, dtype=torch.float64)

    for r in rows:
        sid = r.get("sample_id") or Path(r.get("filename", "")).stem
        a_path = processed_audio_dir / f"{sid}_audio.pt"
        if not a_path.is_file():
            continue
        ad = torch.load(a_path, weights_only=True)
        seg = ad["audio_segments"]  # (K, 104, 100)
        mask = ad.get("audio_mask", torch.ones(seg.shape[0]))

        for k in range(seg.shape[0]):
            if mask[k] > 0.5:
                frames = seg[k].to(torch.float64)
                sum_feat += frames.sum(dim=-1)
                sum_sq_feat += (frames ** 2).sum(dim=-1)
                total_frames += frames.shape[-1]

    if total_frames == 0:
        raise ValueError(f"No valid audio frames found in manifest samples from {manifest_path}")

    mean = (sum_feat / total_frames).to(torch.float32)
    var = ((sum_sq_feat / total_frames) - (mean.to(torch.float64) ** 2)).clamp(min=1e-12)
    std = torch.sqrt(var).to(torch.float32).clamp(min=1e-5)

    stats_payload = {
        "mean": mean,
        "std": std,
        "feature_dim": feat_dim,
        "source_manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "preprocessing_version": preprocessing_version,
        "num_samples": len(rows),
        "total_frames": total_frames,
        "computed_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    safe_torch_save(stats_payload, output_path)
    logger.info(
        "Saved fresh audio norm stats to %s (%d samples, %d frames)",
        output_path,
        len(rows),
        total_frames,
    )
    return stats_payload


def train_model(
    config: dict[str, Any],
    manifest_path: Path,
    output_checkpoint: Path,
    device: torch.device,
    resume_checkpoint: Path | None = None,
    val_manifest_path: Path | None = None,
    audio_norm_stats: Path | dict[str, Any] | None = None,
    experiment_dir: Path | None = None,
) -> dict[str, Any]:
    """Train the AVT²-DWF model using the specified configuration and manifest."""
    import time

    cfg = config
    cfg_train = cfg.get("training", {})
    epochs = int(cfg_train.get("epochs", 15))
    batch_size = int(cfg_train.get("batch_size", 2))
    grad_accum_steps = int(cfg_train.get("grad_accum_steps", 4))
    lr = float(cfg_train.get("lr", 1e-4))
    weight_decay = float(cfg_train.get("weight_decay", 0.01))
    patience = int(cfg_train.get("patience", 15))
    label_smoothing = float(cfg_train.get("label_smoothing", 0.1))
    classifier_dropout = float(cfg_train.get("classifier_dropout", 0.25))
    aux_audio_weight = float(cfg_train.get("aux_audio_weight", 0.3))
    aux_video_weight = float(cfg_train.get("aux_video_weight", 0.1))
    audio_layernorm = bool(cfg_train.get("audio_layernorm", True))
    audio_input_norm = bool(cfg_train.get("audio_input_norm", True))
    mismatch_prob = float(cfg_train.get("mismatch_prob", 0.0))
    checkpoint_metric = str(cfg_train.get("checkpoint_metric", "val_auc"))
    modality_dropout_prob = float(cfg_train.get("modality_dropout_prob", 0.0))
    dwf_min_weight = float(
        cfg.get("model", {}).get("dwf_min_weight", cfg_train.get("dwf_min_weight", 0.0))
    )
    dwf_max_weight = float(
        cfg.get("model", {}).get("dwf_max_weight", cfg_train.get("dwf_max_weight", 1.0))
    )
    seed = int(cfg_train.get("seed", 42))

    # Set deterministic random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    effective_batch_size = batch_size * grad_accum_steps
    logger.info(
        "Starting AVT²-DWF training: epochs=%d, batch=%d, accum=%d (eff=%d), lr=%.6f, "
        "dev=%s, seed=%d",
        epochs,
        batch_size,
        grad_accum_steps,
        effective_batch_size,
        lr,
        device,
        seed,
    )
    logger.info(
        "Phase 4 enhancements: audio_ln=%s, aux_audio_wt=%.2f, aux_video_wt=%.2f, "
        "smoothing=%.2f, dropout=%.2f, wd=%.4f, mismatch_prob=%.2f, metric=%s",
        audio_layernorm,
        aux_audio_weight,
        aux_video_weight,
        label_smoothing,
        classifier_dropout,
        weight_decay,
        mismatch_prob,
        checkpoint_metric,
    )

    if val_manifest_path is not None:
        val_manifest = val_manifest_path
    elif manifest_path.name in ("train.csv", "train_recovery.csv"):
        val_name = (
            "validation.csv"
            if manifest_path.name == "train.csv"
            else "validation_recovery.csv"
        )
        val_candidate = manifest_path.parent / val_name
        val_manifest = val_candidate if val_candidate.exists() else manifest_path
    else:
        val_manifest = manifest_path

    augmenter = MultimodalAugmenter(mismatch_prob=mismatch_prob)
    train_dataset = MultimodalDataset(manifest_path, split="train", augmenter=augmenter)
    val_dataset = MultimodalDataset(val_manifest, split="validation", augmenter=None)

    sampler = create_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=multimodal_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
    )

    pretrained = bool(cfg.get("model", {}).get("pretrained_vit", True))
    auxiliary_heads = (aux_audio_weight > 0.0 or aux_video_weight > 0.0)
    model = AVT2DWF(
        pretrained_vit=pretrained,
        audio_layernorm=audio_layernorm,
        audio_input_norm=audio_input_norm,
        auxiliary_heads=auxiliary_heads,
        classifier_dropout=classifier_dropout,
        dwf_min_weight=dwf_min_weight,
        dwf_max_weight=dwf_max_weight,
    ).to(device)

    # Apply custom audio normalization statistics if provided
    if audio_norm_stats is not None:
        if isinstance(audio_norm_stats, (str, Path)):
            s_path = Path(audio_norm_stats)
            if s_path.is_file():
                s_dict = torch.load(s_path, weights_only=True)
                model.audio_encoder.input_norm.set_stats(s_dict["mean"], s_dict["std"])
                logger.info("Applied custom audio norm stats from %s", s_path)
            else:
                logger.warning("Specified audio_norm_stats file not found: %s", s_path)
        elif (
            isinstance(audio_norm_stats, dict)
            and "mean" in audio_norm_stats
            and "std" in audio_norm_stats
        ):
            model.audio_encoder.input_norm.set_stats(
                audio_norm_stats["mean"], audio_norm_stats["std"]
            )
            logger.info("Applied custom audio norm stats dictionary")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    start_epoch = 1
    best_val_f1 = -1.0
    best_val_auc = -1.0
    best_val_balanced_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 1
    stagnant_epochs = 0
    history: list[dict[str, Any]] = []

    if resume_checkpoint and resume_checkpoint.exists():
        if experiment_dir is not None:
            try:
                resume_checkpoint.resolve().relative_to(Path(experiment_dir).resolve())
            except ValueError:
                raise ValueError(
                    f"Safety violation: Resuming is strictly restricted to checkpoints within "
                    f"the current experiment directory ({experiment_dir}). "
                    f"Checkpoints from other directories ({resume_checkpoint}) cannot be used."
                ) from None

        logger.info("Resuming training from checkpoint: %s", resume_checkpoint)
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "rng_state" in ckpt:
            with contextlib.suppress(Exception):
                torch.set_rng_state(ckpt["rng_state"])
        completed_epoch = int(ckpt.get("epoch", 0))
        start_epoch = completed_epoch + 1

        if output_checkpoint.exists():
            best_ckpt = torch.load(output_checkpoint, map_location=device, weights_only=False)
            best_val_auc = float(best_ckpt.get("val_auc", -1.0))
            best_val_f1 = float(best_ckpt.get("val_f1", -1.0))
            best_val_loss = float(best_ckpt.get("val_loss", float("inf")))
            best_val_balanced_acc = float(best_ckpt.get("val_balanced_acc", -1.0))
            best_epoch = int(best_ckpt.get("epoch", 1))
            stagnant_epochs = max(0, completed_epoch - best_epoch)
        else:
            best_val_auc = float(ckpt.get("val_auc", -1.0))
            best_val_f1 = float(ckpt.get("val_f1", -1.0))
            best_val_loss = float(ckpt.get("val_loss", float("inf")))
            best_val_balanced_acc = float(ckpt.get("val_balanced_acc", -1.0))
            best_epoch = int(ckpt.get("epoch", 1))
            stagnant_epochs = 0

        # Load history for completed epochs if available
        if experiment_dir is not None:
            exp_hist = Path(experiment_dir) / "training_history.csv"
            if exp_hist.exists():
                with open(exp_hist, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        if int(r["epoch"]) <= completed_epoch:
                            history.append(
                                {k: (float(v) if k != "epoch" else int(v)) for k, v in r.items()}
                            )
        else:
            repo_root = Path(__file__).resolve().parent.parent.parent
            hist_file = repo_root / "reports" / "phase4_training_history.csv"
            if hist_file.exists():
                with open(hist_file, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        if int(r["epoch"]) <= completed_epoch:
                            history.append(
                                {k: (float(v) if k != "epoch" else int(v)) for k, v in r.items()}
                            )

        logger.info(
            "Resuming at epoch %d/%d (completed=%d, preloaded history=%d epochs)",
            start_epoch,
            epochs,
            completed_epoch,
            len(history),
        )

    is_phase4 = "phase4" in output_checkpoint.name
    is_phase2_improved = "phase2_improved" in output_checkpoint.name
    if experiment_dir is not None:
        latest_checkpoint = experiment_dir / "latest_checkpoint.pt"
    else:
        latest_checkpoint = (
            output_checkpoint.parent / "phase2_improved_latest_avt2_dwf.pt"
            if is_phase2_improved
            else (
                output_checkpoint.parent / "phase4_latest_avt2_dwf.pt"
                if is_phase4
                else output_checkpoint.parent / "latest_avt2_dwf.pt"
            )
        )
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        current_lr = float(optimizer.param_groups[0]["lr"])

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_res = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_accum_steps=grad_accum_steps,
            aux_audio_weight=aux_audio_weight,
            aux_video_weight=aux_video_weight,
            modality_dropout_prob=modality_dropout_prob,
        )
        val_res = evaluate_epoch(
            model,
            val_loader,
            criterion,
            device,
            aux_audio_weight=aux_audio_weight,
            aux_video_weight=aux_video_weight,
        )
        scheduler.step()

        epoch_duration = round(time.perf_counter() - epoch_start, 2)
        gpu_mem_mb = (
            round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)
            if device.type == "cuda"
            else 0.0
        )

        record = {
            "epoch": epoch,
            "train_loss": round(train_res["loss"], 4),
            "train_loss_fused": round(train_res["loss_fused"], 4),
            "train_loss_audio": round(train_res.get("loss_audio", 0.0), 4),
            "train_loss_video": round(train_res.get("loss_video", 0.0), 4),
            "train_acc": round(train_res["accuracy"], 4),
            "train_prec": round(train_res["precision"], 4),
            "train_rec": round(train_res["recall"], 4),
            "train_f1": round(train_res["f1"], 4),
            "train_auc": round(train_res["auc"], 4),
            "val_loss": round(val_res["loss"], 4),
            "val_loss_fused": round(val_res["loss_fused"], 4),
            "val_loss_audio": round(val_res.get("loss_audio", 0.0), 4),
            "val_loss_video": round(val_res.get("loss_video", 0.0), 4),
            "val_acc": round(val_res["accuracy"], 4),
            "val_prec": round(val_res["precision"], 4),
            "val_rec": round(val_res["recall"], 4),
            "val_f1": round(val_res["f1"], 4),
            "val_auc": round(val_res["auc"], 4),
            "val_balanced_acc": round(val_res.get("balanced_acc", 0.0), 4),
            "val_specificity": round(val_res.get("specificity", 0.0), 4),
            "val_audio_acc": round(val_res.get("audio_acc", 0.0), 4),
            "val_audio_prec": round(val_res.get("audio_prec", 0.0), 4),
            "val_audio_rec": round(val_res.get("audio_rec", 0.0), 4),
            "val_audio_f1": round(val_res.get("audio_f1", 0.0), 4),
            "val_audio_auc": round(val_res.get("audio_auc", 0.0), 4),
            "val_audio_balanced_acc": round(val_res.get("audio_balanced_acc", 0.0), 4),
            "val_video_acc": round(val_res.get("video_acc", 0.0), 4),
            "val_video_prec": round(val_res.get("video_prec", 0.0), 4),
            "val_video_rec": round(val_res.get("video_rec", 0.0), 4),
            "val_video_f1": round(val_res.get("video_f1", 0.0), 4),
            "val_video_auc": round(val_res.get("video_auc", 0.0), 4),
            "val_video_balanced_acc": round(val_res.get("video_balanced_acc", 0.0), 4),
            "mean_w_a": round(val_res.get("mean_w_a", 0.5), 4),
            "mean_w_v": round(val_res.get("mean_w_v", 0.5), 4),
            "std_w_a": round(val_res.get("std_w_a", 0.0), 4),
            "std_w_v": round(val_res.get("std_w_v", 0.0), 4),
            "min_w_a": round(val_res.get("min_w_a", 0.0), 4),
            "max_w_a": round(val_res.get("max_w_a", 1.0), 4),
            "min_w_v": round(val_res.get("min_w_v", 0.0), 4),
            "max_w_v": round(val_res.get("max_w_v", 1.0), 4),
            "mean_w_a_real": round(val_res.get("mean_w_a_real", 0.5), 4),
            "std_w_a_real": round(val_res.get("std_w_a_real", 0.0), 4),
            "mean_w_v_real": round(val_res.get("mean_w_v_real", 0.5), 4),
            "std_w_v_real": round(val_res.get("std_w_v_real", 0.0), 4),
            "mean_w_a_fake": round(val_res.get("mean_w_a_fake", 0.5), 4),
            "std_w_a_fake": round(val_res.get("std_w_a_fake", 0.0), 4),
            "mean_w_v_fake": round(val_res.get("mean_w_v_fake", 0.5), 4),
            "std_w_v_fake": round(val_res.get("std_w_v_fake", 0.0), 4),
            "grad_norm_audio": round(train_res.get("grad_norm_audio", 0.0), 4),
            "grad_norm_video": round(train_res.get("grad_norm_video", 0.0), 4),
            "grad_norm_dwf": round(train_res.get("grad_norm_dwf", 0.0), 4),
            "lr": round(current_lr, 7),
            "duration_s": epoch_duration,
            "gpu_mem_mb": gpu_mem_mb,
        }
        history.append(record)
        if experiment_dir is not None:
            exp_csv = Path(experiment_dir) / "training_history.csv"
            with open(exp_csv, "w", newline="", encoding="utf-8") as f_hist:
                writer = csv.DictWriter(f_hist, fieldnames=list(record.keys()))
                writer.writeheader()
                writer.writerows(history)

        logger.info(
            "Epoch %02d/%02d (%4.1fs) | Train Loss: %.4f (Fused: %.4f, A: %.4f, V: %.4f) "
            "Acc: %.4f F1: %.4f AUC: %.4f | "
            "Val Loss: %.4f Acc: %.4f F1: %.4f AUC: %.4f BAcc: %.4f | "
            "Val Unimodal - Audio AUC: %.4f (F1: %.4f), Video AUC: %.4f (F1: %.4f) | "
            "Weights - w_a: %.4f±%.4f, w_v: %.4f±%.4f | "
            "Grads - ||g_A||: %.4f, ||g_V||: %.4f | LR: %.2e | GPU: %s MB",
            epoch,
            epochs,
            epoch_duration,
            train_res["loss"],
            train_res["loss_fused"],
            train_res.get("loss_audio", 0.0),
            train_res.get("loss_video", 0.0),
            train_res["accuracy"],
            train_res["f1"],
            train_res["auc"],
            val_res["loss"],
            val_res["accuracy"],
            val_res["f1"],
            val_res["auc"],
            val_res.get("balanced_acc", 0.0),
            val_res.get("audio_auc", 0.0),
            val_res.get("audio_f1", 0.0),
            val_res.get("video_auc", 0.0),
            val_res.get("video_f1", 0.0),
            val_res.get("mean_w_a", 0.5),
            val_res.get("std_w_a", 0.0),
            val_res.get("mean_w_v", 0.5),
            val_res.get("std_w_v", 0.0),
            train_res.get("grad_norm_audio", 0.0),
            train_res.get("grad_norm_video", 0.0),
            current_lr,
            gpu_mem_mb,
        )

        checkpoint_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "train_loss": train_res["loss"],
            "train_loss_fused": train_res["loss_fused"],
            "train_loss_audio": train_res.get("loss_audio", 0.0),
            "train_loss_video": train_res.get("loss_video", 0.0),
            "train_acc": train_res["accuracy"],
            "train_f1": train_res["f1"],
            "train_auc": train_res["auc"],
            "val_loss": val_res["loss"],
            "val_loss_fused": val_res["loss_fused"],
            "val_loss_audio": val_res.get("loss_audio", 0.0),
            "val_loss_video": val_res.get("loss_video", 0.0),
            "val_acc": val_res["accuracy"],
            "val_f1": val_res["f1"],
            "val_auc": val_res["auc"],
            "val_balanced_acc": val_res.get("balanced_acc", 0.0),
            "mean_w_a": val_res.get("mean_w_a", 0.5),
            "mean_w_v": val_res.get("mean_w_v", 0.5),
            "std_w_a": val_res.get("std_w_a", 0.0),
            "std_w_v": val_res.get("std_w_v", 0.0),
            "min_w_a": val_res.get("min_w_a", 0.0),
            "max_w_a": val_res.get("max_w_a", 1.0),
            "min_w_v": val_res.get("min_w_v", 0.0),
            "max_w_v": val_res.get("max_w_v", 1.0),
            "mean_w_a_real": val_res.get("mean_w_a_real", 0.5),
            "mean_w_v_real": val_res.get("mean_w_v_real", 0.5),
            "mean_w_a_fake": val_res.get("mean_w_a_fake", 0.5),
            "mean_w_v_fake": val_res.get("mean_w_v_fake", 0.5),
            "config": cfg,
            "manifest_path": str(manifest_path),
            "device": str(device),
            "rng_state": torch.get_rng_state(),
        }

        # Always safely save latest candidate checkpoint
        safe_torch_save(checkpoint_state, latest_checkpoint)

        val_auc = val_res["auc"]
        val_f1 = val_res["f1"]
        val_loss = val_res["loss"]
        val_balanced_acc = val_res.get("balanced_acc", 0.0)

        # Determine if checkpoint improved according to configured primary metric and tie-breaking:
        # 1. Higher validation ROC-AUC
        # 2. Higher validation balanced accuracy
        # 3. Lower validation loss
        # 4. Higher validation F1
        if checkpoint_metric == "val_auc":
            if val_auc > best_val_auc + 1e-4:
                improved = True
            elif abs(val_auc - best_val_auc) <= 1e-4:
                if val_balanced_acc > best_val_balanced_acc + 1e-4:
                    improved = True
                elif abs(val_balanced_acc - best_val_balanced_acc) <= 1e-4:
                    if val_loss < best_val_loss - 1e-4:
                        improved = True
                    elif abs(val_loss - best_val_loss) <= 1e-4:
                        improved = val_f1 > best_val_f1 + 1e-4
                    else:
                        improved = False
                else:
                    improved = False
            else:
                improved = False
        else:
            if val_f1 > best_val_f1 + 1e-4:
                improved = True
            elif abs(val_f1 - best_val_f1) <= 1e-4:
                improved = val_loss < best_val_loss - 1e-4
            else:
                improved = False

        if improved:
            best_val_f1 = val_f1
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_val_balanced_acc = val_balanced_acc
            best_epoch = epoch
            stagnant_epochs = 0
            safe_torch_save(checkpoint_state, output_checkpoint)
            logger.info(
                "Saved new best checkpoint to %s (Epoch %d | "
                "Val AUC: %.4f, Val BAcc: %.4f, Val F1: %.4f, Val Loss: %.4f)",
                output_checkpoint,
                best_epoch,
                best_val_auc,
                best_val_balanced_acc,
                best_val_f1,
                best_val_loss,
            )
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.", patience
                )
                break

    # Save training history to CSV
    if experiment_dir is not None:
        exp_dir = Path(experiment_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        csv_paths = [exp_dir / "training_history.csv"]
        report_json_paths = [exp_dir / "training_report.json"]
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent
        reports_dir = repo_root / "datasets" / "reports"
        root_reports_dir = repo_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        root_reports_dir.mkdir(parents=True, exist_ok=True)

        if is_phase2_improved:
            history_filename = "phase2_improved_training_history.csv"
            report_filename = "phase2_improved_training_report.json"
        elif is_phase4:
            history_filename = "phase4_training_history.csv"
            report_filename = "phase4_training_report.json"
        else:
            history_filename = "training_history.csv"
            report_filename = "training_report.json"

        csv_paths = [
            reports_dir / history_filename,
            root_reports_dir / history_filename,
        ]
        if not is_phase4 and not is_phase2_improved:
            csv_paths.append(reports_dir / "phase2_training_history.csv")
            csv_paths.append(root_reports_dir / "phase2_training_history.csv")

    fieldnames = list(history[0].keys()) if history else [
        "epoch", "train_loss", "train_loss_fused", "train_loss_audio", "train_loss_video",
        "train_acc", "train_prec", "train_rec", "train_f1", "train_auc",
        "val_loss", "val_loss_fused", "val_loss_audio", "val_loss_video",
        "val_acc", "val_prec", "val_rec", "val_f1", "val_auc",
        "val_balanced_acc", "val_specificity",
        "val_audio_acc", "val_audio_prec", "val_audio_rec", "val_audio_f1", "val_audio_auc",
        "val_audio_balanced_acc",
        "val_video_acc", "val_video_prec", "val_video_rec", "val_video_f1", "val_video_auc",
        "val_video_balanced_acc",
        "mean_w_a", "mean_w_v", "std_w_a", "std_w_v", "min_w_a", "max_w_a", "min_w_v", "max_w_v",
        "mean_w_a_real", "std_w_a_real", "mean_w_v_real", "std_w_v_real",
        "mean_w_a_fake", "std_w_a_fake", "mean_w_v_fake", "std_w_v_fake",
        "grad_norm_audio", "grad_norm_video", "grad_norm_dwf",
        "lr", "duration_s", "gpu_mem_mb",
    ]
    for p in csv_paths:
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        logger.info("Saved training history to %s", p)

    if experiment_dir is not None:
        phase_label: str | int = "4.4_clean"
    elif is_phase2_improved:
        phase_label = "2_improved"
    elif is_phase4:
        phase_label = 4
    else:
        phase_label = 2

    training_report = {
        "status": "COMPLETED",
        "phase": phase_label,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_auc": best_val_auc,
        "best_val_balanced_acc": best_val_balanced_acc,
        "best_val_loss": best_val_loss,
        "epochs_completed": len(history),
        "best_checkpoint": str(output_checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
        "training_config": {
            "seed": seed,
            "manifest_path": str(manifest_path),
            "val_manifest_path": str(val_manifest_path) if val_manifest_path else None,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": effective_batch_size,
            "initial_lr": lr,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "classifier_dropout": classifier_dropout,
            "aux_audio_weight": aux_audio_weight,
            "aux_video_weight": aux_video_weight,
            "audio_layernorm": audio_layernorm,
            "audio_input_norm": audio_input_norm,
            "mismatch_prob": mismatch_prob,
            "checkpoint_metric": checkpoint_metric,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
        },
        "history": history,
    }

    if experiment_dir is None:
        report_json_paths = [
            reports_dir / report_filename,
            root_reports_dir / report_filename,
        ]
        if not is_phase4 and not is_phase2_improved:
            report_json_paths.append(reports_dir / "phase2_training_report.json")
            report_json_paths.append(root_reports_dir / "phase2_training_report.json")
    else:
        exp_path = Path(experiment_dir)
        config_path = exp_path / "training_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            import json

            json.dump(training_report["training_config"], f, indent=2)
        logger.info("Saved training config to %s", config_path)

        # Also write experiment_config.json for backwards compatibility
        with open(exp_path / "experiment_config.json", "w", encoding="utf-8") as f:
            json.dump(training_report["training_config"], f, indent=2)

        summary_path = exp_path / "experiment_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(training_report, f, indent=2)
        logger.info("Saved experiment summary to %s", summary_path)

        # Ensure best_model.pt exists
        best_model_pt = exp_path / "best_model.pt"
        if output_checkpoint.exists() and output_checkpoint.name != "best_model.pt":
            shutil.copy2(output_checkpoint, best_model_pt)
            logger.info("Copied %s -> %s", output_checkpoint, best_model_pt)

        # Save reproducibility.json
        import hashlib
        manifest_hashes: dict[str, str] = {}
        for m_name, m_p in [
            ("train", manifest_path),
            ("val", val_manifest),
            ("test", manifest_path.parent / "test_recovery.csv"),
        ]:
            if m_p.exists():
                manifest_hashes[f"{m_name}_manifest_sha256"] = (
                    hashlib.sha256(m_p.read_bytes()).hexdigest()
                )

        norm_file = exp_path / "audio_norm_stats.pt"
        norm_hash = (
            hashlib.sha256(norm_file.read_bytes()).hexdigest() if norm_file.exists() else None
        )

        repro_data = {
            "experiment_name": "phase4_4_clean_avt2_dwf",
            "phase": "4.4",
            "random_seed": seed,
            "software_versions": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            },
            "hardware": {
                "cuda_available": torch.cuda.is_available(),
                "device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
                ),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            },
            "manifest_hashes": manifest_hashes,
            "audio_norm_stats_sha256": norm_hash,
            "best_checkpoint_sha256": (
                hashlib.sha256(output_checkpoint.read_bytes()).hexdigest()
                if output_checkpoint.exists()
                else None
            ),
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
        }
        repro_path = exp_path / "reproducibility.json"
        with open(repro_path, "w", encoding="utf-8") as f:
            json.dump(repro_data, f, indent=2)
        logger.info("Saved reproducibility info to %s", repro_path)

    for rp in report_json_paths:
        with open(rp, "w", encoding="utf-8") as f:
            import json

            json.dump(training_report, f, indent=2)
        logger.info("Saved training report to %s", rp)

    if is_phase2_improved and experiment_dir is None:
        generate_phase2_improved_reports(
            history=history,
            training_report=training_report,
            repo_root=repo_root,
            output_checkpoint=output_checkpoint,
            latest_checkpoint=latest_checkpoint,
        )

    logger.info(
        "Training complete. Best Val AUC: %.4f, BAcc: %.4f, F1: %.4f, Loss: %.4f at Epoch %d",
        best_val_auc,
        best_val_balanced_acc,
        best_val_f1,
        best_val_loss,
        best_epoch,
    )
    return training_report


def generate_phase2_improved_reports(
    history: list[dict[str, Any]],
    training_report: dict[str, Any],
    repo_root: Path,
    output_checkpoint: Path,
    latest_checkpoint: Path,
) -> None:
    """Generate Phase 2 Improved diagnostics, modality comparisons, and integrity reports."""
    import hashlib
    import json
    import time

    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    best_epoch = int(training_report["best_epoch"])
    best_record = next((r for r in history if r["epoch"] == best_epoch), history[-1])

    # 1. Phase 2 Audio Diagnostics
    audio_stats_path = repo_root / "backend" / "app" / "ml" / "weights" / "audio_norm_stats.pt"
    audio_stats_meta: dict[str, Any] = {}
    if audio_stats_path.exists():
        stats = torch.load(audio_stats_path, weights_only=True)
        mean_vec = stats["mean"]
        std_vec = stats["std"]
        audio_stats_meta = {
            "stats_path": str(audio_stats_path.relative_to(repo_root)),
            "channels": int(mean_vec.shape[0]),
            "log_mel_channels_0_to_79": {
                "mean_min": round(float(mean_vec[:80].min().item()), 4),
                "mean_max": round(float(mean_vec[:80].max().item()), 4),
                "std_min": round(float(std_vec[:80].min().item()), 4),
                "std_max": round(float(std_vec[:80].max().item()), 4),
            },
            "mfcc_channels_80_to_103": {
                "channel_80_mfcc0_mean": round(float(mean_vec[80].item()), 4),
                "channel_80_mfcc0_std": round(float(std_vec[80].item()), 4),
                "mean_min": round(float(mean_vec[80:].min().item()), 4),
                "mean_max": round(float(mean_vec[80:].max().item()), 4),
                "std_min": round(float(std_vec[80:].min().item()), 4),
                "std_max": round(float(std_vec[80:].max().item()), 4),
            },
        }

    audio_diag = {
        "status": "COMPLETED",
        "description": "Phase 2 Audio Representation Improvement and Diagnostics",
        "normalization": {
            "applied": True,
            "layer": "AudioInputNorm (clamp [-10.0, 10.0])",
            "post_patch_layernorm": True,
            "statistics": audio_stats_meta,
        },
        "auxiliary_audio_head": {
            "weight": training_report.get("training_config", {}).get("aux_audio_weight", 0.3),
            "architecture": "Linear(256, 128) -> GELU -> Dropout(0.25) -> Linear(128, 2)",
        },
        "gradient_flow": {
            "per_epoch_audio_grad_norms": [
                {"epoch": r["epoch"], "grad_norm_audio": r["grad_norm_audio"]} for r in history
            ],
            "mean_audio_grad_norm": round(
                float(sum(r["grad_norm_audio"] for r in history) / max(1, len(history))), 4
            ),
            "non_zero_gradients_verified": all(r["grad_norm_audio"] > 0 for r in history),
        },
        "audio_validation_performance": {
            "best_epoch": best_epoch,
            "audio_acc": best_record.get("val_audio_acc", 0.0),
            "audio_prec": best_record.get("val_audio_prec", 0.0),
            "audio_rec": best_record.get("val_audio_rec", 0.0),
            "audio_f1": best_record.get("val_audio_f1", 0.0),
            "audio_auc": best_record.get("val_audio_auc", 0.0),
            "audio_balanced_acc": best_record.get("val_audio_balanced_acc", 0.0),
            "history": [
                {
                    "epoch": r["epoch"],
                    "audio_acc": r.get("val_audio_acc", 0.0),
                    "audio_f1": r.get("val_audio_f1", 0.0),
                    "audio_auc": r.get("val_audio_auc", 0.0),
                    "audio_balanced_acc": r.get("val_audio_balanced_acc", 0.0),
                }
                for r in history
            ],
        },
        "baseline_comparison": {
            "baseline_audio_auc": 0.4972,
            "improved_audio_auc": best_record.get("val_audio_auc", 0.0),
            "delta_auc": round(best_record.get("val_audio_auc", 0.0) - 0.4972, 4),
            "modality_collapse_resolved": bool(best_record.get("val_audio_auc", 0.0) > 0.50),
        },
    }
    (reports_dir / "phase2_audio_diagnostics.json").write_text(
        json.dumps(audio_diag, indent=2), encoding="utf-8"
    )

    # 2. Phase 2 DWF Diagnostics
    dwf_diag = {
        "status": "COMPLETED",
        "description": "Dynamic Weight Fusion (DWF) Dynamics and Modality Balance Audit",
        "best_epoch": best_epoch,
        "best_epoch_weights": {
            "mean_w_a": best_record.get("mean_w_a", 0.5),
            "std_w_a": best_record.get("std_w_a", 0.0),
            "mean_w_v": best_record.get("mean_w_v", 0.5),
            "std_w_v": best_record.get("std_w_v", 0.0),
            "min_w_a": best_record.get("min_w_a", 0.0),
            "max_w_a": best_record.get("max_w_a", 1.0),
            "min_w_v": best_record.get("min_w_v", 0.0),
            "max_w_v": best_record.get("max_w_v", 1.0),
            "real_samples": {
                "mean_w_a": best_record.get("mean_w_a_real", 0.5),
                "std_w_a": best_record.get("std_w_a_real", 0.0),
                "mean_w_v": best_record.get("mean_w_v_real", 0.5),
                "std_w_v": best_record.get("std_w_v_real", 0.0),
            },
            "fake_samples": {
                "mean_w_a": best_record.get("mean_w_a_fake", 0.5),
                "std_w_a": best_record.get("std_w_a_fake", 0.0),
                "mean_w_v": best_record.get("mean_w_v_fake", 0.5),
                "std_w_v": best_record.get("std_w_v_fake", 0.0),
            },
        },
        "per_epoch_weights": [
            {
                "epoch": r["epoch"],
                "mean_w_a": r["mean_w_a"],
                "std_w_a": r["std_w_a"],
                "mean_w_v": r["mean_w_v"],
                "std_w_v": r["std_w_v"],
                "min_w_a": r.get("min_w_a", 0.0),
                "max_w_a": r.get("max_w_a", 1.0),
                "real": {
                    "mean_w_a": r.get("mean_w_a_real", 0.5),
                    "mean_w_v": r.get("mean_w_v_real", 0.5),
                },
                "fake": {
                    "mean_w_a": r.get("mean_w_a_fake", 0.5),
                    "mean_w_v": r.get("mean_w_v_fake", 0.5),
                },
            }
            for r in history
        ],
        "collapse_assessment": {
            "baseline_mean_w_a": 0.0041,
            "baseline_mean_w_v": 0.9959,
            "improved_mean_w_a": best_record.get("mean_w_a", 0.5),
            "improved_mean_w_v": best_record.get("mean_w_v", 0.5),
            "is_collapsed_to_video": bool(best_record.get("mean_w_a", 0.5) < 0.05),
            "verdict": (
                "HEALTHY_DYNAMIC_FUSION"
                if best_record.get("mean_w_a", 0.5) >= 0.05
                else "VIDEO_DOMINATED"
            ),
        },
    }
    (reports_dir / "phase2_dwf_diagnostics.json").write_text(
        json.dumps(dwf_diag, indent=2), encoding="utf-8"
    )

    # 3. Phase 2 Modality Comparison
    best_audio_auc = best_record.get("val_audio_auc", 0.0)
    best_video_auc = best_record.get("val_video_auc", 0.0)
    best_fused_auc = best_record.get("val_auc", 0.0)
    best_unimodal_auc = max(best_audio_auc, best_video_auc)

    mod_comp = {
        "status": "COMPLETED",
        "description": "Phase 2 Unimodal vs Multimodal Comparison at Best Checkpoint",
        "best_epoch": best_epoch,
        "modalities": {
            "audio_only": {
                "accuracy": best_record.get("val_audio_acc", 0.0),
                "precision": best_record.get("val_audio_prec", 0.0),
                "recall": best_record.get("val_audio_rec", 0.0),
                "f1": best_record.get("val_audio_f1", 0.0),
                "auc": best_audio_auc,
                "balanced_acc": best_record.get("val_audio_balanced_acc", 0.0),
            },
            "video_only": {
                "accuracy": best_record.get("val_video_acc", 0.0),
                "precision": best_record.get("val_video_prec", 0.0),
                "recall": best_record.get("val_video_rec", 0.0),
                "f1": best_record.get("val_video_f1", 0.0),
                "auc": best_video_auc,
                "balanced_acc": best_record.get("val_video_balanced_acc", 0.0),
            },
            "multimodal_fused": {
                "accuracy": best_record.get("val_acc", 0.0),
                "precision": best_record.get("val_prec", 0.0),
                "recall": best_record.get("val_rec", 0.0),
                "f1": best_record.get("val_f1", 0.0),
                "auc": best_fused_auc,
                "balanced_acc": best_record.get("val_balanced_acc", 0.0),
            },
        },
        "synergy_analysis": {
            "best_unimodal_auc": best_unimodal_auc,
            "multimodal_auc": best_fused_auc,
            "delta_auc_vs_best_unimodal": round(best_fused_auc - best_unimodal_auc, 4),
            "cross_modal_synergy": bool(best_fused_auc >= best_unimodal_auc),
        },
        "baseline_comparison": {
            "baseline_val_auc": 0.6974,
            "baseline_val_acc": 0.6579,
            "baseline_val_f1": 0.7174,
            "baseline_val_loss": 1.8823,
            "improved_val_auc": best_fused_auc,
            "improved_val_acc": best_record.get("val_acc", 0.0),
            "improved_val_f1": best_record.get("val_f1", 0.0),
            "improved_val_loss": best_record.get("val_loss", 0.0),
            "delta_val_auc": round(best_fused_auc - 0.6974, 4),
        },
    }
    (reports_dir / "phase2_modality_comparison.json").write_text(
        json.dumps(mod_comp, indent=2), encoding="utf-8"
    )

    # 4. Phase 2 Checkpoint Integrity
    def compute_sha256(p: Path) -> str:
        if not p.exists():
            return "NOT_FOUND"
        h = hashlib.sha256()
        with open(p, "rb") as bf:
            while chunk := bf.read(65536):
                h.update(chunk)
        return h.hexdigest()

    protected_files = {
        "baseline_checkpoint": {
            "path": "checkpoints/baseline_avt2_dwf.pt",
            "expected_sha256": "73b062c36ffbd4d4098ea640d09f3f11078bf80d09c1f7f4e10d902a98f0fa7f",
        },
        "legacy_latest_checkpoint": {
            "path": "checkpoints/latest_avt2_dwf.pt",
            "expected_sha256": "0e43b9592f05ac5fb26dad5a80d33d2512e4fcf84e14af4d8713c6681c7c9949",
        },
        "phase4_best_checkpoint": {
            "path": "checkpoints/phase4_best_avt2_dwf.pt",
            "expected_sha256": "f38448a19d6440393de738302f0215c646d26410eff5138bba1acb88c8e1bedc",
        },
        "phase4_latest_checkpoint": {
            "path": "checkpoints/phase4_latest_avt2_dwf.pt",
            "expected_sha256": "4360a4532ef246a04fa92a50fdd240b3de34a93d651f6d09a538f424714568c4",
        },
        "production_weights": {
            "path": "backend/app/ml/weights/avt2_dwf.pt",
            "expected_sha256": "6de2e663799ee8687c2ca06bbb4309034937c8fbf7a18ba3166ce745052ef354",
        },
    }

    integrity_audit: dict[str, Any] = {}
    all_protected_pass = True
    for key, spec in protected_files.items():
        fp = repo_root / spec["path"]
        current_hash = compute_sha256(fp)
        match = (current_hash == spec["expected_sha256"])
        if not match:
            all_protected_pass = False
        integrity_audit[key] = {
            "path": spec["path"],
            "sha256": current_hash,
            "size_bytes": fp.stat().st_size if fp.exists() else 0,
            "matches_reference": match,
            "status": "PASS_PRESERVED" if match else "FAIL_MODIFIED",
        }

    phase2_artifacts = {
        "phase2_improved_best_checkpoint": {
            "path": str(output_checkpoint.resolve().relative_to(repo_root.resolve())),
            "sha256": compute_sha256(output_checkpoint),
            "size_bytes": output_checkpoint.stat().st_size if output_checkpoint.exists() else 0,
        },
        "phase2_improved_latest_checkpoint": {
            "path": str(latest_checkpoint.resolve().relative_to(repo_root.resolve())),
            "sha256": compute_sha256(latest_checkpoint),
            "size_bytes": latest_checkpoint.stat().st_size if latest_checkpoint.exists() else 0,
        },
    }

    chk_audit = {
        "status": "PASS" if all_protected_pass else "FAIL",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protected_checkpoints_integrity": integrity_audit,
        "new_phase2_improved_checkpoints": phase2_artifacts,
        "zero_overwrites_verified": all_protected_pass,
    }
    (reports_dir / "phase2_checkpoint_integrity.json").write_text(
        json.dumps(chk_audit, indent=2), encoding="utf-8"
    )
    logger.info("Generated all 4 Phase 2 diagnostic reports in %s", reports_dir)



def main() -> None:
    parser = argparse.ArgumentParser(description="Train AVT²-DWF Deepfake Detection Model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/configs/default.yaml"),
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("storage/manifest.csv"),
        help="Path to dataset manifest CSV",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("app/ml/weights/avt2_dwf.pt"),
        help="Target output checkpoint path",
    )
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 2-step synthetic smoke training loop and save checkpoint",
    )

    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    logger.info("Using device: %s", device)

    if args.smoke:
        run_smoke_training(args.output_checkpoint, device)
        sys.exit(0)

    cfg = load_config(args.config)
    if args.epochs:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        cfg.setdefault("training", {})["lr"] = args.lr

    train_model(
        config=cfg,
        manifest_path=args.manifest,
        output_checkpoint=args.output_checkpoint,
        device=device,
    )


if __name__ == "__main__":
    main()
