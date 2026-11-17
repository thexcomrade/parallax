"""Offline training script for the AVT²-DWF multimodal deepfake detection model.

Supports full training with AdamW, CosineAnnealingLR, class-balanced sampling,
early stopping on validation AUC, and a --smoke mode for fast verification (README §5, §8.7).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

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
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


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
        res["audio_acc"] = float(sum(p == t for p, t in zip(y_audio_pred, y_audio_true, strict=False)) / len(y_audio_true))
        res["audio_f1"] = float(f1_score(y_audio_true, y_audio_pred, zero_division=0.0))
        res["audio_auc"] = compute_auc(y_audio_true, y_audio_scores)

    if y_video_true:
        res["video_acc"] = float(sum(p == t for p, t in zip(y_video_pred, y_video_true, strict=False)) / len(y_video_true))
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
    """Evaluate on validation set, returning dictionary of fused, unimodal, and fusion weight metrics."""
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
        "mean_w_a": mean_w_a,
        "mean_w_v": mean_w_v,
        "std_w_a": std_w_a,
        "std_w_v": std_w_v,
    }

    if y_audio_true:
        res["audio_acc"] = float(sum(p == t for p, t in zip(y_audio_pred, y_audio_true, strict=False)) / len(y_audio_true))
        res["audio_prec"] = float(precision_score(y_audio_true, y_audio_pred, zero_division=0.0))
        res["audio_rec"] = float(recall_score(y_audio_true, y_audio_pred, zero_division=0.0))
        res["audio_f1"] = float(f1_score(y_audio_true, y_audio_pred, zero_division=0.0))
        res["audio_auc"] = compute_auc(y_audio_true, y_audio_scores)
        res["audio_balanced_acc"] = float(balanced_accuracy_score(y_audio_true, y_audio_pred))

    if y_video_true:
        res["video_acc"] = float(sum(p == t for p, t in zip(y_video_pred, y_video_true, strict=False)) / len(y_video_true))
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
    logger.info("Smoke train step complete: loss=%.4f, acc=%.2f", train_res["loss"], train_res["accuracy"])

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


def train_model(
    config: dict[str, Any],
    manifest_path: Path,
    output_checkpoint: Path,
    device: torch.device,
    resume_checkpoint: Path | None = None,
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
    mismatch_prob = float(cfg_train.get("mismatch_prob", 0.0))
    checkpoint_metric = str(cfg_train.get("checkpoint_metric", "val_auc"))

    effective_batch_size = batch_size * grad_accum_steps
    logger.info(
        "Starting AVT²-DWF training: epochs=%d, batch=%d, accum=%d (eff=%d), lr=%.6f, dev=%s",
        epochs,
        batch_size,
        grad_accum_steps,
        effective_batch_size,
        lr,
        device,
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

    val_manifest = manifest_path
    if manifest_path.name == "train.csv":
        val_candidate = manifest_path.parent / "validation.csv"
        if val_candidate.exists():
            val_manifest = val_candidate

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
        auxiliary_heads=auxiliary_heads,
        classifier_dropout=classifier_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    start_epoch = 1
    best_val_f1 = -1.0
    best_val_auc = -1.0
    best_val_loss = float("inf")
    best_epoch = 1
    stagnant_epochs = 0
    history: list[dict[str, Any]] = []

    if resume_checkpoint and resume_checkpoint.exists():
        logger.info("Resuming training from checkpoint: %s", resume_checkpoint)
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        completed_epoch = int(ckpt.get("epoch", 0))
        start_epoch = completed_epoch + 1
        best_val_auc = float(ckpt.get("val_auc", -1.0))
        best_val_f1 = float(ckpt.get("val_f1", -1.0))
        best_val_loss = float(ckpt.get("val_loss", float("inf")))
        best_epoch = int(ckpt.get("epoch", 1))

        # Reconstruct history for completed epochs if available
        repo_root = Path(__file__).resolve().parent.parent.parent
        hist_file = repo_root / "reports" / "phase4_training_history.csv"
        if hist_file.exists():
            import csv

            with open(hist_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if int(r["epoch"]) <= completed_epoch:
                        history.append({k: (float(v) if k != "epoch" else int(v)) for k, v in r.items()})

        if len(history) < completed_epoch:
            log_task_file = Path("C:/Users/deva5/.gemini/antigravity/brain/d55fd788-3822-4b05-a697-412e0c065c58/.system_generated/tasks/task-11651.log")
            if log_task_file.exists():
                import re

                log_text = log_task_file.read_text(encoding="utf-8", errors="ignore")
                pattern = re.compile(
                    r"Epoch (\d+)/(\d+) \((\d+\.\d+)s\) \| Train Loss: (\d+\.\d+) \(Fused: (\d+\.\d+), A: (\d+\.\d+), V: (\d+\.\d+)\) Acc: (\d+\.\d+) F1: (\d+\.\d+) AUC: (\d+\.\d+) \| Val Loss: (\d+\.\d+) Acc: (\d+\.\d+) F1: (\d+\.\d+) AUC: (\d+\.\d+) BAcc: (\d+\.\d+) \| Val Unimodal - Audio AUC: (\d+\.\d+) \(F1: (\d+\.\d+)\), Video AUC: (\d+\.\d+) \(F1: (\d+\.\d+)\) \| Weights - w_a: (\d+\.\d+)±(\d+\.\d+), w_v: (\d+\.\d+)±(\d+\.\d+) \| Grads - \|\|g_A\|\|: (\d+\.\d+), \|\|g_V\|\|: (\d+\.\d+) \| LR: ([\d\.e\-\+]+) \| GPU: (\d+\.\d+) MB"
                )
                seen_epochs: set[int] = set()
                for match in pattern.finditer(log_text):
                    ep = int(match.group(1))
                    if ep <= completed_epoch and ep not in seen_epochs:
                        seen_epochs.add(ep)
                        history.append({
                            "epoch": ep,
                            "train_loss": float(match.group(4)),
                            "train_loss_fused": float(match.group(5)),
                            "train_loss_audio": float(match.group(6)),
                            "train_loss_video": float(match.group(7)),
                            "train_acc": float(match.group(8)),
                            "train_prec": 0.0,
                            "train_rec": 0.0,
                            "train_f1": float(match.group(9)),
                            "train_auc": float(match.group(10)),
                            "val_loss": float(match.group(11)),
                            "val_loss_fused": float(match.group(11)),
                            "val_loss_audio": 0.0,
                            "val_loss_video": 0.0,
                            "val_acc": float(match.group(12)),
                            "val_prec": 0.0,
                            "val_rec": 0.0,
                            "val_f1": float(match.group(13)),
                            "val_auc": float(match.group(14)),
                            "val_balanced_acc": float(match.group(15)),
                            "val_audio_acc": 0.0,
                            "val_audio_f1": float(match.group(17)),
                            "val_audio_auc": float(match.group(16)),
                            "val_video_acc": 0.0,
                            "val_video_f1": float(match.group(19)),
                            "val_video_auc": float(match.group(18)),
                            "mean_w_a": float(match.group(20)),
                            "mean_w_v": float(match.group(22)),
                            "std_w_a": float(match.group(21)),
                            "std_w_v": float(match.group(23)),
                            "grad_norm_audio": float(match.group(24)),
                            "grad_norm_video": float(match.group(25)),
                            "grad_norm_dwf": 0.0,
                            "lr": float(match.group(26)),
                            "duration_s": float(match.group(3)),
                            "gpu_mem_mb": float(match.group(27)),
                        })

        logger.info(
            "Resuming at epoch %d/%d (completed=%d, preloaded history=%d epochs)",
            start_epoch,
            epochs,
            completed_epoch,
            len(history),
        )

    is_phase4 = "phase4" in output_checkpoint.name
    latest_checkpoint = (
        output_checkpoint.parent / "phase4_latest_avt2_dwf.pt"
        if is_phase4
        else output_checkpoint.parent / "latest_avt2_dwf.pt"
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
            "val_audio_acc": round(val_res.get("audio_acc", 0.0), 4),
            "val_audio_f1": round(val_res.get("audio_f1", 0.0), 4),
            "val_audio_auc": round(val_res.get("audio_auc", 0.0), 4),
            "val_video_acc": round(val_res.get("video_acc", 0.0), 4),
            "val_video_f1": round(val_res.get("video_f1", 0.0), 4),
            "val_video_auc": round(val_res.get("video_auc", 0.0), 4),
            "mean_w_a": round(val_res.get("mean_w_a", 0.5), 4),
            "mean_w_v": round(val_res.get("mean_w_v", 0.5), 4),
            "std_w_a": round(val_res.get("std_w_a", 0.0), 4),
            "std_w_v": round(val_res.get("std_w_v", 0.0), 4),
            "grad_norm_audio": round(train_res.get("grad_norm_audio", 0.0), 4),
            "grad_norm_video": round(train_res.get("grad_norm_video", 0.0), 4),
            "grad_norm_dwf": round(train_res.get("grad_norm_dwf", 0.0), 4),
            "lr": round(current_lr, 7),
            "duration_s": epoch_duration,
            "gpu_mem_mb": gpu_mem_mb,
        }
        history.append(record)

        logger.info(
            "Epoch %02d/%02d (%4.1fs) | Train Loss: %.4f (Fused: %.4f, A: %.4f, V: %.4f) Acc: %.4f F1: %.4f AUC: %.4f | "
            "Val Loss: %.4f Acc: %.4f F1: %.4f AUC: %.4f BAcc: %.4f | "
            "Val Unimodal - Audio AUC: %.4f (F1: %.4f), Video AUC: %.4f (F1: %.4f) | "
            "Weights - w_a: %.4f±%.4f, w_v: %.4f±%.4f | Grads - ||g_A||: %.4f, ||g_V||: %.4f | LR: %.2e | GPU: %s MB",
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
            "config": cfg,
            "manifest_path": str(manifest_path),
            "device": str(device),
        }

        # Always save latest candidate checkpoint (phase4_latest_avt2_dwf.pt)
        torch.save(checkpoint_state, latest_checkpoint)

        val_auc = val_res["auc"]
        val_f1 = val_res["f1"]
        val_loss = val_res["loss"]

        # Determine if checkpoint improved according to configured primary metric
        if checkpoint_metric == "val_auc":
            improved = (
                (val_auc > best_val_auc + 1e-4)
                or (abs(val_auc - best_val_auc) <= 1e-4 and val_f1 > best_val_f1 + 1e-4)
                or (
                    abs(val_auc - best_val_auc) <= 1e-4
                    and abs(val_f1 - best_val_f1) <= 1e-4
                    and val_loss < best_val_loss
                )
            )
        else:
            improved = (val_f1 > best_val_f1 + 1e-4) or (
                abs(val_f1 - best_val_f1) <= 1e-4 and val_loss < best_val_loss
            )

        if improved:
            best_val_f1 = val_f1
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_epoch = epoch
            stagnant_epochs = 0
            torch.save(checkpoint_state, output_checkpoint)
            logger.info(
                "Saved new best checkpoint to %s (Epoch %d | Val AUC: %.4f, Val F1: %.4f, Val Loss: %.4f)",
                output_checkpoint,
                best_epoch,
                best_val_auc,
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

    # Save training history to CSV (datasets/reports and reports paths)
    repo_root = Path(__file__).resolve().parent.parent.parent
    reports_dir = repo_root / "datasets" / "reports"
    root_reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    root_reports_dir.mkdir(parents=True, exist_ok=True)

    history_filename = "phase4_training_history.csv" if is_phase4 else "training_history.csv"
    report_filename = "phase4_training_report.json" if is_phase4 else "training_report.json"

    csv_paths = [
        reports_dir / history_filename,
        root_reports_dir / history_filename,
    ]
    if not is_phase4:
        csv_paths.append(reports_dir / "phase2_training_history.csv")
        csv_paths.append(root_reports_dir / "phase2_training_history.csv")

    fieldnames = list(history[0].keys()) if history else [
        "epoch", "train_loss", "train_loss_fused", "train_loss_audio", "train_loss_video",
        "train_acc", "train_prec", "train_rec", "train_f1", "train_auc",
        "val_loss", "val_loss_fused", "val_loss_audio", "val_loss_video",
        "val_acc", "val_prec", "val_rec", "val_f1", "val_auc", "val_balanced_acc",
        "val_audio_acc", "val_audio_f1", "val_audio_auc",
        "val_video_acc", "val_video_f1", "val_video_auc",
        "mean_w_a", "mean_w_v", "std_w_a", "std_w_v",
        "grad_norm_audio", "grad_norm_video", "grad_norm_dwf",
        "lr", "duration_s", "gpu_mem_mb",
    ]
    for p in csv_paths:
        with open(p, "w", newline="", encoding="utf-8") as f:
            import csv

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        logger.info("Saved training history to %s", p)

    training_report = {
        "status": "COMPLETED",
        "phase": 4 if is_phase4 else 2,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_auc": best_val_auc,
        "best_val_loss": best_val_loss,
        "epochs_completed": len(history),
        "best_checkpoint": str(output_checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
        "training_config": {
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
            "mismatch_prob": mismatch_prob,
            "checkpoint_metric": checkpoint_metric,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
        },
        "history": history,
    }

    report_json_paths = [
        reports_dir / report_filename,
        root_reports_dir / report_filename,
    ]
    if not is_phase4:
        report_json_paths.append(reports_dir / "phase2_training_report.json")
        report_json_paths.append(root_reports_dir / "phase2_training_report.json")

    for rp in report_json_paths:
        with open(rp, "w", encoding="utf-8") as f:
            import json

            json.dump(training_report, f, indent=2)
        logger.info("Saved training report to %s", rp)

    logger.info(
        "Training complete. Best Val AUC: %.4f, F1: %.4f, Loss: %.4f at Epoch %d",
        best_val_auc,
        best_val_f1,
        best_val_loss,
        best_epoch,
    )
    return training_report



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
