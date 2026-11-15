"""Offline training script for the AVT²-DWF multimodal deepfake detection model.

Supports full training with AdamW, CosineAnnealingLR, class-balanced sampling,
early stopping on validation AUC, and a --smoke mode for fast verification (README §5, §8.7).
"""

from __future__ import annotations

import argparse
import logging
import sys
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
    max_steps: int | None = None,
) -> tuple[float, float]:
    """Train for one epoch, returning (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        video_windows = batch["video_windows"].to(device)
        audio_segments = batch["audio_segments"].to(device)
        audio_mask = batch["audio_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        out: AVT2Result = model(video_windows, audio_segments, audio_mask)
        loss = criterion(out.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        preds = torch.argmax(out.logits, dim=-1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    avg_loss = total_loss / max(1, step + 1 if max_steps else len(loader))
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def evaluate_epoch(
    model: AVT2DWF,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    max_steps: int | None = None,
) -> tuple[float, float, float]:
    """Evaluate on validation set, returning (avg_loss, accuracy, auc)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_scores: list[float] = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break

            video_windows = batch["video_windows"].to(device)
            audio_segments = batch["audio_segments"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            labels = batch["labels"].to(device)

            out: AVT2Result = model(video_windows, audio_segments, audio_mask)
            loss = criterion(out.logits, labels)

            total_loss += float(loss.item())
            preds = torch.argmax(out.logits, dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().tolist()
            y_scores.extend(probs)
            y_true.extend(labels.cpu().tolist())

    steps_count = max(1, step + 1 if max_steps else len(loader))
    avg_loss = total_loss / steps_count
    accuracy = correct / max(1, total)
    auc = compute_auc(y_true, y_scores)
    return avg_loss, accuracy, auc


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
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, criterion, device, max_steps=2
    )
    logger.info("Smoke train step complete: loss=%.4f, acc=%.2f", train_loss, train_acc)

    logger.info("Running 1 smoke validation step...")
    val_loss, val_acc, val_auc = evaluate_epoch(model, val_loader, criterion, device, max_steps=1)
    logger.info(
        "Smoke validation complete: loss=%.4f, acc=%.2f, auc=%.2f",
        val_loss,
        val_acc,
        val_auc,
    )

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_checkpoint)
    logger.info("Smoke checkpoint saved successfully to %s", output_checkpoint)


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
    epochs = args.epochs or int(cfg.get("training", {}).get("epochs", 20))
    batch_size = args.batch_size or int(cfg.get("training", {}).get("batch_size", 8))
    lr = args.lr or float(cfg.get("training", {}).get("lr", 1e-4))
    weight_decay = float(cfg.get("training", {}).get("weight_decay", 0.01))
    patience = int(cfg.get("training", {}).get("patience", 5))

    logger.info(
        "Starting AVT²-DWF training: epochs=%d, batch_size=%d, lr=%.6f",
        epochs,
        batch_size,
        lr,
    )

    augmenter = MultimodalAugmenter()
    train_dataset = MultimodalDataset(args.manifest, split="train", augmenter=augmenter)
    val_dataset = MultimodalDataset(args.manifest, split="val", augmenter=None)

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
    model = AVT2DWF(pretrained_vit=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_auc = -1.0
    stagnant_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %02d/%02d | Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f AUC: %.4f",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_auc,
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            stagnant_epochs = 0
            args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output_checkpoint)
            logger.info("Saved new best checkpoint with Val AUC: %.4f", best_val_auc)
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    patience,
                )
                break

    logger.info("Training complete. Best Val AUC: %.4f", best_val_auc)


if __name__ == "__main__":
    main()
