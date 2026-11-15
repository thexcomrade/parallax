"""Model evaluation, ablation studies, and model card generation.

Computes classification metrics, cross-dataset evaluations,
static fusion ablation, and writes live metrics to model_card.json (README §5, §8.7).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from app.ml.models.avt2_dwf import AVT2DWF, AVT2Result
from training.dataset import MultimodalDataset, multimodal_collate_fn
from training.manifest import build_smoke_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("parallax.training.evaluate")


def compute_metrics(
    y_true: list[int],
    y_scores: list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute Accuracy, Precision, Recall, F1, AUC, and Confusion Matrix."""
    if not y_true:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.5,
            "confusion_matrix": [[0, 0], [0, 0]],
            "total_samples": 0,
        }

    y_pred = [1 if s >= threshold else 0 for s in y_scores]

    tp = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == 1 and yp == 0)

    total = len(y_true)
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * (precision * recall) / max(1e-8, precision + recall)

    # AUC calculation
    if len(set(y_true)) < 2:
        auc = 0.5
    else:
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            pos_scores = [s for y, s in zip(y_true, y_scores, strict=False) if y == 1]
            neg_scores = [s for y, s in zip(y_true, y_scores, strict=False) if y == 0]
            n_pos = len(pos_scores)
            n_neg = len(neg_scores)
            if n_pos == 0 or n_neg == 0:
                auc = 0.5
            else:
                correct = sum(1.0 for p in pos_scores for n in neg_scores if p > n)
                ties = sum(0.5 for p in pos_scores for n in neg_scores if p == n)
                auc = float((correct + ties) / (n_pos * n_neg))

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "total_samples": total,
    }


def forward_static_fusion(
    model: AVT2DWF,
    video_windows: Tensor,
    audio_segments: Tensor,
    audio_mask: Tensor,
) -> Tensor:
    """Ablation forward pass: replace DWF dynamic gating with static equal weights (0.5/0.5)."""
    B, K, _T, _C, _H, _W = video_windows.shape

    # 1. Face encoder
    video_flat = video_windows.reshape(B * K, _T, _C, _H, _W)
    V_flat, _ = model.face_encoder(video_flat)
    V_clip = V_flat.reshape(B, K, -1).mean(dim=1)  # [B, 256]

    # 2. Audio encoder
    A_clip, _ = model.audio_encoder(audio_segments)  # [B, 256]

    # 3. Static fusion: w_a = 0.5, w_v = 0.5 (or w_a = 0.0 if audio masked)
    if audio_mask.ndim > 1:
        audio_present = audio_mask.any(dim=-1, keepdim=True).float()  # [B, 1]
    else:
        audio_present = audio_mask.any().float().unsqueeze(0).expand(B, 1)

    w_a = 0.5 * audio_present
    w_v = 1.0 - w_a
    fused = w_a * A_clip + w_v * V_clip

    # 4. Classifier head
    logits_raw: Tensor = model.classifier(fused)
    temperature = model.log_temperature.exp().clamp(min=0.5, max=5.0)
    logits: Tensor = logits_raw / temperature
    return logits


def run_evaluation(
    model: AVT2DWF,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    static_fusion: bool = False,
) -> tuple[dict[str, Any], list[float], list[int]]:
    """Run evaluation loop across a DataLoader, yielding metrics."""
    model.eval()
    y_true: list[int] = []
    y_scores: list[float] = []

    with torch.no_grad():
        for batch in loader:
            video_windows = batch["video_windows"].to(device)
            audio_segments = batch["audio_segments"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            labels: Tensor = batch["labels"]

            if static_fusion:
                logits = forward_static_fusion(model, video_windows, audio_segments, audio_mask)
            else:
                out: AVT2Result = model(video_windows, audio_segments, audio_mask)
                logits = out.logits

            probs = F.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            y_scores.extend(probs)
            y_true.extend(labels.tolist())

    metrics = compute_metrics(y_true, y_scores)
    return metrics, y_scores, y_true


def update_model_card(
    card_path: Path,
    checkpoint_path: Path,
    live_metrics: dict[str, Any],
) -> None:
    """Update weights/model_card.json with live evaluation results and provenance hash."""
    card_data: dict[str, Any] = {}
    if card_path.exists():
        try:
            with open(card_path, encoding="utf-8") as f:
                card_data = json.load(f)
        except Exception:
            card_data = {}

    sha = ""
    if checkpoint_path.exists():
        sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:16]

    card_data["model_mode"] = "trained" if checkpoint_path.exists() else "untrained"
    card_data["checkpoint_hash"] = sha or None
    card_data["live_metrics"] = live_metrics
    card_data["evaluated_at"] = datetime.now(UTC).isoformat()

    card_path.parent.mkdir(parents=True, exist_ok=True)
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card_data, f, indent=2)

    logger.info("Updated model card at %s with live metrics and sha256:%s", card_path, sha)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AVT²-DWF Deepfake Detection Model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("app/ml/weights/avt2_dwf.pt"),
        help="Path to trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("storage/manifest.csv"),
        help="Path to test manifest CSV",
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        default=Path("app/ml/weights/model_card.json"),
        help="Path to model_card.json to update",
    )
    parser.add_argument(
        "--static-fusion",
        action="store_true",
        help="Run static fusion ablation (wa=0.5, wv=0.5) to isolate DWF performance",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Evaluate on synthetic smoke manifest",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    parser.add_argument("--batch-size", type=int, default=2)

    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    logger.info("Evaluation device: %s", device)

    # 1. Dataset setup
    if args.smoke:
        manifest_path = build_smoke_manifest(Path("storage"))
        eval_split = "val"
    else:
        manifest_path = args.manifest
        eval_split = "val"

    dataset = MultimodalDataset(manifest_path, split=eval_split, augmenter=None)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
    )

    # 2. Model setup
    model = AVT2DWF(pretrained_vit=False).to(device)
    if args.checkpoint.exists() and args.checkpoint.stat().st_size > 0:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        logger.info("Loaded weights from %s", args.checkpoint)
    else:
        logger.warning("No checkpoint found at %s — evaluating untrained weights.", args.checkpoint)

    # 3. Dynamic Weight Fusion Evaluation
    metrics_dwf, _, _ = run_evaluation(model, loader, device, static_fusion=False)

    # 4. Static Fusion Ablation Evaluation
    metrics_static, _, _ = run_evaluation(model, loader, device, static_fusion=True)
    dwf_gain = round(float(metrics_dwf["auc"]) - float(metrics_static["auc"]), 4)

    # Print evaluation summary
    print("\n" + "=" * 60)
    print("         PARALLAX AVT²-DWF EVALUATION REPORT")
    print("=" * 60)
    print(f"Dataset Split:       {eval_split} ({metrics_dwf['total_samples']} samples)")
    print(f"Checkpoint:          {args.checkpoint.name}")
    print("-" * 60)
    print(f"DWF Accuracy:        {metrics_dwf['accuracy']:.4f}")
    print(f"DWF Precision:       {metrics_dwf['precision']:.4f}")
    print(f"DWF Recall:          {metrics_dwf['recall']:.4f}")
    print(f"DWF F1-Score:        {metrics_dwf['f1']:.4f}")
    print(f"DWF AUC-ROC:         {metrics_dwf['auc']:.4f}")
    print(
        f"Confusion Matrix:    TN={metrics_dwf['confusion_matrix'][0][0]}, "
        f"FP={metrics_dwf['confusion_matrix'][0][1]}, "
        f"FN={metrics_dwf['confusion_matrix'][1][0]}, "
        f"TP={metrics_dwf['confusion_matrix'][1][1]}"
    )
    print("-" * 60)
    print("ABLATION STUDY: Dynamic Weight Fusion vs Static Equal Fusion")
    print(f"Static Fusion AUC:   {metrics_static['auc']:.4f} (wa=0.5, wv=0.5)")
    print(f"Dynamic DWF AUC:     {metrics_dwf['auc']:.4f}")
    print(f"DWF Net AUC Gain:    {dwf_gain:+.4f}")
    print("=" * 60 + "\n")

    # Combine into live_metrics dict
    chosen_metrics = metrics_static if args.static_fusion else metrics_dwf
    live_metrics_record = {
        "accuracy": chosen_metrics["accuracy"],
        "precision": chosen_metrics["precision"],
        "recall": chosen_metrics["recall"],
        "f1": chosen_metrics["f1"],
        "auc": chosen_metrics["auc"],
        "confusion_matrix": chosen_metrics["confusion_matrix"],
        "static_fusion_auc": metrics_static["auc"],
        "dynamic_dwf_auc": metrics_dwf["auc"],
        "dwf_gain_over_static": dwf_gain,
        "mode": "static_fusion" if args.static_fusion else "dynamic_dwf",
    }

    update_model_card(args.model_card, args.checkpoint, live_metrics_record)


if __name__ == "__main__":
    main()
