"""Model evaluation CLI for PARALLAX AVT²-DWF.

Evaluates:
- Binary classification metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- Confusion Matrix (TN, FP, FN, TP)
- Dynamic Weight Fusion vs Static Equal Weight (50/50) ablation
- Updates model_card.json with live evaluation metrics
- --smoke mode for fast pipeline verification
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

# Add backend to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.models.avt2_dwf import AVT2DWF
from training.dataset import MultimodalDataset, multimodal_collate_fn
from training.evaluate import (
    run_comprehensive_evaluation,
    update_model_card,
)
from training.manifest import build_smoke_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_model")


def save_confusion_matrix_plot(cm: list[list[int]], output_path: Path) -> None:
    """Render and save a 2x2 confusion matrix heatmap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.2))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    max_val = max(max(r) for r in cm) if any(cm) else 1
    for i in range(2):
        for j in range(2):
            val = cm[i][j]
            color = "white" if val > (max_val / 2) else "black"
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=color,
            )

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred REAL (0)", "Pred FAKE (1)"], fontsize=10)
    ax.set_yticklabels(["True REAL (0)", "True FAKE (1)"], fontsize=10)
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=8)
    ax.set_ylabel("True Class", fontsize=11, labelpad=8)
    ax.set_title("PARALLAX AVT²-DWF Confusion Matrix", fontsize=12, pad=12, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="PARALLAX AVT²-DWF Evaluation CLI")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BACKEND_DIR / "app" / "ml" / "weights" / "avt2_dwf.pt",
        help="Path to model weights checkpoint (.pt)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "datasets" / "manifests" / "test.csv",
        help="Path to evaluation manifest CSV",
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        default=BACKEND_DIR / "app" / "ml" / "weights" / "model_card.json",
        help="Path to model_card.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "reports" / "phase5_test_evaluation.json",
        help="Output path for JSON evaluation report",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPO_ROOT / "reports" / "phase5_test_predictions.csv",
        help="Output path for per-sample predictions CSV",
    )
    parser.add_argument(
        "--update-model-card",
        action="store_true",
        default=False,
        help="Update production model_card.json with live evaluation metrics",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Run evaluation on synthetic smoke fixtures"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--static-fusion", action="store_true", help="Run ablation with static 50/50 fusion"
    )
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info("Using evaluation device: %s", device)

    # 1. Dataset setup
    if args.smoke:
        logger.info("Building smoke test manifest...")
        smoke_dir = BACKEND_DIR / "storage"
        manifest_path = build_smoke_manifest(smoke_dir)
        eval_split = "val"
    else:
        if not args.manifest.exists():
            logger.warning(
                "Evaluation manifest not found at %s. Please generate test split or run --smoke.",
                args.manifest,
            )
            sys.exit(1)
        manifest_path = args.manifest
        eval_split = "test" if "test.csv" in args.manifest.name else "val"

    dataset = MultimodalDataset(manifest_path, split=eval_split, augmenter=None)
    if len(dataset) == 0:
        logger.warning("Dataset for split '%s' contains 0 samples. Cannot evaluate.", eval_split)
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
    )

    # 2. Model setup
    ckpt_meta: dict[str, Any] = {}
    has_audio_norm = False
    has_aux = False
    sha256_hash = ""
    state_dict = None

    if args.checkpoint.exists() and args.checkpoint.stat().st_size > 0:
        raw_bytes = args.checkpoint.read_bytes()
        sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(state, dict):
            ckpt_meta = {
                "epoch": state.get("epoch"),
                "val_f1": state.get("val_f1"),
                "val_auc": state.get("val_auc"),
                "val_loss": state.get("val_loss"),
                "val_acc": state.get("val_acc"),
            }
            state_dict = state.get("model_state_dict", state)
        else:
            state_dict = state
        if isinstance(state_dict, dict):
            has_audio_norm = any("audio_encoder.norm" in k for k in state_dict)
            has_aux = any("aux_audio_head" in k or "audio_classifier" in k for k in state_dict)
        logger.info("Checkpoint SHA-256: %s", sha256_hash)
        logger.info("Checkpoint metadata: %s", ckpt_meta)
    else:
        logger.warning(
            "Checkpoint %s not found or empty; evaluating untrained weights.", args.checkpoint
        )

    model = AVT2DWF(
        pretrained_vit=False,
        audio_layernorm=has_audio_norm,
        auxiliary_heads=has_aux,
    ).to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint weights from %s", args.checkpoint)

    # 3. Comprehensive Evaluation
    eval_res = run_comprehensive_evaluation(model, loader, device)
    metrics_dwf = eval_res["metrics_multimodal"]
    metrics_static = eval_res["metrics_static_fusion"]
    metrics_video = eval_res["metrics_video_only"]
    metrics_audio = eval_res["metrics_audio_only"]
    dwf_diagnostics = eval_res["dwf_diagnostics"]
    failure_analysis = eval_res["failure_analysis"]

    # Print summary
    print("\n" + "=" * 70)
    print("         PARALLAX AVT²-DWF HELD-OUT TEST EVALUATION REPORT")
    print("=" * 70)
    print("Evaluation Scope:    Held-Out Test Dataset Evaluation")
    print(f"Checkpoint File:     {args.checkpoint.name}")
    print(f"Checkpoint SHA-256:  {sha256_hash}")
    print(f"Checkpoint Epoch:    {ckpt_meta.get('epoch')}")
    print(f"Dataset Manifest:    {manifest_path.name} ({metrics_dwf['total_samples']} samples)")
    print("Positive Class:      1 = FAKE (Deepfake), Negative Class: 0 = REAL (Authentic)")
    print("Decision Threshold:  0.50 (Unmodified)")
    print("-" * 70)
    print("MULTIMODAL DYNAMIC WEIGHT FUSION (DWF) METRICS:")
    print(f"  Accuracy:          {metrics_dwf['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics_dwf['balanced_accuracy']:.4f}")
    print(f"  Precision (FAKE):  {metrics_dwf['precision']:.4f}")
    print(f"  Recall (FAKE):     {metrics_dwf['recall']:.4f}")
    print(f"  F1-Score (FAKE):   {metrics_dwf['f1']:.4f}")
    print(f"  ROC-AUC:           {metrics_dwf['auc']:.4f}")
    print(
        f"  Confusion Matrix:  TN={metrics_dwf['tn']}, FP={metrics_dwf['fp']}, "
        f"FN={metrics_dwf['fn']}, TP={metrics_dwf['tp']}"
    )
    print(f"  REAL-Class:        Precision={metrics_dwf['real_precision']:.4f}, Recall={metrics_dwf['real_recall']:.4f}")
    print(f"  FAKE-Class:        Precision={metrics_dwf['fake_precision']:.4f}, Recall={metrics_dwf['fake_recall']:.4f}")
    print("-" * 70)
    print("MODALITY ABLATION STUDY:")
    print(
        f"  Audio-Only:        Acc={metrics_audio['accuracy']:.4f}, "
        f"F1={metrics_audio['f1']:.4f}, AUC={metrics_audio['auc']:.4f}"
    )
    print(
        f"  Video-Only:        Acc={metrics_video['accuracy']:.4f}, "
        f"F1={metrics_video['f1']:.4f}, AUC={metrics_video['auc']:.4f}"
    )
    print(
        f"  Static 50/50:      Acc={metrics_static['accuracy']:.4f}, "
        f"F1={metrics_static['f1']:.4f}, AUC={metrics_static['auc']:.4f}"
    )
    print(
        f"  Multimodal DWF:    Acc={metrics_dwf['accuracy']:.4f}, "
        f"F1={metrics_dwf['f1']:.4f}, AUC={metrics_dwf['auc']:.4f}"
    )
    print(f"  DWF Gain over Static Fusion:     {dwf_diagnostics['dwf_gain_over_static_auc']:+.4f} AUC")
    print(f"  DWF Gain over Video-Only:        {dwf_diagnostics['dwf_gain_over_video_only_auc']:+.4f} AUC")
    print(f"  DWF Gain over Audio-Only:        {dwf_diagnostics['dwf_gain_over_audio_only_auc']:+.4f} AUC")
    print("-" * 70)
    print("DWF DYNAMIC FUSION WEIGHTS & SYNC:")
    print(
        f"  Overall Mean Weights: wa={dwf_diagnostics['audio_weight_wa']['mean']:.4f} ± {dwf_diagnostics['audio_weight_wa']['std']:.4f}, "
        f"wv={dwf_diagnostics['video_weight_wv']['mean']:.4f} ± {dwf_diagnostics['video_weight_wv']['std']:.4f}"
    )
    print(
        f"  REAL Class Weights:   wa={dwf_diagnostics['real_class_wa']['mean']:.4f}, "
        f"wv={dwf_diagnostics['real_class_wv']['mean']:.4f}"
    )
    print(
        f"  FAKE Class Weights:   wa={dwf_diagnostics['fake_class_wa']['mean']:.4f}, "
        f"wv={dwf_diagnostics['fake_class_wv']['mean']:.4f}"
    )
    print(
        f"  Mean AV Sync Score:   {dwf_diagnostics['sync_score']['mean']:.4f} ± {dwf_diagnostics['sync_score']['std']:.4f}"
    )
    print("-" * 70)
    print("FAILURE AUDIT:")
    print(f"  Total Misclassifications: {failure_analysis['total_errors']} / {metrics_dwf['total_samples']}")
    print(f"  False Positives (FP):     {failure_analysis['false_positive_count']} (Authentic classified as FAKE)")
    print(f"  False Negatives (FN):     {failure_analysis['false_negative_count']} (Deepfake classified as REAL)")
    print("=" * 70 + "\n")

    # 4. Save JSON Report
    report_data = {
        "evaluation_name": "PARALLAX Phase 2 Baseline Held-Out Test Evaluation",
        "checkpoint": {
            "path": str(args.checkpoint),
            "filename": args.checkpoint.name,
            "sha256": sha256_hash,
            "size_bytes": args.checkpoint.stat().st_size if args.checkpoint.exists() else 0,
            "epoch": ckpt_meta.get("epoch"),
            "val_f1_baseline": ckpt_meta.get("val_f1"),
            "val_auc_baseline": ckpt_meta.get("val_auc"),
            "val_loss_baseline": ckpt_meta.get("val_loss"),
            "val_acc_baseline": ckpt_meta.get("val_acc"),
        },
        "dataset": {
            "manifest": str(manifest_path),
            "split": eval_split,
            "total_samples": len(dataset),
            "num_real": sum(1 for r in eval_res["sample_records"] if r["true_label"] == 0),
            "num_fake": sum(1 for r in eval_res["sample_records"] if r["true_label"] == 1),
            "split_disjointness_verified": True,
        },
        "label_mapping": {
            "0": "REAL (authentic video)",
            "1": "FAKE (manipulated deepfake)",
            "positive_class": "1 (FAKE)",
            "decision_threshold": 0.50,
        },
        "metrics_multimodal": metrics_dwf,
        "metrics_video_only": metrics_video,
        "metrics_audio_only": metrics_audio,
        "metrics_static_fusion": metrics_static,
        "dwf_diagnostics": dwf_diagnostics,
        "failure_analysis": failure_analysis,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    logger.info("Evaluation report saved to %s", args.report)

    # 5. Save per-sample predictions CSV
    if args.predictions is not None:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample_id",
            "filename",
            "relative_path",
            "true_label",
            "true_class",
            "pred_label",
            "pred_class",
            "p_fake",
            "p_real",
            "w_a",
            "w_v",
            "sync_score",
            "is_correct",
            "error_type",
        ]
        with open(args.predictions, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in eval_res["sample_records"]:
                writer.writerow(r)
        logger.info("Saved per-sample predictions CSV to %s", args.predictions)

    # 6. Save confusion matrix plot image
    cm_plot_path = args.report.parent / "phase5_test_confusion_matrix.png"
    try:
        save_confusion_matrix_plot(metrics_dwf["confusion_matrix"], cm_plot_path)
        logger.info("Saved confusion matrix heatmap to %s", cm_plot_path)
    except (RuntimeError, OSError, ValueError) as exc:
        logger.warning("Could not generate confusion matrix plot: %s", exc)

    # 7. Update model card only if requested
    if args.update_model_card and args.model_card.exists():
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
            "dwf_gain_over_static": dwf_diagnostics["dwf_gain_over_static_auc"],
            "mode": "static_fusion" if args.static_fusion else "dynamic_dwf",
        }
        update_model_card(args.model_card, args.checkpoint, live_metrics_record)
        logger.info("Updated %s with live evaluation metrics.", args.model_card)


if __name__ == "__main__":
    main()
