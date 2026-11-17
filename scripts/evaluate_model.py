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
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add backend to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.models.avt2_dwf import AVT2DWF  # noqa: E402
from training.dataset import MultimodalDataset, multimodal_collate_fn  # noqa: E402
from training.evaluate import run_evaluation, update_model_card  # noqa: E402
from training.manifest import build_smoke_manifest  # noqa: E402

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
        default=REPO_ROOT / "datasets" / "reports" / "evaluation_report.json",
        help="Output path for JSON evaluation report",
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
    model = AVT2DWF(pretrained_vit=False).to(device)
    if args.checkpoint.exists() and args.checkpoint.stat().st_size > 0:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        logger.info("Loaded checkpoint weights from %s", args.checkpoint)
    else:
        logger.warning(
            "Checkpoint %s not found or empty; evaluating untrained weights.", args.checkpoint
        )

    # 3. Dynamic Weight Fusion Evaluation
    metrics_dwf, _, _ = run_evaluation(model, loader, device, static_fusion=False)

    # 4. Static Fusion Ablation Evaluation
    metrics_static, _, _ = run_evaluation(model, loader, device, static_fusion=True)
    dwf_gain = round(float(metrics_dwf["auc"]) - float(metrics_static["auc"]), 4)

    # Print summary
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

    report_data = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(manifest_path),
        "split": eval_split,
        "metrics_dwf": metrics_dwf,
        "metrics_static_fusion": metrics_static,
        "dwf_net_auc_gain": dwf_gain,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    logger.info("Evaluation report saved to %s", args.report)

    # Save confusion matrix plot image
    cm_plot_path = args.report.parent / "confusion_matrix.png"
    try:
        save_confusion_matrix_plot(metrics_dwf["confusion_matrix"], cm_plot_path)
        logger.info("Saved confusion matrix heatmap to %s", cm_plot_path)
    except Exception as exc:
        logger.warning("Could not generate confusion matrix plot: %s", exc)

    # Update model card if requested
    if args.model_card.exists():
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
        logger.info("Updated %s with live evaluation metrics.", args.model_card)


if __name__ == "__main__":
    main()
