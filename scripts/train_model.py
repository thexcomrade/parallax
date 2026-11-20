"""Model training CLI for PARALLAX AVT²-DWF.

Supports:
- Offline training with AdamW and Cosine Annealing learning rate
- Multimodal data augmentation & cross-modal mismatch detection
- Class-balanced sampling
- Early stopping on validation AUC
- Atomic candidate checkpoint saving to checkpoints/candidate/
- --smoke mode for fast end-to-end pipeline verification
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Add backend to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.train import (
    compute_and_save_audio_norm_stats,
    load_config,
    run_smoke_training,
    train_model,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_model")


def main() -> None:
    parser = argparse.ArgumentParser(description="PARALLAX AVT²-DWF Training Script")
    parser.add_argument(
        "--config",
        type=Path,
        default=BACKEND_DIR / "training" / "configs" / "default.yaml",
        help="Path to training YAML configuration file",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "datasets" / "manifests" / "train.csv",
        help="Path to training manifest CSV",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "baseline_avt2_dwf.pt",
        help="Destination path for baseline model checkpoint",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast 2-step smoke test using synthetic fixtures",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--aux-audio-weight", type=float, default=0.3, help="Weight for auxiliary audio classification loss")
    parser.add_argument("--aux-video-weight", type=float, default=0.1, help="Weight for auxiliary video classification loss")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon for CrossEntropyLoss")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay regularizer")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout probability for classifier heads")
    parser.add_argument("--audio-layernorm", dest="audio_layernorm", action="store_true", default=True, help="Enable LayerNorm on audio patch embeddings")
    parser.add_argument("--no-audio-layernorm", dest="audio_layernorm", action="store_false", help="Disable LayerNorm on audio patch embeddings")
    parser.add_argument("--audio-input-norm", dest="audio_input_norm", action="store_true", default=True, help="Enable AudioInputNorm channel standardization")
    parser.add_argument("--no-audio-input-norm", dest="audio_input_norm", action="store_false", help="Disable AudioInputNorm channel standardization")
    parser.add_argument("--mismatch-prob", type=float, default=0.0, help="Probability of cross-modal mismatch augmentation")
    parser.add_argument("--checkpoint-metric", type=str, default="val_auc", choices=["val_auc", "val_f1"], help="Primary metric for checkpoint selection")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint from which to resume training")
    parser.add_argument("--val-manifest", type=Path, default=None, help="Path to validation manifest CSV")
    parser.add_argument("--audio-norm-stats", type=Path, default=None, help="Path to audio normalization statistics file (.pt)")
    parser.add_argument("--experiment-dir", type=Path, default=None, help="Directory for isolated experiment artifacts")
    parser.add_argument("--compute-audio-norm", action="store_true", default=False, help="Precompute audio normalization statistics from training manifest before training")
    parser.add_argument("--dwf-min-weight", type=float, default=0.0, help="Minimum weight bound for DWF fusion (prevents modality starvation)")
    parser.add_argument("--dwf-max-weight", type=float, default=1.0, help="Maximum weight bound for DWF fusion")
    parser.add_argument("--modality-dropout", type=float, default=0.0, help="Probability of modality dropout during training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info("Using compute device: %s", device)

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.experiment_dir is not None:
        args.experiment_dir.mkdir(parents=True, exist_ok=True)

    if args.compute_audio_norm:
        norm_stats_path = args.audio_norm_stats
        if norm_stats_path is None:
            if args.experiment_dir is not None:
                norm_stats_path = args.experiment_dir / "audio_norm_stats.pt"
            else:
                norm_stats_path = args.output_checkpoint.parent / "audio_norm_stats.pt"
        logger.info("Computing audio normalization statistics from %s -> %s", args.manifest, norm_stats_path)
        compute_and_save_audio_norm_stats(args.manifest, norm_stats_path)
        args.audio_norm_stats = norm_stats_path
    elif args.audio_norm_stats is None and args.experiment_dir is not None:
        candidate_norm = args.experiment_dir / "audio_norm_stats.pt"
        if candidate_norm.exists():
            args.audio_norm_stats = candidate_norm
            logger.info("Using existing audio normalization statistics from %s", candidate_norm)

    if args.smoke:
        logger.info("Starting AVT²-DWF SMOKE TRAINING verification...")
        run_smoke_training(args.output_checkpoint, device)
        logger.info(
            "[SUCCESS] Smoke training completed. Candidate checkpoint saved to %s",
            args.output_checkpoint,
        )
        sys.exit(0)

    # Full training mode
    if not args.manifest.exists() or args.manifest.stat().st_size == 0:
        logger.warning(
            "Training manifest not found or empty at %s.\n"
            "Cannot train without raw video data. Please place REAL samples in\n"
            "'datasets/raw/real/' and FAKE samples in 'datasets/raw/fake/',\n"
            "then run 'python scripts/prepare_dataset.py'.\n"
            "To verify the training code execution with synthetic data, run:\n"
            "  python scripts/train_model.py --smoke",
            args.manifest,
        )
        sys.exit(1)

    config = load_config(args.config)
    config.setdefault("training", {})
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        config["training"]["grad_accum_steps"] = args.grad_accum
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.patience is not None:
        config["training"]["patience"] = args.patience
    if args.aux_audio_weight is not None:
        config["training"]["aux_audio_weight"] = args.aux_audio_weight
    if args.aux_video_weight is not None:
        config["training"]["aux_video_weight"] = args.aux_video_weight
    if args.label_smoothing is not None:
        config["training"]["label_smoothing"] = args.label_smoothing
    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay
    if args.dropout is not None:
        config["training"]["classifier_dropout"] = args.dropout
    if args.audio_layernorm is not None:
        config["training"]["audio_layernorm"] = args.audio_layernorm
    if args.audio_input_norm is not None:
        config["training"]["audio_input_norm"] = args.audio_input_norm
    if args.mismatch_prob is not None:
        config["training"]["mismatch_prob"] = args.mismatch_prob
    if args.checkpoint_metric is not None:
        config["training"]["checkpoint_metric"] = args.checkpoint_metric
    if args.dwf_min_weight is not None:
        config["training"]["dwf_min_weight"] = args.dwf_min_weight
    if args.dwf_max_weight is not None:
        config["training"]["dwf_max_weight"] = args.dwf_max_weight
    if args.modality_dropout is not None:
        config["training"]["modality_dropout_prob"] = args.modality_dropout
    if args.seed is not None:
        config["training"]["seed"] = args.seed

    logger.info("Starting full model training with config: %s", args.config)
    summary = train_model(
        config=config,
        manifest_path=args.manifest,
        output_checkpoint=args.output_checkpoint,
        device=device,
        resume_checkpoint=args.resume,
        val_manifest_path=args.val_manifest,
        audio_norm_stats=args.audio_norm_stats,
        experiment_dir=args.experiment_dir,
    )
    summary_filename = f"{args.output_checkpoint.stem}_summary.json"
    summary_path = args.output_checkpoint.parent / summary_filename
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Training complete. Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
