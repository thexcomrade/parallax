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
    """Compute Accuracy, Precision, Recall, F1, AUC, Balanced Accuracy, and Confusion Matrix."""
    if not y_true:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.5,
            "balanced_accuracy": 0.5,
            "confusion_matrix": [[0, 0], [0, 0]],
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "real_precision": 0.0,
            "real_recall": 0.0,
            "fake_precision": 0.0,
            "fake_recall": 0.0,
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

    real_precision = tn / max(1, tn + fn)
    real_recall = tn / max(1, tn + fp)
    fake_precision = precision
    fake_recall = recall
    balanced_accuracy = 0.5 * (real_recall + fake_recall)

    # AUC calculation
    if len(set(y_true)) < 2:
        auc = 0.5
    else:
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

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
        "balanced_accuracy": round(balanced_accuracy, 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "real_precision": round(real_precision, 4),
        "real_recall": round(real_recall, 4),
        "fake_precision": round(fake_precision, 4),
        "fake_recall": round(fake_recall, 4),
        "total_samples": total,
    }


def forward_video_only(
    model: AVT2DWF,
    video_windows: Tensor,
) -> Tensor:
    """Video-only ablation forward pass."""
    B, K, _T, _C, _H, _W = video_windows.shape
    video_flat = video_windows.reshape(B * K, _T, _C, _H, _W)
    V_flat, _ = model.face_encoder(video_flat)
    V_clip = V_flat.reshape(B, K, -1).mean(dim=1)  # [B, 256]
    if model.auxiliary_heads and model.video_classifier is not None:
        logits_raw: Tensor = model.video_classifier(V_clip)
    else:
        logits_raw = model.classifier(V_clip)
    temperature = model.log_temperature.exp().clamp(min=0.5, max=5.0)
    res: Tensor = logits_raw / temperature
    return res


def forward_audio_only(
    model: AVT2DWF,
    audio_segments: Tensor,
) -> Tensor:
    """Audio-only ablation forward pass."""
    A_clip, _ = model.audio_encoder(audio_segments)  # [B, 256]
    if model.auxiliary_heads and model.audio_classifier is not None:
        logits_raw: Tensor = model.audio_classifier(A_clip)
    else:
        logits_raw = model.classifier(A_clip)
    temperature = model.log_temperature.exp().clamp(min=0.5, max=5.0)
    res: Tensor = logits_raw / temperature
    return res


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


def run_comprehensive_evaluation(
    model: AVT2DWF,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Run full evaluation suite across a DataLoader.

    Computes:
    - Multimodal DWF predictions, probabilities, and metrics
    - Video-only ablation predictions and metrics
    - Audio-only ablation predictions and metrics
    - Static 50/50 fusion ablation predictions and metrics
    - Per-sample predictions (including w_a, w_v, sync score, error type)
    - DWF weights and sync score statistics (overall, real, fake)
    - Full list of False Positives and False Negatives
    """
    import math

    model.eval()

    sample_records: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_scores_multimodal: list[float] = []
    y_scores_video: list[float] = []
    y_scores_audio: list[float] = []
    y_scores_static: list[float] = []

    wa_all: list[float] = []
    wv_all: list[float] = []
    sync_all: list[float] = []

    wa_real: list[float] = []
    wv_real: list[float] = []
    wa_fake: list[float] = []
    wv_fake: list[float] = []

    repo_root = Path(__file__).resolve().parent.parent.parent

    with torch.no_grad():
        for batch in loader:
            video_windows = batch["video_windows"].to(device)
            audio_segments = batch["audio_segments"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            labels: Tensor = batch["labels"]
            clip_paths: list[str] = batch["clip_paths"]
            sample_ids: list[str] = batch.get(
                "sample_ids", [Path(p).stem for p in clip_paths]
            )

            # 1. Full multimodal DWF
            out: AVT2Result = model(video_windows, audio_segments, audio_mask)
            p_fake_batch = F.softmax(out.logits, dim=-1)[:, 1].cpu().tolist()

            # Flatten weights and sync scores
            wa_batch = [float(x) for x in out.w_a.view(-1).cpu().tolist()]
            wv_batch = [float(x) for x in out.w_v.view(-1).cpu().tolist()]
            if out.sync_score.ndim == 0:
                sc_val = float(out.sync_score.item())
                sync_batch = [sc_val] * len(sample_ids)
            else:
                sync_batch = [float(x) for x in out.sync_score.view(-1).cpu().tolist()]

            # 2. Video-only ablation
            v_logits = forward_video_only(model, video_windows)
            p_fake_v = F.softmax(v_logits, dim=-1)[:, 1].cpu().tolist()

            # 3. Audio-only ablation
            a_logits = forward_audio_only(model, audio_segments)
            p_fake_a = F.softmax(a_logits, dim=-1)[:, 1].cpu().tolist()

            # 4. Static 50/50 fusion ablation
            s_logits = forward_static_fusion(
                model, video_windows, audio_segments, audio_mask
            )
            p_fake_s = F.softmax(s_logits, dim=-1)[:, 1].cpu().tolist()

            labels_list = labels.cpu().tolist()
            y_true.extend(labels_list)
            y_scores_multimodal.extend(p_fake_batch)
            y_scores_video.extend(p_fake_v)
            y_scores_audio.extend(p_fake_a)
            y_scores_static.extend(p_fake_s)

            wa_all.extend(wa_batch)
            wv_all.extend(wv_batch)
            sync_all.extend(sync_batch)

            for sid, cp, y, pf, wa, wv, sc in zip(
                sample_ids,
                clip_paths,
                labels_list,
                p_fake_batch,
                wa_batch,
                wv_batch,
                sync_batch,
                strict=False,
            ):
                if y == 0:
                    wa_real.append(float(wa))
                    wv_real.append(float(wv))
                else:
                    wa_fake.append(float(wa))
                    wv_fake.append(float(wv))

                pred_lbl = 1 if pf >= 0.5 else 0
                is_corr = bool(pred_lbl == y)
                err_type = "None"
                if not is_corr:
                    err_type = (
                        "False Positive (FP)"
                        if pred_lbl == 1 and y == 0
                        else "False Negative (FN)"
                    )

                try:
                    rel_path = str(Path(cp).relative_to(repo_root)).replace("\\", "/")
                except Exception:
                    rel_path = str(cp).replace("\\", "/")

                sample_records.append(
                    {
                        "sample_id": sid,
                        "filename": Path(cp).name,
                        "relative_path": rel_path,
                        "true_label": int(y),
                        "true_class": "FAKE" if y == 1 else "REAL",
                        "pred_label": int(pred_lbl),
                        "pred_class": "FAKE" if pred_lbl == 1 else "REAL",
                        "p_fake": round(float(pf), 6),
                        "p_real": round(float(1.0 - pf), 6),
                        "w_a": round(float(wa), 6),
                        "w_v": round(float(wv), 6),
                        "sync_score": round(float(sc), 6),
                        "is_correct": is_corr,
                        "error_type": err_type,
                    }
                )

    metrics_multimodal = compute_metrics(y_true, y_scores_multimodal)
    metrics_video = compute_metrics(y_true, y_scores_video)
    metrics_audio = compute_metrics(y_true, y_scores_audio)
    metrics_static = compute_metrics(y_true, y_scores_static)

    def stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        m = sum(vals) / len(vals)
        variance = sum((x - m) ** 2 for x in vals) / max(1, len(vals))
        return {
            "mean": round(m, 4),
            "std": round(math.sqrt(variance), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    false_positives = [
        r for r in sample_records if r["error_type"] == "False Positive (FP)"
    ]
    false_negatives = [
        r for r in sample_records if r["error_type"] == "False Negative (FN)"
    ]

    return {
        "metrics_multimodal": metrics_multimodal,
        "metrics_video_only": metrics_video,
        "metrics_audio_only": metrics_audio,
        "metrics_static_fusion": metrics_static,
        "sample_records": sample_records,
        "dwf_diagnostics": {
            "audio_weight_wa": stats(wa_all),
            "video_weight_wv": stats(wv_all),
            "real_class_wa": stats(wa_real),
            "real_class_wv": stats(wv_real),
            "fake_class_wa": stats(wa_fake),
            "fake_class_wv": stats(wv_fake),
            "sync_score": stats(sync_all),
            "dwf_gain_over_static_auc": round(
                float(metrics_multimodal["auc"]) - float(metrics_static["auc"]), 4
            ),
            "dwf_gain_over_video_only_auc": round(
                float(metrics_multimodal["auc"]) - float(metrics_video["auc"]), 4
            ),
            "dwf_gain_over_audio_only_auc": round(
                float(metrics_multimodal["auc"]) - float(metrics_audio["auc"]), 4
            ),
        },
        "failure_analysis": {
            "total_errors": len(false_positives) + len(false_negatives),
            "false_positive_count": len(false_positives),
            "false_negative_count": len(false_negatives),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
    }


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
