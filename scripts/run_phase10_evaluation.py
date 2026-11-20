"""Independent Model Evaluation & Performance Validation Engine for AVT2-DWF (Phase 10).

Executes comprehensive evaluation protocol across multimodal deepfake detection models.
Calculates classification metrics, bootstrap confidence intervals, calibration,
ablation branches (audio-only, visual-only, static fusion), DWF dynamics, robustness,
explainability, and hardware latency profiling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str((REPO_ROOT / "backend").resolve()))

from app.ml.models.avt2_dwf import AVT2DWF
from training.dataset import MultimodalDataset, multimodal_collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("parallax.phase10_evaluation")

def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()

def compute_all_metrics(
    y_true: list[int],
    y_scores: list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate comprehensive primary, class-wise, calibration, and agreement metrics."""
    y_true_np = np.array(y_true, dtype=int)
    y_scores_np = np.array(y_scores, dtype=float)
    y_pred_np = (y_scores_np >= threshold).astype(int)

    total = len(y_true_np)
    if total == 0:
        return {}

    tp = int(np.sum((y_true_np == 1) & (y_pred_np == 1)))
    tn = int(np.sum((y_true_np == 0) & (y_pred_np == 0)))
    fp = int(np.sum((y_true_np == 0) & (y_pred_np == 1)))
    fn = int(np.sum((y_true_np == 1) & (y_pred_np == 0)))

    accuracy = float((tp + tn) / max(1, total))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    specificity = float(tn / max(1, tn + fp))
    sensitivity = recall
    f1 = float(2 * precision * recall / max(1e-8, precision + recall))
    balanced_acc = float(0.5 * (recall + specificity))
    fpr = float(fp / max(1, fp + tn))
    fnr = float(fn / max(1, fn + tp))

    # Real (class 0) metrics
    real_precision = float(tn / max(1, tn + fn))
    real_recall = float(tn / max(1, tn + fp))
    real_f1 = float(2 * real_precision * real_recall / max(1e-8, real_precision + real_recall))
    real_support = int(np.sum(y_true_np == 0))

    # Fake (class 1) metrics
    fake_precision = precision
    fake_recall = recall
    fake_f1 = f1
    fake_support = int(np.sum(y_true_np == 1))

    # Macro & Weighted F1
    macro_f1 = float(0.5 * (real_f1 + fake_f1))
    weighted_f1 = float((real_f1 * real_support + fake_f1 * fake_support) / max(1, total))

    # Advanced correlation
    mcc = float(matthews_corrcoef(y_true_np, y_pred_np)) if len(set(y_pred_np)) > 1 else 0.0
    kappa = float(cohen_kappa_score(y_true_np, y_pred_np))

    # ROC-AUC & AP
    if len(set(y_true_np)) > 1:
        roc_auc = float(roc_auc_score(y_true_np, y_scores_np))
        avg_precision = float(average_precision_score(y_true_np, y_scores_np))
    else:
        roc_auc = 0.5
        avg_precision = float(np.mean(y_true_np))

    # Calibration: Brier score
    brier = float(np.mean((y_scores_np - y_true_np) ** 2))

    # Expected Calibration Error (ECE - 10 bins)
    n_bins = 10
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    cal_bins = []
    for i in range(n_bins):
        b_low, b_high = bin_limits[i], bin_limits[i + 1]
        in_bin = (y_scores_np >= b_low) & (y_scores_np < b_high) if i < n_bins - 1 else (y_scores_np >= b_low) & (y_scores_np <= b_high)
        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(y_true_np[in_bin]))
            bin_conf = float(np.mean(y_scores_np[in_bin]))
            ece += (bin_count / total) * abs(bin_acc - bin_conf)
            cal_bins.append({
                "bin": i,
                "range": [round(b_low, 2), round(b_high, 2)],
                "count": bin_count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
            })
        else:
            cal_bins.append({
                "bin": i,
                "range": [round(b_low, 2), round(b_high, 2)],
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
            })

    # Raw and Row-Normalized Confusion Matrix
    cm_raw = [[tn, fp], [fn, tp]]
    cm_norm = [
        [round(tn / max(1, real_support), 4), round(fp / max(1, real_support), 4)],
        [round(fn / max(1, fake_support), 4), round(tp / max(1, fake_support), 4)],
    ]

    return {
        "total_samples": total,
        "operating_threshold": threshold,
        "overall": {
            "accuracy": round(accuracy, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "specificity": round(specificity, 4),
            "sensitivity": round(sensitivity, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "roc_auc": round(roc_auc, 4),
            "average_precision": round(avg_precision, 4),
            "mcc": round(mcc, 4),
            "cohen_kappa": round(kappa, 4),
            "brier_score": round(brier, 4),
            "ece": round(float(ece), 4),
        },
        "classwise": {
            "real": {
                "support": real_support,
                "correct": tn,
                "incorrect": fp,
                "precision": round(real_precision, 4),
                "recall": round(real_recall, 4),
                "f1": round(real_f1, 4),
            },
            "fake": {
                "support": fake_support,
                "correct": tp,
                "incorrect": fn,
                "precision": round(fake_precision, 4),
                "recall": round(fake_recall, 4),
                "f1": round(fake_f1, 4),
            },
        },
        "confusion_matrix": {
            "raw": cm_raw,
            "normalized_percentages": cm_norm,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "calibration": {
            "brier_score": round(brier, 4),
            "ece": round(float(ece), 4),
            "bins": cal_bins,
        },
    }

def compute_bootstrap_confidence_intervals(
    y_true: list[int],
    y_scores: list[float],
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute 95% bootstrap confidence intervals for primary metrics."""
    rng = np.random.RandomState(seed)
    y_true_np = np.array(y_true, dtype=int)
    y_scores_np = np.array(y_scores, dtype=float)
    n = len(y_true_np)

    accs = []
    bal_accs = []
    f1s = []
    aucs = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt_sample = y_true_np[idx]
        ys_sample = y_scores_np[idx]

        yp_sample = (ys_sample >= threshold).astype(int)
        acc = np.mean(yt_sample == yp_sample)
        accs.append(acc)

        # Sensitivity & Specificity
        tp = np.sum((yt_sample == 1) & (yp_sample == 1))
        tn = np.sum((yt_sample == 0) & (yp_sample == 0))
        fp = np.sum((yt_sample == 0) & (yp_sample == 1))
        fn = np.sum((yt_sample == 1) & (yp_sample == 0))

        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        bal_accs.append(0.5 * (sens + spec))

        prec = tp / max(1, tp + fp)
        rec = sens
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        f1s.append(f1)

        if len(set(yt_sample)) > 1:
            try:
                auc = roc_auc_score(yt_sample, ys_sample)
                aucs.append(auc)
            except ValueError:
                continue

    def get_ci(data: list[float]) -> dict[str, Any]:
        arr = np.array(data)
        low = float(np.percentile(arr, 2.5))
        high = float(np.percentile(arr, 97.5))
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        return {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci_95": [round(low, 4), round(high, 4)],
            "interval_width": round(high - low, 4),
        }

    return {
        "n_bootstraps": n_bootstraps,
        "seed": seed,
        "sample_size": n,
        "accuracy": get_ci(accs),
        "balanced_accuracy": get_ci(bal_accs),
        "f1": get_ci(f1s),
        "roc_auc": get_ci(aucs) if aucs else {},
    }

def run_forward_passes(
    model: AVT2DWF,
    video_windows: torch.Tensor,
    audio_segments: torch.Tensor,
    audio_mask: torch.Tensor,
) -> tuple[float, float, float, float, float, float, float]:
    """Execute fused model, audio branch, video branch, and static fusion passes."""
    with torch.no_grad():
        # 1. Full Fused Forward
        res = model(video_windows, audio_segments, audio_mask)
        p_fused = float(torch.softmax(res.logits, dim=-1)[0, 1].item())
        w_a = float(res.w_a.squeeze().item()) if hasattr(res.w_a, "item") else float(res.w_a)
        w_v = float(res.w_v.squeeze().item()) if hasattr(res.w_v, "item") else float(res.w_v)
        sync_score = float(res.sync_score.squeeze().item()) if hasattr(res.sync_score, "item") else 0.0

        # 2. Audio Only Forward (using auxiliary audio head)
        if res.audio_logits is not None:
            p_audio = float(torch.softmax(res.audio_logits, dim=-1)[0, 1].item())
        else:
            A_clip, _ = model.audio_encoder(audio_segments)
            temp = model.log_temperature.exp().clamp(min=0.5, max=5.0)
            a_logits = model.classifier(A_clip) / temp
            p_audio = float(torch.softmax(a_logits, dim=-1)[0, 1].item())

        # 3. Video Only Forward (using auxiliary video head)
        if res.video_logits is not None:
            p_video = float(torch.softmax(res.video_logits, dim=-1)[0, 1].item())
        else:
            B, K, _T, _C, _H, _W = video_windows.shape
            video_flat = video_windows.reshape(B * K, _T, _C, _H, _W)
            V_flat, _ = model.face_encoder(video_flat)
            V_clip = V_flat.reshape(B, K, -1).mean(dim=1)
            temp = model.log_temperature.exp().clamp(min=0.5, max=5.0)
            v_logits = model.classifier(V_clip) / temp
            p_video = float(torch.softmax(v_logits, dim=-1)[0, 1].item())

        # 4. Static 50/50 Fusion
        B, K, _T, _C, _H, _W = video_windows.shape
        video_flat = video_windows.reshape(B * K, _T, _C, _H, _W)
        V_flat, _ = model.face_encoder(video_flat)
        V_clip = V_flat.reshape(B, K, -1).mean(dim=1)
        A_clip, _ = model.audio_encoder(audio_segments)

        if audio_mask.ndim > 1:
            a_present = audio_mask.any(dim=-1, keepdim=True).float()
        else:
            a_present = audio_mask.any().float().unsqueeze(0).expand(B, 1)

        w_a_stat = 0.5 * a_present
        w_v_stat = 1.0 - w_a_stat
        fused_stat = w_a_stat * A_clip + w_v_stat * V_clip
        temp = model.log_temperature.exp().clamp(min=0.5, max=5.0)
        s_logits = model.classifier(fused_stat) / temp
        p_static = float(torch.softmax(s_logits, dim=-1)[0, 1].item())

    return p_fused, p_audio, p_video, p_static, w_a, w_v, sync_score

def run_robustness_suite(
    model: AVT2DWF,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Execute controlled synthetic robustness perturbations on isolated tensor copies."""
    logger.info("Executing robustness perturbation experiments across test set...")
    scenarios = {
        "video_gaussian_noise": "Additive Gaussian noise (sigma=0.05) to visual tensor",
        "video_frame_drop": "Frame dropout (zeroing alternate frames in temporal windows)",
        "audio_volume_reduction": "Audio volume scaled to 50% (-6 dB)",
        "audio_noise_injection": "Audio Gaussian noise injection (sigma=0.05)",
        "complete_audio_silence": "Complete audio absence (audio_mask=0, audio_segments=0)",
        "temporal_sync_offset": "Audio window shifted by +1 temporal window",
    }

    results: dict[str, Any] = {}
    base_acc = baseline_metrics["overall"]["accuracy"]
    base_auc = baseline_metrics["overall"]["roc_auc"]
    base_f1 = baseline_metrics["overall"]["f1"]

    for sc_name, sc_desc in scenarios.items():
        y_true = []
        y_scores = []

        with torch.inference_mode():
            for batch in loader:
                v = batch["video_windows"].clone().to(device)
                a = batch["audio_segments"].clone().to(device)
                m = batch["audio_mask"].clone().to(device)
                lbl = int(batch["labels"][0].item())

                # Apply isolated non-destructive transformation
                if sc_name == "video_gaussian_noise":
                    v = v + 0.05 * torch.randn_like(v)
                elif sc_name == "video_frame_drop":
                    v[:, :, 1::2, :, :, :] = 0.0  # drop every 2nd frame
                elif sc_name == "audio_volume_reduction":
                    a = a * 0.5
                elif sc_name == "audio_noise_injection":
                    a = a + 0.05 * torch.randn_like(a)
                elif sc_name == "complete_audio_silence":
                    a = torch.zeros_like(a)
                    m = torch.zeros_like(m)
                elif sc_name == "temporal_sync_offset":
                    a = torch.roll(a, shifts=1, dims=1)

                res = model(v, a, m)
                p = float(torch.softmax(res.logits, dim=-1)[0, 1].item())
                y_true.append(lbl)
                y_scores.append(p)

        m_sc = compute_all_metrics(y_true, y_scores, threshold=0.5)
        acc = m_sc["overall"]["accuracy"]
        auc = m_sc["overall"]["roc_auc"]
        f1 = m_sc["overall"]["f1"]

        results[sc_name] = {
            "description": sc_desc,
            "accuracy": acc,
            "roc_auc": auc,
            "f1": f1,
            "delta_accuracy": round(acc - base_acc, 4),
            "delta_roc_auc": round(auc - base_auc, 4),
            "delta_f1": round(f1 - base_f1, 4),
        }
        logger.info("Robustness [%s]: Acc=%.4f (Δ=%.4f), AUC=%.4f (Δ=%.4f)", sc_name, acc, acc - base_acc, auc, auc - base_auc)

    return results

def evaluate_split(
    manifest_csv: Path,
    checkpoint_path: Path,
    out_dir: Path,
    split_name: str,
    device_str: str = "cuda",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Execute complete evaluation protocol on specified manifest split."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info("Initializing AVT2-DWF model for evaluation on device: %s", device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # 1. Load Model
    model = AVT2DWF(
        pretrained_vit=False,
        audio_layernorm=True,
        audio_input_norm=True,
        auxiliary_heads=True,
        classifier_dropout=0.25,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # 2. Load Dataset
    logger.info("Loading manifest from: %s", manifest_csv)
    dataset = MultimodalDataset(manifest_csv=manifest_csv)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=0,
    )
    logger.info("Loaded %d samples for %s evaluation", len(dataset), split_name)

    # 3. Evaluate Loop
    sample_records = []
    y_true = []
    y_fused = []
    y_audio = []
    y_video = []
    y_static = []
    w_a_list = []
    w_v_list = []
    sync_scores = []
    latencies = []

    t0_eval = time.time()
    for idx, batch in enumerate(loader):
        v = batch["video_windows"].to(device)
        a = batch["audio_segments"].to(device)
        m = batch["audio_mask"].to(device)
        lbl = int(batch["labels"][0].item())
        sample_id = batch["identities"][0] if "identities" in batch else f"sample_{idx}"

        # Latency benchmark
        t_start = time.perf_counter()
        p_fused, p_audio, p_video, p_static, w_a, w_v, sync = run_forward_passes(model, v, a, m)
        t_elapsed = (time.perf_counter() - t_start) * 1000.0  # ms

        latencies.append(t_elapsed)
        y_true.append(lbl)
        y_fused.append(p_fused)
        y_audio.append(p_audio)
        y_video.append(p_video)
        y_static.append(p_static)
        w_a_list.append(w_a)
        w_v_list.append(w_v)
        sync_scores.append(sync)

        pred_fused = 1 if p_fused >= threshold else 0
        is_correct = bool(pred_fused == lbl)
        err_type = "CORRECT" if is_correct else ("FP (REAL as FAKE)" if lbl == 0 else "FN (FAKE as REAL)")

        sample_records.append({
            "sample_id": sample_id,
            "ground_truth": lbl,
            "predicted_fused": pred_fused,
            "p_fused": round(p_fused, 6),
            "p_audio": round(p_audio, 6),
            "p_video": round(p_video, 6),
            "p_static": round(p_static, 6),
            "weight_audio": round(w_a, 4),
            "weight_video": round(w_v, 4),
            "sync_score": round(sync, 4),
            "is_correct": is_correct,
            "error_type": err_type,
            "latency_ms": round(t_elapsed, 2),
        })

    eval_duration = time.time() - t0_eval
    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0.0

    logger.info("Completed forward passes for %d samples in %.2fs (avg %.2f ms/sample, peak VRAM: %.2f MB)",
                len(y_true), eval_duration, np.mean(latencies), peak_vram_mb)

    # 4. Metrics Calculation
    fused_metrics = compute_all_metrics(y_true, y_fused, threshold=threshold)
    audio_metrics = compute_all_metrics(y_true, y_audio, threshold=threshold)
    video_metrics = compute_all_metrics(y_true, y_video, threshold=threshold)
    static_metrics = compute_all_metrics(y_true, y_static, threshold=threshold)

    # Bootstrap CIs for Fused Model
    cis = compute_bootstrap_confidence_intervals(y_true, y_fused, threshold=threshold, n_bootstraps=1000, seed=42)

    # DWF Fusion Statistics
    w_a_np = np.array(w_a_list)
    w_v_np = np.array(w_v_list)
    yt_np = np.array(y_true)
    correct_mask = np.array([r["is_correct"] for r in sample_records])

    dwf_stats = {
        "overall": {
            "mean_weight_audio": round(float(np.mean(w_a_np)), 4),
            "mean_weight_video": round(float(np.mean(w_v_np)), 4),
            "median_weight_audio": round(float(np.median(w_a_np)), 4),
            "median_weight_video": round(float(np.median(w_v_np)), 4),
            "std_weight_audio": round(float(np.std(w_a_np)), 4),
            "std_weight_video": round(float(np.std(w_v_np)), 4),
            "min_weight_audio": round(float(np.min(w_a_np)), 4),
            "max_weight_audio": round(float(np.max(w_a_np)), 4),
        },
        "by_class": {
            "real_mean_w_a": round(float(np.mean(w_a_np[yt_np == 0])), 4),
            "real_mean_w_v": round(float(np.mean(w_v_np[yt_np == 0])), 4),
            "fake_mean_w_a": round(float(np.mean(w_a_np[yt_np == 1])), 4),
            "fake_mean_w_v": round(float(np.mean(w_v_np[yt_np == 1])), 4),
        },
        "by_accuracy": {
            "correct_mean_w_a": round(float(np.mean(w_a_np[correct_mask])), 4) if np.sum(correct_mask) > 0 else 0.0,
            "correct_mean_w_v": round(float(np.mean(w_v_np[correct_mask])), 4) if np.sum(correct_mask) > 0 else 0.0,
            "incorrect_mean_w_a": round(float(np.mean(w_a_np[~correct_mask])), 4) if np.sum(~correct_mask) > 0 else 0.0,
            "incorrect_mean_w_v": round(float(np.mean(w_v_np[~correct_mask])), 4) if np.sum(~correct_mask) > 0 else 0.0,
        },
    }

    # Latency & Hardware Statistics
    lat_np = np.array(latencies)
    hardware_stats = {
        "device": device_str,
        "sample_count": len(latencies),
        "mean_latency_ms": round(float(np.mean(lat_np)), 2),
        "median_latency_ms": round(float(np.median(lat_np)), 2),
        "p90_latency_ms": round(float(np.percentile(lat_np, 90)), 2),
        "p95_latency_ms": round(float(np.percentile(lat_np, 95)), 2),
        "min_latency_ms": round(float(np.min(lat_np)), 2),
        "max_latency_ms": round(float(np.max(lat_np)), 2),
        "std_latency_ms": round(float(np.std(lat_np)), 2),
        "total_eval_duration_s": round(eval_duration, 2),
        "peak_gpu_vram_mb": round(peak_vram_mb, 2),
        "peak_gpu_vram_gb": round(peak_vram_mb / 1024, 2),
    }

    # Modality comparison summary
    modality_comparison = {
        "fused_dwf": fused_metrics["overall"],
        "video_only": video_metrics["overall"],
        "audio_only": audio_metrics["overall"],
        "static_50_50": static_metrics["overall"],
    }

    # 5. Robustness Experiments
    robustness_results = {}
    if split_name == "test":
        robustness_results = run_robustness_suite(model, loader, device, fused_metrics)

    # 6. Save Target Files (Specifically for test split under experiments/phase10_independent_evaluation/test/)
    target_dir = (out_dir / "test") if split_name == "test" else (out_dir / "val")
    target_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = target_dir

    prefix = "test" if split_name == "test" else "val"

    # Save CSVs
    pred_csv_path = target_dir / f"{prefix}_predictions.csv"
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_records[0].keys()))
        writer.writeheader()
        writer.writerows(sample_records)
    logger.info("Saved predictions to: %s", pred_csv_path)

    err_csv_path = target_dir / f"{prefix}_error_analysis.csv"
    err_records = [r for r in sample_records if not r["is_correct"]]
    with open(err_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_records[0].keys()))
        writer.writeheader()
        writer.writerows(err_records)
    logger.info("Saved error analysis (%d errors) to: %s", len(err_records), err_csv_path)

    # Save Metrics JSONs
    with open(target_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "overall": fused_metrics["overall"],
            "classwise": fused_metrics["classwise"],
            "confusion_matrix": fused_metrics["confusion_matrix"],
            "calibration": fused_metrics["calibration"],
            "modality_comparison": modality_comparison,
            "dwf_stats": dwf_stats,
        }, f, indent=2)

    with open(target_dir / f"{prefix}_confidence_intervals.json", "w", encoding="utf-8") as f:
        json.dump(cis, f, indent=2)

    with open(target_dir / f"{prefix}_hardware_metrics.json", "w", encoding="utf-8") as f:
        json.dump(hardware_stats, f, indent=2)

    if split_name == "test":
        with open(target_dir / "test_robustness_metrics.json", "w", encoding="utf-8") as f:
            json.dump(robustness_results, f, indent=2)

    # 7. Generate High-Resolution Plots
    generate_evaluation_plots(y_true, y_fused, y_audio, y_video, y_static, w_a_list, w_v_list, latencies, plots_dir, prefix, threshold)

    # 8. Explainability Samples (Select 6 Representative Samples)
    if split_name == "test":
        generate_explainability_records(sample_records, out_dir / "explainability" / "representative_examples")

    # 9. Test Evaluation Report Markdown
    if split_name == "test":
        report_md_path = target_dir / "test_evaluation_report.md"
        generate_test_report_md(fused_metrics, cis, modality_comparison, dwf_stats, hardware_stats, robustness_results, report_md_path)
        logger.info("Generated scientific test evaluation report: %s", report_md_path)

        # 10. Test Artifact Hashes
        generate_test_artifact_hashes(target_dir)

    return {
        "split": split_name,
        "sample_count": len(y_true),
        "fused_metrics": fused_metrics,
        "modality_comparison": modality_comparison,
        "confidence_intervals": cis,
        "dwf_stats": dwf_stats,
        "hardware_stats": hardware_stats,
        "robustness_results": robustness_results,
        "error_count": len(err_records),
    }

def generate_explainability_records(sample_records: list[dict[str, Any]], exp_dir: Path) -> None:
    """Identify and persist representative explainability examples."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    reals = [r for r in sample_records if r["ground_truth"] == 0]
    fakes = [r for r in sample_records if r["ground_truth"] == 1]

    correct_reals = [r for r in reals if r["is_correct"]]
    correct_fakes = [r for r in fakes if r["is_correct"]]
    fps = [r for r in reals if not r["is_correct"]]
    fns = [r for r in fakes if not r["is_correct"]]

    examples = {
        "correct_real": min(correct_reals, key=lambda x: x["p_fused"]) if correct_reals else None,
        "correct_fake": max(correct_fakes, key=lambda x: x["p_fused"]) if correct_fakes else None,
        "false_positive": max(fps, key=lambda x: x["p_fused"]) if fps else None,
        "false_negative": min(fns, key=lambda x: x["p_fused"]) if fns else None,
        "audio_dominant": max(sample_records, key=lambda x: x["weight_audio"]) if sample_records else None,
        "visual_dominant": max(sample_records, key=lambda x: x["weight_video"]) if sample_records else None,
    }

    with open(exp_dir / "representative_examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    logger.info("Saved 6 representative explainability examples to: %s", exp_dir)

def generate_test_artifact_hashes(test_dir: Path) -> None:
    """Compute and save SHA-256 hashes of all artifacts in test directory."""
    hashes = {}
    for p in sorted(test_dir.iterdir()):
        if p.is_file() and p.name != "test_artifact_hashes.json":
            hashes[p.name] = {
                "sha256": get_file_sha256(p),
                "size_bytes": p.stat().st_size,
            }
    with open(test_dir / "test_artifact_hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)
    logger.info("Saved artifact hashes to: %s", test_dir / "test_artifact_hashes.json")

def generate_test_report_md(
    fused_metrics: dict[str, Any],
    cis: dict[str, Any],
    modality: dict[str, Any],
    dwf: dict[str, Any],
    hw: dict[str, Any],
    robust: dict[str, Any],
    out_file: Path,
) -> None:
    """Generate comprehensive scientific markdown report."""
    ov = fused_metrics["overall"]
    cw = fused_metrics["classwise"]
    cm = fused_metrics["confusion_matrix"]

    lines = [
        "# PARALLAX — PHASE 10: INDEPENDENT MODEL EVALUATION REPORT",
        f"**Date:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        "**Model:** AVT²-DWF (Vision Transformer ViT-B/16 + Audio Transformer + Dynamic Weight Fusion)  ",
        "**Candidate Checkpoint:** `experiments/phase4_4_clean_avt2_dwf/best_checkpoint.pt` (Epoch 6)  ",
        "**Test Manifest:** `datasets/manifests/recovery_4_3a/test_recovery.csv` (150 samples)  ",
        f"**Hardware Environment:** {hw['device'].upper()} | Peak VRAM: {hw['peak_gpu_vram_mb']:.2f} MB  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "Independent single-pass evaluation was executed across all 150 test samples (75 REAL, 75 FAKE) at the fixed threshold of 0.50.",
        f"- **Test Accuracy:** **{ov['accuracy'] * 100:.2f}%** (95% CI: [{cis['accuracy']['ci_95'][0] * 100:.1f}%, {cis['accuracy']['ci_95'][1] * 100:.1f}%])",
        f"- **Balanced Accuracy:** **{ov['balanced_accuracy'] * 100:.2f}%**",
        f"- **ROC-AUC:** **{ov['roc_auc']:.4f}** (95% CI: [{cis['roc_auc']['ci_95'][0]:.4f}, {cis['roc_auc']['ci_95'][1]:.4f}])",
        f"- **Average Precision (PR-AUC):** **{ov['average_precision']:.4f}**",
        f"- **F1 Score:** **{ov['f1']:.4f}** (Precision: {ov['precision']:.4f}, Recall: {ov['recall']:.4f})",
        f"- **Specificity:** **{ov['specificity']:.4f}** | **Sensitivity:** **{ov['sensitivity']:.4f}**",
        f"- **Brier Score:** **{ov['brier_score']:.4f}** | **ECE (10 bins):** **{ov['ece']:.4f}**",
        f"- **Mean Latency:** **{hw['mean_latency_ms']:.2f} ms/sample**",
        "",
        "## 2. Confusion Matrix",
        "| True \\ Predicted | Predicted REAL | Predicted FAKE | Support |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Actual REAL** | **{cm['tn']}** ({cm['normalized_percentages'][0][0] * 100:.1f}%) | **{cm['fp']}** ({cm['normalized_percentages'][0][1] * 100:.1f}%) | {cw['real']['support']} |",
        f"| **Actual FAKE** | **{cm['fn']}** ({cm['normalized_percentages'][1][0] * 100:.1f}%) | **{cm['tp']}** ({cm['normalized_percentages'][1][1] * 100:.1f}%) | {cw['fake']['support']} |",
        "",
        "## 3. Modality & Ablation Comparison",
        "| Configuration | Accuracy | Balanced Acc | F1 Score | ROC-AUC | PR-AUC | Specificity | Sensitivity |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **AVT²-DWF Fused** | **{modality['fused_dwf']['accuracy']:.4f}** | **{modality['fused_dwf']['balanced_accuracy']:.4f}** | **{modality['fused_dwf']['f1']:.4f}** | **{modality['fused_dwf']['roc_auc']:.4f}** | **{modality['fused_dwf']['average_precision']:.4f}** | {modality['fused_dwf']['specificity']:.4f} | {modality['fused_dwf']['sensitivity']:.4f} |",
        f"| **Visual-Only Branch** | {modality['video_only']['accuracy']:.4f} | {modality['video_only']['balanced_accuracy']:.4f} | {modality['video_only']['f1']:.4f} | {modality['video_only']['roc_auc']:.4f} | {modality['video_only']['average_precision']:.4f} | {modality['video_only']['specificity']:.4f} | {modality['video_only']['sensitivity']:.4f} |",
        f"| **Audio-Only Branch** | {modality['audio_only']['accuracy']:.4f} | {modality['audio_only']['balanced_accuracy']:.4f} | {modality['audio_only']['f1']:.4f} | {modality['audio_only']['roc_auc']:.4f} | {modality['audio_only']['average_precision']:.4f} | {modality['audio_only']['specificity']:.4f} | {modality['audio_only']['sensitivity']:.4f} |",
        f"| **Static 50/50 Fusion** | {modality['static_50_50']['accuracy']:.4f} | {modality['static_50_50']['balanced_accuracy']:.4f} | {modality['static_50_50']['f1']:.4f} | {modality['static_50_50']['roc_auc']:.4f} | {modality['static_50_50']['average_precision']:.4f} | {modality['static_50_50']['specificity']:.4f} | {modality['static_50_50']['sensitivity']:.4f} |",
        "",
        "## 4. DWF Dynamics",
        f"- **Mean Weights:** Audio $w_a = {dwf['overall']['mean_weight_audio']:.4f}$, Video $w_v = {dwf['overall']['mean_weight_video']:.4f}$",
        f"- **REAL Class Weights:** $w_a = {dwf['by_class']['real_mean_w_a']:.4f}$, $w_v = {dwf['by_class']['real_mean_w_v']:.4f}$",
        f"- **FAKE Class Weights:** $w_a = {dwf['by_class']['fake_mean_w_a']:.4f}$, $w_v = {dwf['by_class']['fake_mean_w_v']:.4f}$",
        "",
        "## 5. Robustness & Input Degradation Suite",
        "| Perturbation Scenario | Accuracy | ROC-AUC | F1 | Delta Accuracy | Delta ROC-AUC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for k, v in robust.items():
        lines.append(f"| **{k}** | {v['accuracy']:.4f} | {v['roc_auc']:.4f} | {v['f1']:.4f} | {v['delta_accuracy']:+.4f} | {v['delta_roc_auc']:+.4f} |")

    lines.extend([
        "",
        "## 6. Hardware & Latency Benchmarks",
        f"- **Mean Latency:** {hw['mean_latency_ms']:.2f} ms",
        f"- **Median Latency:** {hw['median_latency_ms']:.2f} ms",
        f"- **P90 Latency:** {hw['p90_latency_ms']:.2f} ms",
        f"- **P95 Latency:** {hw['p95_latency_ms']:.2f} ms",
        f"- **Peak VRAM Allocated:** {hw['peak_gpu_vram_mb']:.2f} MB ({hw['peak_gpu_vram_gb']:.2f} GB)",
        f"- **Total Runtime (150 samples):** {hw['total_eval_duration_s']:.2f} s",
    ])

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def generate_evaluation_plots(
    y_true: list[int],
    y_fused: list[float],
    y_audio: list[float],
    y_video: list[float],
    y_static: list[float],
    w_a_list: list[float],
    w_v_list: list[float],
    latencies: list[float],
    plots_dir: Path,
    prefix: str,
    threshold: float = 0.5,
) -> None:
    """Generate high-resolution PNG plots for ROC, PR, calibration, confusion matrix, fusion weights."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. ROC Curves (Multimodal vs Video vs Audio vs Static)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for scores, label, color in [
        (y_fused, "AVT²-DWF Fused", "#0284c7"),
        (y_video, "Visual Only Branch", "#10b981"),
        (y_audio, "Audio Only Branch", "#f59e0b"),
        (y_static, "Static 50/50 Fusion", "#8b5cf6"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.4f})", color=color, linewidth=2.0)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title(f"ROC Curves — {prefix.capitalize()} Split", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_roc_curve.png")
    plt.close(fig)

    # 2. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for scores, label, color in [
        (y_fused, "AVT²-DWF Fused", "#0284c7"),
        (y_video, "Visual Only Branch", "#10b981"),
        (y_audio, "Audio Only Branch", "#f59e0b"),
        (y_static, "Static 50/50 Fusion", "#8b5cf6"),
    ]:
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(rec, prec, label=f"{label} (AP = {ap:.4f})", color=color, linewidth=2.0)
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title(f"Precision-Recall Curves — {prefix.capitalize()} Split", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_precision_recall_curve.png")
    plt.close(fig)

    # 3. Raw Confusion Matrix Heatmap
    y_pred = [1 if s >= threshold else 0 for s in y_fused]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    classes = ["REAL", "FAKE"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title(f"Confusion Matrix (Raw Counts) — {prefix.capitalize()}", fontsize=12, fontweight="bold")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_confusion_matrix.png")
    plt.close(fig)

    # 4. Normalized Confusion Matrix Heatmap
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title(f"Confusion Matrix (Row Normalized %) — {prefix.capitalize()}", fontsize=12, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i, j] * 100:.1f}%", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_normalized_confusion_matrix.png")
    plt.close(fig)

    # 5. Calibration Reliability Diagram
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    n_bins = 10
    bin_limits = np.linspace(0, 1, n_bins + 1)
    y_true_np = np.array(y_true)
    y_scores_np = np.array(y_fused)
    bin_accs = []
    bin_confs = []
    for i in range(n_bins):
        b_low, b_high = bin_limits[i], bin_limits[i + 1]
        mask = (y_scores_np >= b_low) & (y_scores_np < b_high) if i < n_bins - 1 else (y_scores_np >= b_low) & (y_scores_np <= b_high)
        if np.sum(mask) > 0:
            bin_accs.append(np.mean(y_true_np[mask]))
            bin_confs.append(np.mean(y_scores_np[mask]))
        else:
            bin_accs.append(0.0)
            bin_confs.append((b_low + b_high) / 2.0)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
    ax.plot(bin_confs, bin_accs, "s-", color="#0284c7", linewidth=2, label="AVT²-DWF Model")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    ax.set_ylabel("Empirical True Fraction", fontsize=11, fontweight="bold")
    ax.set_title(f"Reliability Diagram (Calibration) — {prefix.capitalize()}", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_calibration_curve.png")
    plt.close(fig)

    # 6. Fusion Weight Distribution
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.hist(w_a_list, bins=25, alpha=0.6, label="Audio Weight ($w_a$)", color="#f59e0b", edgecolor="black")
    ax.hist(w_v_list, bins=25, alpha=0.6, label="Video Weight ($w_v$)", color="#0284c7", edgecolor="black")
    ax.set_xlabel("Dynamic Weight Value", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sample Frequency", fontsize=11, fontweight="bold")
    ax.set_title(f"Dynamic Weight Fusion (DWF) Distribution — {prefix.capitalize()}", fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_fusion_weight_distribution.png")
    plt.close(fig)

    # 7. Latency Distribution
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.hist(latencies, bins=20, color="#10b981", edgecolor="black", alpha=0.75)
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    ax.axvline(mean_lat, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_lat:.1f} ms")
    ax.axvline(p95_lat, color="orange", linestyle=":", linewidth=1.5, label=f"P95: {p95_lat:.1f} ms")
    ax.set_xlabel("Single-Video Forward Latency (ms)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sample Count", fontsize=11, fontweight="bold")
    ax.set_title(f"Inference Latency Distribution — {prefix.capitalize()}", fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_latency_distribution.png")
    plt.close(fig)

    logger.info("Generated high-resolution evaluation plots in: %s", plots_dir)

def main() -> None:
    parser = argparse.ArgumentParser(description="AVT2-DWF Phase 10 Independent Evaluation")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Dataset split to evaluate")
    parser.add_argument("--authorized", action="store_true", help="Explicit confirmation for test-set evaluation")
    parser.add_argument("--device", default="cuda", help="Device to run inference on ('cuda' or 'cpu')")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()

    out_base = REPO_ROOT / "experiments" / "phase10_independent_evaluation"
    checkpoint = REPO_ROOT / "experiments" / "phase4_4_clean_avt2_dwf" / "best_checkpoint.pt"

    if args.split == "test":
        if not args.authorized:
            logger.error("TEST SET ACCESS DENIED: Operator authorization (--authorized) is required before accessing test_recovery.csv.")
            print("\n[BLOCKED] Operator authorization required before accessing the independent test split.\n")
            sys.exit(1)
        manifest = REPO_ROOT / "datasets" / "manifests" / "recovery_4_3a" / "test_recovery.csv"
        split_label = "test"
    else:
        manifest = REPO_ROOT / "datasets" / "manifests" / "recovery_4_3a" / "validation_recovery.csv"
        split_label = "val"

    logger.info("Executing Phase 10 evaluation on split: %s", split_label)
    results = evaluate_split(
        manifest_csv=manifest,
        checkpoint_path=checkpoint,
        out_dir=out_base,
        split_name=split_label,
        device_str=args.device,
        threshold=args.threshold,
    )

    print(f"\n=== PHASE 10 EVALUATION COMPLETE ({split_label.upper()} SPLIT) ===")
    print(f"Samples Evaluated: {results['sample_count']}")
    print(f"Accuracy:          {results['fused_metrics']['overall']['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['fused_metrics']['overall']['balanced_accuracy']:.4f}")
    print(f"ROC-AUC:           {results['fused_metrics']['overall']['roc_auc']:.4f}")
    print(f"Average Precision: {results['fused_metrics']['overall']['average_precision']:.4f}")
    print(f"F1 Score:          {results['fused_metrics']['overall']['f1']:.4f}")
    print(f"Precision:         {results['fused_metrics']['overall']['precision']:.4f}")
    print(f"Recall:            {results['fused_metrics']['overall']['recall']:.4f}")
    print(f"Specificity:       {results['fused_metrics']['overall']['specificity']:.4f}")
    print(f"Brier Score:       {results['fused_metrics']['overall']['brier_score']:.4f}")
    print(f"ECE (10 bins):     {results['fused_metrics']['overall']['ece']:.4f}")
    print(f"Mean Latency:      {results['hardware_stats']['mean_latency_ms']:.2f} ms")
    if results.get("split") == "test":
        print(f"Peak VRAM:         {results['hardware_stats']['peak_gpu_vram_mb']:.2f} MB")
        print("Artifacts Saved:   experiments/phase10_independent_evaluation/test/")

if __name__ == "__main__":
    main()
