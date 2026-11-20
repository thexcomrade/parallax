"""Generate detailed False Negative (False REAL) failure analysis for Phase 10 test cases."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "experiments" / "model_reliability_improvement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load predictions
preds_path = REPO_ROOT / "experiments" / "phase10_independent_evaluation" / "test" / "test_predictions.csv"
assert preds_path.exists(), f"Missing {preds_path}"

fn_cases: list[dict[str, str]] = []
with open(preds_path, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r.get("error_type") == "FN (FAKE as REAL)":
            fn_cases.append(r)

# 2. Load dataset manifest
manifest_path = REPO_ROOT / "datasets" / "manifests" / "dataset_manifest.csv"
manifest_lookup: dict[str, dict[str, str]] = {}
if manifest_path.exists():
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_lookup = {r["sample_id"]: r for r in csv.DictReader(f)}

# 3. Load integrity metadata
integrity_path = REPO_ROOT / "reports" / "phase1_dataset_integrity.csv"
integrity_lookup: dict[str, dict[str, str]] = {}
if integrity_path.exists():
    with open(integrity_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fname = r["filename"].lower()
            integrity_lookup[fname] = r
            sid = r["filename"].upper().replace(".MP4", "")
            integrity_lookup[sid] = r

analysis_rows: list[dict[str, str]] = []

for fn in sorted(fn_cases, key=lambda x: x["sample_id"]):
    sid = fn["sample_id"]
    m = manifest_lookup.get(sid, {})
    integ = integrity_lookup.get(sid, {}) or integrity_lookup.get(f"{sid.lower()}.mp4", {})

    p_fused = float(fn["p_fused"])
    p_video = float(fn["p_video"])
    p_audio = float(fn["p_audio"])
    w_audio = float(fn["weight_audio"])
    w_video = float(fn["weight_video"])
    sync_score = fn.get("sync_score", "N/A")

    duration = integ.get("duration", "N/A")
    width = integ.get("width", "N/A")
    height = integ.get("height", "N/A")
    resolution = f"{width}x{height}" if width != "N/A" else "N/A"
    fps = integ.get("fps", "N/A")
    has_audio = integ.get("has_audio", "True")
    silent_audio = integ.get("silent_audio", "False")
    source_group = m.get("group_id", "source_identity_unknown")

    # Failure Categorization logic
    if p_video >= 0.50 and w_audio >= 0.80 and p_fused < 0.50:
        primary_cause = "Audio branch misleading fusion model (Visual branch was correct)"
        failure_category = "Audio branch misleading the fusion model"
        root_cause_detail = (
            f"Visual branch correctly predicted FAKE with probability {p_video:.3f}, "
            f"but unbounded DWF gating assigned {w_audio*100:.1f}% weight to audio (p_audio={p_audio:.3f}), "
            f"collapsing fused probability to {p_fused:.3f}."
        )
    elif p_video >= 0.40 and w_audio >= 0.80:
        primary_cause = "DWF gating failure suppressing moderate visual manipulation evidence"
        failure_category = "DWF gating failure"
        root_cause_detail = (
            f"Visual branch had elevated fake score ({p_video:.3f}), but DWF assigned "
            f"{w_audio*100:.1f}% weight to audio ({p_audio:.3f}), suppressing visual evidence."
        )
    elif width != "N/A" and (int(width) < 640 or int(height) < 480):
        primary_cause = "Low resolution / heavy spatial compression"
        failure_category = "Compression or low resolution"
        root_cause_detail = f"Video resolution {resolution} degraded fine facial boundary artifacts."
    elif p_video < 0.15 and p_audio < 0.15:
        primary_cause = "Weak facial manipulation evidence / unseen manipulation method"
        failure_category = "Weak facial manipulation evidence"
        root_cause_detail = (
            f"Both visual ({p_video:.3f}) and audio ({p_audio:.3f}) branches produced low fake probabilities. "
            "Manipulation artifacts were imperceptible to pretrained ViT or from an unseen generator pipeline."
        )
    elif silent_audio == "True" or has_audio == "False":
        primary_cause = "Silent or absent audio misclassified by audio encoder"
        failure_category = "Audio branch misleading the fusion model"
        root_cause_detail = "Clip has silent/missing audio, but audio branch predicted authentic instead of neutral."
    else:
        primary_cause = "Unseen manipulation method / subtle deepfake synthesis"
        failure_category = "Unseen manipulation method"
        root_cause_detail = f"Visual branch ({p_video:.3f}) failed to recognize deepfake synthesis signatures."

    analysis_rows.append({
        "video_id": sid,
        "ground_truth": "FAKE",
        "predicted_label": "REAL",
        "p_fused": f"{p_fused:.4f}",
        "p_video": f"{p_video:.4f}",
        "p_audio": f"{p_audio:.4f}",
        "w_video": f"{w_video:.4f}",
        "w_audio": f"{w_audio:.4f}",
        "sync_score": sync_score,
        "video_duration": duration,
        "resolution": resolution,
        "fps": fps,
        "has_audio": has_audio,
        "silent_audio": silent_audio,
        "source_group": source_group,
        "failure_category": failure_category,
        "primary_cause": primary_cause,
        "root_cause_detail": root_cause_detail,
    })

# Save CSV
csv_out = OUTPUT_DIR / "false_real_failure_analysis.csv"
fieldnames = [
    "video_id",
    "ground_truth",
    "predicted_label",
    "p_fused",
    "p_video",
    "p_audio",
    "w_video",
    "w_audio",
    "sync_score",
    "video_duration",
    "resolution",
    "fps",
    "has_audio",
    "silent_audio",
    "source_group",
    "failure_category",
    "primary_cause",
    "root_cause_detail",
]

with open(csv_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(analysis_rows)

print(f"Generated {len(analysis_rows)} False Negative failure records to {csv_out}")

# Summary of categories
cat_counts: dict[str, int] = {}
for r in analysis_rows:
    c = r["failure_category"]
    cat_counts[c] = cat_counts.get(c, 0) + 1

print("\nFailure Category Breakdown:")
for c, cnt in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {c:<50} : {cnt} ({cnt/len(analysis_rows)*100:.1f}%)")
