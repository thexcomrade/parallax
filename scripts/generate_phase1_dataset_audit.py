"""Dataset quality and generalization audit script across all 1,000 canonical samples."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "experiments" / "model_reliability_improvement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

manifest_path = REPO_ROOT / "datasets" / "manifests" / "dataset_manifest.csv"
integrity_path = REPO_ROOT / "reports" / "phase1_dataset_integrity.csv"

assert manifest_path.exists(), f"Missing {manifest_path}"
assert integrity_path.exists(), f"Missing {integrity_path}"

# Load manifest
manifest_records: list[dict[str, str]] = []
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest_records = list(csv.DictReader(f))

# Load integrity
integrity_records: dict[str, dict[str, str]] = {}
with open(integrity_path, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        sid = r["filename"].upper().replace(".MP4", "")
        integrity_records[sid] = r

dataset_rows: list[dict[str, str]] = []

for m in manifest_records:
    sid = m["sample_id"]
    integ = integrity_records.get(sid, {})

    duration = float(integ.get("duration", 0.0))
    width = int(integ.get("width", 0))
    height = int(integ.get("height", 0))
    fps = float(integ.get("fps", 0.0))
    file_size = int(integ.get("file_size", 0))
    has_audio = integ.get("has_audio", "False") == "True"
    silent_audio = integ.get("silent_audio", "False") == "True"
    group_id = m.get("group_id", "source_identity_unknown")
    split = m.get("split", "unknown")
    label = m.get("label", "0")
    class_name = "FAKE" if label == "1" else "REAL"

    # Quality and generalization checks
    is_standard_hd = (width == 1920 and height == 1080)
    audio_status = "NORMAL" if (has_audio and not silent_audio) else ("SILENT" if silent_audio else "NO_AUDIO")

    dataset_rows.append({
        "sample_id": sid,
        "filename": m.get("filename", ""),
        "split": split,
        "label": label,
        "class_name": class_name,
        "group_id": group_id,
        "duration_s": f"{duration:.3f}",
        "width": str(width),
        "height": str(height),
        "resolution": f"{width}x{height}",
        "fps": f"{fps:.2f}",
        "file_size_bytes": str(file_size),
        "file_size_mb": f"{file_size / (1024*1024):.2f}",
        "has_audio": str(has_audio),
        "silent_audio": str(silent_audio),
        "audio_status": audio_status,
        "is_standard_hd": str(is_standard_hd),
        "sha256": integ.get("sha256", ""),
        "decode_status": integ.get("decode_status", "OK"),
    })

# Save dataset_quality_analysis.csv
out_csv = OUTPUT_DIR / "dataset_quality_analysis.csv"
fieldnames = [
    "sample_id", "filename", "split", "label", "class_name", "group_id",
    "duration_s", "width", "height", "resolution", "fps", "file_size_bytes",
    "file_size_mb", "has_audio", "silent_audio", "audio_status",
    "is_standard_hd", "sha256", "decode_status",
]

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(dataset_rows)

print(f"Saved {len(dataset_rows)} quality analysis records to {out_csv}")

# Print distribution summary
splits = {}
resolutions = {}
audio_counts = {}
group_counts = {}

for r in dataset_rows:
    sp = r["split"]
    splits.setdefault(sp, {"REAL": 0, "FAKE": 0})
    splits[sp][r["class_name"]] += 1

    res = r["resolution"]
    resolutions[res] = resolutions.get(res, 0) + 1

    aud = r["audio_status"]
    audio_counts[aud] = audio_counts.get(aud, 0) + 1

    grp = r["group_id"]
    group_counts[grp] = group_counts.get(grp, 0) + 1

print("\nSplit Distribution:")
for sp, counts in sorted(splits.items()):
    tot = counts["REAL"] + counts["FAKE"]
    print(f"  {sp:<12}: Total={tot:>4} | REAL={counts['REAL']:>3} | FAKE={counts['FAKE']:>3}")

print("\nResolution Distribution:")
for res, cnt in sorted(resolutions.items(), key=lambda x: x[1], reverse=True):
    print(f"  {res:<12}: {cnt}")

print("\nAudio Status Distribution:")
for aud, cnt in sorted(audio_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {aud:<12}: {cnt}")

print(f"\nSource Group Distribution: {len(group_counts)} distinct groups")
print(f"  'source_identity_unknown': {group_counts.get('source_identity_unknown', 0)} / {len(dataset_rows)} clips")
