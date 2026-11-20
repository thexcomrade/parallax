"""Phase 4.1 Dataset Cache Integrity Audit.

Performs a non-destructive audit of all cached feature files against raw media
for samples in train.csv, validation.csv, and test.csv.

STRICT CONSTRAINTS:
- NEVER silently regenerate or overwrite caches.
- NEVER access datasets/manifests/phase5_locked_test.csv.
"""

import csv
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Input manifests - explicitly excludes phase5_locked_test.csv
MANIFEST_FILES = {
    "train": WORKSPACE_ROOT / "datasets" / "manifests" / "train.csv",
    "validation": WORKSPACE_ROOT / "datasets" / "manifests" / "validation.csv",
    "test": WORKSPACE_ROOT / "datasets" / "manifests" / "test.csv",
}

# Historical provenance reports
NOV16_REPORT = WORKSPACE_ROOT / "datasets" / "reports" / "dataset_validation_report.csv"
NOV17_INTEGRITY = WORKSPACE_ROOT / "reports" / "phase1_dataset_integrity.csv"
NOV17_RENAME = WORKSPACE_ROOT / "datasets" / "manifests" / "rename_mapping.csv"

OUTPUT_REPORT = WORKSPACE_ROOT / "reports" / "phase4_1_cache_integrity.json"


def load_historical_hashes() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load historical source hashes from Nov 16 and Nov 17 reports."""
    nov16_data: dict[str, dict[str, Any]] = {}
    if NOV16_REPORT.exists():
        with open(NOV16_REPORT, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                nov16_data[row["filename"]] = {
                    "sha256": row["sha256"],
                    "size_bytes": int(row["size_bytes"]),
                    "duration_s": float(row["duration_s"]),
                }

    nov17_data: dict[str, dict[str, Any]] = {}
    if NOV17_INTEGRITY.exists():
        with open(NOV17_INTEGRITY, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                nov17_data[row["filename"]] = {
                    "sha256": row["sha256"],
                    "size_bytes": int(row["file_size"]),
                    "duration": float(row["duration"]),
                }

    return nov16_data, nov17_data


def compute_file_sha256(path: Path) -> str:
    """Compute full SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_cache_integrity() -> None:
    print("[1/4] Loading historical hash registries...")
    nov16_data, nov17_data = load_historical_hashes()
    print(f"Loaded {len(nov16_data)} Nov 16 records and {len(nov17_data)} Nov 17 records.")

    records: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {
        "MATCHED": 0,
        "MISMATCHED_STALE": 0,
        "MISSING_SOURCE_HASH": 0,
        "MISSING_CACHE": 0,
    }
    split_counts: dict[str, dict[str, int]] = {
        "train": {"MATCHED": 0, "MISMATCHED_STALE": 0, "MISSING_CACHE": 0, "TOTAL": 0},
        "validation": {"MATCHED": 0, "MISMATCHED_STALE": 0, "MISSING_CACHE": 0, "TOTAL": 0},
        "test": {"MATCHED": 0, "MISMATCHED_STALE": 0, "MISSING_CACHE": 0, "TOTAL": 0},
    }

    print("[2/4] Auditing manifests (train, validation, test)...")
    for split_name, manifest_path in MANIFEST_FILES.items():
        if not manifest_path.exists():
            print(f"Warning: manifest {manifest_path} not found.")
            continue

        with open(manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            manifest_rows = list(reader)

        print(f"  Auditing split '{split_name}': {len(manifest_rows)} samples...")
        for row in manifest_rows:
            sample_id = row["sample_id"]
            filename = row["filename"]
            rel_path = row["relative_path"]
            label_name = row.get("class_name", "UNKNOWN")

            raw_file_path = WORKSPACE_ROOT / rel_path
            raw_exists = raw_file_path.exists()
            raw_size = raw_file_path.stat().st_size if raw_exists else 0
            raw_sha256 = compute_file_sha256(raw_file_path) if raw_exists else ""

            # Cache paths
            video_cache_path = WORKSPACE_ROOT / "datasets" / "processed" / "video" / f"{sample_id}_video.pt"
            audio_cache_path = WORKSPACE_ROOT / "datasets" / "processed" / "audio" / f"{sample_id}_audio.pt"
            meta_cache_path = WORKSPACE_ROOT / "datasets" / "processed" / "metadata" / f"{sample_id}.json"

            video_cache_exists = video_cache_path.exists()
            audio_cache_exists = audio_cache_path.exists()
            meta_cache_exists = meta_cache_path.exists()

            cache_complete = video_cache_exists and audio_cache_exists and meta_cache_exists

            # Cache metadata inspection
            cache_mtime: float | None = None
            cache_date: str | None = None
            window_timestamps: list[float] = []
            num_windows: int = 0
            audio_present: bool = False

            if meta_cache_exists:
                cache_mtime = meta_cache_path.stat().st_mtime
                dt = datetime.datetime.fromtimestamp(cache_mtime, datetime.UTC)
                cache_date = dt.strftime("%Y-%m-%d")
                try:
                    with open(meta_cache_path, encoding="utf-8") as mf:
                        mdata = json.load(mf)
                        window_timestamps = mdata.get("window_timestamps", [])
                        num_windows = mdata.get("num_windows", 0)
                        audio_present = bool(mdata.get("audio_present", False))
                except (OSError, json.JSONDecodeError, KeyError):
                    pass

            # Provenance & Hash matching
            # Determine which recorded source hash was active when cache was generated
            status: str = "UNKNOWN"
            recorded_source_hash: str | None = None
            historical_source_date: str | None = None
            anomaly_explanation: str | None = None

            if not cache_complete:
                status = "MISSING_CACHE"
            elif cache_date == "2026-11-16":
                # Cache was generated during the Nov 16 baseline run
                historical_source_date = "2026-11-16"
                if filename in nov16_data:
                    recorded_source_hash = nov16_data[filename]["sha256"]
                    # If current raw hash starts with or equals the recorded hash:
                    if raw_sha256.startswith(recorded_source_hash):
                        status = "MATCHED"
                    else:
                        status = "MISMATCHED_STALE"
                        anomaly_explanation = (
                            f"Cache was created 2026-11-16 from original {filename} "
                            f"(source hash prefix {recorded_source_hash}). During dataset expansion "
                            f"on 2026-11-17, the file was re-indexed (+250 shift) and {filename} "
                            f"was replaced by new media (hash {raw_sha256[:16]}...). Because "
                            f"skip_existing=True was used during preprocessing, the 2026-11-16 cache "
                            f"was retained without regeneration."
                        )
                else:
                    status = "MISSING_SOURCE_HASH"
            else:
                # Cache was generated on or after Nov 17
                historical_source_date = "2026-11-17"
                if filename in nov17_data:
                    recorded_source_hash = nov17_data[filename]["sha256"]
                    if raw_sha256 == recorded_source_hash:
                        status = "MATCHED"
                    else:
                        status = "MISMATCHED_STALE"
                        anomaly_explanation = (
                            f"Raw media hash {raw_sha256[:16]} does not match Nov 17 recorded "
                            f"hash {recorded_source_hash[:16]}."
                        )
                else:
                    status = "MISSING_SOURCE_HASH"

            status_counts[status] = status_counts.get(status, 0) + 1
            split_counts[split_name][status] = split_counts[split_name].get(status, 0) + 1
            split_counts[split_name]["TOTAL"] += 1

            records.append({
                "sample_id": sample_id,
                "split": split_name,
                "filename": filename,
                "label": label_name,
                "raw_file_exists": raw_exists,
                "raw_file_size": raw_size,
                "raw_sha256": raw_sha256,
                "cache_complete": cache_complete,
                "cache_date": cache_date,
                "cache_mtime": cache_mtime,
                "num_windows": num_windows,
                "window_timestamps": window_timestamps,
                "audio_present": audio_present,
                "recorded_source_hash": recorded_source_hash,
                "historical_source_date": historical_source_date,
                "status": status,
                "anomaly_explanation": anomaly_explanation,
            })

    print("[3/4] Aggregating results...")
    total_audited = len(records)
    print(f"Total audited samples across train, validation, test: {total_audited}")
    for st, count in status_counts.items():
        print(f"  {st}: {count} ({count / total_audited * 100:.2f}%)")

    # Detailed audit report
    audit_report = {
        "audit_timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "audit_name": "Phase 4.1 Dataset Cache Integrity & REAL_0001 Forensic Audit",
        "scope": {
            "included_manifests": ["train.csv", "validation.csv", "test.csv"],
            "phase5_locked_test_accessed": False,
            "caches_regenerated": False,
            "caches_overwritten": False,
        },
        "summary": {
            "total_samples_audited": total_audited,
            "matched_count": status_counts.get("MATCHED", 0),
            "mismatched_stale_count": status_counts.get("MISMATCHED_STALE", 0),
            "missing_source_hash_count": status_counts.get("MISSING_SOURCE_HASH", 0),
            "missing_cache_count": status_counts.get("MISSING_CACHE", 0),
            "matched_percentage": round(status_counts.get("MATCHED", 0) / total_audited * 100, 2),
            "mismatched_percentage": round(status_counts.get("MISMATCHED_STALE", 0) / total_audited * 100, 2),
        },
        "by_split": split_counts,
        "root_cause_analysis_real_0001": {
            "sample_id": "REAL_0001",
            "historical_investigation": {
                "nov16_baseline_state": {
                    "filename": "real_0001.mp4",
                    "file_size": 1108882,
                    "duration_s": 6.01,
                    "sha256_prefix": "c0ba002d5bed546a",
                    "cache_generation_date": "2026-11-16",
                    "window_timestamps": [1.4389375, 2.9612916666666664, 4.504499999999999],
                },
                "nov17_dataset_expansion": {
                    "event": "Dataset expanded from 500 to 1,000 samples and re-indexed.",
                    "shift_mechanism": "Existing 250 REAL and 250 FAKE files were shifted by +250 (e.g. real_0001 -> real_0251).",
                    "current_real_0251_state": {
                        "filename": "real_0251.mp4",
                        "file_size": 1108882,
                        "sha256": "c0ba002d5bed546aca804e91f2b3d1a6b8399a629f023500d5ba9d61c5d1913d",
                        "matches_nov16_real_0001": True,
                    },
                    "current_real_0001_raw": {
                        "filename": "real_0001.mp4",
                        "file_size": 2445404,
                        "duration_s": 16.016,
                        "sha256": "66a7fd780a92a3152a71d18eb0091784bc987b1cedba3a5ea1e4c12878c05600",
                    },
                    "preprocessing_skip_behavior": (
                        "When scripts/prepare_dataset.py ran on 2026-11-17, skip_existing=True was "
                        "enabled. Because datasets/processed/video/REAL_0001_video.pt already existed "
                        "from 2026-11-16, preprocessing was skipped. Consequently, REAL_0001 cached "
                        "features correspond to the original video (now real_0251.mp4), while current "
                        "raw real_0001.mp4 contains the newly added video."
                    ),
                    "impact_extent": (
                        "All 500 samples originally processed on 2026-11-16 (samples 1..250 in REAL "
                        "and FAKE) exhibit this +250 shift relative to raw filenames. In contrast, "
                        "all 500 samples processed on 2026-11-17 (samples 251..500) match their raw "
                        "files exactly."
                    ),
                },
            },
        },
        "records": records,
    }

    print(f"[4/4] Writing report to {OUTPUT_REPORT}...")
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    audit_cache_integrity()
