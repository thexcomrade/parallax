"""Dataset validation and quality control CLI for PARALLAX.

Validates:
- REAL and FAKE raw folder contents (500 videos)
- File readability, duration, resolution, FPS, codecs, audio presence via FFprobe
- SHA-256 duplicate detection
- Class balance and distribution
- Manifest and split consistency (train/validation/test)
- Data leakage audit
- Emits structured reports to:
    datasets/reports/dataset_validation_report.json
    datasets/reports/dataset_validation_report.csv
    datasets/reports/dataset_summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add backend to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_dataset")

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def audit_manifests(
    manifests_dir: Path, readable_paths: set[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit dataset manifests for consistency, duplicates, and leakage."""
    manifest_csv = manifests_dir / "dataset_manifest.csv"
    train_csv = manifests_dir / "train.csv"
    val_csv = manifests_dir / "validation.csv"
    test_csv = manifests_dir / "test.csv"

    audit_summary: dict[str, Any] = {
        "manifests_present": True,
        "global_manifest_count": 0,
        "train_count": 0,
        "validation_count": 0,
        "test_count": 0,
        "split_leakage_detected": False,
        "leakage_details": [],
        "missing_files_in_manifest": [],
        "unassigned_media_files": [],
    }

    if not all(p.exists() for p in (manifest_csv, train_csv, val_csv, test_csv)):
        audit_summary["manifests_present"] = False
        return audit_summary, []

    def read_csv_samples(path: Path) -> list[dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    global_rows = read_csv_samples(manifest_csv)
    train_rows = read_csv_samples(train_csv)
    val_rows = read_csv_samples(val_csv)
    test_rows = read_csv_samples(test_csv)

    audit_summary["global_manifest_count"] = len(global_rows)
    audit_summary["train_count"] = len(train_rows)
    audit_summary["validation_count"] = len(val_rows)
    audit_summary["test_count"] = len(test_rows)

    train_ids = {r.get("sample_id") for r in train_rows if r.get("sample_id")}
    val_ids = {r.get("sample_id") for r in val_rows if r.get("sample_id")}
    test_ids = {r.get("sample_id") for r in test_rows if r.get("sample_id")}

    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        audit_summary["split_leakage_detected"] = True
        audit_summary["leakage_details"] = [
            {"train_val_overlap": list(train_val_overlap)},
            {"train_test_overlap": list(train_test_overlap)},
            {"val_test_overlap": list(val_test_overlap)},
        ]

    # Verify every manifest sample exists on disk
    missing: list[str] = []
    manifest_paths: set[str] = set()
    for row in global_rows:
        rel = row.get("relative_path", "")
        norm_rel = rel.replace("\\", "/")
        manifest_paths.add(norm_rel)
        if norm_rel not in readable_paths and not (REPO_ROOT / rel).exists():
            missing.append(rel)

    audit_summary["missing_files_in_manifest"] = missing

    # Media files on disk not registered in manifest
    unassigned = [p for p in readable_paths if p not in manifest_paths]
    audit_summary["unassigned_media_files"] = unassigned

    return audit_summary, global_rows


def validate_raw_dataset(
    real_dir: Path,
    fake_dir: Path,
    manifests_dir: Path,
    report_json: Path,
    report_csv: Path,
) -> dict[str, Any]:
    """Inspect all raw video samples in real/ and fake/ directories."""
    from app.ml.preprocess.probe import probe_media

    real_files = sorted(
        [p for p in real_dir.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )
    fake_files = sorted(
        [p for p in fake_dir.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )

    total_files = len(real_files) + len(fake_files)
    logger.info(
        "Found %d raw files (REAL: %d, FAKE: %d)", total_files, len(real_files), len(fake_files)
    )

    seen_hashes: dict[str, Path] = {}
    duplicates: list[dict[str, str]] = []
    readable_samples: list[dict[str, Any]] = []
    unreadable_samples: list[dict[str, str]] = []

    all_candidates = [(p, 0, "REAL") for p in real_files] + [(p, 1, "FAKE") for p in fake_files]

    for idx, (path, label_id, label_str) in enumerate(all_candidates, start=1):
        if idx % 50 == 0 or idx == total_files:
            logger.info("Probing video %d/%d (%s)...", idx, total_files, path.name)

        file_hash = compute_file_sha256(path)
        if file_hash in seen_hashes:
            duplicates.append(
                {
                    "original": safe_rel_path(seen_hashes[file_hash]),
                    "duplicate": safe_rel_path(path),
                    "sha256": file_hash[:16],
                }
            )
            continue
        seen_hashes[file_hash] = path

        try:
            probe = probe_media(path, max_duration_s=300.0)
            sample_rel = safe_rel_path(path)
            readable_samples.append(
                {
                    "sample_name": path.stem,
                    "filename": path.name,
                    "relative_path": sample_rel,
                    "label": label_str,
                    "label_id": label_id,
                    "duration_s": round(probe.duration_s, 2),
                    "resolution": f"{probe.width}x{probe.height}",
                    "width": probe.width,
                    "height": probe.height,
                    "fps": round(probe.fps, 2),
                    "audio_present": probe.audio_present,
                    "video_codec": probe.video_codec,
                    "audio_codec": probe.audio_codec or "none",
                    "size_bytes": probe.size_bytes,
                    "sha256": file_hash[:16],
                    "status": "VALID",
                }
            )
        except Exception as exc:
            logger.warning("Unreadable file %s: %s", path.name, exc)
            unreadable_samples.append(
                {
                    "path": safe_rel_path(path),
                    "error": str(exc),
                }
            )

    readable_paths = {s["relative_path"] for s in readable_samples}
    manifest_audit, _ = audit_manifests(manifests_dir, readable_paths)

    has_audio_count = sum(1 for s in readable_samples if s["audio_present"])
    real_count = sum(1 for s in readable_samples if s["label_id"] == 0)
    fake_count = sum(1 for s in readable_samples if s["label_id"] == 1)

    is_valid = (
        total_files > 0
        and len(unreadable_samples) == 0
        and real_count > 0
        and fake_count > 0
        and not manifest_audit.get("split_leakage_detected", False)
    )

    summary: dict[str, Any] = {
        "status": "VALID" if is_valid else ("INVALID" if total_files > 0 else "EMPTY"),
        "total_files_found": total_files,
        "readable_count": len(readable_samples),
        "unreadable_count": len(unreadable_samples),
        "duplicate_count": len(duplicates),
        "real_count": real_count,
        "fake_count": fake_count,
        "audio_present_count": has_audio_count,
        "audio_missing_count": len(readable_samples) - has_audio_count,
        "class_balance_ratio": round(fake_count / max(1, real_count), 2) if real_count > 0 else 0.0,
        "manifest_audit": manifest_audit,
        "duplicates": duplicates,
        "unreadable": unreadable_samples,
        "samples": readable_samples,
    }

    # Save JSON report
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("JSON validation report written to %s", report_json)

    # Save CSV report
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "sample_name",
        "filename",
        "relative_path",
        "label",
        "label_id",
        "duration_s",
        "resolution",
        "fps",
        "audio_present",
        "video_codec",
        "audio_codec",
        "size_bytes",
        "sha256",
        "status",
    ]
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for sample in readable_samples:
            row = {k: sample.get(k, "") for k in csv_fields}
            writer.writerow(row)
    logger.info("CSV validation report written to %s", report_csv)

    # Also update dataset_summary.json for backward compatibility
    dataset_summary_path = report_json.parent / "dataset_summary.json"
    dataset_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def validate_processed_dataset(
    processed_dir: Path, report_file: Path | None = None
) -> dict[str, Any]:
    """Validate processed feature caches."""
    video_dir = processed_dir / "video"
    audio_dir = processed_dir / "audio"

    video_samples = (
        list(video_dir.glob("*/*.pt")) + list(video_dir.glob("*.pt")) if video_dir.exists() else []
    )
    audio_samples = (
        list(audio_dir.glob("*/*.pt")) + list(audio_dir.glob("*.pt")) if audio_dir.exists() else []
    )

    summary = {
        "processed_dir": str(processed_dir),
        "video_tensors_count": len(video_samples),
        "audio_tensors_count": len(audio_samples),
        "status": "READY"
        if len(video_samples) > 0 and len(video_samples) == len(audio_samples)
        else "INCOMPLETE",
    }

    if report_file:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="PARALLAX Dataset Validation and Quality Control")
    parser.add_argument("--real-dir", type=Path, default=REPO_ROOT / "datasets" / "raw" / "real")
    parser.add_argument("--fake-dir", type=Path, default=REPO_ROOT / "datasets" / "raw" / "fake")
    parser.add_argument("--manifests-dir", type=Path, default=REPO_ROOT / "datasets" / "manifests")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "datasets" / "processed")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=REPO_ROOT / "datasets" / "reports" / "dataset_validation_report.json",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=REPO_ROOT / "datasets" / "reports" / "dataset_validation_report.csv",
    )
    parser.add_argument(
        "--processed",
        action="store_true",
        help="Validate processed feature tensors instead of raw media",
    )
    args = parser.parse_args()

    if args.processed:
        res = validate_processed_dataset(args.processed_dir, args.report_json)
        print(json.dumps(res, indent=2))
        sys.exit(0 if res["status"] == "READY" else 1)
    else:
        res = validate_raw_dataset(
            args.real_dir,
            args.fake_dir,
            args.manifests_dir,
            args.report_json,
            args.report_csv,
        )
        print("\n" + "=" * 60)
        print("          PARALLAX DATASET VALIDATION REPORT")
        print("=" * 60)
        print(f"Overall Status:        {res['status']}")
        print(f"Total Media Inspected: {res['total_files_found']}")
        print(f"Readable Videos:       {res['readable_count']}")
        print(f"  • REAL (Label 0):    {res['real_count']}")
        print(f"  • FAKE (Label 1):    {res['fake_count']}")
        print(f"Audio Streams Present: {res['audio_present_count']} / {res['readable_count']}")
        print(f"Unreadable Corrupted:  {res['unreadable_count']}")
        print(f"SHA-256 Duplicates:    {res['duplicate_count']}")
        print("-" * 60)
        manifest_audit = res.get("manifest_audit", {})
        print(f"Global Manifest Count: {manifest_audit.get('global_manifest_count', 0)}")
        print(f"Train Split Count:     {manifest_audit.get('train_count', 0)}")
        print(f"Validation Split Count:{manifest_audit.get('validation_count', 0)}")
        print(f"Test Split Count:      {manifest_audit.get('test_count', 0)}")
        print(f"Data Leakage Detected: {manifest_audit.get('split_leakage_detected', False)}")
        print("-" * 60)
        print(f"JSON Report File:      {args.report_json}")
        print(f"CSV Report File:       {args.report_csv}")
        print("=" * 60)

        if res["status"] == "EMPTY":
            print("\n[NOTE] No raw training media found in datasets/raw/real or datasets/raw/fake.")
            sys.exit(0)
        elif res["status"] == "INVALID":
            print("\n[WARNING] Dataset validation failed quality thresholds.")
            sys.exit(1)
        else:
            print("\n[SUCCESS] Dataset validated and ready for preprocessing.")
            sys.exit(0)


if __name__ == "__main__":
    main()
