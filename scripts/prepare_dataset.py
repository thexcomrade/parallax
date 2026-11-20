"""Dataset preparation CLI for PARALLAX.

Phase 1 Dataset Preparation, Manifest Generation, Group-Aware Splitting & Preprocessing.
Supports stages:
1. --stage manifest: Scan raw/real and raw/fake (1,000 videos), incorporate source groups and Phase 5 locks, generate dataset_manifest.csv
2. --stage split: Generate group-aware, identity-disjoint train.csv (70%), validation.csv (15%), test.csv (15%)
3. --stage preprocess: Preprocess video and audio into processed/ caches (video, audio, metadata)
4. --stage all: Run manifest -> split -> preprocess in sequence
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

# Add backend to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_dataset")

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


def safe_rel_path(path: Path) -> str:
    """Return forward-slash relative path from repo root."""
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def generate_manifest(
    real_dir: Path,
    fake_dir: Path,
    output_csv: Path,
    source_groups_csv: Path | None = None,
    phase5_locked_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Scan raw directories and build master dataset manifest."""
    real_files = sorted(
        [p for p in real_dir.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )
    fake_files = sorted(
        [p for p in fake_dir.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )

    total = len(real_files) + len(fake_files)
    if total == 0:
        logger.warning("No media files found in %s or %s", real_dir, fake_dir)
        return []

    # Load source groups if available
    group_map: dict[str, str] = {}
    if source_groups_csv and source_groups_csv.exists():
        with open(source_groups_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                group_map[r["filename"]] = r.get("group_id", "source_identity_unknown")

    # Load Phase 5 locked filenames if available
    locked_filenames: set[str] = set()
    if phase5_locked_csv and phase5_locked_csv.exists():
        with open(phase5_locked_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                locked_filenames.add(r["filename"])

    rows: list[dict[str, Any]] = []

    for idx, path in enumerate(real_files, start=1):
        sample_id = f"REAL_{idx:04d}"
        rel_path = safe_rel_path(path)
        gid = group_map.get(path.name, "source_identity_unknown")
        is_locked = path.name in locked_filenames
        rows.append(
            {
                "sample_id": sample_id,
                "filename": path.name,
                "relative_path": rel_path,
                "clip_path": str(path.resolve()),
                "label": "REAL",
                "label_id": 0,
                "class_name": "REAL",
                "group_id": gid,
                "phase5_locked": is_locked,
                "split": "phase5_locked" if is_locked else "unassigned",
            }
        )

    for idx, path in enumerate(fake_files, start=1):
        sample_id = f"FAKE_{idx:04d}"
        rel_path = safe_rel_path(path)
        gid = group_map.get(path.name, "source_identity_unknown")
        is_locked = path.name in locked_filenames
        rows.append(
            {
                "sample_id": sample_id,
                "filename": path.name,
                "relative_path": rel_path,
                "clip_path": str(path.resolve()),
                "label": "FAKE",
                "label_id": 1,
                "class_name": "FAKE",
                "group_id": gid,
                "phase5_locked": is_locked,
                "split": "phase5_locked" if is_locked else "unassigned",
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "filename",
        "relative_path",
        "clip_path",
        "label",
        "label_id",
        "class_name",
        "group_id",
        "phase5_locked",
        "split",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Generated master manifest at %s with %d records", output_csv, len(rows))
    return rows


def generate_splits(
    manifest_csv: Path,
    output_dir: Path,
    report_file: Path,
    phase5_locked_csv: Path | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """Partition manifest strictly by group_id and class balance, excluding Phase 5 test samples."""
    random.seed(seed)
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    with open(manifest_csv, encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    if not records:
        logger.warning("Empty manifest at %s", manifest_csv)
        return {"status": "EMPTY"}

    # Load locked Phase 5 filenames
    locked_filenames: set[str] = set()
    if phase5_locked_csv and phase5_locked_csv.exists():
        with open(phase5_locked_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                locked_filenames.add(r["filename"])

    # Identify any groups that contain locked test samples
    group_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        group_to_records[r["group_id"]].append(r)

    locked_groups: set[str] = set()
    for r in records:
        if r["filename"] in locked_filenames and r["group_id"] != "source_identity_unknown":
            locked_groups.add(r["group_id"])

    excluded_filenames = set(locked_filenames)
    paired_excluded_filenames: set[str] = set()
    for gid in locked_groups:
        for r in group_to_records[gid]:
            if r["filename"] not in locked_filenames:
                paired_excluded_filenames.add(r["filename"])
            excluded_filenames.add(r["filename"])

    logger.info(
        "Excluding %d samples (%d Phase 5 locked test + %d paired near-duplicate counterparts)",
        len(excluded_filenames),
        len(locked_filenames),
        len(paired_excluded_filenames),
    )

    # Separate eligible units into multi-sample groups and singletons
    multi_units: dict[str, list[dict[str, Any]]] = {}
    single_reals: list[dict[str, Any]] = []
    single_fakes: list[dict[str, Any]] = []

    for gid, rows in group_to_records.items():
        if gid in locked_groups or gid == "source_identity_unknown":
            continue
        multi_units[gid] = rows

    for r in records:
        fn = r["filename"]
        if fn in excluded_filenames:
            continue
        if r["group_id"] == "source_identity_unknown":
            if str(r.get("label", "")) in ("0", "REAL") or str(r.get("class_name", "")) == "REAL":
                single_reals.append(r)
            else:
                single_fakes.append(r)

    random.shuffle(single_reals)
    random.shuffle(single_fakes)

    m_keys = sorted(multi_units.keys())
    random.shuffle(m_keys)

    total_eligible = sum(len(rows) for rows in multi_units.values()) + len(single_reals) + len(single_fakes)
    total_eligible_real = sum(1 for rows in multi_units.values() for x in rows if str(x.get("label")) == "0" or x.get("class_name") == "REAL") + len(single_reals)
    total_eligible_fake = sum(1 for rows in multi_units.values() for x in rows if str(x.get("label")) == "1" or x.get("class_name") == "FAKE") + len(single_fakes)

    target_train_total = round(total_eligible * train_ratio)
    target_val_total = round(total_eligible * val_ratio)

    # Target class distributions
    target_train_real = target_train_total // 2
    target_train_fake = target_train_total - target_train_real

    target_val_real = (target_val_total + 1) // 2
    target_val_fake = target_val_total - target_val_real

    target_test_real = total_eligible_real - target_train_real - target_val_real
    target_test_fake = total_eligible_fake - target_train_fake - target_val_fake

    splits: dict[str, dict[str, Any]] = {
        "train": {"rows": [], "target_real": target_train_real, "target_fake": target_train_fake},
        "val": {"rows": [], "target_real": target_val_real, "target_fake": target_val_fake},
        "test": {"rows": [], "target_real": target_test_real, "target_fake": target_test_fake},
    }

    # Greedily assign multi-sample groups to split with most remaining room
    for gid in m_keys:
        rows = multi_units[gid]
        r_count = sum(1 for x in rows if str(x.get("label")) == "0" or x.get("class_name") == "REAL")
        f_count = sum(1 for x in rows if str(x.get("label")) == "1" or x.get("class_name") == "FAKE")

        best_split = None
        best_margin = -9999
        for s_name in ["train", "val", "test"]:
            curr_r = sum(1 for x in splits[s_name]["rows"] if str(x.get("label")) == "0" or x.get("class_name") == "REAL")
            curr_f = sum(1 for x in splits[s_name]["rows"] if str(x.get("label")) == "1" or x.get("class_name") == "FAKE")
            rem_r = splits[s_name]["target_real"] - curr_r
            rem_f = splits[s_name]["target_fake"] - curr_f
            if rem_r >= r_count and rem_f >= f_count:
                margin = min(rem_r - r_count, rem_f - f_count)
                if margin > best_margin:
                    best_margin = margin
                    best_split = s_name
        if best_split is None:
            best_split = "train"
        splits[best_split]["rows"].extend(rows)

    # Fill remaining slots with singletons
    for s_name in ["train", "val", "test"]:
        curr_r = sum(1 for x in splits[s_name]["rows"] if str(x.get("label")) == "0" or x.get("class_name") == "REAL")
        needed_r = splits[s_name]["target_real"] - curr_r
        for _ in range(needed_r):
            if single_reals:
                splits[s_name]["rows"].append(single_reals.pop())

        curr_f = sum(1 for x in splits[s_name]["rows"] if str(x.get("label")) == "1" or x.get("class_name") == "FAKE")
        needed_f = splits[s_name]["target_fake"] - curr_f
        for _ in range(needed_f):
            if single_fakes:
                splits[s_name]["rows"].append(single_fakes.pop())

    train_rows = splits["train"]["rows"]
    val_rows = splits["val"]["rows"]
    test_rows = splits["test"]["rows"]

    for r in train_rows:
        r["split"] = "train"
    for r in val_rows:
        r["split"] = "val"
    for r in test_rows:
        r["split"] = "test"

    # Assign split in master manifest records
    assigned_map = {r["filename"]: r["split"] for r in (train_rows + val_rows + test_rows)}
    for r in records:
        fn = r["filename"]
        if fn in assigned_map:
            r["split"] = assigned_map[fn]
        elif fn in locked_filenames:
            r["split"] = "phase5_locked"
        elif fn in paired_excluded_filenames:
            r["split"] = "phase5_paired_excluded"

    # Write master manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    master_fields = [
        "sample_id",
        "filename",
        "relative_path",
        "clip_path",
        "label",
        "class_name",
        "group_id",
        "phase5_locked",
        "split",
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=master_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    # Write split CSVs
    split_fields = [
        "sample_id",
        "filename",
        "relative_path",
        "label",
        "class_name",
        "split",
    ]

    def write_split_csv(path: Path, data: list[dict[str, Any]], split_label: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=split_fields)
            w.writeheader()
            for r in data:
                w.writerow({
                    "sample_id": r["sample_id"],
                    "filename": r["filename"],
                    "relative_path": r["relative_path"],
                    "label": r.get("label", 0),
                    "class_name": r.get("class_name", "REAL"),
                    "split": split_label,
                })

    write_split_csv(output_dir / "train.csv", train_rows, "train")
    write_split_csv(output_dir / "validation.csv", val_rows, "val")
    write_split_csv(output_dir / "test.csv", test_rows, "test")

    # Leakage check
    train_ids = {r["sample_id"] for r in train_rows}
    val_ids = {r["sample_id"] for r in val_rows}
    test_ids = {r["sample_id"] for r in test_rows}
    locked_ids = {r["sample_id"] for r in records if r["filename"] in locked_filenames}

    split_report = {
        "status": "SPLIT_CREATED",
        "seed": seed,
        "total_manifest_samples": len(records),
        "excluded_phase5_locked_samples": len(locked_filenames),
        "excluded_phase5_paired_samples": len(paired_excluded_filenames),
        "total_eligible_samples": total_eligible,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "train": {
            "samples": len(train_rows),
            "real_count": sum(1 for r in train_rows if str(r.get("label")) == "0" or r.get("class_name") == "REAL"),
            "fake_count": sum(1 for r in train_rows if str(r.get("label")) == "1" or r.get("class_name") == "FAKE"),
        },
        "validation": {
            "samples": len(val_rows),
            "real_count": sum(1 for r in val_rows if str(r.get("label")) == "0" or r.get("class_name") == "REAL"),
            "fake_count": sum(1 for r in val_rows if str(r.get("label")) == "1" or r.get("class_name") == "FAKE"),
        },
        "test": {
            "samples": len(test_rows),
            "real_count": sum(1 for r in test_rows if str(r.get("label")) == "0" or r.get("class_name") == "REAL"),
            "fake_count": sum(1 for r in test_rows if str(r.get("label")) == "1" or r.get("class_name") == "FAKE"),
        },
        "leakage_check": {
            "train_val_overlap": len(train_ids.intersection(val_ids)),
            "train_test_overlap": len(train_ids.intersection(test_ids)),
            "val_test_overlap": len(val_ids.intersection(test_ids)),
            "train_locked_overlap": len(train_ids.intersection(locked_ids)),
            "val_locked_overlap": len(val_ids.intersection(locked_ids)),
            "test_locked_overlap": len(test_ids.intersection(locked_ids)),
            "is_zero_leakage": bool(
                len(train_ids.intersection(val_ids)) == 0
                and len(train_ids.intersection(test_ids)) == 0
                and len(val_ids.intersection(test_ids)) == 0
                and len(train_ids.intersection(locked_ids)) == 0
                and len(val_ids.intersection(locked_ids)) == 0
                and len(test_ids.intersection(locked_ids)) == 0
            ),
        },
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(split_report, indent=2), encoding="utf-8")
    logger.info("Split report written to %s", report_file)
    return split_report


def _process_single_sample(
    args: tuple[dict[str, Any], Path, Path, Path, bool],
) -> tuple[str, bool, bool, dict[str, Any] | None, str | None]:
    """Worker function for concurrent sample preprocessing.

    Returns: (sample_id, is_success, is_skipped, metadata_dict, error_msg)
    """
    from app.ml.preprocess.align import align_audio_video
    from app.ml.preprocess.audio import extract_audio_features
    from app.ml.preprocess.video import extract_video_frames

    rec, video_out, audio_out, metadata_out, skip_existing = args
    sample_id = rec.get("sample_id") or Path(rec.get("filename", "sample")).stem
    raw_clip = rec.get("clip_path") or rec.get("relative_path") or ""
    clip_path = Path(raw_clip)
    if not clip_path.is_absolute():
        clip_path = REPO_ROOT / clip_path

    label_raw = rec.get("label_id") if rec.get("label_id") is not None else rec.get("label", 0)
    try:
        label_id = int(label_raw)
    except ValueError:
        label_id = 1 if str(label_raw).upper() == "FAKE" else 0
    label_str = rec.get("class_name") or ("FAKE" if label_id == 1 else "REAL")

    v_target = video_out / f"{sample_id}_video.pt"
    a_target = audio_out / f"{sample_id}_audio.pt"
    m_target = metadata_out / f"{sample_id}.json"

    raw_sha256 = rec.get("sha256")
    if not raw_sha256 and clip_path.is_file():
        h = hashlib.sha256()
        with open(clip_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        raw_sha256 = h.hexdigest()

    if skip_existing and v_target.is_file() and a_target.is_file() and m_target.is_file():
        try:
            m_data = json.loads(m_target.read_text(encoding="utf-8"))
            if (
                m_data.get("source_sha256") == raw_sha256
                and m_data.get("preprocessing_version") == "2.1.0"
            ):
                return sample_id, True, True, None, None
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    try:
        video_res = extract_video_frames(clip_path, target_count=32, window_size=16, stride=8)
        audio_res = extract_audio_features(clip_path)
        aligned = align_audio_video(video_res, audio_res, window_size=16, stride=8)

        torch.save(aligned.video_windows, v_target)
        torch.save(
            {
                "audio_segments": aligned.audio_segments,
                "audio_mask": aligned.audio_mask,
                "audio_present": aligned.audio_present,
            },
            a_target,
        )

        meta = {
            "sample_id": sample_id,
            "raw_source_path": safe_rel_path(clip_path),
            "source_sha256": raw_sha256,
            "label": label_str,
            "label_id": label_id,
            "split": rec.get("split", "unassigned"),
            "preprocessing_version": "2.1.0",
            "feature_configuration": {
                "target_count": 32,
                "window_size": 16,
                "stride": 8,
                "image_size": [224, 224],
                "sample_rate": 16000,
                "n_mfcc": 40,
                "n_mels": 64,
            },
            "extraction_timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "face_coverage": float(aligned.face_coverage),
            "num_windows": int(aligned.video_windows.shape[0]),
            "video_shape": list(aligned.video_windows.shape),
            "video_dtype": "float32",
            "audio_shape": list(aligned.audio_segments.shape),
            "audio_dtype": "float32",
            "audio_present": bool(aligned.audio_present),
            "window_timestamps": [float(t) for t in aligned.window_timestamps],
        }
        m_target.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return sample_id, True, False, meta, None
    except Exception as exc:  # noqa: BLE001
        return sample_id, False, False, None, str(exc)


def preprocess_dataset(
    manifest_csv: Path,
    processed_dir: Path,
    report_file: Path,
    smoke: bool = False,
    skip_existing: bool = True,
    limit: int | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Run video/audio preprocessing and cache features."""
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    with open(manifest_csv, encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    if not records:
        logger.warning("Empty manifest for preprocessing at %s", manifest_csv)
        return {"status": "EMPTY"}

    if smoke:
        real_recs = [
            r for r in records if str(r.get("label", r.get("class_name", ""))) in ("0", "REAL")
        ][:2]
        fake_recs = [
            r for r in records if str(r.get("label", r.get("class_name", ""))) in ("1", "FAKE")
        ][:2]
        records = real_recs + fake_recs
        logger.info(
            "SMOKE MODE: selected %d samples for verification (%d real, %d fake)",
            len(records),
            len(real_recs),
            len(fake_recs),
        )
    elif limit is not None:
        records = records[:limit]

    video_out = processed_dir / "video"
    audio_out = processed_dir / "audio"
    metadata_out = processed_dir / "metadata"
    for d in (video_out, audio_out, metadata_out):
        d.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skipped_count = 0
    failed_count = 0
    failures: list[dict[str, str]] = []

    tasks = [(r, video_out, audio_out, metadata_out, skip_existing) for r in records]
    total_tasks = len(tasks)
    logger.info("Preprocessing %d samples with %d workers (skip_existing=%s)...", total_tasks, workers, skip_existing)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_process_single_sample, t): t[0].get("sample_id") for t in tasks}
        for done_count, fut in enumerate(as_completed(future_map), start=1):
            sample_id, success, skipped, _, err_msg = fut.result()
            if skipped:
                skipped_count += 1
            elif success:
                success_count += 1
            else:
                failed_count += 1
                failures.append({"sample_id": sample_id, "error": str(err_msg)})

            if done_count % 50 == 0 or done_count == total_tasks:
                logger.info(
                    "Preprocessing progress: %d/%d (new: %d, skipped: %d, failed: %d)",
                    done_count,
                    total_tasks,
                    success_count,
                    skipped_count,
                    failed_count,
                )

    report_data = {
        "status": "COMPLETED" if failed_count == 0 else "PARTIAL_FAILURE",
        "total_manifest_samples": total_tasks,
        "successfully_processed": success_count,
        "skipped_existing": skipped_count,
        "failed_count": failed_count,
        "failures": failures,
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    logger.info("Preprocessing report written to %s", report_file)
    return report_data


def main() -> None:
    parser = argparse.ArgumentParser(description="PARALLAX Dataset Preparation CLI")
    parser.add_argument(
        "--stage", choices=["manifest", "split", "preprocess", "all"], default="all"
    )
    parser.add_argument("--real-dir", type=Path, default=REPO_ROOT / "datasets" / "raw" / "real")
    parser.add_argument("--fake-dir", type=Path, default=REPO_ROOT / "datasets" / "raw" / "fake")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "datasets" / "manifests" / "dataset_manifest.csv",
    )
    parser.add_argument("--manifests-dir", type=Path, default=REPO_ROOT / "datasets" / "manifests")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "datasets" / "processed")
    parser.add_argument(
        "--source-groups",
        type=Path,
        default=REPO_ROOT / "datasets" / "manifests" / "source_groups.csv",
    )
    parser.add_argument(
        "--phase5-locked",
        type=Path,
        default=REPO_ROOT / "datasets" / "manifests" / "phase5_locked_test.csv",
    )
    parser.add_argument(
        "--split-report",
        type=Path,
        default=REPO_ROOT / "reports" / "phase1_source_leakage_audit.json",
    )
    parser.add_argument(
        "--preprocess-report",
        type=Path,
        default=REPO_ROOT / "reports" / "phase1_cache_audit.json",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke", action="store_true", help="Run fast preprocessing smoke test on 4 samples"
    )
    parser.add_argument(
        "--force-reprocess", action="store_true", help="Overwrite existing cached features"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples to process"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers for preprocessing"
    )
    args = parser.parse_args()

    if args.stage in ("manifest", "all"):
        logger.info("--- STAGE 1: MANIFEST GENERATION ---")
        generate_manifest(
            args.real_dir,
            args.fake_dir,
            args.manifest,
            args.source_groups,
            args.phase5_locked,
        )

    if args.stage in ("split", "all"):
        logger.info("--- STAGE 2: IDENTITY-DISJOINT SPLIT GENERATION ---")
        if args.manifest.exists():
            generate_splits(
                args.manifest,
                args.manifests_dir,
                args.split_report,
                args.phase5_locked,
                args.train_ratio,
                args.val_ratio,
                args.test_ratio,
                args.seed,
            )
        else:
            logger.warning("Manifest not found at %s. Skipping split stage.", args.manifest)

    if args.stage in ("preprocess", "all"):
        logger.info("--- STAGE 3: DATASET PREPROCESSING ---")
        if args.manifest.exists():
            preprocess_dataset(
                args.manifest,
                args.processed_dir,
                args.preprocess_report,
                smoke=args.smoke,
                skip_existing=not args.force_reprocess,
                limit=args.limit,
                workers=args.workers,
            )
        else:
            logger.warning("Manifest not found at %s. Skipping preprocessing stage.", args.manifest)


if __name__ == "__main__":
    main()
