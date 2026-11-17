"""Dataset preparation CLI for PARALLAX.

Supports stages:
1. --stage manifest: Scan raw/real and raw/fake, extract identities, generate dataset_manifest.csv
2. --stage split: Generate identity-disjoint train.csv, validation.csv, test.csv + split_report.json
3. --stage preprocess: Preprocess video and audio into processed/ caches
4. --stage all: Run manifest -> split -> preprocess in sequence
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
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


def extract_identity_key(file_path: Path) -> str:
    """Extract identity key from filename prefix or parent directory."""
    stem = file_path.stem
    if "_" in stem:
        part = stem.split("_")[0]
        if part.isalnum():
            return part
    # If filename is just a number or simple name, check parent
    parent = file_path.parent.name.lower()
    if parent not in ("real", "fake", "authentic", "synthetic", "videos", "clips", "raw"):
        return parent
    return stem


def generate_manifest(
    real_dir: Path,
    fake_dir: Path,
    output_csv: Path,
) -> list[dict[str, Any]]:
    """Scan raw directories and build master dataset manifest."""
    real_files = [
        p for p in real_dir.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    fake_files = [
        p for p in fake_dir.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    total = len(real_files) + len(fake_files)
    if total == 0:
        logger.warning("No media files found in %s or %s", real_dir, fake_dir)
        return []

    rows: list[dict[str, Any]] = []

    for idx, path in enumerate(real_files, start=1):
        sample_id = f"real_{idx:05d}"
        identity = extract_identity_key(path)
        try:
            rel_path = str(path.relative_to(REPO_ROOT))
        except ValueError:
            rel_path = str(path)
        rows.append(
            {
                "sample_id": sample_id,
                "clip_path": str(path.resolve()),
                "relative_path": rel_path,
                "label": "REAL",
                "label_id": 0,
                "identity": identity,
                "audio_present": True,
            }
        )

    for idx, path in enumerate(fake_files, start=1):
        sample_id = f"fake_{idx:05d}"
        identity = extract_identity_key(path)
        try:
            rel_path = str(path.relative_to(REPO_ROOT))
        except ValueError:
            rel_path = str(path)
        rows.append(
            {
                "sample_id": sample_id,
                "clip_path": str(path.resolve()),
                "relative_path": rel_path,
                "label": "FAKE",
                "label_id": 1,
                "identity": identity,
                "audio_present": True,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "clip_path",
        "relative_path",
        "label",
        "label_id",
        "identity",
        "audio_present",
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
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """Partition manifest strictly by identity to avoid identity leakage across splits."""
    random.seed(seed)
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    with open(manifest_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    if not records:
        logger.warning("Empty manifest at %s", manifest_csv)
        return {"status": "EMPTY"}

    # Group by identity
    id_to_records: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        ident = r["identity"]
        id_to_records.setdefault(ident, []).append(r)

    identities = sorted(id_to_records.keys())
    random.shuffle(identities)

    n_total_ids = len(identities)
    n_train_ids = max(1, int(n_total_ids * train_ratio))
    n_val_ids = max(1, int(n_total_ids * val_ratio)) if n_total_ids > 2 else 0

    train_id_set = set(identities[:n_train_ids])
    val_id_set = set(identities[n_train_ids : n_train_ids + n_val_ids])
    test_id_set = set(identities[n_train_ids + n_val_ids :])

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for ident, recs in id_to_records.items():
        if ident in train_id_set:
            for r in recs:
                rc = dict(r)
                rc["split"] = "train"
                train_rows.append(rc)
        elif ident in val_id_set:
            for r in recs:
                rc = dict(r)
                rc["split"] = "val"
                val_rows.append(rc)
        else:
            for r in recs:
                rc = dict(r)
                rc["split"] = "test"
                test_rows.append(rc)

    # Write split CSVs
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [*list(records[0].keys()), "split"]

    def write_csv(path: Path, data: list[dict[str, Any]]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    write_csv(output_dir / "train.csv", train_rows)
    write_csv(output_dir / "validation.csv", val_rows)
    write_csv(output_dir / "test.csv", test_rows)

    split_report = {
        "status": "SPLIT_CREATED",
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "total_identities": n_total_ids,
        "total_samples": len(records),
        "train": {
            "identities": len(train_id_set),
            "samples": len(train_rows),
            "real_count": sum(1 for r in train_rows if str(r["label_id"]) == "0"),
            "fake_count": sum(1 for r in train_rows if str(r["label_id"]) == "1"),
        },
        "validation": {
            "identities": len(val_id_set),
            "samples": len(val_rows),
            "real_count": sum(1 for r in val_rows if str(r["label_id"]) == "0"),
            "fake_count": sum(1 for r in val_rows if str(r["label_id"]) == "1"),
        },
        "test": {
            "identities": len(test_id_set),
            "samples": len(test_rows),
            "real_count": sum(1 for r in test_rows if str(r["label_id"]) == "0"),
            "fake_count": sum(1 for r in test_rows if str(r["label_id"]) == "1"),
        },
        "leakage_check": {
            "train_val_overlap": len(train_id_set.intersection(val_id_set)),
            "train_test_overlap": len(train_id_set.intersection(test_id_set)),
            "val_test_overlap": len(val_id_set.intersection(test_id_set)),
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

    if skip_existing and v_target.is_file() and a_target.is_file() and m_target.is_file():
        return sample_id, True, True, None, None

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
            "label": label_str,
            "label_id": label_id,
            "face_coverage": float(aligned.face_coverage),
            "num_windows": int(aligned.video_windows.shape[0]),
            "video_shape": list(aligned.video_windows.shape),
            "audio_shape": list(aligned.audio_segments.shape),
            "audio_present": bool(aligned.audio_present),
            "window_timestamps": [float(t) for t in aligned.window_timestamps],
        }
        m_target.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return sample_id, True, False, meta, None
    except Exception as exc:
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
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    with open(manifest_csv, encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    if not records:
        logger.warning("Empty manifest for preprocessing at %s", manifest_csv)
        return {"status": "EMPTY"}

    if smoke:
        # Pick 2 REAL and 2 FAKE for verification smoke test
        real_recs = [
            r for r in records if str(r.get("label", r.get("label_id", ""))) in ("0", "REAL")
        ][:2]
        fake_recs = [
            r for r in records if str(r.get("label", r.get("label_id", ""))) in ("1", "FAKE")
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
    processed_details: list[dict[str, Any]] = []

    total = len(records)
    logger.info("Beginning batch preprocessing of %d media samples (workers=%d)...", total, workers)

    task_args = [(rec, video_out, audio_out, metadata_out, skip_existing) for rec in records]

    completed_done = 0
    if workers <= 1 or smoke:
        for args in task_args:
            sample_id, is_ok, is_skip, meta, err = _process_single_sample(args)
            completed_done += 1
            if is_skip:
                skipped_count += 1
            elif is_ok and meta:
                success_count += 1
                processed_details.append(meta)
                if smoke:
                    logger.info(
                        "Sample %s -> video_shape: %s, audio_shape: %s",
                        sample_id,
                        meta["video_shape"],
                        meta["audio_shape"],
                    )
            else:
                failed_count += 1
                failures.append({"sample_id": sample_id, "error": str(err)})
            if completed_done % 25 == 0 or completed_done == total:
                logger.info("Processed %d/%d samples...", completed_done, total)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_id = {
                executor.submit(_process_single_sample, arg): arg[0].get("sample_id", f"sample_{i}")
                for i, arg in enumerate(task_args, 1)
            }
            for future in as_completed(future_to_id):
                completed_done += 1
                sample_id, is_ok, is_skip, meta, err = future.result()
                if is_skip:
                    skipped_count += 1
                elif is_ok and meta:
                    success_count += 1
                    processed_details.append(meta)
                else:
                    failed_count += 1
                    failures.append({"sample_id": sample_id, "error": str(err)})

                if completed_done % 50 == 0 or completed_done == total:
                    logger.info(
                        "Progress: %d/%d samples complete (new: %d, cached: %d, failed: %d)",
                        completed_done,
                        total,
                        success_count,
                        skipped_count,
                        failed_count,
                    )

    report = {
        "status": "COMPLETED" if failed_count == 0 else "PARTIAL",
        "total_manifest_samples": total,
        "preprocessed_new": success_count,
        "skipped_existing": skipped_count,
        "failed_count": failed_count,
        "failures": failures,
        "sample_shapes": processed_details[:4],
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Preprocessing complete: %d new, %d cached, %d failed. Report: %s",
        success_count,
        skipped_count,
        failed_count,
        report_file,
    )
    return report


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
        "--split-report",
        type=Path,
        default=REPO_ROOT / "datasets" / "reports" / "split_report.json",
    )
    parser.add_argument(
        "--preprocess-report",
        type=Path,
        default=REPO_ROOT / "datasets" / "reports" / "preprocessing_report.json",
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
        generate_manifest(args.real_dir, args.fake_dir, args.manifest)

    if args.stage in ("split", "all"):
        logger.info("--- STAGE 2: IDENTITY-DISJOINT SPLIT GENERATION ---")
        if args.manifest.exists():
            generate_splits(
                args.manifest,
                args.manifests_dir,
                args.split_report,
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
