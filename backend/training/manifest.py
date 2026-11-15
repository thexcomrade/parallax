"""Dataset manifest generation with identity-disjoint train/val splitting.

Rules (README §5, §8.7):
- Manifest columns: clip_path, label, identity, audio_present, split
- Split policy: Partition strictly by subject identity (80/20) to prevent
  facial identity leakage across training and validation sets.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

logger = logging.getLogger("parallax.training.manifest")


def extract_identity_from_path(path: Path) -> str:
    """Extract identity key from file name or parent directory."""
    name = path.stem
    # If format is id00123_001.mp4 or similar
    if "_" in name and name.split("_")[0].isalnum():
        return name.split("_")[0]
    # Fallback to parent directory name if indicative of identity
    if path.parent.name not in ("real", "fake", "authentic", "synthetic", "videos", "clips"):
        return path.parent.name
    return name


def build_manifest(
    dataset_dir: Path,
    output_csv: Path,
    split_ratio: float = 0.8,
    seed: int = 42,
) -> Path:
    """Scan dataset directory, extract identities, perform disjoint split, and write CSV."""
    random.seed(seed)
    extensions = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
    all_files = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in extensions]

    if not all_files:
        raise ValueError(f"No video files found in {dataset_dir}")

    # Map identity -> list of entries
    id_to_records: dict[str, list[dict[str, str | int | bool]]] = {}

    for file_path in all_files:
        path_str = str(file_path.resolve())
        # Infer label: check if path contains "fake", "synthetic", "manipulated"
        lower_parts = [p.lower() for p in file_path.parts]
        fake_keywords = ("fake", "synthetic", "manipulated", "swap", "wav2lip")
        is_fake = any(k in lower_parts or k in file_path.stem.lower() for k in fake_keywords)
        label = 1 if is_fake else 0
        identity = extract_identity_from_path(file_path)

        record: dict[str, str | int | bool] = {
            "clip_path": path_str,
            "label": label,
            "identity": identity,
            "audio_present": True,
        }
        id_to_records.setdefault(identity, []).append(record)

    # Disjoint split by identity
    all_identities = sorted(id_to_records.keys())
    random.shuffle(all_identities)

    n_train = max(1, int(len(all_identities) * split_ratio))
    train_ids = set(all_identities[:n_train])

    rows: list[dict[str, str | int | bool]] = []
    for ident, records in id_to_records.items():
        split = "train" if ident in train_ids else "val"
        for rec in records:
            rec_with_split = dict(rec)
            rec_with_split["split"] = split
            rows.append(rec_with_split)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip_path", "label", "identity", "audio_present", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Manifest created at %s: %d clips across %d identities.",
        output_csv,
        len(rows),
        len(all_identities),
    )
    return output_csv


def build_smoke_manifest(output_dir: Path) -> Path:
    """Build a synthetic smoke test manifest using the repo's test fixture."""
    fixture_candidates = [
        Path("tests/fixtures/test_sample.mp4"),
        Path("backend/tests/fixtures/test_sample.mp4"),
        Path("../frontend/tests/fixtures/test_sample.mp4"),
        Path("frontend/tests/fixtures/test_sample.mp4"),
    ]
    fixture_path: Path | None = None
    for p in fixture_candidates:
        if p.exists():
            fixture_path = p.resolve()
            break

    if not fixture_path:
        raise FileNotFoundError("Could not locate test_sample.mp4 fixture for smoke manifest")

    output_csv = output_dir / "smoke_manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4 synthetic rows: 2 train (authentic & fake), 2 val (authentic & fake)
    rows = [
        {
            "clip_path": str(fixture_path),
            "label": 0,
            "identity": "subj_smoke_A",
            "audio_present": True,
            "split": "train",
        },
        {
            "clip_path": str(fixture_path),
            "label": 1,
            "identity": "subj_smoke_B",
            "audio_present": True,
            "split": "train",
        },
        {
            "clip_path": str(fixture_path),
            "label": 0,
            "identity": "subj_smoke_C",
            "audio_present": True,
            "split": "val",
        },
        {
            "clip_path": str(fixture_path),
            "label": 1,
            "identity": "subj_smoke_D",
            "audio_present": True,
            "split": "val",
        },
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip_path", "label", "identity", "audio_present", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build identity-disjoint dataset manifest")
    parser.add_argument("--dataset-dir", type=Path, help="Root directory containing dataset clips")
    parser.add_argument("--output-csv", type=Path, default=Path("storage/manifest.csv"))
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--smoke", action="store_true", help="Generate synthetic smoke manifest")

    args = parser.parse_args()
    if args.smoke:
        out = build_smoke_manifest(args.output_csv.parent)
        print(f"Smoke manifest generated at {out}")
    else:
        if not args.dataset_dir:
            parser.error("--dataset-dir is required when not running with --smoke")
        out = build_manifest(args.dataset_dir, args.output_csv, args.split_ratio)
        print(f"Manifest written to {out}")


if __name__ == "__main__":
    main()
