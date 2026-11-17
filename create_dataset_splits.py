
from pathlib import Path
import csv
import random
import shutil
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "datasets"
RAW_ROOT = DATASET_ROOT / "raw"
MANIFEST_ROOT = DATASET_ROOT / "manifests"

REAL_DIR = RAW_ROOT / "real"
FAKE_DIR = RAW_ROOT / "fake"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SUPPORTED_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"
}

CSV_FIELDS = [
    "sample_id",
    "video_path",
    "filename",
    "label",
    "label_name",
    "split"
]


# ============================================================
# VALIDATION
# ============================================================

def validate_directories():
    for directory in [REAL_DIR, FAKE_DIR]:
        if not directory.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {directory}"
            )

    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError("Split ratios must add up to 1.0")


def get_video_files(directory):
    return sorted(
        [
            file for file in directory.iterdir()
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda x: x.name.lower()
    )


# ============================================================
# SAFE FILE RENAMING
# ============================================================

def rename_files(directory, prefix):
    files = get_video_files(directory)

    if not files:
        raise RuntimeError(f"No videos found in {directory}")

    print(f"\nRenaming {prefix.upper()} files: {len(files)}")

    # First rename everything to temporary names.
    # This prevents filename collision during renaming.
    temporary_files = []

    for index, old_path in enumerate(files, start=1):
        temp_path = directory / (
            f"__temporary_{prefix}_{index:05d}{old_path.suffix.lower()}"
        )

        if temp_path.exists():
            raise FileExistsError(
                f"Temporary filename already exists: {temp_path}"
            )

        old_path.rename(temp_path)
        temporary_files.append(temp_path)

    # Rename temporary files to final sequential names.
    renamed_files = []

    for index, temp_path in enumerate(temporary_files, start=1):
        new_name = f"{prefix}_{index:04d}{temp_path.suffix.lower()}"
        new_path = directory / new_name

        if new_path.exists():
            raise FileExistsError(
                f"Target filename already exists: {new_path}"
            )

        temp_path.rename(new_path)
        renamed_files.append(new_path)

    print(f"Successfully renamed {len(renamed_files)} files.")

    return renamed_files


# ============================================================
# DATASET SPLITTING
# ============================================================

def split_class(files, label, label_name):
    rng = random.Random(RANDOM_SEED + label)
    files = list(files)
    rng.shuffle(files)

    total = len(files)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    def create_records(selected_files, split_name):
        records = []

        for file_path in selected_files:
            relative_path = file_path.relative_to(PROJECT_ROOT)

            records.append({
                "sample_id": file_path.stem,
                "video_path": relative_path.as_posix(),
                "filename": file_path.name,
                "label": label,
                "label_name": label_name,
                "split": split_name
            })

        return records

    return (
        create_records(train_files, "train"),
        create_records(val_files, "validation"),
        create_records(test_files, "test")
    )


# ============================================================
# CSV BACKUP AND WRITING
# ============================================================

def backup_existing_csvs():
    existing_files = [
        MANIFEST_ROOT / "train.csv",
        MANIFEST_ROOT / "validation.csv",
        MANIFEST_ROOT / "test.csv",
        MANIFEST_ROOT / "dataset_manifest.csv"
    ]

    existing_files = [f for f in existing_files if f.exists()]

    if not existing_files:
        return

    backup_dir = MANIFEST_ROOT / (
        "backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    backup_dir.mkdir(parents=True, exist_ok=True)

    for file_path in existing_files:
        shutil.copy2(file_path, backup_dir / file_path.name)

    print(f"\nExisting CSV backups saved to: {backup_dir}")


def write_csv(file_path, records):
    with file_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_FIELDS
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"Created: {file_path.name} ({len(records)} records)")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PARALLAX DATASET RENAME & SPLIT")
    print("=" * 60)

    validate_directories()
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    real_files = get_video_files(REAL_DIR)
    fake_files = get_video_files(FAKE_DIR)

    print(f"\nREAL videos found: {len(real_files)}")
    print(f"FAKE videos found: {len(fake_files)}")

    if len(real_files) != 500 or len(fake_files) != 500:
        raise ValueError(
            "Expected exactly 500 REAL and 500 FAKE videos. "
            f"Found REAL={len(real_files)}, FAKE={len(fake_files)}. "
            "No files have been renamed."
        )

    # Backup old manifests before changing anything.
    backup_existing_csvs()

    # Rename files.
    real_files = rename_files(REAL_DIR, "real")
    fake_files = rename_files(FAKE_DIR, "fake")

    # Create stratified splits.
    real_train, real_val, real_test = split_class(
        real_files, 0, "REAL"
    )

    fake_train, fake_val, fake_test = split_class(
        fake_files, 1, "FAKE"
    )

    train_records = real_train + fake_train
    val_records = real_val + fake_val
    test_records = real_test + fake_test

    # Deterministic shuffle within each split.
    random.Random(RANDOM_SEED).shuffle(train_records)
    random.Random(RANDOM_SEED + 1).shuffle(val_records)
    random.Random(RANDOM_SEED + 2).shuffle(test_records)

    all_records = train_records + val_records + test_records

    # Write CSV manifests.
    write_csv(MANIFEST_ROOT / "train.csv", train_records)
    write_csv(MANIFEST_ROOT / "validation.csv", val_records)
    write_csv(MANIFEST_ROOT / "test.csv", test_records)
    write_csv(MANIFEST_ROOT / "dataset_manifest.csv", all_records)

    # Final verification.
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)

    for split_name, records in [
        ("TRAIN", train_records),
        ("VALIDATION", val_records),
        ("TEST", test_records)
    ]:
        real_count = sum(r["label"] == 0 for r in records)
        fake_count = sum(r["label"] == 1 for r in records)

        print(
            f"{split_name:12} | "
            f"REAL: {real_count:3} | "
            f"FAKE: {fake_count:3} | "
            f"TOTAL: {len(records):3}"
        )

    print("-" * 60)
    print(f"TOTAL VIDEOS: {len(all_records)}")
    print("RENAME STATUS: SUCCESS")
    print("SPLIT STATUS: SUCCESS")
    print("CSV STATUS: UPDATED")
    print("=" * 60)


if __name__ == "__main__":
    main()
