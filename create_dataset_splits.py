
from pathlib import Path
import csv
import random
import uuid
import sys

# ============================================================
# PARALLAX DATASET PREPARATION
# Rename + Manifest Generation + Train/Validation/Test Split
# ============================================================

# Project root = folder containing this script
PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "datasets" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "manifests"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v"
}

RANDOM_SEED = 42

# Split proportions
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# STEP 1: GET VIDEO FILES
# ============================================================

def get_video_files(folder_name):

    folder = DATASET_DIR / folder_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    files = [
        file for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in VIDEO_EXTENSIONS
    ]

    return sorted(files, key=lambda f: f.name.lower())


# ============================================================
# STEP 2: SAFE FILE RENAMING
# ============================================================

def rename_files(folder_name, prefix):

    folder = DATASET_DIR / folder_name
    files = get_video_files(folder_name)

    if not files:
        print(f"[WARNING] No videos found in {folder_name}")
        return []

    print(f"\nRenaming {folder_name.upper()} files...")
    print(f"Found: {len(files)} videos")

    # Generate unique temporary names first.
    # This avoids collisions with existing filenames.
    temp_files = []

    for file in files:

        temp_name = (
            f"__parallax_tmp_{uuid.uuid4().hex}"
            f"{file.suffix.lower()}"
        )

        temp_path = folder / temp_name

        file.rename(temp_path)

        temp_files.append((file, temp_path))

    renamed_files = []

    try:

        # Apply final sequential filenames
        for index, (original_path, temp_path) in enumerate(
            temp_files, start=1
        ):

            new_name = f"{prefix}_{index:04d}{temp_path.suffix.lower()}"
            new_path = folder / new_name

            if new_path.exists():
                raise FileExistsError(
                    f"Target filename already exists: {new_path}"
                )

            temp_path.rename(new_path)

            renamed_files.append(new_path)

            print(f"[RENAMED] {original_path.name} -> {new_name}")

    except Exception:

        print("\n[ERROR] Renaming failed.")
        print("Attempting to restore original filenames...")

        # Restore already-renamed files and remaining temp files
        for original_path, temp_path in temp_files:

            current_path = None

            if temp_path.exists():
                current_path = temp_path
            else:
                # Find the final renamed path by matching the
                # temporary file's extension and its position.
                position = temp_files.index((original_path, temp_path))
                candidate = folder / (
                    f"{prefix}_{position + 1:04d}"
                    f"{temp_path.suffix.lower()}"
                )

                if candidate.exists():
                    current_path = candidate

            if current_path is not None and not original_path.exists():
                current_path.rename(original_path)

        raise

    print(f"\n[SUCCESS] {folder_name.upper()}: {len(renamed_files)} files")

    return renamed_files


# ============================================================
# STEP 3: CREATE DATASET SAMPLE RECORDS
# ============================================================

def create_samples(files, prefix, label, class_name):

    samples = []

    for index, file in enumerate(files, start=1):

        relative_path = file.relative_to(PROJECT_ROOT)

        samples.append({
            "sample_id": f"{prefix}_{index:04d}",
            "filename": file.name,
            "relative_path": relative_path.as_posix(),
            "label": label,
            "class_name": class_name
        })

    return samples


# ============================================================
# STEP 4: SPLIT EACH CLASS
# ============================================================

def split_class(samples, rng):

    samples = samples.copy()
    rng.shuffle(samples)

    total = len(samples)

    train_end = int(total * TRAIN_RATIO)
    validation_end = int(
        total * (TRAIN_RATIO + VALIDATION_RATIO) + 0.5
    )

    train = samples[:train_end]
    validation = samples[train_end:validation_end]
    test = samples[validation_end:]

    return train, validation, test


# ============================================================
# STEP 5: WRITE CSV FILES
# ============================================================

def write_csv(filename, samples):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / filename

    fieldnames = [
        "sample_id",
        "filename",
        "relative_path",
        "label",
        "class_name",
        "split"
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(samples)

    print(f"[CREATED] {output_path.relative_to(PROJECT_ROOT)}")


# ============================================================
# STEP 6: MAIN DATASET PREPARATION
# ============================================================

def prepare_dataset():

    print("\n" + "=" * 60)
    print("       PARALLAX DATASET PREPARATION")
    print("=" * 60)

    # Check folders before renaming
    real_files_before = get_video_files("real")
    fake_files_before = get_video_files("fake")

    if not real_files_before or not fake_files_before:
        raise ValueError(
            "Both REAL and FAKE folders must contain videos."
        )

    print("\nINITIAL DATASET")
    print(f"REAL videos: {len(real_files_before)}")
    print(f"FAKE videos: {len(fake_files_before)}")
    print(f"TOTAL videos: {len(real_files_before) + len(fake_files_before)}")

    # --------------------------------------------------------
    # Rename REAL files
    # --------------------------------------------------------

    real_files = rename_files("real", "real")

    # --------------------------------------------------------
    # Rename FAKE files
    # --------------------------------------------------------

    fake_files = rename_files("fake", "fake")

    # --------------------------------------------------------
    # Create labelled records
    # REAL = 0
    # FAKE = 1
    # --------------------------------------------------------

    real_samples = create_samples(
        real_files,
        "REAL",
        0,
        "REAL"
    )

    fake_samples = create_samples(
        fake_files,
        "FAKE",
        1,
        "FAKE"
    )

    # --------------------------------------------------------
    # Split each class independently
    # --------------------------------------------------------

    rng = random.Random(RANDOM_SEED)

    real_train, real_val, real_test = split_class(
        real_samples, rng
    )

    fake_train, fake_val, fake_test = split_class(
        fake_samples, rng
    )

    splits = {
        "train": real_train + fake_train,
        "validation": real_val + fake_val,
        "test": real_test + fake_test
    }

    # Shuffle rows inside each split
    for samples in splits.values():
        rng.shuffle(samples)

    # Add split name to every record
    for split_name, samples in splits.items():

        for sample in samples:
            sample["split"] = split_name

    # --------------------------------------------------------
    # Generate complete manifest
    # --------------------------------------------------------

    all_samples = (
        splits["train"]
        + splits["validation"]
        + splits["test"]
    )

    write_csv("dataset_manifest.csv", all_samples)

    # --------------------------------------------------------
    # Generate individual CSV files
    # --------------------------------------------------------

    write_csv("train.csv", splits["train"])
    write_csv("validation.csv", splits["validation"])
    write_csv("test.csv", splits["test"])

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("           FINAL DATASET SUMMARY")
    print("=" * 60)

    for split_name, samples in splits.items():

        real_count = sum(
            sample["label"] == 0 for sample in samples
        )

        fake_count = sum(
            sample["label"] == 1 for sample in samples
        )

        print(f"\n{split_name.upper()}")
        print("-" * 35)
        print(f"REAL:  {real_count}")
        print(f"FAKE:  {fake_count}")
        print(f"TOTAL: {len(samples)}")

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 60)

    print(f"REAL videos: {len(real_files)}")
    print(f"FAKE videos: {len(fake_files)}")
    print(f"TOTAL: {len(all_samples)}")

    print("\nGenerated files:")
    print("1. dataset_manifest.csv")
    print("2. train.csv")
    print("3. validation.csv")
    print("4. test.csv")

    print("\nDataset location:", DATASET_DIR)
    print("Manifest location:", OUTPUT_DIR)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        prepare_dataset()

    except Exception as error:

        print("\n[FAILED]")
        print(error)
        sys.exit(1)
