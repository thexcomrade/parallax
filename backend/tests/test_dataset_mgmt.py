"""Tests for dataset management, validation, manifest generation, and CLI workflows."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from app.ml.models.avt2_dwf import AVT2DWF

# Repo paths
TESTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TESTS_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_dataset_validator_empty_dir(tmp_path: Path) -> None:
    """Validate that empty raw directories report status='EMPTY'."""
    from scripts.validate_dataset import validate_raw_dataset  # type: ignore[import-not-found]

    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()

    report_file = tmp_path / "report.json"
    res = validate_raw_dataset(real_dir, fake_dir, report_file)
    assert res["status"] == "EMPTY"
    assert res["total_files_found"] == 0
    assert report_file.exists()


def test_manifest_and_identity_disjoint_split(tmp_path: Path) -> None:
    """Test manifest generation and identity-disjoint split ensuring zero leakage."""
    from scripts.prepare_dataset import (  # type: ignore[import-not-found]
        generate_manifest,
        generate_splits,
    )

    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()

    # Create dummy video files with identity prefixes
    # id001, id002, id003 (real)
    # id004, id005, id006 (fake)
    for i in range(1, 4):
        (real_dir / f"id{i:03d}_clip01.mp4").write_bytes(b"dummy")
        (real_dir / f"id{i:03d}_clip02.mp4").write_bytes(b"dummy")
    for i in range(4, 7):
        (fake_dir / f"id{i:03d}_clip01.mp4").write_bytes(b"dummy")
        (fake_dir / f"id{i:03d}_clip02.mp4").write_bytes(b"dummy")

    manifest_csv = tmp_path / "manifest.csv"
    manifest_rows = generate_manifest(real_dir, fake_dir, manifest_csv)
    assert len(manifest_rows) == 12

    # Verify labels
    real_rows = [r for r in manifest_rows if r["label"] == "REAL"]
    fake_rows = [r for r in manifest_rows if r["label"] == "FAKE"]
    assert len(real_rows) == 6
    assert len(fake_rows) == 6
    assert all(r["label_id"] == 0 for r in real_rows)
    assert all(r["label_id"] == 1 for r in fake_rows)

    # Test identity-disjoint split
    splits_dir = tmp_path / "splits"
    split_report_file = tmp_path / "split_report.json"
    report = generate_splits(
        manifest_csv=manifest_csv,
        output_dir=splits_dir,
        report_file=split_report_file,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        seed=42,
    )

    assert report["status"] == "SPLIT_CREATED"
    # Zero identity leakage
    assert report["leakage_check"]["train_val_overlap"] == 0
    assert report["leakage_check"]["train_test_overlap"] == 0
    assert report["leakage_check"]["val_test_overlap"] == 0

    assert (splits_dir / "train.csv").exists()
    assert (splits_dir / "validation.csv").exists()
    assert (splits_dir / "test.csv").exists()


def test_promote_model_safety_gate(tmp_path: Path) -> None:
    """Test that promote_model script requires --confirm flag and validates checkpoint."""
    from scripts.promote_model import main as promote_main  # type: ignore[import-not-found]

    # Create dummy checkpoint
    model = AVT2DWF(pretrained_vit=False)
    candidate_ckpt = tmp_path / "candidate.pt"
    torch.save(model.state_dict(), candidate_ckpt)

    prod_target = tmp_path / "prod" / "avt2_dwf.pt"
    archive_dir = tmp_path / "archived"

    # Running without --confirm exits with code 1
    sys_argv_backup = sys.argv
    try:
        sys.argv = [
            "promote_model.py",
            "--checkpoint",
            str(candidate_ckpt),
            "--target",
            str(prod_target),
            "--archive-dir",
            str(archive_dir),
        ]
        with pytest.raises(SystemExit) as exc_info:
            promote_main()
        assert exc_info.value.code == 1

        # Running with --confirm succeeds
        sys.argv.append("--confirm")
        promote_main()
        assert prod_target.exists()
        assert prod_target.stat().st_size == candidate_ckpt.stat().st_size
    finally:
        sys.argv = sys_argv_backup
