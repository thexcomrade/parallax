"""Phase 3 controlled, approved repository cleanup execution script.

Executes file-by-file deletion of only explicitly approved disposable items.
Enforces all absolute protection rules before every single removal.
Generates complete cleanup_execution_manifest.json and storage comparison.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEANUP_DIR = REPO_ROOT / "experiments" / "phase3_repository_cleanup"
PROPOSAL_PATH = CLEANUP_DIR / "cleanup_proposal.json"
STORAGE_BEFORE_PATH = CLEANUP_DIR / "storage_before.json"

PROTECTED_PATHS = {
    "backend/app/ml/weights/avt2_dwf.pt",
    "experiments/phase4_4_clean_avt2_dwf/best_checkpoint.pt",
    "experiments/phase1_model_improvement/best_checkpoint.pt",
    "datasets/manifests/phase5_locked_test.csv",
    "backend/storage/parallax.db",
    "backend/app/ml/weights/audio_norm_stats.pt",
    "backend/app/ml/weights/model_card.json",
}


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest().upper()
    except OSError as e:
        return f"[ERROR: {e}]"


def is_protected(rel_path: str) -> bool:
    if rel_path in PROTECTED_PATHS:
        return True
    if rel_path.startswith("datasets/"):
        return True
    if rel_path.startswith("backend/.venv/"):
        return True
    if rel_path.startswith("frontend/node_modules/") and not rel_path.startswith("frontend/node_modules/.cache"):
        return True
    return False


def main() -> None:
    print("=== PARALLAX Phase 3 Controlled Cleanup Execution ===")
    t0 = time.time()

    assert PROPOSAL_PATH.exists(), f"Proposal not found at {PROPOSAL_PATH}"
    with open(PROPOSAL_PATH, "r", encoding="utf-8") as f:
        proposal = json.load(f)

    items = proposal.get("items", [])
    print(f"Loaded {len(items)} approved items for cleanup.")

    deletion_log: list[dict[str, Any]] = []
    deleted_bytes = 0
    deleted_files = 0
    errors: list[dict[str, Any]] = []

    for item in items:
        rel_path = item["relative_path"]
        fpath = REPO_ROOT / Path(rel_path)

        # Protection assertion
        if is_protected(rel_path):
            print(f"CRITICAL SAFETY ABORT: Attempted to delete protected file: {rel_path}")
            errors.append({"path": rel_path, "error": "Protected file rule violation"})
            continue

        if not fpath.exists():
            deletion_log.append({
                "path": rel_path,
                "size_bytes": 0,
                "sha256": item.get("sha256", ""),
                "timestamp": datetime.now(UTC).isoformat(),
                "result": "SKIPPED_NOT_FOUND",
            })
            continue

        # If it's a file, compute sha256 and delete
        try:
            stat = fpath.stat()
            fsize = stat.st_size
            sha = get_sha256(fpath) if fsize < 500 * 1024 * 1024 else item.get("sha256", "")

            # Specific check for best_model.pt
            if rel_path == "experiments/phase1_model_improvement/best_model.pt":
                assert sha == "8513C207753A208834DF0FA0BD1EC94445BA4FA5759BC3FB1F246D7A56B68E4A", (
                    f"Unexpected hash for {rel_path}: {sha}"
                )

            fpath.unlink()
            deleted_bytes += fsize
            deleted_files += 1

            deletion_log.append({
                "path": rel_path,
                "size_bytes": fsize,
                "size_fmt": format_size(fsize),
                "sha256": sha,
                "timestamp": datetime.now(UTC).isoformat(),
                "approval_reference": "USER_APPROVED_PHASE_3_MANIFEST",
                "result": "SUCCESS",
            })
        except Exception as e:
            print(f"Error deleting {rel_path}: {e}")
            errors.append({"path": rel_path, "error": str(e)})
            deletion_log.append({
                "path": rel_path,
                "size_bytes": item.get("size_bytes", 0),
                "timestamp": datetime.now(UTC).isoformat(),
                "result": f"FAILED: {e}",
            })

    # Cleanup empty cache directories if empty
    cache_dirs_to_check = [
        ".ruff_cache",
        "backend/.ruff_cache",
        ".pytest_cache",
        "backend/.pytest_cache",
        ".mypy_cache",
    ]
    for root, dirs, files in os.walk(REPO_ROOT, topdown=False):
        parts = Path(root).relative_to(REPO_ROOT).parts
        if "__pycache__" in parts or any(c in parts for c in cache_dirs_to_check):
            try:
                if not os.listdir(root):
                    os.rmdir(root)
            except OSError:
                pass

    print(f"\nSuccessfully deleted {deleted_files} files ({format_size(deleted_bytes)}).")
    if errors:
        print(f"Encountered {len(errors)} errors during deletion.")

    # Save cleanup execution manifest
    manifest_data = {
        "execution_timestamp": datetime.now(UTC).isoformat(),
        "total_approved_items": len(items),
        "deleted_files_count": deleted_files,
        "deleted_bytes": deleted_bytes,
        "deleted_size_fmt": format_size(deleted_bytes),
        "errors_count": len(errors),
        "errors": errors,
        "deletions": deletion_log,
    }
    manifest_path = CLEANUP_DIR / "cleanup_execution_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved execution manifest to {manifest_path}")

    # Measure post-cleanup storage
    print("\nMeasuring post-cleanup storage...")
    total_after_files = 0
    total_after_dirs = 0
    total_after_bytes = 0

    for root, dirs, files in os.walk(REPO_ROOT):
        total_after_dirs += 1
        for fname in files:
            total_after_files += 1
            fpath = Path(root) / fname
            try:
                total_after_bytes += fpath.stat().st_size
            except OSError:
                pass

    with open(STORAGE_BEFORE_PATH, "r", encoding="utf-8") as f:
        sb = json.load(f)

    storage_after = {
        "timestamp": datetime.now(UTC).isoformat(),
        "repository_path": str(REPO_ROOT),
        "total_files": total_after_files,
        "total_directories": total_after_dirs,
        "total_bytes": total_after_bytes,
        "total_size_fmt": format_size(total_after_bytes),
        "net_files_reduction": sb["total_files"] - total_after_files,
        "net_bytes_reduction": sb["total_bytes"] - total_after_bytes,
        "net_size_reduction_fmt": format_size(sb["total_bytes"] - total_after_bytes),
    }

    storage_after_path = CLEANUP_DIR / "storage_after.json"
    with open(storage_after_path, "w", encoding="utf-8") as f:
        json.dump(storage_after, f, indent=2)
    print(f"Saved storage_after.json to {storage_after_path}")

    # Write storage_comparison.csv
    comparison_csv_path = CLEANUP_DIR / "storage_comparison.csv"
    with open(comparison_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Before", "After", "Difference"])
        writer.writerow([
            "Repository Size (Bytes)",
            sb["total_bytes"],
            total_after_bytes,
            sb["total_bytes"] - total_after_bytes,
        ])
        writer.writerow([
            "Repository Size (Formatted)",
            sb["total_size_fmt"],
            format_size(total_after_bytes),
            f"-{format_size(sb['total_bytes'] - total_after_bytes)}",
        ])
        writer.writerow([
            "File Count",
            sb["total_files"],
            total_after_files,
            sb["total_files"] - total_after_files,
        ])
        writer.writerow([
            "Directory Count",
            sb["total_directories"],
            total_after_dirs,
            sb["total_directories"] - total_after_dirs,
        ])
    print(f"Saved storage_comparison.csv to {comparison_csv_path}")

    # Verify Production Weights SHA-256 after cleanup
    prod_path = REPO_ROOT / "backend" / "app" / "ml" / "weights" / "avt2_dwf.pt"
    post_prod_sha = get_sha256(prod_path)
    print(f"\n[Integrity Check] Production weights SHA-256 after cleanup: {post_prod_sha}")
    assert post_prod_sha == "6DE2E663799EE8687C2CA06BBB4309034937C8FBF7A18BA3166CE745052EF354", (
        "CRITICAL: Production weights hash changed!"
    )
    print("  -> Production weights hash VERIFIED UNCHANGED.")

    # Verify Candidate Weights SHA-256 after cleanup
    cand_path = REPO_ROOT / "experiments" / "phase4_4_clean_avt2_dwf" / "best_checkpoint.pt"
    post_cand_sha = get_sha256(cand_path)
    print(f"[Integrity Check] Candidate weights SHA-256 after cleanup:  {post_cand_sha}")
    assert post_cand_sha == "3D7CECEDAC4B3E065758E203EDF2362F9425F0EFB4FA2F44B6A35011AE5D4877", (
        "CRITICAL: Candidate weights hash changed!"
    )
    print("  -> Candidate weights hash VERIFIED UNCHANGED.")

    dt = time.time() - t0
    print(f"\nCleanup execution & post-measurement complete in {dt:.2f}s.")


if __name__ == "__main__":
    main()
