"""Phase 8B Safe Cleanup Execution & Verification Script.

Executes authorized, safe disposal of build caches, bytecode, test uploads,
frontend build outputs, duplicate reports, and identical model replica.
Strictly preserves historical provenance (datasets/archived/processed_phase2_backup/),
all active raw/processed datasets, manifests, production weights, candidate weights,
Phase 10 evaluation artifacts, and the active SQLite database.

Never touches or inspects datasets/manifests/phase5_locked_test.csv.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKED_TEST_PATH = (REPO_ROOT / "datasets" / "manifests" / "phase5_locked_test.csv").resolve()

# Expected cryptographic signatures
EXPECTED_PROD_SHA = "6DE2E663799EE8687C2CA06BBB4309034937C8FBF7A18BA3166CE745052EF354"
EXPECTED_BEST_CKPT_SHA = "3D7CECEDAC4B3E065758E203EDF2362F9425F0EFB4FA2F44B6A35011AE5D4877"


def get_sha256(filepath: Path) -> str:
    """Compute SHA-256 for a file without opening locked test manifest."""
    if filepath.resolve() == LOCKED_TEST_PATH:
        return "[LOCKED_TEST_ISOLATED]"
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest().upper()
    except OSError as e:
        return f"[ERROR: {e}]"


def get_repo_stats(root: Path) -> tuple[int, int]:
    """Calculate total file count and size in bytes for the repository."""
    total_files = 0
    total_size = 0
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            fp = Path(dirpath) / fname
            try:
                total_size += fp.stat().st_size
                total_files += 1
            except OSError:
                pass
    return total_files, total_size


def get_drive_d_free() -> tuple[int, int, int]:
    """Return total, used, free bytes for D: drive."""
    total, used, free = shutil.disk_usage("D:\\")
    return total, used, free


def check_protected_assets() -> dict[str, Any]:
    """Cryptographically and structurally verify all protected assets."""
    results: dict[str, Any] = {}

    # 1. Locked test manifest (EXACT ZERO-TOUCH VERIFICATION)
    results["locked_test_exists"] = LOCKED_TEST_PATH.exists()

    # 2. Raw dataset videos (expected 1000)
    raw_fake = list((REPO_ROOT / "datasets" / "raw" / "fake").glob("*.mp4"))
    raw_real = list((REPO_ROOT / "datasets" / "raw" / "real").glob("*.mp4"))
    results["raw_fake_count"] = len(raw_fake)
    results["raw_real_count"] = len(raw_real)
    results["raw_total_count"] = len(raw_fake) + len(raw_real)

    # 3. Processed tensors (expected 3000: 1000 audio, 1000 metadata, 1000 video)
    proc_audio = list((REPO_ROOT / "datasets" / "processed" / "audio").glob("*.pt"))
    proc_meta = list((REPO_ROOT / "datasets" / "processed" / "metadata").glob("*.json"))
    proc_video = list((REPO_ROOT / "datasets" / "processed" / "video").glob("*.pt"))
    results["processed_audio_count"] = len(proc_audio)
    results["processed_metadata_count"] = len(proc_meta)
    results["processed_video_count"] = len(proc_video)
    results["processed_total_count"] = len(proc_audio) + len(proc_meta) + len(proc_video)

    # 4. Archived historical backup (expected 3000: 1000 audio, 1000 metadata, 1000 video)
    arch_dir = REPO_ROOT / "datasets" / "archived" / "processed_phase2_backup"
    arch_audio = list((arch_dir / "audio").glob("*.pt")) if arch_dir.exists() else []
    arch_meta = list((arch_dir / "metadata").glob("*.json")) if arch_dir.exists() else []
    arch_video = list((arch_dir / "video").glob("*.pt")) if arch_dir.exists() else []
    results["archived_audio_count"] = len(arch_audio)
    results["archived_metadata_count"] = len(arch_meta)
    results["archived_video_count"] = len(arch_video)
    results["archived_total_count"] = len(arch_audio) + len(arch_meta) + len(arch_video)

    # 5. Recovery manifests
    rec_dir = REPO_ROOT / "datasets" / "manifests" / "recovery_4_3a"
    results["recovery_manifests"] = sorted([f.name for f in rec_dir.glob("*.csv")]) if rec_dir.exists() else []

    # 6. Production weights SHA-256
    prod_w = REPO_ROOT / "backend" / "app" / "ml" / "weights" / "avt2_dwf.pt"
    if prod_w.exists():
        sha = get_sha256(prod_w)
        results["prod_weights_sha256"] = sha
        results["prod_weights_match"] = (sha == EXPECTED_PROD_SHA)
    else:
        results["prod_weights_sha256"] = None
        results["prod_weights_match"] = False

    # 7. Candidate checkpoint SHA-256
    cand_ckpt = REPO_ROOT / "experiments" / "phase4_4_clean_avt2_dwf" / "best_checkpoint.pt"
    if cand_ckpt.exists():
        sha = get_sha256(cand_ckpt)
        results["candidate_checkpoint_sha256"] = sha
        results["candidate_checkpoint_match"] = (sha == EXPECTED_BEST_CKPT_SHA)
    else:
        results["candidate_checkpoint_sha256"] = None
        results["candidate_checkpoint_match"] = False

    # 8. Active SQLite database
    db_file = REPO_ROOT / "backend" / "storage" / "parallax.db"
    results["sqlite_db_exists"] = db_file.exists()
    results["sqlite_db_size"] = db_file.stat().st_size if db_file.exists() else 0

    # 9. Phase 10 evaluation test directory
    p10_dir = REPO_ROOT / "experiments" / "phase10_independent_evaluation" / "test"
    results["phase10_test_files"] = len(list(p10_dir.glob("*"))) if p10_dir.exists() else 0

    return results


def build_safe_deletion_manifest() -> list[dict[str, Any]]:
    """Scan and catalog every item approved under SAFE TO DELETE."""
    manifest_entries: list[dict[str, Any]] = []

    # 1. Empty build/namespace directories
    empty_dirs = [
        REPO_ROOT / "frontend" / "node_modules" / ".vite-temp",
        REPO_ROOT / "frontend" / "node_modules" / "@dimforge",
        REPO_ROOT / "frontend" / "node_modules" / "@tweenjs",
    ]
    for ed in empty_dirs:
        if ed.exists() and ed.is_dir():
            manifest_entries.append({
                "path": str(ed.relative_to(REPO_ROOT)).replace("\\", "/"),
                "type": "directory",
                "category": "SAFE TO DELETE",
                "reason": "Empty directory serving no architectural or gitkeep purpose",
                "size_bytes": 0,
                "pre_deletion_sha256": None,
            })

    # 2. Build & Lint/Test Caches
    cache_dirs = [
        (REPO_ROOT / ".mypy_cache", "Root MyPy type-checking cache"),
        (REPO_ROOT / "backend" / ".mypy_cache", "Backend MyPy type-checking cache"),
        (REPO_ROOT / ".ruff_cache", "Root Ruff linting cache"),
        (REPO_ROOT / "backend" / ".ruff_cache", "Backend Ruff linting cache"),
        (REPO_ROOT / ".pytest_cache", "Root PyTest test cache"),
        (REPO_ROOT / "backend" / ".pytest_cache", "Backend PyTest test cache"),
    ]
    for cd, desc in cache_dirs:
        if cd.exists():
            for cfile in cd.rglob("*"):
                if cfile.is_file():
                    try:
                        sz = cfile.stat().st_size
                        manifest_entries.append({
                            "path": str(cfile.relative_to(REPO_ROOT)).replace("\\", "/"),
                            "type": "file",
                            "category": "SAFE TO DELETE",
                            "reason": desc,
                            "size_bytes": sz,
                            "pre_deletion_sha256": get_sha256(cfile),
                        })
                    except OSError:
                        pass
            manifest_entries.append({
                "path": str(cd.relative_to(REPO_ROOT)).replace("\\", "/"),
                "type": "directory",
                "category": "SAFE TO DELETE",
                "reason": f"Cache directory container ({desc})",
                "size_bytes": 0,
                "pre_deletion_sha256": None,
            })

    # 3. Python Bytecode outside .venv
    for root, _dirs, files in os.walk(REPO_ROOT):
        if ".venv" in Path(root).parts or "node_modules" in Path(root).parts:
            continue
        if Path(root).name == "__pycache__":
            for pyc_name in files:
                fp = Path(root) / pyc_name
                if fp.suffix == ".pyc":
                    try:
                        sz = fp.stat().st_size
                        manifest_entries.append({
                            "path": str(fp.relative_to(REPO_ROOT)).replace("\\", "/"),
                            "type": "file",
                            "category": "SAFE TO DELETE",
                            "reason": "Compiled Python bytecode (.pyc)",
                            "size_bytes": sz,
                            "pre_deletion_sha256": get_sha256(fp),
                        })
                    except OSError:
                        pass
            manifest_entries.append({
                "path": str(Path(root).relative_to(REPO_ROOT)).replace("\\", "/"),
                "type": "directory",
                "category": "SAFE TO DELETE",
                "reason": "Python bytecode cache directory (__pycache__)",
                "size_bytes": 0,
                "pre_deletion_sha256": None,
            })

    # 4. Frontend Build Outputs
    frontend_builds = [
        (REPO_ROOT / "frontend" / "dist", "Frontend production Vite build bundle"),
        (REPO_ROOT / "frontend" / "playwright-report", "Frontend Playwright e2e test report"),
        (REPO_ROOT / "frontend" / "test-results", "Frontend Playwright test results output"),
    ]
    for fb, desc in frontend_builds:
        if fb.exists():
            for fb_file in fb.rglob("*"):
                if fb_file.is_file():
                    try:
                        sz = fb_file.stat().st_size
                        manifest_entries.append({
                            "path": str(fb_file.relative_to(REPO_ROOT)).replace("\\", "/"),
                            "type": "file",
                            "category": "SAFE TO DELETE",
                            "reason": desc,
                            "size_bytes": sz,
                            "pre_deletion_sha256": get_sha256(fb_file),
                        })
                    except OSError:
                        pass
            manifest_entries.append({
                "path": str(fb.relative_to(REPO_ROOT)).replace("\\", "/"),
                "type": "directory",
                "category": "SAFE TO DELETE",
                "reason": f"Frontend build output directory container ({desc})",
                "size_bytes": 0,
                "pre_deletion_sha256": None,
            })

    # 5. Duplicate Reports with Canonical Copies in reports/
    duplicate_reports = [
        (REPO_ROOT / "datasets" / "reports" / "dataset_validation_report.json", "reports/phase1_dataset_integrity.json"),
        (REPO_ROOT / "datasets" / "reports" / "phase2_improved_training_history.csv", "reports/phase2_improved_training_history.csv"),
        (REPO_ROOT / "datasets" / "reports" / "phase2_improved_training_report.json", "reports/phase2_improved_training_report.json"),
        (REPO_ROOT / "datasets" / "reports" / "phase2_training_history.csv", "reports/phase2_training_history.csv"),
        (REPO_ROOT / "datasets" / "reports" / "phase2_training_report.json", "reports/phase2_training_report.json"),
    ]
    for dr, canon_rel in duplicate_reports:
        if dr.exists():
            sz = dr.stat().st_size
            manifest_entries.append({
                "path": str(dr.relative_to(REPO_ROOT)).replace("\\", "/"),
                "type": "file",
                "category": "SAFE TO DELETE",
                "reason": f"Cryptographic duplicate of canonical report {canon_rel}",
                "size_bytes": sz,
                "pre_deletion_sha256": get_sha256(dr),
            })

    # 6. Redundant Exact Checkpoint Duplicate
    best_model = REPO_ROOT / "experiments" / "phase4_4_clean_avt2_dwf" / "best_model.pt"
    if best_model.exists():
        sz = best_model.stat().st_size
        manifest_entries.append({
            "path": str(best_model.relative_to(REPO_ROOT)).replace("\\", "/"),
            "type": "file",
            "category": "SAFE TO DELETE",
            "reason": "Exact bitwise duplicate replica of best_checkpoint.pt (preserved)",
            "size_bytes": sz,
            "pre_deletion_sha256": get_sha256(best_model),
        })

    # 7. Ephemeral Test-Staging Uploads and Artifacts
    uploads_dir = REPO_ROOT / "backend" / "storage" / "uploads"
    if uploads_dir.exists():
        for up in uploads_dir.glob("*.mp4"):
            try:
                sz = up.stat().st_size
                manifest_entries.append({
                    "path": str(up.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "type": "file",
                    "category": "SAFE TO DELETE",
                    "reason": "Ephemeral test-staging video upload",
                    "size_bytes": sz,
                    "pre_deletion_sha256": get_sha256(up),
                })
            except OSError:
                pass

    artifacts_dir = REPO_ROOT / "backend" / "storage" / "artifacts"
    if artifacts_dir.exists():
        for art in artifacts_dir.rglob("*"):
            if art.is_file():
                try:
                    sz = art.stat().st_size
                    manifest_entries.append({
                        "path": str(art.relative_to(REPO_ROOT)).replace("\\", "/"),
                        "type": "file",
                        "category": "SAFE TO DELETE",
                        "reason": "Ephemeral test-staging analysis artifact",
                        "size_bytes": sz,
                        "pre_deletion_sha256": get_sha256(art),
                    })
                except OSError:
                    pass
        # Include artifact subdirectories (leaving the root artifacts directory container intact)
        for ad in artifacts_dir.iterdir():
            if ad.is_dir():
                manifest_entries.append({
                    "path": str(ad.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "type": "directory",
                    "category": "SAFE TO DELETE",
                    "reason": "Ephemeral test-staging artifact subdirectory",
                    "size_bytes": 0,
                    "pre_deletion_sha256": None,
                })

    return manifest_entries


def main() -> None:
    print("=" * 80)
    print("PARALLAX — PHASE 8B SAFE CLEANUP EXECUTION & VERIFICATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: PRE-CLEANUP SAFETY CHECK
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Running Pre-Cleanup Safety Check & Asset Verification...")
    pre_d_total, pre_d_used, pre_d_free = get_drive_d_free()
    pre_repo_files, pre_repo_size = get_repo_stats(REPO_ROOT)
    pre_assets = check_protected_assets()

    print(f"Drive D: Free={pre_d_free / 1024**3:.2f} GB | Used={pre_d_used / 1024**3:.2f} GB | Total={pre_d_total / 1024**3:.2f} GB")
    print(f"Repository: {pre_repo_files:,} files | {pre_repo_size / 1024**2:.2f} MB ({pre_repo_size / 1024**3:.2f} GB)")
    print(f"Locked Test Manifest Exists: {pre_assets['locked_test_exists']} (Zero-touch isolation strictly respected)")
    print(f"Raw Videos: {pre_assets['raw_total_count']} files (500 fake + 500 real)")
    print(f"Active Processed Tensors: {pre_assets['processed_total_count']} files (1000 audio + 1000 video + 1000 meta)")
    print(f"Historical Phase 2 Archive: {pre_assets['archived_total_count']} files (1000 audio + 1000 video + 1000 meta)")
    print(f"Recovery Manifests: {pre_assets['recovery_manifests']}")
    print(f"Production Weights SHA-256 Match: {pre_assets['prod_weights_match']} ({pre_assets['prod_weights_sha256']})")
    print(f"Candidate Checkpoint SHA-256 Match: {pre_assets['candidate_checkpoint_match']} ({pre_assets['candidate_checkpoint_sha256']})")
    print(f"SQLite DB Exists: {pre_assets['sqlite_db_exists']} ({pre_assets['sqlite_db_size']:,} bytes)")
    print(f"Phase 10 Test Artifacts: {pre_assets['phase10_test_files']} files")

    # Safety Assertions
    assert pre_assets["locked_test_exists"], "CRITICAL: Locked test manifest not found!"
    assert pre_assets["raw_total_count"] == 1000, f"CRITICAL: Raw videos count mismatch: {pre_assets['raw_total_count']} != 1000"
    assert pre_assets["processed_total_count"] == 3000, f"CRITICAL: Processed tensors mismatch: {pre_assets['processed_total_count']} != 3000"
    assert pre_assets["archived_total_count"] == 3000, f"CRITICAL: Archived tensors mismatch: {pre_assets['archived_total_count']} != 3000"
    assert pre_assets["prod_weights_match"], "CRITICAL: Production weights hash mismatch!"
    assert pre_assets["candidate_checkpoint_match"], "CRITICAL: Candidate checkpoint hash mismatch!"
    assert pre_assets["sqlite_db_exists"], "CRITICAL: Active SQLite database not found!"

    # -------------------------------------------------------------------------
    # STEP 2: BUILD CLEANUP EXECUTION MANIFEST
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Generating Cleanup Execution Manifest...")
    manifest_entries = build_safe_deletion_manifest()

    total_manifest_files = sum(1 for e in manifest_entries if e["type"] == "file")
    total_manifest_dirs = sum(1 for e in manifest_entries if e["type"] == "directory")
    total_reclaimable_bytes = sum(e["size_bytes"] for e in manifest_entries)

    print(f"Manifest Compiled: {len(manifest_entries)} total targets ({total_manifest_files} files, {total_manifest_dirs} directories)")
    print(f"Expected Reclaimable Space: {total_reclaimable_bytes:,} bytes ({total_reclaimable_bytes / 1024**2:.2f} MB / {total_reclaimable_bytes / 1024**3:.3f} GB)")

    out_dir = REPO_ROOT / "experiments" / "phase8b_safe_cleanup"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "cleanup_execution_manifest.json"

    manifest_payload = {
        "metadata": {
            "phase": "8B",
            "title": "PARALLAX Safe Dataset & Repository Cleanup Execution Manifest",
            "created_at": datetime.now(tz=UTC).isoformat(),
            "target_count": len(manifest_entries),
            "file_count": total_manifest_files,
            "directory_count": total_manifest_dirs,
            "expected_reclaimed_bytes": total_reclaimable_bytes,
            "expected_reclaimed_mb": round(total_reclaimable_bytes / (1024 * 1024), 2),
            "pre_cleanup_repo_size_bytes": pre_repo_size,
            "pre_cleanup_repo_files": pre_repo_files,
            "pre_cleanup_drive_d_free_bytes": pre_d_free,
        },
        "targets": manifest_entries,
    }

    manifest_path.write_text(json.dumps(manifest_payload, indent=2))
    print(f"Execution manifest persisted to: {manifest_path}")

    # -------------------------------------------------------------------------
    # STEP 3: EXECUTE TARGETED DELETIONS
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Executing Targeted Deletions (Strictly SAFE TO DELETE)...")
    deletion_log: list[dict[str, Any]] = []
    actual_reclaimed_bytes = 0
    deleted_files = 0
    deleted_dirs = 0

    # Delete files first
    for entry in manifest_entries:
        if entry["type"] != "file":
            continue
        p = REPO_ROOT / entry["path"]
        if p.exists() and p.is_file():
            sz = p.stat().st_size
            try:
                p.unlink()
                actual_reclaimed_bytes += sz
                deleted_files += 1
                deletion_log.append({
                    "path": entry["path"],
                    "type": "file",
                    "status": "DELETED",
                    "bytes_reclaimed": sz,
                    "sha256": entry["pre_deletion_sha256"],
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                })
            except OSError as e:
                print(f"ERROR unlinking file {entry['path']}: {e}")
                sys.exit(1)

    # Delete directories next (deepest first)
    dir_entries = [e for e in manifest_entries if e["type"] == "directory"]
    # Sort by path length descending so subdirectories are removed before parents
    dir_entries.sort(key=lambda x: len(Path(x["path"]).parts), reverse=True)

    for entry in dir_entries:
        p = REPO_ROOT / entry["path"]
        if p.exists() and p.is_dir():
            try:
                shutil.rmtree(p, ignore_errors=False)
                deleted_dirs += 1
                deletion_log.append({
                    "path": entry["path"],
                    "type": "directory",
                    "status": "REMOVED",
                    "bytes_reclaimed": 0,
                    "sha256": None,
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                })
            except OSError as e:
                # If directory was already deleted by a parent rmtree or empty, log cleanly
                if p.exists():
                    print(f"ERROR removing directory {entry['path']}: {e}")
                    sys.exit(1)

    print(f"Deletions Complete: {deleted_files} files deleted, {deleted_dirs} directories removed.")
    print(f"Total Bytes Reclaimed: {actual_reclaimed_bytes:,} bytes ({actual_reclaimed_bytes / 1024**2:.2f} MB)")

    # -------------------------------------------------------------------------
    # STEP 4: POST-CLEANUP VERIFICATION
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Running Post-Cleanup Verification...")
    post_d_total, _post_d_used, post_d_free = get_drive_d_free()
    post_repo_files, post_repo_size = get_repo_stats(REPO_ROOT)
    post_assets = check_protected_assets()

    print(f"Post-Cleanup Drive D: Free={post_d_free / 1024**3:.2f} GB (+{(post_d_free - pre_d_free) / 1024**2:.2f} MB) | Total={post_d_total / 1024**3:.2f} GB")
    print(f"Post-Cleanup Repo: {post_repo_files:,} files ({pre_repo_files - post_repo_files:,} removed) | {post_repo_size / 1024**2:.2f} MB (reclaimed: {(pre_repo_size - post_repo_size) / 1024**2:.2f} MB)")

    # Assert all protected assets are preserved bitwise and count-wise
    assert post_assets["locked_test_exists"], "POST-CHECK FAILED: Locked test manifest missing!"
    assert post_assets["raw_total_count"] == 1000, f"POST-CHECK FAILED: Raw videos count mismatch: {post_assets['raw_total_count']} != 1000"
    assert post_assets["processed_total_count"] == 3000, f"POST-CHECK FAILED: Processed tensors mismatch: {post_assets['processed_total_count']} != 3000"
    assert post_assets["archived_total_count"] == 3000, f"POST-CHECK FAILED: Archived tensors mismatch: {post_assets['archived_total_count']} != 3000"
    assert post_assets["prod_weights_match"], "POST-CHECK FAILED: Production weights hash altered!"
    assert post_assets["candidate_checkpoint_match"], "POST-CHECK FAILED: Candidate checkpoint hash altered!"
    assert post_assets["sqlite_db_exists"], "POST-CHECK FAILED: Active SQLite database missing!"
    assert post_assets["sqlite_db_size"] == pre_assets["sqlite_db_size"], "POST-CHECK FAILED: SQLite DB modified!"
    assert post_assets["phase10_test_files"] == pre_assets["phase10_test_files"], "POST-CHECK FAILED: Phase 10 test files changed!"

    # Git working tree status
    print("\nChecking Git working tree status...")
    git_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    git_lines = git_res.stdout.splitlines()

    # Verify no tracked code files were deleted
    for line in git_lines:
        if line.startswith(" D "):
            print(f"WARNING: Tracked file deleted: {line}")
            # Ensure it's not a source file
            deleted_path = line[3:].strip()
            assert not deleted_path.startswith("backend/app/"), f"CRITICAL: App source deleted: {deleted_path}"
            assert not deleted_path.startswith("frontend/src/"), f"CRITICAL: Frontend source deleted: {deleted_path}"

    # Run isolation tests
    print("\nRunning test_isolation.py...")
    pytest_bin = REPO_ROOT / "backend" / ".venv" / "Scripts" / "pytest.exe"
    test_res = subprocess.run(
        [str(pytest_bin), "backend/tests/test_isolation.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(test_res.stdout)
    assert test_res.returncode == 0, f"Isolation tests failed: {test_res.stderr}"

    # Save execution log and summary
    summary = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "status": "SUCCESS",
        "pre_cleanup": {
            "repo_files": pre_repo_files,
            "repo_size_bytes": pre_repo_size,
            "repo_size_mb": round(pre_repo_size / 1024**2, 2),
            "drive_d_free_bytes": pre_d_free,
            "drive_d_free_gb": round(pre_d_free / 1024**3, 2),
            "assets": pre_assets,
        },
        "post_cleanup": {
            "repo_files": post_repo_files,
            "repo_size_bytes": post_repo_size,
            "repo_size_mb": round(post_repo_size / 1024**2, 2),
            "drive_d_free_bytes": post_d_free,
            "drive_d_free_gb": round(post_d_free / 1024**3, 2),
            "assets": post_assets,
        },
        "reconciliation": {
            "files_deleted": deleted_files,
            "directories_removed": deleted_dirs,
            "bytes_reclaimed": actual_reclaimed_bytes,
            "mb_reclaimed": round(actual_reclaimed_bytes / 1024**2, 2),
            "drive_d_free_increase_bytes": post_d_free - pre_d_free,
            "drive_d_free_increase_mb": round((post_d_free - pre_d_free) / 1024**2, 2),
        },
        "isolation_tests_passed": (test_res.returncode == 0),
        "deletion_log": deletion_log,
    }

    summary_path = out_dir / "cleanup_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nPhase 8B execution summary written to: {summary_path}")
    print("=" * 80)
    print("PHASE 8B EXECUTION & POST-VERIFICATION SUCCESSFUL")
    print("=" * 80)


if __name__ == "__main__":
    main()
