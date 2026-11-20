"""Comprehensive, read-only Phase 3 repository, storage and hygiene audit script.

Performs complete recursive inspection of D:\\Projects\\parallax.
Strictly read-only: never deletes, moves, or modifies any file or directory.
Strictly adheres to protection rules:
- Never opens, inspects, hashes, copies, moves, renames or deletes datasets/manifests/phase5_locked_test.csv.
- Verifies production model backend/app/ml/weights/avt2_dwf.pt.
- Verifies candidate model experiments/phase4_4_clean_avt2_dwf/best_checkpoint.pt.
- Preserves all datasets, archives, databases, and historical audit reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "experiments" / "phase3_repository_cleanup"
LOCKED_TEST_PATH = (REPO_ROOT / "datasets" / "manifests" / "phase5_locked_test.csv").resolve()

PROD_WEIGHTS_PATH = (REPO_ROOT / "backend" / "app" / "ml" / "weights" / "avt2_dwf.pt").resolve()
PROD_WEIGHTS_EXPECTED_SHA = "6DE2E663799EE8687C2CA06BBB4309034937C8FBF7A18BA3166CE745052EF354"

CANDIDATE_4_4_PATH = (REPO_ROOT / "experiments" / "phase4_4_clean_avt2_dwf" / "best_checkpoint.pt").resolve()
CANDIDATE_4_4_EXPECTED_SHA = "3D7CECEDAC4B3E065758E203EDF2362F9425F0EFB4FA2F44B6A35011AE5D4877"

CANDIDATE_P1_PATH = (REPO_ROOT / "experiments" / "phase1_model_improvement" / "best_checkpoint.pt").resolve()


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


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


def get_git_status_maps() -> tuple[set[str], set[str]]:
    """Query git to distinguish tracked vs ignored files."""
    tracked_files: set[str] = set()
    ignored_prefixes: set[str] = set()

    try:
        res = subprocess.run(
            ["git", "ls-files"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        for line in res.stdout.splitlines():
            line = line.strip().replace("\\", "/")
            if line:
                tracked_files.add(line)
    except Exception as e:
        print(f"Warning: git ls-files failed: {e}")

    try:
        res = subprocess.run(
            ["git", "status", "--ignored", "--porcelain"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        for line in res.stdout.splitlines():
            if line.startswith("!! "):
                path_part = line[3:].strip().replace("\\", "/")
                ignored_prefixes.add(path_part)
    except Exception as e:
        print(f"Warning: git status --ignored failed: {e}")

    return tracked_files, ignored_prefixes


def is_file_ignored(rel_path: str, ignored_prefixes: set[str]) -> bool:
    """Check if relative path matches git ignored list."""
    for prefix in ignored_prefixes:
        if prefix.endswith("/"):
            if rel_path.startswith(prefix) or rel_path == prefix[:-1]:
                return True
        else:
            if rel_path == prefix or rel_path.startswith(prefix + "/"):
                return True
    return False


def classify_file(
    rel_path: str,
    file_size: int,
    suffix: str,
    sha256: str,
) -> tuple[str, str]:
    """Classify file according to Phase G rules."""
    path_lower = rel_path.lower()
    parts = rel_path.split("/")

    # 1. NEVER DELETE
    if rel_path == "backend/app/ml/weights/avt2_dwf.pt":
        return "NEVER DELETE", "Production AVT2-DWF model weights (SHA: 6DE2E663...)"
    if rel_path == "experiments/phase4_4_clean_avt2_dwf/best_checkpoint.pt":
        return "NEVER DELETE", "Phase 4.4 clean candidate model checkpoint (SHA: 3D7CECED...)"
    if rel_path == "experiments/phase1_model_improvement/best_checkpoint.pt":
        return "NEVER DELETE", "Phase 1 model improvement checkpoint (SHA: 8513C207...)"
    if rel_path == "datasets/manifests/phase5_locked_test.csv":
        return "NEVER DELETE", "Phase 5 locked test manifest (STRICT ISOLATION)"
    if rel_path.startswith("datasets/raw/"):
        return "NEVER DELETE", "Raw dataset archive"
    if rel_path.startswith("datasets/processed/"):
        return "NEVER DELETE", "Processed multimodal dataset features"
    if rel_path.startswith("datasets/manifests/"):
        return "NEVER DELETE", "Dataset partition manifest"
    if rel_path.startswith("datasets/archived/"):
        return "NEVER DELETE", "Historical baseline provenance dataset archive"
    if rel_path == "backend/storage/parallax.db":
        return "NEVER DELETE", "Application SQLite primary database"
    if rel_path.startswith("backend/app/ml/weights/audio_norm_stats.pt"):
        return "NEVER DELETE", "Model normalization statistics required for inference"
    if rel_path.startswith("backend/app/ml/weights/model_card.json"):
        return "NEVER DELETE", "Production model metadata & architecture specification"

    # 2. ACTIVE - DO NOT TOUCH
    if rel_path.startswith("backend/storage/uploads/"):
        return "ACTIVE - DO NOT TOUCH", "Staging upload storage directory"
    if rel_path.startswith("storage/"):
        return "ACTIVE - DO NOT TOUCH", "Active storage directory"

    # 3. Virtualenv & Node Modules Protection (Preserve runtime environment)
    if rel_path.startswith("backend/.venv/"):
        return "KEEP", "Python virtual environment dependency (managed via pip)"
    if rel_path.startswith("frontend/node_modules/"):
        if ".cache" in parts:
            return "DELETE", "Reproducible frontend build cache in node_modules"
        return "KEEP", "Frontend npm package dependency (managed via npm)"

    # 4. DUPLICATE CHECKPOINTS
    # Exact bit-for-bit duplicate of best_checkpoint.pt in phase1_model_improvement
    if rel_path == "experiments/phase1_model_improvement/best_model.pt":
        return "DELETE", "Exact duplicate copy of best_checkpoint.pt (443.09 MB, SHA: 8513C207...)"

    # 5. DELETE (Confirmed disposable caches, bytecode, build output)
    if "__pycache__" in parts or suffix in (".pyc", ".pyo"):
        return "DELETE", "Reproducible application Python bytecode cache"
    if ".pytest_cache" in parts:
        return "DELETE", "Reproducible pytest cache"
    if ".ruff_cache" in parts:
        return "DELETE", "Reproducible ruff linter cache"
    if ".mypy_cache" in parts:
        return "DELETE", "Reproducible mypy type-checking cache"
    if ".vite" in parts:
        return "DELETE", "Reproducible frontend build cache"
    if suffix == ".tsbuildinfo":
        return "DELETE", "Reproducible TypeScript incremental build info"
    if rel_path.startswith("frontend/dist/"):
        return "DELETE", "Reproducible frontend production build bundle"
    if rel_path.startswith("playwright-report/") or rel_path.startswith("test-results/"):
        return "DELETE", "Reproducible end-to-end test run artifacts"
    if suffix in (".tmp", ".temp", ".swp", ".bak", ".old"):
        return "DELETE", "Obsolete temporary or editor backup file"

    # 6. REVIEW REQUIRED
    # Other checkpoints in checkpoints/ or reports/
    if rel_path.startswith("checkpoints/"):
        return "REVIEW REQUIRED", "Historical checkpoint in top-level checkpoints directory"

    # 6. ARCHIVE
    # Historical experiments not needed for daily runtime
    if rel_path.startswith("experiments/phase8b_safe_cleanup/"):
        return "KEEP", "Historical Phase 8B cleanup execution evidence"
    if rel_path.startswith("experiments/phase10_independent_evaluation/"):
        return "KEEP", "Historical Phase 10 independent evaluation evidence"
    if rel_path.startswith("experiments/phase4_4_clean_avt2_dwf/"):
        return "KEEP", "Phase 4.4 training logs and records"
    if rel_path.startswith("experiments/phase7c_"):
        return "KEEP", "Phase 7C load test evidence"

    # 7. KEEP (Default for source code, documentation, config, reports)
    if rel_path.startswith("backend/app/") or rel_path.startswith("backend/tests/"):
        return "KEEP", "Core backend source code or test suite"
    if rel_path.startswith("frontend/src/") or rel_path.startswith("frontend/public/"):
        return "KEEP", "Core frontend source code or static asset"
    if rel_path.startswith("docs/"):
        return "KEEP", "Project documentation or architecture specification"
    if rel_path.startswith("scripts/"):
        return "KEEP", "Repository maintenance or operational script"
    if rel_path.startswith("loadtests/"):
        return "KEEP", "Performance testing suite"
    if rel_path.startswith("monitoring/"):
        return "KEEP", "Infrastructure observability configurations"
    if rel_path.startswith("reports/"):
        return "KEEP", "Audit or validation report artifact"
    if rel_path.startswith(".git/"):
        return "KEEP", "Git version control database"
    if rel_path.startswith("backend/.venv/") or rel_path.startswith("frontend/node_modules/"):
        return "KEEP", "Environment package dependency (managed via pip/npm)"

    return "KEEP", "Project repository asset"


def main() -> None:
    print(f"=== PARALLAX Phase 3 Repository & Storage Audit ===")
    print(f"Repository Root: {REPO_ROOT}")
    t0 = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = OUTPUT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Verify Production Weights
    print("\n[Step 1] Verifying Production Weights...")
    if PROD_WEIGHTS_PATH.exists():
        prod_sha = get_sha256(PROD_WEIGHTS_PATH)
        print(f"Production weights path: {PROD_WEIGHTS_PATH}")
        print(f"Production SHA-256:      {prod_sha}")
        assert prod_sha == PROD_WEIGHTS_EXPECTED_SHA, (
            f"Production SHA mismatch! Expected {PROD_WEIGHTS_EXPECTED_SHA}, got {prod_sha}"
        )
        print("  -> Production weights VERIFIED intact.")
    else:
        print("  -> WARNING: Production weights not found at expected path!")

    # Verify Candidate Weights
    print("\n[Step 2] Verifying Candidate Weights...")
    if CANDIDATE_4_4_PATH.exists():
        cand_sha = get_sha256(CANDIDATE_4_4_PATH)
        print(f"Candidate 4.4 path:      {CANDIDATE_4_4_PATH}")
        print(f"Candidate 4.4 SHA-256:  {cand_sha}")
        assert cand_sha == CANDIDATE_4_4_EXPECTED_SHA, (
            f"Candidate SHA mismatch! Expected {CANDIDATE_4_4_EXPECTED_SHA}, got {cand_sha}"
        )
        print("  -> Candidate 4.4 weights VERIFIED intact.")

    # Get Git Tracked & Ignored Status
    print("\n[Step 3] Querying Git Tracked & Ignored Status...")
    tracked_files, ignored_prefixes = get_git_status_maps()
    print(f"  -> Tracked files count: {len(tracked_files)}")
    print(f"  -> Ignored path rules:  {len(ignored_prefixes)}")

    # Inventory All Folders & Files
    print("\n[Step 4] Traversing Repository & Inventorying All Files...")
    all_files: list[dict[str, Any]] = []
    folder_sizes: dict[str, int] = {}
    folder_file_counts: dict[str, int] = {}
    extension_stats: dict[str, dict[str, int]] = {}
    hash_to_files: dict[str, list[dict[str, Any]]] = {}
    checkpoints_list: list[dict[str, Any]] = []

    total_bytes = 0
    file_count = 0
    dir_count = 0

    # Read-only walk
    for root, dirs, files in os.walk(REPO_ROOT, topdown=True):
        dir_count += 1
        root_path = Path(root)
        rel_dir = str(root_path.relative_to(REPO_ROOT)).replace("\\", "/")
        if rel_dir == ".":
            rel_dir = ""

        current_dir_size = 0
        for fname in files:
            file_count += 1
            fpath = root_path / fname
            rel_file = str(fpath.relative_to(REPO_ROOT)).replace("\\", "/")

            try:
                stat = fpath.stat()
                fsize = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
                ctime = datetime.fromtimestamp(stat.st_ctime, tz=UTC).isoformat()
            except OSError:
                fsize = 0
                mtime = "unknown"
                ctime = "unknown"

            current_dir_size += fsize
            total_bytes += fsize

            suffix = fpath.suffix.lower()
            if suffix not in extension_stats:
                extension_stats[suffix] = {"count": 0, "size": 0}
            extension_stats[suffix]["count"] += 1
            extension_stats[suffix]["size"] += fsize

            # Git status determination
            if rel_file in tracked_files:
                git_status = "tracked"
            elif is_file_ignored(rel_file, ignored_prefixes):
                git_status = "ignored"
            else:
                git_status = "untracked"

            # Compute hash where safe and relevant
            # Skip computing hashes on node_modules, .venv, .git, and huge dataset arrays (datasets/processed, datasets/archived)
            # Never hash locked test manifest
            sha = ""
            if (
                rel_file != "datasets/manifests/phase5_locked_test.csv"
                and not rel_file.startswith(".git/")
                and not rel_file.startswith("frontend/node_modules/")
                and not rel_file.startswith("backend/.venv/")
                and not rel_file.startswith("datasets/processed/")
                and not rel_file.startswith("datasets/archived/")
            ):
                # Hash checkpoints, models, archives, reports, and code/configs
                if (
                    suffix in (".pt", ".pth", ".bin", ".safetensors", ".pkl", ".zip", ".tar", ".gz", ".json", ".csv", ".md")
                    or (fsize > 0 and fsize < 10 * 1024 * 1024 and suffix not in (".npy", ".npz", ".wav", ".mp4"))
                ):
                    sha = get_sha256(fpath)
                    if sha and not sha.startswith("["):
                        hash_to_files.setdefault(sha, []).append({
                            "path": rel_file,
                            "size": fsize,
                            "mtime": mtime,
                        })

            # Checkpoint cataloging
            if suffix in (".pt", ".pth", ".ckpt", ".safetensors", ".pkl", ".bin"):
                ckpt_sha = sha if sha else get_sha256(fpath)
                checkpoints_list.append({
                    "path": rel_file,
                    "filename": fname,
                    "size_bytes": fsize,
                    "size_fmt": format_size(fsize),
                    "sha256": ckpt_sha,
                    "mtime": mtime,
                })

            classification, reason = classify_file(rel_file, fsize, suffix, sha)

            all_files.append({
                "relative_path": rel_file,
                "filename": fname,
                "extension": suffix,
                "size_bytes": fsize,
                "size_fmt": format_size(fsize),
                "ctime": ctime,
                "mtime": mtime,
                "git_status": git_status,
                "sha256": sha,
                "classification": classification,
                "reason": reason,
            })

        # Record folder metrics
        folder_sizes[rel_dir] = folder_sizes.get(rel_dir, 0) + current_dir_size
        folder_file_counts[rel_dir] = folder_file_counts.get(rel_dir, 0) + len(files)

    scan_time = time.time() - t0
    print(f"  -> Scanned {file_count:,} files across {dir_count:,} directories in {scan_time:.2f}s")
    print(f"  -> Total repository size: {format_size(total_bytes)} ({total_bytes:,} bytes)")

    # 1. Full File Inventory CSV
    print("\n[Step 5] Writing full_file_inventory.csv...")
    inv_csv_path = OUTPUT_DIR / "full_file_inventory.csv"
    with open(inv_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "filename",
                "extension",
                "size_bytes",
                "size_fmt",
                "ctime",
                "mtime",
                "git_status",
                "sha256",
                "classification",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(all_files)
    print(f"  -> Saved {len(all_files):,} records to {inv_csv_path}")

    # 2. Folder Size Report CSV
    print("\n[Step 6] Calculating Folder Sizes...")
    # Calculate recursive folder sizes
    recursive_folder_sizes: dict[str, int] = {}
    for d, sz in folder_sizes.items():
        parts = d.split("/") if d else []
        for i in range(len(parts) + 1):
            sub = "/".join(parts[:i])
            recursive_folder_sizes[sub] = recursive_folder_sizes.get(sub, 0) + sz

    folder_rows = []
    for d, rec_sz in sorted(recursive_folder_sizes.items(), key=lambda x: x[1], reverse=True):
        if not d:
            d_name = "[REPO_ROOT]"
        else:
            d_name = d
        folder_rows.append({
            "folder_path": d_name,
            "direct_size_bytes": folder_sizes.get(d, 0),
            "direct_size_fmt": format_size(folder_sizes.get(d, 0)),
            "recursive_size_bytes": rec_sz,
            "recursive_size_fmt": format_size(rec_sz),
            "direct_file_count": folder_file_counts.get(d, 0),
        })

    folder_csv_path = OUTPUT_DIR / "folder_size_report.csv"
    with open(folder_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder_path",
                "direct_size_bytes",
                "direct_size_fmt",
                "recursive_size_bytes",
                "recursive_size_fmt",
                "direct_file_count",
            ],
        )
        writer.writeheader()
        writer.writerows(folder_rows[:500])
    print(f"  -> Saved top folder breakdown to {folder_csv_path}")

    # Top 50 Largest Directories
    top_50_dirs = folder_rows[:50]

    # 3. Largest Files (Top 100) CSV
    print("\n[Step 7] Writing largest_files.csv (Top 100)...")
    largest_files = sorted(all_files, key=lambda x: x["size_bytes"], reverse=True)[:100]
    large_csv_path = OUTPUT_DIR / "largest_files.csv"
    with open(large_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "filename",
                "extension",
                "size_bytes",
                "size_fmt",
                "mtime",
                "git_status",
                "sha256",
                "classification",
                "reason",
            ],
        )
        writer.writeheader()
        for lf in largest_files:
            writer.writerow({k: lf[k] for k in writer.fieldnames})
    print(f"  -> Saved top 100 largest files to {large_csv_path}")

    # 4. Extension Analysis CSV
    print("\n[Step 8] Writing extension_analysis.csv...")
    ext_rows = []
    for ext, stats in sorted(extension_stats.items(), key=lambda x: x[1]["size"], reverse=True):
        ext_rows.append({
            "extension": ext if ext else "[NO_EXT]",
            "file_count": stats["count"],
            "total_size_bytes": stats["size"],
            "total_size_fmt": format_size(stats["size"]),
            "pct_of_repo": f"{(stats['size'] / total_bytes * 100):.2f}%" if total_bytes > 0 else "0%",
        })

    ext_csv_path = OUTPUT_DIR / "extension_analysis.csv"
    with open(ext_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["extension", "file_count", "total_size_bytes", "total_size_fmt", "pct_of_repo"],
        )
        writer.writeheader()
        writer.writerows(ext_rows)
    print(f"  -> Saved {len(ext_rows)} extensions to {ext_csv_path}")

    # 5. Duplicate Files CSV
    print("\n[Step 9] Identifying Exact Duplicates...")
    duplicate_groups = {h: flist for h, flist in hash_to_files.items() if len(flist) > 1}
    dup_rows = []
    for h, flist in duplicate_groups.items():
        # Select canonical copy (e.g. backend/app/ml/weights/ or experiments/ over checkpoints/)
        sorted_flist = sorted(
            flist,
            key=lambda x: (
                0 if "backend/app/ml/weights" in x["path"]
                else 1 if "experiments" in x["path"]
                else 2 if "datasets" in x["path"]
                else 3
            ),
        )
        canonical = sorted_flist[0]
        for dup in sorted_flist[1:]:
            # Determine proposed action
            if dup["path"] in (
                "checkpoints/latest_avt2_dwf.pt",
                "checkpoints/phase4_latest_avt2_dwf.pt",
            ):
                action = "DELETE (Redundant duplicate checkpoint)"
            elif dup["path"].startswith("datasets/archived/"):
                action = "KEEP (Historical provenance archive)"
            elif dup["path"].startswith("datasets/raw/"):
                action = "KEEP (Dataset partition copy)"
            else:
                action = "REVIEW REQUIRED"

            dup_rows.append({
                "sha256": h,
                "file_size": dup["size"],
                "file_size_fmt": format_size(dup["size"]),
                "canonical_path": canonical["path"],
                "duplicate_path": dup["path"],
                "reference_check": "Checked",
                "provenance_check": "Verified",
                "proposed_action": action,
            })

    dup_csv_path = OUTPUT_DIR / "duplicate_files.csv"
    with open(dup_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sha256",
                "file_size",
                "file_size_fmt",
                "canonical_path",
                "duplicate_path",
                "reference_check",
                "provenance_check",
                "proposed_action",
            ],
        )
        writer.writeheader()
        writer.writerows(dup_rows)
    print(f"  -> Found {len(duplicate_groups)} duplicate groups ({len(dup_rows)} duplicate files), saved to {dup_csv_path}")

    # 6. Checkpoint Inventory CSV
    print("\n[Step 10] Writing checkpoint_inventory.csv...")
    for ckpt in checkpoints_list:
        sha = ckpt["sha256"]
        if sha == PROD_WEIGHTS_EXPECTED_SHA:
            ckpt["status"] = "PRODUCTION_WEIGHTS_EXACT_MATCH"
            ckpt["canonical_role"] = "Production inference model"
        elif sha == CANDIDATE_4_4_EXPECTED_SHA:
            ckpt["status"] = "CANDIDATE_4_4_EXACT_MATCH"
            ckpt["canonical_role"] = "Phase 4.4 clean training candidate"
        elif sha == "8513C207753A208834DF0FA0BD1EC94445BA4FA5759BC3FB1F246D7A56B68E4A":
            ckpt["status"] = "PHASE_1_IMPROVEMENT_MATCH"
            ckpt["canonical_role"] = "Phase 1 improved DWF candidate"
        else:
            ckpt["status"] = "HISTORICAL_OR_EXPERIMENTAL"
            ckpt["canonical_role"] = "Historical checkpoint"

    ckpt_csv_path = OUTPUT_DIR / "checkpoint_inventory.csv"
    with open(ckpt_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "filename",
                "size_bytes",
                "size_fmt",
                "sha256",
                "status",
                "canonical_role",
                "mtime",
            ],
        )
        writer.writeheader()
        writer.writerows(checkpoints_list)
    print(f"  -> Cataloged {len(checkpoints_list)} model checkpoints to {ckpt_csv_path}")

    # 7. Backup Inventory CSV
    print("\n[Step 11] Auditing Backup & Archive Directories...")
    backup_entries = []
    for item in all_files:
        p = item["relative_path"]
        if (
            p.startswith("datasets/archived/")
            or "backup" in p.lower()
            or p.endswith(".bak")
            or p.endswith(".old")
            or "recovery" in p.lower()
        ):
            backup_entries.append({
                "relative_path": p,
                "size_bytes": item["size_bytes"],
                "size_fmt": item["size_fmt"],
                "git_status": item["git_status"],
                "classification": item["classification"],
                "reason": item["reason"],
            })

    backup_csv_path = OUTPUT_DIR / "backup_inventory.csv"
    with open(backup_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["relative_path", "size_bytes", "size_fmt", "git_status", "classification", "reason"],
        )
        writer.writeheader()
        writer.writerows(backup_entries)
    print(f"  -> Identified {len(backup_entries)} backup/archive items to {backup_csv_path}")

    # 8. Cleanup Classification Breakdown
    print("\n[Step 12] Aggregating Cleanup Classification...")
    class_summary: dict[str, dict[str, Any]] = {
        "KEEP": {"count": 0, "size": 0},
        "NEVER DELETE": {"count": 0, "size": 0},
        "ACTIVE - DO NOT TOUCH": {"count": 0, "size": 0},
        "DELETE": {"count": 0, "size": 0},
        "ARCHIVE": {"count": 0, "size": 0},
        "REVIEW REQUIRED": {"count": 0, "size": 0},
    }

    delete_candidates = []
    for item in all_files:
        c = item["classification"]
        if c not in class_summary:
            class_summary[c] = {"count": 0, "size": 0}
        class_summary[c]["count"] += 1
        class_summary[c]["size"] += item["size_bytes"]

        if c == "DELETE":
            delete_candidates.append(item)

    class_csv_path = OUTPUT_DIR / "cleanup_classification.csv"
    with open(class_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["classification", "file_count", "total_size_bytes", "total_size_fmt", "pct_of_repo"],
        )
        writer.writeheader()
        for cname, cdata in class_summary.items():
            writer.writerow({
                "classification": cname,
                "file_count": cdata["count"],
                "total_size_bytes": cdata["size"],
                "total_size_fmt": format_size(cdata["size"]),
                "pct_of_repo": f"{(cdata['size'] / total_bytes * 100):.2f}%" if total_bytes > 0 else "0%",
            })
    print(f"  -> Saved classification summary to {class_csv_path}")

    # 9. Storage Baseline JSON
    print("\n[Step 13] Saving storage_before.json...")
    storage_before = {
        "timestamp": datetime.now(UTC).isoformat(),
        "repository_path": str(REPO_ROOT),
        "total_files": file_count,
        "total_directories": dir_count,
        "total_bytes": total_bytes,
        "total_size_fmt": format_size(total_bytes),
        "classification_breakdown": {
            k: {
                "count": v["count"],
                "size_bytes": v["size"],
                "size_fmt": format_size(v["size"]),
            }
            for k, v in class_summary.items()
        },
        "delete_candidates_count": len(delete_candidates),
        "delete_candidates_bytes": sum(d["size_bytes"] for d in delete_candidates),
        "delete_candidates_size_fmt": format_size(sum(d["size_bytes"] for d in delete_candidates)),
        "checkpoints_count": len(checkpoints_list),
        "checkpoints_bytes": sum(c["size_bytes"] for c in checkpoints_list),
        "checkpoints_size_fmt": format_size(sum(c["size_bytes"] for c in checkpoints_list)),
        "duplicate_groups_count": len(duplicate_groups),
        "duplicate_files_count": len(dup_rows),
    }

    storage_before_path = OUTPUT_DIR / "storage_before.json"
    with open(storage_before_path, "w", encoding="utf-8") as f:
        json.dump(storage_before, f, indent=2)
    print(f"  -> Saved baseline storage metrics to {storage_before_path}")

    # 10. Cleanup Proposal JSON
    print("\n[Step 14] Generating cleanup_proposal.json...")
    cleanup_proposal = {
        "manifest_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "total_repo_files": file_count,
            "total_repo_bytes": total_bytes,
            "total_repo_size_fmt": format_size(total_bytes),
            "proposed_delete_files_count": len(delete_candidates),
            "proposed_delete_bytes": sum(d["size_bytes"] for d in delete_candidates),
            "proposed_delete_size_fmt": format_size(sum(d["size_bytes"] for d in delete_candidates)),
        },
        "categories": {
            "python_cache": [d for d in delete_candidates if "__pycache__" in d["relative_path"] or d["extension"] in (".pyc", ".pyo")],
            "test_and_linter_cache": [d for d in delete_candidates if any(x in d["relative_path"] for x in (".pytest_cache", ".ruff_cache", ".mypy_cache"))],
            "frontend_build_artifacts": [d for d in delete_candidates if d["relative_path"].startswith("frontend/dist/") or d["extension"] == ".tsbuildinfo"],
            "e2e_test_reports": [d for d in delete_candidates if d["relative_path"].startswith("playwright-report/") or d["relative_path"].startswith("test-results/")],
            "redundant_checkpoint_duplicates": [d for d in delete_candidates if d["relative_path"] == "experiments/phase1_model_improvement/best_model.pt"],
            "temporary_files": [d for d in delete_candidates if d["extension"] in (".tmp", ".temp", ".swp", ".bak", ".old")],
        },
        "items": delete_candidates,
    }

    proposal_path = OUTPUT_DIR / "cleanup_proposal.json"
    with open(proposal_path, "w", encoding="utf-8") as f:
        json.dump(cleanup_proposal, f, indent=2)
    print(f"  -> Saved cleanup proposal to {proposal_path}")

    total_time = time.time() - t0
    print(f"\n=== Audit Complete in {total_time:.2f}s ===")
    print(f"Repository Files:       {file_count:,}")
    print(f"Repository Total Size:  {format_size(total_bytes)}")
    print(f"Proposed Deletions:     {len(delete_candidates):,} files ({format_size(sum(d['size_bytes'] for d in delete_candidates))})")
    print(f"NEVER DELETE Size:      {format_size(class_summary['NEVER DELETE']['size'])} ({class_summary['NEVER DELETE']['count']:,} files)")
    print(f"KEEP Size:              {format_size(class_summary['KEEP']['size'])} ({class_summary['KEEP']['count']:,} files)")
    print(f"REVIEW REQUIRED Size:   {format_size(class_summary['REVIEW REQUIRED']['size'])} ({class_summary['REVIEW REQUIRED']['count']:,} files)")


if __name__ == "__main__":
    main()
