"""Comprehensive, read-only repository, dataset, storage and file hygiene audit script.

Performs recursive inspection of D:\\Projects\\parallax.
Strictly read-only: never deletes, moves, or modifies any file or directory.
Never opens or inspects datasets/manifests/phase5_locked_test.csv.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKED_TEST_PATH = (REPO_ROOT / "datasets" / "manifests" / "phase5_locked_test.csv").resolve()

def get_sha256(filepath: Path) -> str:
    """Compute SHA-256 for a file without opening locked test manifest."""
    if filepath.resolve() == LOCKED_TEST_PATH:
        return "[LOCKED_TEST_ISOLATED]"
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        return f"[ERROR: {e}]"

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

def main() -> None:
    print(f"Starting PARALLAX Repository Cleanup Audit on: {REPO_ROOT}")
    t0 = time.time()

    all_dirs: list[dict[str, Any]] = []
    empty_dirs: list[dict[str, Any]] = []
    empty_files: list[dict[str, Any]] = []
    temp_cache_files: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    dataset_breakdown: dict[str, Any] = {}

    total_repo_files = 0
    total_repo_dirs = 0
    total_repo_size = 0

    hash_to_files: dict[str, list[dict[str, Any]]] = {}

    temp_exts = {".tmp", ".temp", ".bak", ".old", ".orig", ".swp", ".pyc", ".pyo"}
    temp_dir_names = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "temp", "tmp", "cache", ".vite"}

    for root, dirs, files in os.walk(REPO_ROOT, topdown=True):
        root_path = Path(root)
        rel_root = root_path.relative_to(REPO_ROOT)
        rel_root_str = str(rel_root).replace("\\", "/") if str(rel_root) != "." else "."

        dir_file_count = len(files)
        dir_subfolder_count = len(dirs)
        dir_size = 0

        if dir_file_count == 0 and dir_subfolder_count == 0:
            mtime = datetime.fromtimestamp(root_path.stat().st_mtime, tz=UTC).isoformat()
            empty_dirs.append({
                "path": rel_root_str,
                "mtime": mtime,
            })

        for f in files:
            file_path = root_path / f
            try:
                stat = file_path.stat()
                file_size = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
            except OSError:
                file_size = 0
                mtime = "unknown"

            dir_size += file_size
            total_repo_files += 1
            total_repo_size += file_size

            rel_file_str = str(file_path.relative_to(REPO_ROOT)).replace("\\", "/")

            if file_size == 0 and file_path.resolve() != LOCKED_TEST_PATH:
                empty_files.append({
                    "path": rel_file_str,
                    "mtime": mtime,
                })

            is_temp = False
            suffix = file_path.suffix.lower()
            if suffix in temp_exts or any(td in file_path.parts for td in temp_dir_names) or suffix == ".log":
                is_temp = True

            if is_temp:
                temp_cache_files.append({
                    "path": rel_file_str,
                    "size": file_size,
                    "suffix": suffix,
                    "mtime": mtime,
                })

            if suffix in (".pt", ".pth", ".bin", ".safetensors") and file_path.resolve() != LOCKED_TEST_PATH:
                sha = get_sha256(file_path)
                checkpoints.append({
                    "path": rel_file_str,
                    "size": file_size,
                    "sha256": sha,
                    "mtime": mtime,
                })

            if (
                suffix in (".pt", ".csv", ".json", ".zip", ".tar", ".gz")
                and file_size > 0
                and file_path.resolve() != LOCKED_TEST_PATH
                and ".git" not in file_path.parts
                and "node_modules" not in file_path.parts
                and ".venv" not in file_path.parts
            ):
                sha = get_sha256(file_path)
                if not sha.startswith("["):
                    entry = {"path": rel_file_str, "size": file_size, "mtime": mtime}
                    hash_to_files.setdefault(sha, []).append(entry)

        mtime_dir = datetime.fromtimestamp(root_path.stat().st_mtime, tz=UTC).isoformat()
        all_dirs.append({
            "path": rel_root_str,
            "files": dir_file_count,
            "subfolders": dir_subfolder_count,
            "size": dir_size,
            "mtime": mtime_dir,
        })
        total_repo_dirs += 1

    duplicates = {
        h: flist for h, flist in hash_to_files.items() if len(flist) > 1
    }

    dataset_root = REPO_ROOT / "datasets"
    if dataset_root.exists():
        for d in dataset_root.rglob("*"):
            if d.is_dir():
                rel_d = str(d.relative_to(REPO_ROOT)).replace("\\", "/")
                d_files = [f for f in d.iterdir() if f.is_file()]
                d_size = sum(f.stat().st_size for f in d_files)
                real_count = sum(1 for f in d_files if "real" in f.name.lower())
                fake_count = sum(1 for f in d_files if "fake" in f.name.lower())
                zero_count = sum(1 for f in d_files if f.stat().st_size == 0 and f.resolve() != LOCKED_TEST_PATH)
                dataset_breakdown[rel_d] = {
                    "path": rel_d,
                    "file_count": len(d_files),
                    "size_bytes": d_size,
                    "size_fmt": format_size(d_size),
                    "real_count": real_count,
                    "fake_count": fake_count,
                    "zero_count": zero_count,
                }

    dt = time.time() - t0
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "scan_duration_s": round(dt, 2),
        "total_repo_files": total_repo_files,
        "total_repo_dirs": total_repo_dirs,
        "total_repo_size_bytes": total_repo_size,
        "total_repo_size_fmt": format_size(total_repo_size),
        "empty_dirs_count": len(empty_dirs),
        "empty_files_count": len(empty_files),
        "temp_cache_files_count": len(temp_cache_files),
        "temp_cache_size_bytes": sum(f["size"] for f in temp_cache_files),
        "temp_cache_size_fmt": format_size(sum(f["size"] for f in temp_cache_files)),
        "checkpoints_count": len(checkpoints),
        "checkpoints_size_bytes": sum(c["size"] for c in checkpoints),
        "checkpoints_size_fmt": format_size(sum(c["size"] for c in checkpoints)),
        "duplicate_groups_count": len(duplicates),
        "duplicate_files_count": sum(len(flist) for flist in duplicates.values()),
    }

    out_json = REPO_ROOT / "reports" / "repository_cleanup_inventory.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "summary": summary,
        "empty_dirs": empty_dirs,
        "empty_files": empty_files,
        "temp_cache_files_sample": temp_cache_files[:100],
        "checkpoints": checkpoints,
        "duplicates": duplicates,
        "dataset_breakdown": dataset_breakdown,
        "all_dirs_count": len(all_dirs),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"Audit completed in {dt:.2f}s")
    print(f"Total Files: {total_repo_files:,} | Total Dirs: {total_repo_dirs:,} | Total Size: {format_size(total_repo_size)}")
    print(f"Empty Dirs: {len(empty_dirs)} | Empty Files: {len(empty_files)} | Temp/Cache Files: {len(temp_cache_files)} ({format_size(sum(f['size'] for f in temp_cache_files))})")
    print(f"Checkpoints: {len(checkpoints)} ({format_size(sum(c['size'] for c in checkpoints))}) | Duplicate Groups: {len(duplicates)}")
    print(f"Full inventory saved to: {out_json}")

if __name__ == "__main__":
    main()
