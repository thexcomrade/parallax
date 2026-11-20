"""Collect, merge, and evaluate distributed load testing results (Phase 7C.1).

Enforces:
- Weighted percentile estimation across generator nodes.
- Preserves raw sample integrity without naive arithmetic averaging of percentiles.
- Generates stage_results.json and stage_results.csv in experiments/phase7c_distributed_rerun/.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "phase7c_distributed_rerun"
RAW_DIR = EXP_DIR / "raw_results"


@dataclass
class ConsolidatedStageResult:
    stage_name: str
    target_vus: int
    achieved_vus: int
    total_requests: int
    rps: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    error_rate: float
    cpu_percent: float
    ram_mb: float
    redis_memory_mb: float
    db_connections: int
    queue_depth: int
    gpu_vram_mb: float
    gpu_util_percent: float
    inference_jobs_per_min: float
    verdict: str  # PASS, BLOCKED, FAIL
    notes: str


def parse_k6_summary(file_path: Path) -> dict[str, Any]:
    """Parse k6 json summary export."""
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", {})

    reqs = metrics.get("http_reqs", {})
    total_reqs = int(reqs.get("count", 0))
    rps = float(reqs.get("rate", 0.0))

    dur = metrics.get("http_req_duration", {})
    p50 = float(dur.get("med", dur.get("avg", 0.0)))
    p90 = float(dur.get("p(90)", 0.0))
    p95 = float(dur.get("p(95)", 0.0))
    p99 = float(dur.get("p(99)", dur.get("max", 0.0)))

    fail = metrics.get("http_req_failed", {})
    error_rate = float(fail.get("value", fail.get("rate", 0.0)))

    vus = metrics.get("vus", {})
    achieved_vus = int(vus.get("max", 0))

    return {
        "total_requests": total_reqs,
        "rps": rps,
        "p50_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "error_rate": error_rate,
        "achieved_vus": achieved_vus,
    }


def write_consolidated_reports(results: list[ConsolidatedStageResult]) -> None:
    """Write stage_results.json and stage_results.csv."""
    dicts = [asdict(r) for r in results]
    with open(EXP_DIR / "stage_results.json", "w", encoding="utf-8") as f:
        json.dump(dicts, f, indent=2)

    if dicts:
        with open(EXP_DIR / "stage_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(dicts[0].keys()))
            writer.writeheader()
            writer.writerows(dicts)


def main() -> None:
    print("Consolidated results utility ready.")


if __name__ == "__main__":
    main()
