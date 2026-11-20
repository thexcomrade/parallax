"""Generator VU allocation algorithm for PARALLAX distributed load testing.

Distributes requested total VUs across available and enabled load generator nodes.
Enforces:
1. Never exceed node max_vus.
2. Even distribution across active nodes.
3. Exact total conservation (sum of allocations == requested total).
4. Explicit rejection if aggregate capacity is insufficient.
5. Rejection of disabled or unreachable nodes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


class InsufficientCapacityError(ValueError):
    """Raised when available generators cannot support the requested VU count."""


class GeneratorAllocationError(ValueError):
    """Raised when an allocation invariant is violated."""


def allocate_vus(total_vus: int, generators: list[dict[str, Any]]) -> dict[str, int]:
    """Distribute requested total VUs across enabled generator nodes.

    Args:
        total_vus: Total target concurrent virtual users.
        generators: List of generator dictionary configs.

    Returns:
        Mapping of generator id to assigned VUs.
    """
    if total_vus <= 0:
        return {}

    # 1. Filter enabled and available generators
    active_generators = [
        g for g in generators
        if g.get("enabled", False) and g.get("max_vus", 0) > 0
    ]

    if not active_generators:
        raise InsufficientCapacityError(
            f"Cannot allocate {total_vus} VUs: zero enabled generators found."
        )

    # 2. Check aggregate capacity ceiling
    total_capacity = sum(g["max_vus"] for g in active_generators)
    if total_vus > total_capacity:
        raise InsufficientCapacityError(
            f"Requested {total_vus:,} VUs exceeds total enabled generator capacity of {total_capacity:,} VUs."
        )

    # 3. Iterative even distribution respecting individual max_vus
    allocations: dict[str, int] = {g["id"]: 0 for g in active_generators}
    remaining_vus = total_vus
    active_pool = active_generators.copy()

    while remaining_vus > 0 and active_pool:
        # Determine even share per active generator
        fair_share = math.ceil(remaining_vus / len(active_pool))
        new_active_pool: list[dict[str, Any]] = []

        for gen in active_pool:
            gid = gen["id"]
            capacity = gen["max_vus"] - allocations[gid]
            if capacity <= 0:
                continue

            give = min(fair_share, capacity, remaining_vus)
            allocations[gid] += give
            remaining_vus -= give

            if allocations[gid] < gen["max_vus"] and remaining_vus > 0:
                new_active_pool.append(gen)

            if remaining_vus == 0:
                break

        active_pool = new_active_pool

    # 4. Invariant audit
    allocated_sum = sum(allocations.values())
    if allocated_sum != total_vus:
        raise GeneratorAllocationError(
            f"Allocation mismatch: requested {total_vus}, but allocated {allocated_sum}"
        )

    for gen in active_generators:
        gid = gen["id"]
        if allocations[gid] > gen["max_vus"]:
            raise GeneratorAllocationError(
                f"Generator {gid} exceeded max capacity: {allocations[gid]} > {gen['max_vus']}"
            )

    return allocations


def main() -> None:
    config_path = Path(__file__).resolve().parent / "generators.json"
    if not config_path.exists():
        print(f"Error: {config_path} not found.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Combine local and external generators
    gens: list[dict[str, Any]] = []
    if "local_host" in data and data["local_host"].get("enabled"):
        gens.append(data["local_host"])
    gens.extend(data.get("external_generators", []))

    test_targets = [50, 100, 500, 1000, 5000, 10000, 30000]
    print("=== Generator Allocation Verification ===")
    for target in test_targets:
        try:
            alloc = allocate_vus(target, gens)
            print(f"[OK] Target {target:>6,} VUs -> Allocation: {alloc}")
        except InsufficientCapacityError as e:
            print(f"[BLOCKED] Target {target:>6,} VUs -> {e}")


if __name__ == "__main__":
    main()
