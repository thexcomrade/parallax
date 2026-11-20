# PARALLAX Distributed Load Testing Framework (Phase 7C.1)

This directory provides the distributed load generation architecture for the PARALLAX Audio-Visual Deepfake Detection Platform.

## Architecture & Components
- `generators.json`: Fleet topology configuration containing local container generator limits and candidate external Linux worker nodes.
- `allocate_vus.py`: Even, capacity-bounded allocation algorithm that distributes target VUs across active generators without exceeding `max_vus`. Fails closed if capacity is insufficient.
- `verify_generators.py`: Network and service audit tool verifying SSH reachability, staging API connectivity, and k6 installation on all candidate nodes.
- `collect_results.py`: Weighted results aggregation utility that preserves exact histogram integrity and outputs consolidated stage metrics.
- `run_distributed.ps1`: Orchestrator coordinating fleet verification, stage progression, and results capture.

## Operational Constraints
1. **Target Allowlist**: The orchestrator strictly validates `BASE_URL` against allowed staging destinations (`staging-api`, `localhost`, `127.0.0.1`, `host.docker.internal`). Never targets production.
2. **Hardware Boundary**: Single-host execution safely sustains up to 1,000 VUs. Stages $\ge 5,000$ VUs require external distributed nodes and fail closed with `BLOCKED` status if external nodes are offline.
3. **Synthetic Credentials**: Tests strictly use isolated, synthetic staging accounts (`staging_load_bot@staging.parallax.local`). Real user accounts and passwords are permanently quarantined.
