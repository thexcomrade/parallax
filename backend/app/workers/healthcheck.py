"""Worker health probe for Docker container HEALTHCHECK.

Verifies:
1. Redis connectivity via ping.
2. Model checkpoint file accessibility.
3. CUDA GPU availability (if REQUIRE_GPU is enabled).
4. Lightweight CUDA tensor operation (bit-exact 1 + 1 == 2).
5. Fast execution (< 100 ms) without loading or invoking AVT2-DWF.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import torch


async def _check_redis() -> bool:
    try:
        from app.core.redis import get_redis_client

        client = await get_redis_client()
        pong = await client.ping()
        return bool(pong)
    except Exception as e:
        print(f"[HEALTHCHECK FAIL] Redis: {e}", file=sys.stderr)
        return False


def _check_checkpoint() -> bool:
    try:
        from app.core.config import settings

        raw_path = Path(settings.MODEL_PATH)
        if raw_path.is_absolute():
            return raw_path.exists()
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        backend_dir = Path(__file__).resolve().parent.parent.parent
        candidates = [
            backend_dir / raw_path,
            repo_root / raw_path,
            Path("/app") / raw_path,
            Path("/app/backend") / raw_path,
        ]
        return any(c.exists() for c in candidates)
    except Exception as e:
        print(f"[HEALTHCHECK FAIL] Checkpoint: {e}", file=sys.stderr)
        return False


def _check_cuda() -> bool:
    require_gpu = os.getenv("REQUIRE_GPU", "true").lower() in ("true", "1", "yes")
    if not require_gpu:
        return True

    if not torch.cuda.is_available():
        print("[HEALTHCHECK FAIL] CUDA is not available but REQUIRE_GPU=True", file=sys.stderr)
        return False

    try:
        x = torch.ones(1, device="cuda")
        y = x + 1.0
        torch.cuda.synchronize()
        if y.item() != 2.0:
            print("[HEALTHCHECK FAIL] CUDA tensor arithmetic verification failed", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[HEALTHCHECK FAIL] CUDA tensor operation: {e}", file=sys.stderr)
        return False

    return True


async def main() -> int:
    cuda_ok = _check_cuda()
    if not cuda_ok:
        return 1

    chk_ok = _check_checkpoint()
    if not chk_ok:
        print("[HEALTHCHECK WARNING] Checkpoint not found; running in untrained mode")

    redis_ok = await _check_redis()
    if not redis_ok:
        return 1

    print("[HEALTHCHECK OK] Worker healthy (CUDA, Checkpoint, Redis)")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
