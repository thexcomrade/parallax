"""Centralized inference device management for PARALLAX.

Ensures consistent device selection across model initialization, checkpoint loading,
input tensor placement, explainability extraction, and worker startup verification.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch

from app.core.config import settings

logger = logging.getLogger(__name__)


class DeviceInfo(NamedTuple):
    device: torch.device
    device_name: str
    is_cuda: bool
    total_memory_mb: float
    compute_capability: tuple[int, int] | None


_cached_device_info: DeviceInfo | None = None


def reset_device_cache() -> None:
    """Reset the module-level device cache (primarily for unit tests)."""
    global _cached_device_info
    _cached_device_info = None


def get_inference_device(
    *,
    force_device: str | None = None,
    require_gpu: bool | None = None,
) -> torch.device:
    """Resolve and return the torch.device for model and tensor placement."""
    info = get_device_info(force_device=force_device, require_gpu=require_gpu)
    return info.device


def get_device_info(
    *,
    force_device: str | None = None,
    require_gpu: bool | None = None,
) -> DeviceInfo:
    """Return structured device metadata, enforcing GPU requirements if configured."""
    global _cached_device_info

    # If parameters are explicitly passed, compute directly without caching
    if force_device is not None or require_gpu is not None:
        return _resolve_device_info(force_device=force_device, require_gpu=require_gpu)

    if _cached_device_info is None:
        _cached_device_info = _resolve_device_info()
    return _cached_device_info


def _resolve_device_info(
    *,
    force_device: str | None = None,
    require_gpu: bool | None = None,
) -> DeviceInfo:
    req_gpu = require_gpu if require_gpu is not None else getattr(settings, "REQUIRE_GPU", False)
    raw_target = (
        force_device
        if force_device is not None
        else getattr(settings, "INFERENCE_DEVICE", "auto")
    )
    target = str(raw_target) if raw_target is not None else "auto"

    cuda_available = torch.cuda.is_available()

    if req_gpu and not cuda_available:
        msg = "GPU execution is required (REQUIRE_GPU=True), but CUDA is not available."
        logger.error(msg)
        raise RuntimeError(msg)

    if target.lower() in ("cpu", "none"):
        return DeviceInfo(
            device=torch.device("cpu"),
            device_name="CPU",
            is_cuda=False,
            total_memory_mb=0.0,
            compute_capability=None,
        )

    if target.lower().startswith("cuda"):
        if not cuda_available:
            if req_gpu:
                raise RuntimeError(f"Requested device '{target}', but CUDA is not available.")
            logger.warning("CUDA requested ('%s') but unavailable. Falling back to CPU.", target)
            return DeviceInfo(
                device=torch.device("cpu"),
                device_name="CPU",
                is_cuda=False,
                total_memory_mb=0.0,
                compute_capability=None,
            )
        dev = torch.device(target)
        dev_idx = dev.index if dev.index is not None else 0
        name = torch.cuda.get_device_name(dev_idx)
        vram = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 * 1024)
        cap = torch.cuda.get_device_capability(dev_idx)
        return DeviceInfo(
            device=dev,
            device_name=name,
            is_cuda=True,
            total_memory_mb=round(vram, 2),
            compute_capability=cap,
        )

    # auto mode: prioritize CUDA if present
    if cuda_available:
        dev = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        cap = torch.cuda.get_device_capability(0)
        return DeviceInfo(
            device=dev,
            device_name=name,
            is_cuda=True,
            total_memory_mb=round(vram, 2),
            compute_capability=cap,
        )

    return DeviceInfo(
        device=torch.device("cpu"),
        device_name="CPU",
        is_cuda=False,
        total_memory_mb=0.0,
        compute_capability=None,
    )
