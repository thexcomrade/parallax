"""Automated tests for CUDA-enabled inference worker and device management.

Covers Section 10:
- Test A: Device selection (auto, explicit, GPU-required failure)
- Test B: Model placement on target device & eval mode
- Test C: Input tensor device placement & dtype compatibility
- Test D: CPU regression inference
- Test E: CUDA inference on physical GPU (when available)
- Test F: Worker startup validation & concurrency guard
- Test G: End-to-end pipeline execution with device placement
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from app.core.config import settings
from app.ml.device import get_device_info, reset_device_cache
from app.ml.inference import (
    InferenceResult,
    get_model,
    load_model,
    run_inference,
)
from app.ml.pipeline import run_pipeline
from app.workers.inference_worker import InferenceWorker


@pytest.fixture(autouse=True)
def clean_device_cache() -> None:
    reset_device_cache()


# ---------------------------------------------------------------------------
# Test A: Device Selection
# ---------------------------------------------------------------------------


def test_device_selection_auto_and_explicit() -> None:
    """Device selector must resolve cuda:0 when available or cpu in development."""
    info = get_device_info()
    if torch.cuda.is_available():
        assert info.is_cuda is True
        assert info.device.type == "cuda"
        assert info.device.index == 0
        assert "GeForce" in info.device_name or len(info.device_name) > 0
    else:
        assert info.is_cuda is False
        assert info.device.type == "cpu"

    # Explicit CPU request
    cpu_info = get_device_info(force_device="cpu")
    assert cpu_info.device == torch.device("cpu")
    assert cpu_info.is_cuda is False


def test_device_selection_require_gpu_enforcement() -> None:
    """When REQUIRE_GPU=True and CUDA is absent, selector must raise RuntimeError."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError, match="GPU execution is required"),
    ):
        get_device_info(require_gpu=True)


# ---------------------------------------------------------------------------
# Test B: Model Placement
# ---------------------------------------------------------------------------


def test_model_placement_on_target_device() -> None:
    """load_model must place all model parameters on target device in eval mode."""
    target_dev = torch.device("cpu")
    load_model(device=target_dev)
    model = get_model()

    assert model.training is False
    param = next(model.parameters())
    assert param.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Physical CUDA GPU required")
def test_model_placement_on_cuda() -> None:
    """Model must place all parameters on cuda:0 when targeted."""
    cuda_dev = torch.device("cuda:0")
    load_model(device=cuda_dev)
    model = get_model()

    assert model.training is False
    param = next(model.parameters())
    assert param.device.type == "cuda"
    assert param.device.index == 0


# ---------------------------------------------------------------------------
# Test C: Input Tensor Placement
# ---------------------------------------------------------------------------


def test_input_tensor_device_placement_cpu() -> None:
    """Input tensors created on CPU must execute through run_inference on target device."""
    target_dev = torch.device("cpu")
    load_model(device=target_dev)

    b, k = 1, 3
    video_windows = torch.randn(b, k, 16, 3, 224, 224, dtype=torch.float32)
    audio_segments = torch.randn(b, k, 104, 100, dtype=torch.float32)
    audio_mask = torch.ones(k, dtype=torch.float32)

    res = run_inference(
        video_windows=video_windows,
        audio_segments=audio_segments,
        audio_mask=audio_mask,
        device=target_dev,
    )
    assert isinstance(res, InferenceResult)
    assert res.verdict in ("FAKE", "AUTHENTIC", "INCONCLUSIVE")
    assert 0.0 <= res.p_fake <= 1.0


# ---------------------------------------------------------------------------
# Test D: CPU Regression
# ---------------------------------------------------------------------------


def test_cpu_regression_inference_structure() -> None:
    """CPU execution must produce valid inference structure with finite scores."""
    target_dev = torch.device("cpu")
    load_model(device=target_dev)

    video_win = torch.zeros(1, 2, 16, 3, 224, 224)
    audio_seg = torch.zeros(1, 2, 104, 100)
    audio_mask = torch.tensor([1.0, 1.0])

    res = run_inference(video_win, audio_seg, audio_mask, device=target_dev)
    assert 0.0 <= res.p_fake <= 1.0
    assert 0.0 <= res.p_authentic <= 1.0
    assert len(res.per_window_scores) == 2


# ---------------------------------------------------------------------------
# Test E: CUDA Inference on Physical Hardware
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Physical CUDA GPU required")
def test_cuda_inference_on_physical_gpu() -> None:
    """Real AVT²-DWF forward pass on physical GPU must produce finite outputs on cuda:0."""
    cuda_dev = torch.device("cuda:0")
    load_model(device=cuda_dev)
    model = get_model()

    param = next(model.parameters())
    assert param.device.type == "cuda"

    # Create test input tensors on CPU to test automatic boundary transfer
    b, k = 1, 2
    video_win = torch.randn(b, k, 16, 3, 224, 224, device="cpu")
    audio_seg = torch.randn(b, k, 104, 100, device="cpu")
    audio_mask = torch.tensor([1.0, 1.0], device="cpu")

    mem_before = torch.cuda.memory_allocated(cuda_dev)

    res = run_inference(video_win, audio_seg, audio_mask, device=cuda_dev)
    torch.cuda.synchronize(cuda_dev)

    assert isinstance(res, InferenceResult)
    assert res.verdict in ("FAKE", "AUTHENTIC", "INCONCLUSIVE")
    assert 0.0 <= res.p_fake <= 1.0
    assert 0.0 <= res.p_authentic <= 1.0
    assert len(res.per_window_scores) == k

    # Memory was allocated for inference
    mem_after = torch.cuda.memory_allocated(cuda_dev)
    assert mem_after >= mem_before


# ---------------------------------------------------------------------------
# Test F: Worker Startup & Concurrency
# ---------------------------------------------------------------------------


def test_worker_initialization_metadata() -> None:
    """Worker must record device and enforce concurrency = 1."""
    worker = InferenceWorker(worker_id="test-worker-cuda", max_concurrency=1)
    assert worker.max_concurrency == 1
    assert worker._semaphore._value == 1  # type: ignore[attr-defined]
    assert worker.device_type in ("cuda", "cpu")


def test_worker_refuses_gpu_required_startup_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker initialization must fail if REQUIRE_GPU=True and CUDA is missing."""
    monkeypatch.setattr(settings, "REQUIRE_GPU", True)
    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError, match="GPU execution is required"),
    ):
        InferenceWorker(worker_id="strict-gpu-worker", max_concurrency=1)


# ---------------------------------------------------------------------------
# Test G: Pipeline Regression
# ---------------------------------------------------------------------------


def test_pipeline_device_flow(synthetic_video_with_audio: Path, tmp_path: Path) -> None:
    """Pipeline execution must yield events through 'done' with valid result."""
    events = list(run_pipeline(synthetic_video_with_audio, tmp_path))
    assert len(events) >= 8

    done_event = events[-1]
    assert done_event.stage == "done"
    assert done_event.result is not None
    assert done_event.result.inference.verdict in ("FAKE", "AUTHENTIC", "INCONCLUSIVE")
    assert (tmp_path / "timeline.json").exists()
