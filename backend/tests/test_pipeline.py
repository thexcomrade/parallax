from pathlib import Path

import pytest
import torch

from app.ml import inference as inference_module
from app.ml.models.audio_encoder import AudioEncoder
from app.ml.models.avt2_dwf import AVT2DWF
from app.ml.models.dwf import DynamicWeightFusion
from app.ml.models.face_encoder import FaceEncoder
from app.ml.preprocess.align import align_audio_video
from app.ml.preprocess.audio import extract_audio_features
from app.ml.preprocess.probe import CorruptMediaError, MediaLimitsExceededError, probe_media
from app.ml.preprocess.video import extract_video_frames


def test_probe_valid_synthetic_video(synthetic_video_with_audio: Path) -> None:
    res = probe_media(synthetic_video_with_audio)
    assert 2.5 <= res.duration_s <= 3.5
    assert res.width == 320
    assert res.height == 240
    assert 24.0 <= res.fps <= 26.0
    assert res.audio_present is True
    assert res.size_bytes > 0
    assert "h264" in res.video_codec.lower() or "avc" in res.video_codec.lower()


def test_probe_video_no_audio(synthetic_video_no_audio: Path) -> None:
    res = probe_media(synthetic_video_no_audio)
    assert res.audio_present is False
    assert 2.5 <= res.duration_s <= 3.5
    assert res.width == 320
    assert res.height == 240


def test_probe_corrupt_file(corrupt_file: Path) -> None:
    with pytest.raises(CorruptMediaError):
        probe_media(corrupt_file)


def test_probe_nonexistent_file() -> None:
    with pytest.raises(CorruptMediaError):
        probe_media("nonexistent_video_path_12345.mp4")


def test_probe_duration_limits_exceeded(
    synthetic_video_with_audio: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "MAX_DURATION_S", 1.0)
    with pytest.raises(MediaLimitsExceededError):
        probe_media(synthetic_video_with_audio)


def test_video_preprocess_shapes_and_windows(synthetic_video_with_audio: Path) -> None:
    video_res = extract_video_frames(
        synthetic_video_with_audio, target_count=32, window_size=16, stride=8
    )

    # Output frames tensor shape: (32, 3, 224, 224)
    assert video_res.frames_tensor.shape == (32, 3, 224, 224)
    assert video_res.frames_tensor.dtype == torch.float32

    # Verify sliding windows (window=16, stride=8 over 32 frames -> 3 windows)
    assert len(video_res.windows) == 3
    for win in video_res.windows:
        assert win.shape == (16, 3, 224, 224)
        assert win.dtype == torch.float32

    # Verify raw crops and timestamps
    assert len(video_res.raw_crops) == 32
    assert len(video_res.timestamps) == 32
    assert video_res.raw_crops[0].shape == (224, 224, 3)

    # Face coverage metric
    assert 0.0 <= video_res.face_coverage <= 1.0


def test_video_preprocess_no_face_fallback(synthetic_video_no_audio: Path) -> None:
    # synthetic testsrc has no human face, should gracefully fall back to center crop
    video_res = extract_video_frames(synthetic_video_no_audio, target_count=32)
    assert video_res.frames_tensor.shape == (32, 3, 224, 224)
    assert video_res.face_coverage < 0.40  # Low face coverage flagged for no-face clips
    assert len(video_res.raw_crops) == 32


def test_audio_preprocess_features(synthetic_video_with_audio: Path) -> None:
    audio_res = extract_audio_features(synthetic_video_with_audio)

    assert audio_res.audio_present is True
    assert audio_res.sample_rate == 16000
    assert 2.5 <= audio_res.duration_s <= 3.5

    # 40 MFCCs and 64 Log-Mel channels
    assert audio_res.mfcc.shape[0] == 40
    assert audio_res.log_mel.shape[0] == 64
    assert audio_res.mfcc.dtype == torch.float32
    assert audio_res.log_mel.dtype == torch.float32

    # 1.0s segments: combined [MFCC; Log-Mel] has 104 channels and 100 time frames
    assert len(audio_res.segments) >= 3
    for seg in audio_res.segments:
        assert seg.shape == (104, 100)
        assert seg.dtype == torch.float32

    assert len(audio_res.segment_timestamps) == len(audio_res.segments)


def test_audio_preprocess_no_audio_clip(synthetic_video_no_audio: Path) -> None:
    audio_res = extract_audio_features(synthetic_video_no_audio)

    assert audio_res.audio_present is False
    assert audio_res.duration_s == 0.0
    assert len(audio_res.segments) == 0
    assert audio_res.mfcc.shape[1] == 0
    assert audio_res.log_mel.shape[1] == 0


def test_multimodal_alignment_full_audio_video(synthetic_video_with_audio: Path) -> None:
    video_res = extract_video_frames(synthetic_video_with_audio, target_count=32)
    audio_res = extract_audio_features(synthetic_video_with_audio)

    aligned = align_audio_video(video_res, audio_res)

    assert aligned.audio_present is True
    assert aligned.video_tensor.shape == (32, 3, 224, 224)
    num_windows = aligned.video_windows.shape[0]
    assert num_windows == 3
    assert aligned.video_windows.shape == (num_windows, 16, 3, 224, 224)
    assert aligned.audio_segments.shape == (num_windows, 104, 100)

    # Audio mask is 1.0 for all windows when audio is present
    assert aligned.audio_mask.shape == (num_windows,)
    assert (aligned.audio_mask == 1.0).all()
    assert len(aligned.window_timestamps) == num_windows


def test_multimodal_alignment_no_audio_masked(synthetic_video_no_audio: Path) -> None:
    video_res = extract_video_frames(synthetic_video_no_audio, target_count=32)
    audio_res = extract_audio_features(synthetic_video_no_audio)

    aligned = align_audio_video(video_res, audio_res)

    assert aligned.audio_present is False
    num_windows = aligned.video_windows.shape[0]
    assert num_windows == 3
    assert aligned.video_windows.shape == (num_windows, 16, 3, 224, 224)

    # Audio features are padded zeros of correct shape (K, 104, 100)
    assert aligned.audio_segments.shape == (num_windows, 104, 100)
    assert (aligned.audio_segments == 0.0).all()

    # Audio mask is strictly 0.0 for all windows so DWF sets w_a = 0
    assert aligned.audio_mask.shape == (num_windows,)
    assert (aligned.audio_mask == 0.0).all()


# =============================================================================
# Phase 5 — Model + Inference + Explainability tests
# =============================================================================


# Shared model fixture (untrained, eval mode)
@pytest.fixture(scope="module")
def avt2_model() -> AVT2DWF:
    model = AVT2DWF(pretrained_vit=False)
    model.eval()
    return model


@pytest.fixture(scope="module")
def aligned_data_full(synthetic_video_with_audio: Path) -> "AlignedData":
    video_res = extract_video_frames(synthetic_video_with_audio, target_count=32)
    audio_res = extract_audio_features(synthetic_video_with_audio)
    return align_audio_video(video_res, audio_res)


@pytest.fixture(scope="module")
def aligned_data_no_audio(synthetic_video_no_audio: Path) -> "AlignedData":
    video_res = extract_video_frames(synthetic_video_no_audio, target_count=32)
    audio_res = extract_audio_features(synthetic_video_no_audio)
    return align_audio_video(video_res, audio_res)


# Type alias for readability
from app.ml.preprocess.align import AlignedMultimodalData as AlignedData  # noqa: E402


def _make_batch(aligned: AlignedData) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Wrap AlignedMultimodalData tensors into B=1 batch for model input."""
    # video_windows: (K, 16, 3, 224, 224) → (1, K, 16, 3, 224, 224)
    video_windows = aligned.video_windows.unsqueeze(0)
    # audio_segments: (K, 104, 100) → (1, K, 104, 100)
    audio_segments = aligned.audio_segments.unsqueeze(0)
    # audio_mask: (K,) stays (K,)
    audio_mask = aligned.audio_mask
    return video_windows, audio_segments, audio_mask


def test_face_encoder_output_shapes() -> None:
    """FaceEncoder must return V[B,256] and tokens[B,K,256] for valid input."""
    encoder = FaceEncoder(pretrained=False)
    encoder.eval()
    # B=1, T=16 frames, 3x224x224
    x = torch.zeros(1, 16, 3, 224, 224)
    with torch.inference_mode():
        V, tokens = encoder(x)
    assert V.shape == (1, 256), f"V shape mismatch: {V.shape}"
    # K = ceil(16/4) = 4
    assert tokens.shape == (1, 4, 256), f"tokens shape mismatch: {tokens.shape}"


def test_audio_encoder_output_shapes() -> None:
    """AudioEncoder must return A[B,256] and tokens[B,K,256] for K=3 segments."""
    encoder = AudioEncoder()
    encoder.eval()
    x = torch.zeros(1, 3, 104, 100)
    with torch.inference_mode():
        A, tokens = encoder(x)
    assert A.shape == (1, 256), f"A shape mismatch: {A.shape}"
    assert tokens.shape == (1, 3, 256), f"tokens shape mismatch: {tokens.shape}"


def test_dwf_weights_sum_to_one() -> None:
    """DWF must produce w_a + w_v = 1.0 (after masking/renormalisation)."""
    dwf = DynamicWeightFusion()
    dwf.eval()
    A = torch.randn(1, 256)
    V = torch.randn(1, 256)
    audio_mask = torch.ones(3)  # audio present
    with torch.inference_mode():
        _, w_a, w_v, _ = dwf(A, V, audio_mask)
    total = (w_a + w_v).item()
    assert abs(total - 1.0) < 1e-5, f"w_a + w_v = {total} ≠ 1.0"


def test_dwf_audio_mask_zero_forces_wa_zero() -> None:
    """When audio_mask is all zeros, DWF must set w_a == 0 exactly."""
    dwf = DynamicWeightFusion()
    dwf.eval()
    A = torch.randn(1, 256)
    V = torch.randn(1, 256)
    audio_mask = torch.zeros(3)  # audio absent
    with torch.inference_mode():
        _, w_a, w_v, _ = dwf(A, V, audio_mask)
    assert w_a.item() == 0.0, f"w_a should be 0 but is {w_a.item()}"
    assert abs(w_v.item() - 1.0) < 1e-5, f"w_v should be 1.0 but is {w_v.item()}"


def test_avt2_forward_pass_with_audio(
    avt2_model: AVT2DWF,
    aligned_data_full: AlignedData,
) -> None:
    """Full forward pass with audio: logits[B,2], w_a + w_v = 1, scores shape OK."""
    video_windows, audio_segments, audio_mask = _make_batch(aligned_data_full)
    with torch.inference_mode():
        result = avt2_model(video_windows, audio_segments, audio_mask)

    K = video_windows.shape[1]
    assert result.logits.shape == (1, 2), f"logits shape: {result.logits.shape}"
    assert result.w_a.shape == (1, 1)
    assert result.w_v.shape == (1, 1)
    assert result.per_window_scores.shape == (1, K), (
        f"per_window_scores shape: {result.per_window_scores.shape}"
    )
    total = (result.w_a + result.w_v).item()
    assert abs(total - 1.0) < 1e-4, f"w_a + w_v = {total}"


def test_avt2_forward_pass_no_audio(
    avt2_model: AVT2DWF,
    aligned_data_no_audio: AlignedData,
) -> None:
    """With no audio: w_a must be 0, logits still valid shape."""
    video_windows, audio_segments, audio_mask = _make_batch(aligned_data_no_audio)
    # audio_mask should be all zeros
    assert (audio_mask == 0.0).all(), "audio_mask should be all zeros for no-audio clip"

    with torch.inference_mode():
        result = avt2_model(video_windows, audio_segments, audio_mask)

    assert result.logits.shape == (1, 2)
    assert result.w_a.item() == 0.0, f"w_a should be 0 but is {result.w_a.item()}"


def test_inference_module_untrained_mode(tmp_path: Path) -> None:
    """With no checkpoint, MODEL_MODE must be 'untrained'."""
    # Point model path to a non-existent file
    inference_module._model = None  # type: ignore[attr-defined]
    inference_module.MODEL_MODE = "untrained"  # type: ignore[attr-defined]
    inference_module.CHECKPOINT_HASH = None  # type: ignore[attr-defined]

    from unittest.mock import patch

    with patch.object(inference_module.settings, "MODEL_PATH", str(tmp_path / "no_checkpoint.pt")):
        inference_module.load_model()

    assert inference_module.MODEL_MODE == "untrained"
    assert inference_module.CHECKPOINT_HASH is None


def test_inference_run_returns_valid_result(
    aligned_data_full: AlignedData,
) -> None:
    """run_inference returns InferenceResult with correct types and ranges."""
    from app.ml.inference import InferenceResult, run_inference

    # Reset to untrained mode with fresh model
    inference_module._model = None  # type: ignore[attr-defined]
    inference_module.load_model()

    video_windows, audio_segments, audio_mask = _make_batch(aligned_data_full)
    result: InferenceResult = run_inference(
        video_windows, audio_segments, audio_mask, model_version="0.0.0"
    )

    assert result.verdict in ("FAKE", "AUTHENTIC", "INCONCLUSIVE")
    assert 0.0 <= result.p_fake <= 1.0
    assert 0.0 <= result.p_authentic <= 1.0
    assert abs(result.p_fake + result.p_authentic - 1.0) < 1e-4
    assert 0.0 <= result.w_audio <= 1.0
    assert 0.0 <= result.w_video <= 1.0
    assert abs(result.w_audio + result.w_video - 1.0) < 1e-4
    assert result.model_mode in ("untrained", "trained")
    assert isinstance(result.per_window_scores, list)
    assert all(0.0 <= s <= 1.0 for s in result.per_window_scores)


def test_heatmap_pngs_written(
    avt2_model: AVT2DWF,
    aligned_data_full: AlignedData,
    synthetic_video_with_audio: Path,
    tmp_path: Path,
) -> None:
    """generate_explanations must write heatmap PNGs and timeline JSON."""
    from app.ml.explain import generate_explanations
    from app.ml.preprocess.audio import extract_audio_features
    from app.ml.preprocess.video import extract_video_frames

    video_res = extract_video_frames(synthetic_video_with_audio, target_count=32)
    audio_res = extract_audio_features(synthetic_video_with_audio)
    aligned = align_audio_video(video_res, audio_res)

    video_windows, audio_segments, audio_mask = _make_batch(aligned)
    with torch.inference_mode():
        result = avt2_model(video_windows, audio_segments, audio_mask)

    per_window_scores = [float(s) for s in result.per_window_scores[0]]
    # Build dummy audio per-window scores (same length)
    audio_pw = per_window_scores[:]
    fused_pw = per_window_scores[:]

    artifact_dir = tmp_path / "artifacts"
    explain_result = generate_explanations(
        model_vit=avt2_model.face_encoder.vit,
        raw_crops=video_res.raw_crops,
        frame_timestamps=video_res.timestamps,
        log_mel=audio_res.log_mel,
        per_window_scores=per_window_scores,
        window_timestamps=aligned.window_timestamps,
        audio_per_window_scores=audio_pw,
        fused_per_window_scores=fused_pw,
        artifact_dir=artifact_dir,
    )

    # At least one heatmap must be written
    assert len(explain_result.heatmap_paths) > 0
    for name in explain_result.heatmap_paths:
        assert (artifact_dir / name).exists(), f"Missing heatmap: {name}"

    # Spectrogram PNG must exist
    assert (artifact_dir / explain_result.spectrogram_path).exists()

    # Timeline JSON must have correct structure
    import json

    timeline_path = artifact_dir / "timeline.json"
    assert timeline_path.exists()
    timeline = json.loads(timeline_path.read_text())
    assert isinstance(timeline, list)
    for entry in timeline:
        assert "t" in entry and "p_face" in entry and "p_audio" in entry and "p_fused" in entry


def test_model_card_json_structure(tmp_path: Path) -> None:
    """model_card.json must have required fields and model_mode=untrained."""
    import json
    from pathlib import Path

    mc_path = Path("app/ml/weights/model_card.json")
    if not mc_path.exists():
        mc_path = (
            Path(__file__).resolve().parent.parent / "app" / "ml" / "weights" / "model_card.json"
        )
    assert mc_path.exists(), "model_card.json missing"
    mc = json.loads(mc_path.read_text())
    required_fields = [
        "model_name",
        "model_version",
        "model_mode",
        "checkpoint_hash",
        "reported_metrics",
        "live_metrics",
    ]
    for field in required_fields:
        assert field in mc, f"model_card.json missing field: {field}"
    assert mc["model_mode"] in ("untrained", "trained")
    if mc["model_mode"] == "trained":
        assert isinstance(mc["checkpoint_hash"], str)
        assert isinstance(mc["live_metrics"], dict)
    else:
        assert mc["checkpoint_hash"] is None
        assert mc["live_metrics"] is None


def test_full_ml_pipeline_stages_and_output(
    synthetic_video_with_audio: Path,
    tmp_path: Path,
) -> None:
    """Full pipeline execution yields all required stages in order."""
    from app.ml.pipeline import run_pipeline

    artifact_dir = tmp_path / "test_pipeline_artifacts"
    events = list(run_pipeline(synthetic_video_with_audio, artifact_dir))

    stage_names = [e.stage for e in events]
    expected_stages = [
        "probe",
        "frames",
        "faces",
        "audio",
        "align",
        "face_enc",
        "audio_enc",
        "fusion",
        "explain",
        "done",
    ]
    for st in expected_stages:
        assert st in stage_names, f"Missing pipeline stage: {st}"

    # Verify final done event
    done_event = events[-1]
    assert done_event.stage == "done"
    assert done_event.progress == 1.0
    assert done_event.result is not None

    res = done_event.result
    assert res.inference.verdict in ("FAKE", "AUTHENTIC", "INCONCLUSIVE")
    assert res.inference.model_mode in ("untrained", "trained")
    assert 0.0 <= res.inference.w_audio <= 1.0
    assert 0.0 <= res.inference.w_video <= 1.0
    assert len(res.explain.heatmap_paths) > 0
    assert len(res.explain.timeline) > 0
