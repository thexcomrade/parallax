from pathlib import Path

import torch

from app.ml.preprocess.align import align_audio_video
from app.ml.preprocess.audio import extract_audio_features
from app.ml.preprocess.video import extract_video_frames


def test_video_audio_preprocessing_shapes(synthetic_video_with_audio: Path) -> None:
    """Verify that video and audio preprocessing produce canonical target dimensions."""
    # 1. Video extraction: 32 frames sampled, window_size=16, stride=8
    v_res = extract_video_frames(
        synthetic_video_with_audio, target_count=32, window_size=16, stride=8
    )
    assert v_res.frames_tensor.shape == (32, 3, 224, 224)
    assert v_res.frames_tensor.dtype == torch.float32

    # 2. Audio extraction: 104 feature channels (40 MFCC + 64 Mel)
    a_res = extract_audio_features(synthetic_video_with_audio)
    assert a_res.audio_present is True
    assert a_res.mfcc.shape[0] == 40
    assert a_res.log_mel.shape[0] == 64
    assert len(a_res.segments) > 0
    assert a_res.segments[0].shape == (104, 100)

    # 3. Multimodal alignment
    aligned = align_audio_video(v_res, a_res, window_size=16, stride=8)
    assert aligned.video_windows.ndim == 5  # [K, 16, 3, 224, 224]
    assert aligned.video_windows.shape[1:] == (16, 3, 224, 224)
    assert aligned.audio_segments.ndim == 3  # [K, 104, 100]
    assert aligned.audio_segments.shape[1:] == (104, 100)
    assert aligned.audio_mask.ndim == 1
    assert aligned.audio_mask.shape[0] == aligned.video_windows.shape[0]
    assert aligned.video_windows.dtype == torch.float32
    assert aligned.audio_segments.dtype == torch.float32


def test_missing_audio_parity(synthetic_video_no_audio: Path) -> None:
    """Verify parity behavior when audio is absent: zeroed segments and mask."""
    v_res = extract_video_frames(
        synthetic_video_no_audio, target_count=32, window_size=16, stride=8
    )
    a_res = extract_audio_features(synthetic_video_no_audio)
    assert a_res.audio_present is False
    assert a_res.mfcc.shape[0] == 40
    assert a_res.log_mel.shape[0] == 64

    aligned = align_audio_video(v_res, a_res, window_size=16, stride=8)
    assert aligned.audio_mask.sum().item() == 0.0
    assert (aligned.audio_segments == 0.0).all().item()


def test_deterministic_preprocessing_parity(synthetic_video_with_audio: Path) -> None:
    """Verify that repeated extractions produce identical deterministic tensors."""
    v1 = extract_video_frames(synthetic_video_with_audio, target_count=32, window_size=16, stride=8)
    v2 = extract_video_frames(synthetic_video_with_audio, target_count=32, window_size=16, stride=8)
    assert torch.allclose(v1.frames_tensor, v2.frames_tensor, atol=1e-5)

    a1 = extract_audio_features(synthetic_video_with_audio)
    a2 = extract_audio_features(synthetic_video_with_audio)
    assert torch.allclose(a1.mfcc, a2.mfcc, atol=1e-5)
    assert torch.allclose(a1.log_mel, a2.log_mel, atol=1e-5)
