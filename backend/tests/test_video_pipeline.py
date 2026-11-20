from pathlib import Path

import torch

from app.ml.models.face_encoder import FaceEncoder
from app.ml.preprocess.video import extract_video_frames


def test_video_frame_extraction(synthetic_video_with_audio: Path) -> None:
    """Verify video extraction produces normalized RGB frames at 224x224."""
    res = extract_video_frames(
        synthetic_video_with_audio, target_count=16, window_size=16, stride=8
    )
    assert res.frames_tensor.shape == (16, 3, 224, 224)
    assert res.frames_tensor.dtype == torch.float32
    assert torch.isfinite(res.frames_tensor).all()
    assert len(res.windows) > 0
    assert res.windows[0].shape == (16, 3, 224, 224)


def test_face_encoder_forward_pass() -> None:
    """Verify FaceEncoder maps frame sequences to V in R^256 with frozen ViT."""
    encoder = FaceEncoder(d_model=256, pretrained=False)
    encoder.eval()

    # ViT weights must be frozen
    for param in encoder.vit.parameters():
        assert not param.requires_grad

    # Synthetic batch: B=2, T=16 frames, 3 channels, 224x224
    dummy_video = torch.randn(2, 16, 3, 224, 224)
    with torch.no_grad():
        V, tokens = encoder(dummy_video)

    assert V.shape == (2, 256)
    # n=4 grouping: 16 / 4 = 4 tokens
    assert tokens.shape == (2, 4, 256)
    assert torch.isfinite(V).all()
    assert torch.isfinite(tokens).all()
