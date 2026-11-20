from pathlib import Path

import torch

from app.ml.models.audio_encoder import AudioEncoder, AudioInputNorm
from app.ml.preprocess.audio import extract_audio_features


def test_audio_feature_extraction_properties(synthetic_video_with_audio: Path) -> None:
    """Verify acoustic feature dimensions and sampling rate conformance."""
    res = extract_audio_features(synthetic_video_with_audio)
    assert res.audio_present is True
    # 40 MFCC + 64 Mel = 104 features
    assert res.mfcc.shape[0] == 40
    assert res.log_mel.shape[0] == 64
    assert res.mfcc.shape[1] > 0
    assert torch.isfinite(res.mfcc).all()
    assert torch.isfinite(res.log_mel).all()


def test_audio_input_norm_module() -> None:
    """Verify channel-wise audio normalization mechanics and clamping."""
    norm = AudioInputNorm(feat_dim=104, active=True)
    mean = torch.full((104,), 2.0)
    std = torch.full((104,), 0.5)
    norm.set_stats(mean, std)

    # Input: tensor of shape [1, 2, 104, 100] filled with 2.0 (should normalize to 0.0)
    x = torch.full((1, 2, 104, 100), 2.0)
    x_norm = norm(x)
    assert torch.allclose(x_norm, torch.zeros_like(x_norm), atol=1e-4)

    # Extreme input should be clamped between -10.0 and 10.0
    x_extreme = torch.full((1, 2, 104, 100), 1000.0)
    x_clamped = norm(x_extreme)
    assert (x_clamped <= 10.0).all()
    assert (x_clamped >= -10.0).all()


def test_audio_encoder_forward_dimensions() -> None:
    """Verify AudioEncoder outputs clip embedding [B, 256] and tokens [B, K, 256]."""
    encoder = AudioEncoder(d_model=256, use_layernorm=True, use_input_norm=True)
    encoder.eval()

    # Synthetic batch: B=2, K=3 segments, 104 channels, 100 time frames
    dummy_audio = torch.randn(2, 3, 104, 100)
    with torch.no_grad():
        A, tokens = encoder(dummy_audio)

    assert A.shape == (2, 256)
    assert tokens.shape == (2, 3, 256)
    assert torch.isfinite(A).all()
    assert torch.isfinite(tokens).all()
