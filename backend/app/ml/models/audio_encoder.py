"""Audio Transformer Encoder.

Architecture (README §5, step 5):
  - Audio input channel normalization: standardization on 104 features (40 MFCC + 64 Log-Mel)
  - MFCC/log-mel patch embedding: linear (104, 100) → d=256 token per segment
  - LayerNorm(256) on projected patch embeddings
  - Learned positional encoding (up to MAX_TOKENS positions)
  - 4-layer, 8-head Transformer (nn.TransformerEncoder)
  - CLS-mean pooled `A ∈ ℝ²⁵⁶` + per-segment token sequence

Input:  x  Tensor[B, K, 104, 100]  — K 1-second audio segments
Output: (A, tokens)
  A      Tensor[B, 256]        — clip-level audio embedding
  tokens Tensor[B, K, 256]     — per-segment tokens
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

AUDIO_FEAT_DIM: int = 104  # N_MFCC(40) + N_MELS(64)
AUDIO_TIME_DIM: int = 100  # 1-second segment time frames
D_MODEL: int = 256
NUM_LAYERS: int = 4
NUM_HEADS: int = 8
MAX_TOKENS: int = 64


class AudioInputNorm(nn.Module):
    """Channel-wise standardization for 104 input audio feature channels.

    Standardizes MFCC (0..39) and Log-Mel (40..103) channels using training-set statistics:
    x_norm = clamp((x - mean) / (std + eps), min=-10.0, max=10.0)
    """

    channel_mean: Tensor
    channel_std: Tensor
    is_active: Tensor

    def __init__(
        self, feat_dim: int = AUDIO_FEAT_DIM, eps: float = 1e-5, active: bool = True
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.eps = eps
        self.register_buffer("channel_mean", torch.zeros(1, 1, feat_dim, 1))
        self.register_buffer("channel_std", torch.ones(1, 1, feat_dim, 1))
        self.register_buffer("is_active", torch.tensor(active))

    def set_stats(self, mean: Tensor, std: Tensor) -> None:
        self.channel_mean.copy_(mean.view(1, 1, self.feat_dim, 1))
        self.channel_std.copy_(std.view(1, 1, self.feat_dim, 1))
        self.is_active.copy_(torch.tensor(True))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, K, 104, 100]
        if bool(self.is_active):
            mean = self.channel_mean.to(device=x.device, dtype=x.dtype)
            std = self.channel_std.to(device=x.device, dtype=x.dtype)
            x_norm = (x - mean) / (std + self.eps)
            return torch.clamp(x_norm, min=-10.0, max=10.0)
        return x


class AudioEncoder(nn.Module):
    """Encodes a batch of MFCC/log-mel segment sequences into clip embedding A ∈ ℝ²⁵⁶."""

    def __init__(
        self,
        feat_dim: int = AUDIO_FEAT_DIM,
        time_dim: int = AUDIO_TIME_DIM,
        d_model: int = D_MODEL,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        use_layernorm: bool = False,
        use_input_norm: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_layernorm = use_layernorm
        self.use_input_norm = use_input_norm

        # Input feature channel standardization
        self.input_norm = AudioInputNorm(feat_dim=feat_dim, active=use_input_norm)
        stats_file = Path(__file__).resolve().parent.parent / "weights" / "audio_norm_stats.pt"
        if stats_file.is_file():
            try:
                stats = torch.load(stats_file, weights_only=True)
                self.input_norm.set_stats(stats["mean"], stats["std"])
            except Exception:
                pass

        # Flatten each segment [104, 100] → 10400 then project to d_model
        self.patch_embed = nn.Linear(feat_dim * time_dim, d_model)
        self.proj_norm: nn.Module = (
            nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        )

        # Learned positional encoding
        self.pos_embedding = nn.Embedding(MAX_TOKENS, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Tensor[B, K, 104, 100]

        Returns:
            A:      Tensor[B, 256]    — clip-level audio embedding
            tokens: Tensor[B, K, 256] — per-segment tokens
        """
        # Channel-wise audio standardization
        x = self.input_norm(x)

        B, K, feat_dim, time_dim = x.shape

        # Flatten each segment: [B, K, 104*100]
        x_flat = x.reshape(B, K, feat_dim * time_dim)

        # Project to d_model and apply LayerNorm
        tokens: Tensor = self.patch_embed(x_flat)  # [B, K, 256]
        tokens = self.proj_norm(tokens)

        # Add positional encoding
        positions = torch.arange(K, device=x.device).unsqueeze(0)  # [1, K]
        tokens = tokens + self.pos_embedding(positions)  # [B, K, 256]

        # Transformer encoder
        tokens = self.transformer(tokens)  # [B, K, 256]

        # Mean-pool for clip-level embedding
        A: Tensor = tokens.mean(dim=1)  # [B, 256]

        return A, tokens
