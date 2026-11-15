"""Audio Transformer Encoder.

Architecture (README §5, step 5):
  - MFCC/log-mel patch embedding: linear (104, 100) → d=256 token per segment
  - Learned positional encoding (up to MAX_TOKENS positions)
  - 4-layer, 8-head Transformer (nn.TransformerEncoder)
  - CLS-mean pooled `A ∈ ℝ²⁵⁶` + per-segment token sequence

Input:  x  Tensor[B, K, 104, 100]  — K 1-second audio segments
Output: (A, tokens)
  A      Tensor[B, 256]        — clip-level audio embedding
  tokens Tensor[B, K, 256]     — per-segment tokens
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

AUDIO_FEAT_DIM: int = 104  # N_MFCC(40) + N_MELS(64)
AUDIO_TIME_DIM: int = 100  # 1-second segment time frames
D_MODEL: int = 256
NUM_LAYERS: int = 4
NUM_HEADS: int = 8
MAX_TOKENS: int = 64


class AudioEncoder(nn.Module):
    """Encodes a batch of MFCC/log-mel segment sequences into clip embedding A ∈ ℝ²⁵⁶."""

    def __init__(
        self,
        feat_dim: int = AUDIO_FEAT_DIM,
        time_dim: int = AUDIO_TIME_DIM,
        d_model: int = D_MODEL,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Flatten each segment [104, 100] → 10400 then project to d_model
        self.patch_embed = nn.Linear(feat_dim * time_dim, d_model)

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
        B, K, feat_dim, time_dim = x.shape

        # Flatten each segment: [B, K, 104*100]
        x_flat = x.reshape(B, K, feat_dim * time_dim)

        # Project to d_model
        tokens: Tensor = self.patch_embed(x_flat)  # [B, K, 256]

        # Add positional encoding
        positions = torch.arange(K, device=x.device).unsqueeze(0)  # [1, K]
        tokens = tokens + self.pos_embedding(positions)  # [B, K, 256]

        # Transformer encoder
        tokens = self.transformer(tokens)  # [B, K, 256]

        # Mean-pool for clip-level embedding
        A: Tensor = tokens.mean(dim=1)  # [B, 256]

        return A, tokens
