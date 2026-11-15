"""Face Transformer Encoder.

Architecture (README §5, step 4):
  - Per-frame ViT-B/16 embedding (pretrained ImageNet, embed_dim=768)
  - n-frame tokenisation: group n=4 consecutive per-frame embeddings → linear → d=256
  - Learned positional encoding (up to MAX_TOKENS positions)
  - 4-layer, 8-head temporal Transformer (nn.TransformerEncoder)
  - CLS-mean pooled `V ∈ ℝ²⁵⁶` + per-window token sequence

Input:  x  Tensor[B, T, 3, 224, 224]  — T pre-processed frames per clip
Output: (V, tokens)
  V      Tensor[B, 256]        — clip-level video embedding
  tokens Tensor[B, K, 256]     — K = ceil(T / n) per-window tokens
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torch import Tensor

# n-frame grouping from README §5
N_FRAMES_PER_TOKEN: int = 4
D_MODEL: int = 256
VIT_EMBED_DIM: int = 768  # ViT-B/16 feature dim
NUM_LAYERS: int = 4
NUM_HEADS: int = 8
MAX_TOKENS: int = 64  # Maximum temporal positions (supports up to 64 windows)


class FaceEncoder(nn.Module):
    """Encodes a batch of face-frame sequences into a clip embedding V in R^256."""

    def __init__(
        self,
        n: int = N_FRAMES_PER_TOKEN,
        d_model: int = D_MODEL,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.n = n
        self.d_model = d_model

        # ViT-B/16 backbone — features_only=True avoids the classifier head
        self.vit: nn.Module = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,  # returns [B, 768] embedding
        )
        # Freeze ViT; only the temporal Transformer and projection are trained
        for param in self.vit.parameters():
            param.requires_grad = False

        # n-frame token projection: n*768 -> d_model
        self.token_proj = nn.Linear(n * VIT_EMBED_DIM, d_model)

        # Learned positional encoding
        self.pos_embedding = nn.Embedding(MAX_TOKENS, d_model)

        # Temporal Transformer
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
        nn.init.xavier_uniform_(self.token_proj.weight)
        nn.init.zeros_(self.token_proj.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Tensor[B, T, 3, 224, 224]

        Returns:
            V:      Tensor[B, 256]    — clip-level embedding
            tokens: Tensor[B, K, 256] — per-window tokens (K = ceil(T / n))
        """
        B, T, C, H, W = x.shape

        # Flatten batch+time for ViT; shape → [B*T, 768]
        x_flat = x.reshape(B * T, C, H, W)
        with torch.no_grad():
            frame_embeds: Tensor = self.vit(x_flat)  # [B*T, 768]
        frame_embeds = frame_embeds.reshape(B, T, VIT_EMBED_DIM)

        # n-frame tokenisation: pad T to nearest multiple of n
        pad_len = (self.n - T % self.n) % self.n
        if pad_len > 0:
            pad = torch.zeros(B, pad_len, VIT_EMBED_DIM, device=x.device, dtype=x.dtype)
            frame_embeds = torch.cat([frame_embeds, pad], dim=1)

        T_padded = frame_embeds.shape[1]
        K = T_padded // self.n  # number of tokens

        # Reshape → [B, K, n*768] → project → [B, K, d_model]
        grouped = frame_embeds.reshape(B, K, self.n * VIT_EMBED_DIM)
        tokens: Tensor = self.token_proj(grouped)  # [B, K, 256]

        # Add positional encoding
        positions = torch.arange(K, device=x.device).unsqueeze(0)  # [1, K]
        tokens = tokens + self.pos_embedding(positions)  # [B, K, 256]

        # Temporal Transformer
        tokens = self.transformer(tokens)  # [B, K, 256]

        # Mean-pool over windows for clip-level embedding
        V: Tensor = tokens.mean(dim=1)  # [B, 256]

        return V, tokens
