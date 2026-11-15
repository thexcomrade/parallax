"""Dynamic Weight Fusion (DWF) module.

Architecture (README §5, step 6):
  g = softmax(MLP([A; V; |A-V|; cos(A,V)]))  ->  (w_a, w_v)
  F = w_a*A + w_v*V
  sync_score = mean cosine similarity between aligned audio/face window tokens

Modality masking: when audio_mask[k] == 0 for all k, force w_a = 0 and renormalise.
The gate MLP input uses a concat of 4 components:
  [A(256); V(256); |A-V|(256); cos_sim(1)] - total 769 dims
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

D_MODEL: int = 256


class DynamicWeightFusion(nn.Module):
    """Fuses audio embedding A and video embedding V with learned dynamic weights."""

    def __init__(self, d_model: int = D_MODEL) -> None:
        super().__init__()
        # Gate input: [A; V; |A-V|; cos_sim(1)] = 3*d_model + 1
        gate_in_dim = 3 * d_model + 1

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.gate_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        A: Tensor,
        V: Tensor,
        audio_mask: Tensor,
        audio_tokens: Tensor | None = None,
        video_tokens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            A:            Tensor[B, 256]    — clip-level audio embedding
            V:            Tensor[B, 256]    — clip-level video embedding
            audio_mask:   Tensor[K]         — 1.0 if audio present, 0.0 if absent
            audio_tokens: Tensor[B, K, 256] — per-segment audio tokens (optional, for sync)
            video_tokens: Tensor[B, K, 256] — per-window video tokens (optional, for sync)

        Returns:
            fused:      Tensor[B, 256]  — fused representation F
            w_a:        Tensor[B, 1]    — audio weight (0 if audio absent)
            w_v:        Tensor[B, 1]    — video weight
            sync_score: Tensor[1]       — mean cosine similarity across aligned tokens
        """
        # Cosine similarity between A and V: [B, 1]
        cos_sim = F.cosine_similarity(A, V, dim=-1, eps=1e-8).unsqueeze(-1)  # [B, 1]

        # Gate MLP input: [A; V; |A-V|; cos_sim]
        gate_input = torch.cat([A, V, (A - V).abs(), cos_sim], dim=-1)  # [B, 3*256+1]
        gate_logits: Tensor = self.gate_mlp(gate_input)  # [B, 2]
        weights = F.softmax(gate_logits, dim=-1)  # [B, 2]  — (w_a, w_v)

        w_a = weights[:, :1]  # [B, 1]
        w_v = weights[:, 1:]  # [B, 1]

        # Modality masking: if audio_mask values are 0, set w_a = 0, w_v = 1
        if audio_mask.ndim > 1:
            audio_present = audio_mask.any(dim=-1, keepdim=True).float()  # [B, 1]
        else:
            audio_present = audio_mask.any().float()  # scalar
        w_a = w_a * audio_present
        # Renormalise so w_a + w_v = 1
        total = w_a + w_v + 1e-8
        w_a = w_a / total
        w_v = w_v / total

        # Fused representation
        fused: Tensor = w_a * A + w_v * V  # [B, 256]

        # Sync score: mean cosine between aligned audio/video tokens
        has_audio = bool((audio_present > 0.5).any())
        if audio_tokens is not None and video_tokens is not None and has_audio:
            # Both [B, K, 256] — compute per-window cosine and average
            sync = F.cosine_similarity(audio_tokens, video_tokens, dim=-1)  # [B, K]
            sync_score: Tensor = sync.mean()
        else:
            sync_score = torch.tensor(0.0, device=A.device)

        return fused, w_a, w_v, sync_score
