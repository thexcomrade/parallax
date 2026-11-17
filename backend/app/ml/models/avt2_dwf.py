"""AVT²-DWF: Audio-Visual Transformer² with Dynamic Weight Fusion.

Assembles FaceEncoder + AudioEncoder + DWF + classifier head.

Input:
  video_windows:  Tensor[B, K, 16, 3, 224, 224] — sliding windows of face frames
  audio_segments: Tensor[B, K, 104, 100]         — aligned 1-second audio segments
  audio_mask:     Tensor[K]                       — 1.0 / 0.0 per window

Output: AVT2Result with fields:
  logits:            Tensor[B, 2]  — [p_authentic, p_fake] pre-softmax
  w_a:               Tensor[B, 1]  — audio fusion weight
  w_v:               Tensor[B, 1]  — video fusion weight
  per_window_scores: Tensor[B, K]  — per-window fake probability (from window tokens)
  sync_score:        Tensor[1]     — lip-sync proxy (cosine of audio/video tokens)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from app.ml.models.audio_encoder import AudioEncoder
from app.ml.models.dwf import DynamicWeightFusion
from app.ml.models.face_encoder import FaceEncoder

D_MODEL: int = 256


@dataclass
class AVT2Result:
    logits: Tensor  # [B, 2]
    w_a: Tensor  # [B, 1]
    w_v: Tensor  # [B, 1]
    per_window_scores: Tensor  # [B, K]
    sync_score: Tensor  # scalar
    audio_logits: Tensor | None = None
    video_logits: Tensor | None = None


class AVT2DWF(nn.Module):
    """Full AVT²-DWF model: face encoder + audio encoder + DWF + classifier."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        pretrained_vit: bool = True,
        audio_layernorm: bool = False,
        auxiliary_heads: bool = False,
        classifier_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.auxiliary_heads = auxiliary_heads
        self.face_encoder = FaceEncoder(d_model=d_model, pretrained=pretrained_vit)
        self.audio_encoder = AudioEncoder(d_model=d_model, use_layernorm=audio_layernorm)
        self.dwf = DynamicWeightFusion(d_model=d_model)

        # Classifier head: fused embedding → 2-class logits
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(d_model // 2, 2),
        )

        # Auxiliary unimodal heads
        if self.auxiliary_heads:
            self.audio_classifier: nn.Module | None = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(p=classifier_dropout),
                nn.Linear(d_model // 2, 2),
            )
            self.video_classifier: nn.Module | None = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(p=classifier_dropout),
                nn.Linear(d_model // 2, 2),
            )
        else:
            self.audio_classifier = None
            self.video_classifier = None

        # Per-window head: individual window/segment tokens → score
        self.window_head = nn.Linear(d_model, 1)

        # Temperature scaling parameter (learned post-training via calibration)
        self.log_temperature = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        if self.audio_classifier is not None:
            for layer in self.audio_classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        if self.video_classifier is not None:
            for layer in self.video_classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.window_head.weight)
        nn.init.zeros_(self.window_head.bias)

    def forward(
        self,
        video_windows: Tensor,
        audio_segments: Tensor,
        audio_mask: Tensor,
    ) -> AVT2Result:
        """
        Args:
            video_windows:  Tensor[B, K, 16, 3, 224, 224]
            audio_segments: Tensor[B, K, 104, 100]
            audio_mask:     Tensor[K]  — per-window audio validity

        Returns:
            AVT2Result
        """
        B, K, _T, _C, _H, _W = video_windows.shape

        # --- Face encoder: process all windows flattened as a batch of frame sequences ---
        # Reshape [B, K, 16, 3, 224, 224] → [B*K, 16, 3, 224, 224]
        video_flat = video_windows.reshape(B * K, _T, _C, _H, _W)
        V_flat, v_tokens_flat = self.face_encoder(video_flat)
        # V_flat: [B*K, 256]; v_tokens_flat: [B*K, n_v_tokens, 256]

        # Reshape back: take the clip-level embedding per window
        V_windows = V_flat.reshape(B, K, D_MODEL)  # [B, K, 256] per-window embeddings
        V_clip = V_windows.mean(dim=1)  # [B, 256] clip-level

        # --- Audio encoder ---
        A_clip, a_tokens = self.audio_encoder(audio_segments)  # [B, 256], [B, K, 256]

        # Reshape v_tokens for sync: pick the mean-pooled token per window
        # v_tokens_flat has shape [B*K, n_v_tokens, 256]; mean pool over n_v_tokens
        V_tokens = v_tokens_flat.mean(dim=1).reshape(B, K, D_MODEL)  # [B, K, 256]

        # --- Dynamic Weight Fusion ---
        fused, w_a, w_v, sync_score = self.dwf(
            A_clip, V_clip, audio_mask, a_tokens, V_tokens
        )  # fused: [B, 256]

        # --- Classifier ---
        logits_raw: Tensor = self.classifier(fused)  # [B, 2]

        # Temperature scaling (calibration)
        temperature = self.log_temperature.exp().clamp(min=0.5, max=5.0)
        logits = logits_raw / temperature  # [B, 2]

        # Auxiliary unimodal logits
        audio_logits: Tensor | None = None
        video_logits: Tensor | None = None
        if self.auxiliary_heads and self.audio_classifier is not None and self.video_classifier is not None:
            audio_logits = self.audio_classifier(A_clip) / temperature
            video_logits = self.video_classifier(V_clip) / temperature

        # --- Per-window fake probability ---
        # Average video and audio window tokens, score with window head
        combined_tokens = (V_windows + a_tokens) / 2.0  # [B, K, 256]
        win_logits: Tensor = self.window_head(combined_tokens).squeeze(-1)  # [B, K]
        per_window_scores: Tensor = torch.sigmoid(win_logits)  # [B, K]

        return AVT2Result(
            logits=logits,
            w_a=w_a,
            w_v=w_v,
            per_window_scores=per_window_scores,
            sync_score=sync_score,
            audio_logits=audio_logits,
            video_logits=video_logits,
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
