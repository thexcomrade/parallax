"""Data augmentations for multimodal deepfake detection training.

Spatial, temporal, audio, and cross-modal mismatch augmentations
derived from the AVT²-DWF specification (README §5, §8.7).
"""

from __future__ import annotations

import random

import torch
from torch import Tensor


class MultimodalAugmenter:
    """Applies stochastic spatial, audio, and cross-modal augmentations.

    Args:
        flip_prob: Probability of horizontal flip across all frames in a window.
        color_jitter_prob: Probability of brightness/contrast jitter on video.
        audio_noise_prob: Probability of additive Gaussian noise on audio features.
        audio_masking_prob: Probability of SpecAugment time/frequency band masking.
        mismatch_prob: Probability of introducing cross-modal mismatch (label=1).
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        color_jitter_prob: float = 0.3,
        audio_noise_prob: float = 0.3,
        audio_masking_prob: float = 0.3,
        mismatch_prob: float = 0.15,
    ) -> None:
        self.flip_prob = flip_prob
        self.color_jitter_prob = color_jitter_prob
        self.audio_noise_prob = audio_noise_prob
        self.audio_masking_prob = audio_masking_prob
        self.mismatch_prob = mismatch_prob

    def augment_video(self, video_windows: Tensor) -> Tensor:
        """Apply spatial transforms consistently across frames in video windows.

        Args:
            video_windows: Tensor[K, 16, 3, 224, 224]

        Returns:
            Augmented video windows tensor of identical shape.
        """
        out = video_windows.clone()

        # 1. Random horizontal flip (flip last dim: width)
        if random.random() < self.flip_prob:
            out = torch.flip(out, dims=[-1])

        # 2. Random contrast and brightness scaling
        if random.random() < self.color_jitter_prob:
            alpha = random.uniform(0.85, 1.15)  # contrast
            beta = random.uniform(-0.1, 0.1)  # brightness
            out = (out * alpha + beta).clamp(-3.0, 3.0)

        return out

    def augment_audio(self, audio_segments: Tensor) -> Tensor:
        """Apply acoustic augmentations to audio feature segments.

        Args:
            audio_segments: Tensor[K, 104, 100] (concatenated MFCC + Mel-spectrogram)

        Returns:
            Augmented audio segments tensor of identical shape.
        """
        out = audio_segments.clone()

        # 1. Random additive Gaussian noise
        if random.random() < self.audio_noise_prob:
            noise_std = random.uniform(0.01, 0.05)
            noise = torch.randn_like(out) * noise_std
            out = out + noise

        # 2. SpecAugment frequency & time masking
        if random.random() < self.audio_masking_prob:
            _k, num_freqs, num_times = out.shape
            # Zero out a random band of frequency channels (up to 8 channels)
            f_size = random.randint(2, min(8, num_freqs))
            f_start = random.randint(0, num_freqs - f_size)
            out[:, f_start : f_start + f_size, :] = 0.0

            # Zero out a random band of time frames (up to 10 frames)
            t_size = random.randint(2, min(10, num_times))
            t_start = random.randint(0, num_times - t_size)
            out[:, :, t_start : t_start + t_size] = 0.0

        return out

    def apply_cross_modal_mismatch(
        self,
        video_windows: Tensor,
        audio_segments: Tensor,
        label: int,
    ) -> tuple[Tensor, Tensor, int, bool]:
        """Optionally inject cross-modal desynchronization/mismatch.

        If mismatch is triggered, circular-rolls audio segments relative to video
        to create an out-of-sync fake sample (forces label = 1).

        Returns:
            (video_windows, audio_segments, new_label, is_mismatched)
        """
        if random.random() < self.mismatch_prob and audio_segments.shape[0] > 1:
            # Shift audio segments by rolling window indices
            rolled_audio = torch.roll(audio_segments, shifts=1, dims=0)
            return video_windows, rolled_audio, 1, True

        return video_windows, audio_segments, label, False
