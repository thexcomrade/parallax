from dataclasses import dataclass

import numpy as np
import torch

from app.ml.preprocess.audio import FRAMES_PER_SEC, N_MELS, N_MFCC, AudioPreprocessResult
from app.ml.preprocess.video import VideoPreprocessResult

FEATURE_DIM = N_MFCC + N_MELS  # 40 + 64 = 104


@dataclass(frozen=True)
class AlignedMultimodalData:
    video_tensor: torch.Tensor  # (32, 3, 224, 224)
    video_windows: torch.Tensor  # (K, 16, 3, 224, 224)
    audio_segments: torch.Tensor  # (K, 104, 100)
    audio_mask: torch.Tensor  # (K,) float: 1.0 if audio valid, 0.0 if missing/masked
    audio_present: bool
    face_coverage: float
    window_timestamps: list[float]  # Midpoint timestamps for each of the K windows


def align_audio_video(
    video_res: VideoPreprocessResult,
    audio_res: AudioPreprocessResult,
    window_size: int = 16,
    stride: int = 8,
) -> AlignedMultimodalData:
    """Align 1-second audio patches to 16-frame sliding video windows by timestamp.

    If audio is missing or corrupted, supplies zero-padded audio features
    and an audio_mask with 0.0s to ensure Dynamic Weight Fusion sets w_a = 0.
    """
    total_frames = video_res.frames_tensor.shape[0]
    num_windows = len(video_res.windows)

    if num_windows == 0:
        # Fallback if video shorter than 16 frames
        video_windows = video_res.frames_tensor.unsqueeze(0)
        window_timestamps = [video_res.timestamps[0] if video_res.timestamps else 0.0]
    else:
        video_windows = torch.stack(video_res.windows, dim=0)  # Shape: (K, 16, 3, 224, 224)
        window_timestamps = []
        for i in range(num_windows):
            start_f = i * stride
            end_f = min(start_f + window_size - 1, total_frames - 1)
            mid_t = (video_res.timestamps[start_f] + video_res.timestamps[end_f]) / 2.0
            window_timestamps.append(mid_t)

    k = video_windows.shape[0]

    if audio_res.audio_present and len(audio_res.segments) > 0:
        aligned_audio_list: list[torch.Tensor] = []
        seg_times = np.array(audio_res.segment_timestamps, dtype=np.float32)

        for win_t in window_timestamps:
            # Find nearest audio segment by timestamp
            closest_idx = int(np.argmin(np.abs(seg_times - win_t)))
            aligned_audio_list.append(audio_res.segments[closest_idx])

        audio_segments = torch.stack(aligned_audio_list, dim=0)  # Shape: (K, 104, 100)
        audio_mask = torch.ones(k, dtype=torch.float32)
        audio_present = True
    else:
        # Audio missing: provide zeros and mask with 0.0
        audio_segments = torch.zeros((k, FEATURE_DIM, FRAMES_PER_SEC), dtype=torch.float32)
        audio_mask = torch.zeros(k, dtype=torch.float32)
        audio_present = False

    return AlignedMultimodalData(
        video_tensor=video_res.frames_tensor,
        video_windows=video_windows,
        audio_segments=audio_segments,
        audio_mask=audio_mask,
        audio_present=audio_present,
        face_coverage=video_res.face_coverage,
        window_timestamps=window_timestamps,
    )
