import contextlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import torch

from app.core.errors import AppException

TARGET_SR = 16000
N_MFCC = 40
N_MELS = 64
WIN_LENGTH = 400  # 25 ms at 16 kHz
HOP_LENGTH = 160  # 10 ms at 16 kHz
N_FFT = 512
FRAMES_PER_SEC = 100  # 1.0s / 0.010s hop = 100 frames


class AudioPreprocessError(AppException):
    def __init__(self, message: str = "Audio preprocessing failed") -> None:
        super().__init__(code="AUDIO_PREPROCESS_ERROR", message=message, status_code=400)


@dataclass(frozen=True)
class AudioPreprocessResult:
    audio_present: bool
    sample_rate: int
    duration_s: float
    raw_audio: np.ndarray  # 1D float32 audio waveform
    mfcc: torch.Tensor  # (40, T_audio)
    log_mel: torch.Tensor  # (64, T_audio)
    segments: list[torch.Tensor]  # list of (104, 100) tensors
    segment_timestamps: list[float]  # start time in seconds for each 1s segment


def _extract_wav_with_ffmpeg(media_path: Path, output_wav: Path, timeout_s: float = 15.0) -> bool:
    """Extract audio from video to 16kHz mono 16-bit PCM WAV.

    Returns False if file contains no audio stream.
    """
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-i",
        str(media_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(TARGET_SR),
        "-ac",
        "1",
        "-f",
        "wav",
        str(output_wav),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise AudioPreprocessError(f"ffmpeg extraction timed out after {timeout_s}s") from e
    except FileNotFoundError as e:
        raise AudioPreprocessError("ffmpeg binary not found on host system") from e

    # If return code is non-zero or output file is empty, no audio is present
    return not (proc.returncode != 0 or not output_wav.exists() or output_wav.stat().st_size == 0)


def extract_audio_features(
    media_path: Path | str,
    timeout_s: float = 15.0,
) -> AudioPreprocessResult:
    """Preprocess audio: FFmpeg 16kHz mono -> denoise/norm -> 40 MFCC + 64 Log-Mel -> 1s segments.

    Gracefully handles clips with no audio by returning audio_present=False.
    """
    path = Path(media_path).resolve()
    if not path.is_file():
        raise AudioPreprocessError(f"Media file not found: {path.name}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_wav_path = Path(tmp_file.name)

    try:
        has_audio = _extract_wav_with_ffmpeg(path, tmp_wav_path, timeout_s=timeout_s)

        if not has_audio:
            # Clean fallback for clips without an audio stream
            return AudioPreprocessResult(
                audio_present=False,
                sample_rate=TARGET_SR,
                duration_s=0.0,
                raw_audio=np.zeros(0, dtype=np.float32),
                mfcc=torch.zeros((N_MFCC, 0), dtype=torch.float32),
                log_mel=torch.zeros((N_MELS, 0), dtype=torch.float32),
                segments=[],
                segment_timestamps=[],
            )

        # Load extracted WAV audio with soundfile
        audio_data, _sr = sf.read(str(tmp_wav_path), dtype="float32")
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        duration_s = float(len(audio_data)) / float(TARGET_SR)
        if len(audio_data) < HOP_LENGTH:
            # Audio is too short to compute features
            return AudioPreprocessResult(
                audio_present=False,
                sample_rate=TARGET_SR,
                duration_s=duration_s,
                raw_audio=audio_data,
                mfcc=torch.zeros((N_MFCC, 0), dtype=torch.float32),
                log_mel=torch.zeros((N_MELS, 0), dtype=torch.float32),
                segments=[],
                segment_timestamps=[],
            )

        # Light denoise via pre-emphasis filter and peak normalization
        # y[n] = x[n] - 0.97 * x[n-1]
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])

        # Peak normalization
        max_val = np.max(np.abs(emphasized_audio))
        norm_audio = (emphasized_audio / max_val) * 0.95 if max_val > 1e-6 else emphasized_audio

        # Compute 40 MFCCs (25 ms window, 10 ms hop)
        mfcc_np = librosa.feature.mfcc(
            y=norm_audio,
            sr=TARGET_SR,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            center=True,
        )

        # Compute 64-band Log-Mel Spectrogram
        mel_np = librosa.feature.melspectrogram(
            y=norm_audio,
            sr=TARGET_SR,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            center=True,
        )
        log_mel_np = librosa.power_to_db(mel_np, ref=np.max)

        mfcc_tensor = torch.from_numpy(mfcc_np).to(dtype=torch.float32)
        log_mel_tensor = torch.from_numpy(log_mel_np).to(dtype=torch.float32)

        # Segment into 1.0-second patches (100 frames per second)
        # Combined feature patch: [MFCC; Log-Mel] -> (104, 100)
        total_time_frames = min(mfcc_tensor.shape[1], log_mel_tensor.shape[1])
        combined_features = torch.cat(
            [mfcc_tensor[:, :total_time_frames], log_mel_tensor[:, :total_time_frames]], dim=0
        )

        segments: list[torch.Tensor] = []
        segment_timestamps: list[float] = []

        # Step by 1-second windows
        for start_f in range(0, total_time_frames, FRAMES_PER_SEC):
            end_f = start_f + FRAMES_PER_SEC
            patch = combined_features[:, start_f:end_f]
            # Pad with zeros if shorter than 100 frames at the tail
            if patch.shape[1] < FRAMES_PER_SEC:
                pad_len = FRAMES_PER_SEC - patch.shape[1]
                patch = torch.nn.functional.pad(patch, (0, pad_len), mode="constant", value=0.0)
            segments.append(patch)
            segment_timestamps.append(float(start_f * HOP_LENGTH) / float(TARGET_SR))

        return AudioPreprocessResult(
            audio_present=True,
            sample_rate=TARGET_SR,
            duration_s=duration_s,
            raw_audio=audio_data,
            mfcc=mfcc_tensor,
            log_mel=log_mel_tensor,
            segments=segments,
            segment_timestamps=segment_timestamps,
        )
    finally:
        if tmp_wav_path.exists():
            with contextlib.suppress(OSError):
                os.remove(tmp_wav_path)
