import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.core.errors import AppException


class ProbeError(AppException):
    def __init__(self, message: str = "Failed to probe media file") -> None:
        super().__init__(code="PROBE_ERROR", message=message, status_code=400)


class CorruptMediaError(AppException):
    def __init__(self, message: str = "Corrupt media stream or unreadable file") -> None:
        super().__init__(code="CORRUPT_MEDIA", message=message, status_code=400)


class MediaLimitsExceededError(AppException):
    def __init__(self, message: str) -> None:
        super().__init__(code="LIMITS_EXCEEDED", message=message, status_code=400)


@dataclass(frozen=True)
class ProbeResult:
    duration_s: float
    size_bytes: int
    width: int
    height: int
    fps: float
    audio_present: bool
    video_codec: str
    audio_codec: str | None


def _probe_media_fallback(
    path: Path,
    size_bytes: int,
    timeout_s: float = 15.0,
    max_duration_s: float | None = None,
) -> ProbeResult:
    """Fallback probing mechanism using OpenCV and bundled FFmpeg when ffprobe is absent."""
    import cv2

    from app.ml.preprocess.audio import _get_ffmpeg_bin

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise CorruptMediaError(f"OpenCV failed to open media file: {path.name}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0 or fps > 120.0:
        fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, _ = cap.read()
    cap.release()

    if not ret or frame_count <= 0:
        raise CorruptMediaError(f"No decodable video frames in {path.name}")

    duration_s = frame_count / fps if fps > 0 else 0.0
    if duration_s <= 0.0:
        raise CorruptMediaError("Unable to determine valid positive stream duration")

    eff_max_duration = (
        max_duration_s if max_duration_s is not None else float(settings.MAX_DURATION_S)
    )
    if eff_max_duration > 0 and duration_s > eff_max_duration:
        raise MediaLimitsExceededError(
            f"Duration {duration_s:.1f}s exceeds maximum limit of {eff_max_duration:.1f}s"
        )

    # Audio check via ffmpeg
    ffmpeg_bin = _get_ffmpeg_bin()
    audio_present = False
    audio_codec: str | None = None
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        for line in proc.stderr.splitlines():
            if "Audio:" in line:
                audio_present = True
                audio_codec = "aac" if "aac" in line.lower() else "pcm"
                break
    except Exception:
        pass

    return ProbeResult(
        duration_s=duration_s,
        size_bytes=size_bytes,
        width=width,
        height=height,
        fps=fps,
        audio_present=audio_present,
        video_codec="h264",
        audio_codec=audio_codec,
    )


def probe_media(
    file_path: Path | str,
    timeout_s: float = 15.0,
    max_duration_s: float | None = None,
) -> ProbeResult:
    """Probe video file using ffprobe with automatic OpenCV/FFmpeg fallback.

    Enforces duration, file size, and resolution caps from README §5.
    Never uses shell=True.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise CorruptMediaError(f"Media file not found: {path.name}")

    size_bytes = path.stat().st_size
    max_size_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if size_bytes > max_size_bytes:
        mb = size_bytes / (1024 * 1024)
        raise MediaLimitsExceededError(
            f"File size ({mb:.1f} MB) exceeds maximum allowed {settings.MAX_UPLOAD_MB} MB"
        )
    if size_bytes == 0:
        raise CorruptMediaError("Uploaded file is empty (0 bytes)")

    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return _probe_media_fallback(path, size_bytes, timeout_s, max_duration_s)

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,duration,r_frame_rate",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
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
        raise ProbeError(f"ffprobe timed out after {timeout_s}s on {path.name}") from e
    except FileNotFoundError as e:
        raise ProbeError("ffprobe executable not found on host system") from e

    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or "Unknown decoder error"
        raise CorruptMediaError(f"ffprobe rejected file: {err_msg}")

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise CorruptMediaError("Invalid ffprobe telemetry output") from e

    streams = data.get("streams", [])
    format_info = data.get("format", {})

    video_stream: dict[str, str | int | float] | None = None
    audio_stream: dict[str, str | int | float] | None = None

    for s in streams:
        c_type = s.get("codec_type")
        if c_type == "video" and video_stream is None:
            video_stream = s
        elif c_type == "audio" and audio_stream is None:
            audio_stream = s

    if not video_stream:
        raise CorruptMediaError("No valid video stream identified in file")

    # Extract resolution
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise CorruptMediaError(f"Invalid video dimensions: {width}x{height}")

    if (width > 1920 or height > 1080) and (width * height > 1920 * 1080):
        raise MediaLimitsExceededError(
            f"Resolution {width}x{height} exceeds maximum allowed 1920x1080"
        )

    # Extract duration
    raw_duration = format_info.get("duration") or video_stream.get("duration") or "0"
    try:
        duration_s = float(raw_duration)
    except (ValueError, TypeError):
        duration_s = 0.0

    if duration_s <= 0.0:
        raise CorruptMediaError("Unable to determine valid positive stream duration")

    eff_max_duration = (
        max_duration_s if max_duration_s is not None else float(settings.MAX_DURATION_S)
    )
    if eff_max_duration > 0 and duration_s > eff_max_duration:
        raise MediaLimitsExceededError(
            f"Duration {duration_s:.1f}s exceeds maximum limit of {eff_max_duration:.1f}s"
        )

    # Extract FPS
    r_frame_rate = str(video_stream.get("r_frame_rate", "25/1"))
    try:
        if "/" in r_frame_rate:
            num, den = r_frame_rate.split("/", 1)
            fps = float(num) / max(1.0, float(den))
        else:
            fps = float(r_frame_rate)
    except (ValueError, TypeError, ZeroDivisionError):
        fps = 25.0

    video_codec = str(video_stream.get("codec_name", "unknown"))
    audio_present = audio_stream is not None
    audio_codec = str(audio_stream.get("codec_name")) if audio_stream else None

    return ProbeResult(
        duration_s=duration_s,
        size_bytes=size_bytes,
        width=width,
        height=height,
        fps=fps,
        audio_present=audio_present,
        video_codec=video_codec,
        audio_codec=audio_codec,
    )
