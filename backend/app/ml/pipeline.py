"""ML pipeline orchestrator for AVT²-DWF.

Yields progress events across stages (README §5, §8.6):
  probe -> frames -> faces -> audio -> align -> face_enc -> audio_enc -> fusion -> explain -> done
"""

from __future__ import annotations

import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from app.ml.explain import ExplainResult, generate_explanations
from app.ml.inference import InferenceResult, get_model, run_inference
from app.ml.preprocess.align import AlignedMultimodalData, align_audio_video
from app.ml.preprocess.audio import AudioPreprocessResult, extract_audio_features
from app.ml.preprocess.probe import ProbeResult, probe_media
from app.ml.preprocess.video import VideoPreprocessResult, extract_video_frames


@dataclass(frozen=True)
class PipelineEvent:
    stage: str
    progress: float
    message: str
    elapsed_s: float
    result: PipelineResult | None = None


@dataclass(frozen=True)
class PipelineResult:
    inference: InferenceResult
    probe: ProbeResult
    video: VideoPreprocessResult
    audio: AudioPreprocessResult
    aligned: AlignedMultimodalData
    explain: ExplainResult
    reasons: list[str]


def run_pipeline(
    video_path: Path,
    artifact_dir: Path,
) -> Generator[PipelineEvent, None, None]:
    """Execute the full AVT²-DWF forensic pipeline on a media file.

    Yields a PipelineEvent at each pipeline stage.
    """
    t0 = time.time()

    def elapsed() -> float:
        return round(time.time() - t0, 2)

    # 1. Probe
    yield PipelineEvent(
        stage="probe", progress=0.05, message="Probing media container", elapsed_s=elapsed()
    )
    probe_info = probe_media(video_path)

    # 2. Frames
    yield PipelineEvent(
        stage="frames",
        progress=0.15,
        message="Decoding and sampling video frames",
        elapsed_s=elapsed(),
    )
    video_res = extract_video_frames(video_path, target_count=32, window_size=16, stride=8)

    # 3. Faces
    yield PipelineEvent(
        stage="faces",
        progress=0.30,
        message="Tracking face landmarks and crops",
        elapsed_s=elapsed(),
    )
    reasons: list[str] = []
    if video_res.face_coverage < 0.40:
        reasons.append(
            f"Low face coverage ({video_res.face_coverage:.1%}). Detection confidence is reduced."
        )

    # 4. Audio
    yield PipelineEvent(
        stage="audio",
        progress=0.45,
        message="Extracting MFCC and log-mel spectrogram",
        elapsed_s=elapsed(),
    )
    audio_res = extract_audio_features(video_path)
    if not audio_res.audio_present:
        reasons.append(
            "Audio stream missing or silent. Analysis performed using visual modality only."
        )

    # 5. Align
    yield PipelineEvent(
        stage="align",
        progress=0.55,
        message="Aligning cross-modal temporal windows",
        elapsed_s=elapsed(),
    )
    aligned = align_audio_video(video_res, audio_res)

    # 6. Face encoder
    yield PipelineEvent(
        stage="face_enc",
        progress=0.65,
        message="Computing facial ViT embeddings",
        elapsed_s=elapsed(),
    )

    # 7. Audio encoder
    yield PipelineEvent(
        stage="audio_enc",
        progress=0.75,
        message="Computing audio Transformer embeddings",
        elapsed_s=elapsed(),
    )

    # 8. Dynamic Weight Fusion
    yield PipelineEvent(
        stage="fusion",
        progress=0.85,
        message="Dynamic Weight Fusion and classification",
        elapsed_s=elapsed(),
    )
    video_windows = aligned.video_windows.unsqueeze(0)  # [1, K, 16, 3, 224, 224]
    audio_segments = aligned.audio_segments.unsqueeze(0)  # [1, K, 104, 100]
    audio_mask = aligned.audio_mask  # [K]

    inference_res = run_inference(
        video_windows=video_windows,
        audio_segments=audio_segments,
        audio_mask=audio_mask,
        model_version="0.0.0",
    )

    if inference_res.model_mode == "untrained":
        reasons.append("Model running in untrained mode. Forensic scores are not calibrated.")

    # 9. Explainability
    yield PipelineEvent(
        stage="explain",
        progress=0.92,
        message="Generating attention heatmaps & spectrogram overlay",
        elapsed_s=elapsed(),
    )
    model = get_model()

    per_window_video = inference_res.per_window_scores
    per_window_audio = [s * (1.0 if aligned.audio_present else 0.0) for s in per_window_video]
    per_window_fused = [
        inference_res.w_video * v + inference_res.w_audio * a
        for v, a in zip(per_window_video, per_window_audio, strict=False)
    ]

    explain_res = generate_explanations(
        model_vit=model.face_encoder.vit,
        raw_crops=video_res.raw_crops,
        frame_timestamps=video_res.timestamps,
        log_mel=audio_res.log_mel,
        per_window_scores=per_window_video,
        window_timestamps=aligned.window_timestamps,
        audio_per_window_scores=per_window_audio,
        fused_per_window_scores=per_window_fused,
        artifact_dir=artifact_dir,
    )

    # 10. Done
    full_result = PipelineResult(
        inference=inference_res,
        probe=probe_info,
        video=video_res,
        audio=audio_res,
        aligned=aligned,
        explain=explain_res,
        reasons=reasons,
    )
    yield PipelineEvent(
        stage="done",
        progress=1.0,
        message="Analysis complete",
        elapsed_s=elapsed(),
        result=full_result,
    )
