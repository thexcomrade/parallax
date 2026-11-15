from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from app.core.errors import AppException

# Standard ImageNet normalization parameters
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class VideoDecodeError(AppException):
    def __init__(self, message: str = "Failed to decode video frames") -> None:
        super().__init__(code="VIDEO_DECODE_ERROR", message=message, status_code=400)


@dataclass(frozen=True)
class VideoPreprocessResult:
    frames_tensor: torch.Tensor  # (32, 3, 224, 224)
    face_coverage: float  # e.g. 0.95
    raw_crops: list[np.ndarray]  # 32 uint8 RGB images (224, 224, 3)
    timestamps: list[float]  # 32 timestamps in seconds
    windows: list[torch.Tensor]  # list of (16, 3, 224, 224) tensors


def _compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two (x, y, w, h) bounding boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    yb = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area) / float(union_area)


def _get_cascade_classifier() -> cv2.CascadeClassifier:
    """Load OpenCV Haar cascade for robust local frontal face detection."""
    weights_path = (
        Path(__file__).resolve().parent.parent / "weights" / "haarcascade_frontalface_default.xml"
    )
    if weights_path.is_file():
        classifier = cv2.CascadeClassifier(str(weights_path))
        if not classifier.empty():
            return classifier

    return cv2.CascadeClassifier()


def _crop_face_with_margin(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    margin_ratio: float = 1.3,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Crop face box with 1.3x margin and resize to target dimension."""
    h_img, w_img = frame.shape[:2]
    x, y, w, h = box

    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(w, h) * margin_ratio

    x1 = max(0, int(cx - side / 2.0))
    y1 = max(0, int(cy - side / 2.0))
    x2 = min(w_img, int(cx + side / 2.0))
    y2 = min(h_img, int(cy + side / 2.0))

    if x2 <= x1 or y2 <= y1:
        # Fallback to center square crop if bounding math degenerates
        side_min = min(h_img, w_img)
        x1 = (w_img - side_min) // 2
        y1 = (h_img - side_min) // 2
        x2 = x1 + side_min
        y2 = y1 + side_min

    cropped = frame[y1:y2, x1:x2]
    resized: np.ndarray = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
    return resized


def extract_video_frames(
    video_path: Path | str,
    target_count: int = 32,
    window_size: int = 16,
    stride: int = 8,
) -> VideoPreprocessResult:
    """Decode video, uniformly sample T=32 frames, detect/track face, crop 1.3x and normalise.

    Returns:
        VideoPreprocessResult with (32, 3, 224, 224) tensor, face coverage, and sliding windows.
    """
    path_str = str(Path(video_path).resolve())
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise VideoDecodeError(f"Could not open video file: {Path(video_path).name}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0 or np.isnan(fps):
        fps = 25.0

    if total_frames <= 0:
        # If frame count header missing, read sequentially to count
        frames_list: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_list.append(frame)
        cap.release()
        total_frames = len(frames_list)
        if total_frames == 0:
            raise VideoDecodeError("Video stream contains 0 readable frames")
    else:
        frames_list = []

    # Uniformly pick target_count indices
    if total_frames >= target_count:
        sampled_indices = np.linspace(0, total_frames - 1, target_count, dtype=int).tolist()
    else:
        # If shorter than 32 frames, duplicate last frame
        sampled_indices = list(range(total_frames))
        while len(sampled_indices) < target_count:
            sampled_indices.append(total_frames - 1)

    raw_sampled_frames: list[np.ndarray] = []
    timestamps: list[float] = [idx / fps for idx in sampled_indices]

    if frames_list:
        # Already decoded into memory
        for idx in sampled_indices:
            raw_sampled_frames.append(frames_list[idx])
    else:
        # Seek and read with VideoCapture
        current_frame_pos = 0
        for idx in sampled_indices:
            if idx != current_frame_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                current_frame_pos = idx
            ret, frame = cap.read()
            current_frame_pos += 1
            if ret and frame is not None:
                raw_sampled_frames.append(frame)
            else:
                # Use duplicate of previous frame or black canvas if read fails
                fallback = (
                    raw_sampled_frames[-1]
                    if raw_sampled_frames
                    else np.zeros((224, 224, 3), dtype=np.uint8)
                )
                raw_sampled_frames.append(fallback)
        cap.release()

    classifier = _get_cascade_classifier()

    cropped_rgb_frames: list[np.ndarray] = []
    face_detected_mask: list[bool] = []
    last_known_box: tuple[int, int, int, int] | None = None

    for frame in raw_sampled_frames:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect candidate faces
        detected_faces: Any = classifier.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(32, 32),
        )

        chosen_box: tuple[int, int, int, int]

        if len(detected_faces) > 0:
            if last_known_box is not None:
                # Track continuity: find candidate with maximum IoU to previous box
                best_iou = -1.0
                best_box: tuple[int, int, int, int] | None = None
                for face_arr in detected_faces:
                    box_tuple = (
                        int(face_arr[0]),
                        int(face_arr[1]),
                        int(face_arr[2]),
                        int(face_arr[3]),
                    )
                    iou = _compute_iou(last_known_box, box_tuple)
                    if iou > best_iou:
                        best_iou = iou
                        best_box = box_tuple
                if best_box is not None:
                    chosen_box = best_box
                else:
                    first = detected_faces[0]
                    chosen_box = (int(first[0]), int(first[1]), int(first[2]), int(first[3]))
            else:
                # Select largest face by area
                boxes = [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in detected_faces]
                chosen_box = max(boxes, key=lambda b: b[2] * b[3])

            face_detected_mask.append(True)
            last_known_box = chosen_box
        else:
            face_detected_mask.append(False)
            if last_known_box is not None:
                # Use last known stable track location
                chosen_box = last_known_box
            else:
                # Center square crop fallback
                side = min(h, w)
                chosen_box = ((w - side) // 2, (h - side) // 2, side, side)

        # Crop face and convert BGR to RGB
        crop = _crop_face_with_margin(frame, chosen_box, margin_ratio=1.3, target_size=(224, 224))
        crop_rgb: np.ndarray = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        cropped_rgb_frames.append(crop_rgb)

    face_coverage = float(sum(face_detected_mask)) / float(target_count)

    # Convert crops to normalized PyTorch tensor: (32, 3, 224, 224)
    tensor_list: list[torch.Tensor] = []
    for crop in cropped_rgb_frames:
        t = torch.from_numpy(crop).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
        t_norm = (t - IMAGENET_MEAN) / IMAGENET_STD
        tensor_list.append(t_norm)

    frames_tensor = torch.stack(tensor_list, dim=0)  # Shape: (32, 3, 224, 224)

    # Sliding windows: window_size=16, stride=8
    # Window 0: 0..16, Window 1: 8..24, Window 2: 16..32
    windows: list[torch.Tensor] = []
    for start_idx in range(0, target_count - window_size + 1, stride):
        win = frames_tensor[start_idx : start_idx + window_size]
        windows.append(win)

    return VideoPreprocessResult(
        frames_tensor=frames_tensor,
        face_coverage=face_coverage,
        raw_crops=cropped_rgb_frames,
        timestamps=timestamps,
        windows=windows,
    )
