"""PyTorch Dataset and DataLoader collation for multimodal deepfake detection.

Uses the same preprocessing pipeline as inference to prevent train/serve skew (README §5, §8.7).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from app.ml.preprocess.align import align_audio_video
from app.ml.preprocess.audio import extract_audio_features
from app.ml.preprocess.video import extract_video_frames
from training.augment import MultimodalAugmenter

logger = logging.getLogger("parallax.training.dataset")


class MultimodalDataset(Dataset[dict[str, Any]]):
    """Multimodal dataset reading from an identity-disjoint manifest CSV.

    Yields:
        dict with keys:
          - 'video_windows': Tensor[K, 16, 3, 224, 224]
          - 'audio_segments': Tensor[K, 104, 100]
          - 'audio_mask': Tensor[K]
          - 'label': int (0 for authentic, 1 for fake)
          - 'identity': str
          - 'clip_path': str
    """

    def __init__(
        self,
        manifest_csv: Path | str,
        split: str | None = None,
        augmenter: MultimodalAugmenter | None = None,
        sample_frames: int = 32,
        window_size: int = 16,
        window_stride: int = 8,
    ) -> None:
        self.manifest_path = Path(manifest_csv)
        self.split = split
        self.augmenter = augmenter
        self.sample_frames = sample_frames
        self.window_size = window_size
        self.window_stride = window_stride

        self.records: list[dict[str, Any]] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_split = row.get("split", "train")
                if self.split is not None and row_split != self.split:
                    continue

                self.records.append(
                    {
                        "clip_path": row["clip_path"],
                        "label": int(row["label"]),
                        "identity": row.get("identity", "unknown"),
                        "audio_present": row.get("audio_present", "True").lower() == "true",
                        "split": row_split,
                    }
                )

        logger.info(
            "Loaded %d samples for split='%s' from %s",
            len(self.records),
            self.split,
            self.manifest_path.name,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        clip_path = Path(rec["clip_path"])
        label = int(rec["label"])
        identity = str(rec["identity"])

        # 1. Video extraction using inference preprocessing
        video_res = extract_video_frames(
            clip_path,
            target_count=self.sample_frames,
            window_size=self.window_size,
            stride=self.window_stride,
        )

        # 2. Audio extraction
        audio_res = extract_audio_features(clip_path)

        # 3. Multimodal temporal alignment
        aligned = align_audio_video(
            video_res,
            audio_res,
            window_size=self.window_size,
            stride=self.window_stride,
        )

        video_windows = aligned.video_windows  # [K, 16, 3, 224, 224]
        audio_segments = aligned.audio_segments  # [K, 104, 100]
        audio_mask = aligned.audio_mask  # [K]

        # 4. Apply augmentations if training
        if self.augmenter is not None:
            video_windows = self.augmenter.augment_video(video_windows)
            audio_segments = self.augmenter.augment_audio(audio_segments)
            video_windows, audio_segments, label, _ = self.augmenter.apply_cross_modal_mismatch(
                video_windows, audio_segments, label
            )

        return {
            "video_windows": video_windows,
            "audio_segments": audio_segments,
            "audio_mask": audio_mask,
            "label": label,
            "identity": identity,
            "clip_path": str(clip_path),
        }

    def get_labels(self) -> list[int]:
        """Return list of integer labels for sampler weighting."""
        return [int(r["label"]) for r in self.records]


def multimodal_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collates a batch of multimodal samples into batched PyTorch tensors.

    Output shapes:
        video_windows:  Tensor[B, K, 16, 3, 224, 224]
        audio_segments: Tensor[B, K, 104, 100]
        audio_mask:     Tensor[B, K]
        labels:         Tensor[B] (torch.long)
        identities:     list[str]
        clip_paths:     list[str]
    """
    video_windows = torch.stack([b["video_windows"] for b in batch], dim=0)
    audio_segments = torch.stack([b["audio_segments"] for b in batch], dim=0)
    audio_mask = torch.stack([b["audio_mask"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    identities = [str(b["identity"]) for b in batch]
    clip_paths = [str(b["clip_path"]) for b in batch]

    return {
        "video_windows": video_windows,
        "audio_segments": audio_segments,
        "audio_mask": audio_mask,
        "labels": labels,
        "identities": identities,
        "clip_paths": clip_paths,
    }


def create_balanced_sampler(dataset: MultimodalDataset) -> WeightedRandomSampler:
    """Compute inverse class frequency weights and return a balanced sampler."""
    labels = dataset.get_labels()
    if not labels:
        raise ValueError("Cannot create balanced sampler for empty dataset")

    count_neg = max(1, sum(1 for lbl in labels if lbl == 0))
    count_pos = max(1, sum(1 for lbl in labels if lbl == 1))
    total = len(labels)

    weight_neg = total / (2.0 * count_neg)
    weight_pos = total / (2.0 * count_pos)

    weights = [weight_pos if lbl == 1 else weight_neg for lbl in labels]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    return sampler
