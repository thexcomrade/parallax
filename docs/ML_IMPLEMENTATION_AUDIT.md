# PARALLAX — ML Implementation & Architecture Audit Report

**Author:** Antigravity (ML Systems & Deep Learning Architect)  
**Date:** 2026-11-16  
**System Target:** PARALLAX Audio-Visual Deepfake Detection (AVT²-DWF)  
**Repository State:** Verified on actual source tree  

---

## 1. Executive Summary

This audit assesses the readiness of the complete offline machine learning lifecycle of PARALLAX:
`Dataset Validation → Video Preprocessing → Audio Preprocessing → Feature Storage → AVT²-DWF Training → Validation → Testing → Candidate Checkpoint`.

A real 500-video dataset has been ingested under `datasets/raw/` (250 REAL, 250 FAKE) with manifests generated under `datasets/manifests/` (Train: 350, Validation: 76, Test: 74).

---

## 2. Component-by-Component Implementation Audit

| Component | Architecture Spec | Actual Source Location | Implemented? | Status & Observations |
|---|---|---|---|---|
| **FaceEncoder** | ViT-B/16 (768) + 4-frame grouping (3072→256) + 4-layer 8-head temporal Transformer | `backend/app/ml/models/face_encoder.py` | **YES** | Pretrained ViT-B/16 frozen; linear projection $3072 \to 256$; learned positional encodings ($64$ max tokens); temporal mean pooling $\to \mathbf{v} \in \mathbb{R}^{B \times 256}$. Verified. |
| **AudioEncoder** | 40 MFCC + 64 Log-Mel ($104 \times 100 \to 10400 \to 256$) + 4-layer 8-head acoustic Transformer | `backend/app/ml/models/audio_encoder.py` | **YES** | Patch embedding linear $10400 \to 256$; learned positional encodings; acoustic mean pooling $\to \mathbf{a} \in \mathbb{R}^{B \times 256}$. Verified. |
| **Dynamic Weight Fusion (DWF)** | Gating MLP over $[A; V; \|A-V\|; \cos(A,V)] \in \mathbb{R}^{769} \to (w_a, w_v)$ | `backend/app/ml/models/dwf.py` | **YES** | Gating MLP Linear(769, 256) $\to$ ReLU $\to$ Linear(256, 2) $\to$ Softmax. Automatic silent-audio fallback ($w_a=0, w_v=1.0$). Cross-modal token sync cosine similarity. Verified. |
| **AVT²-DWF Model** | Full end-to-end assembly + classifier head | `backend/app/ml/models/avt2_dwf.py` | **YES** | Assembles FaceEncoder, AudioEncoder, DWF, 2-layer classifier MLP, per-window head, learnable temperature scaling $T \in [0.5, 5.0]$. Verified. |
| **Video Preprocessing** | 16 frames / 2.0s window, Haar/MediaPipe face crop, 224x224 RGB ImageNet norm | `backend/app/ml/preprocess/video.py` | **YES** | OpenCV `VideoCapture`, face detection with Haar cascade (tracking with IoU across frames), 1.3x margin crop, center fallback, sliding windows ($16$ frames, stride $8$). Output $(32, 3, 224, 224)$ and $(K, 16, 3, 224, 224)$. Verified. |
| **Audio Preprocessing** | 16 kHz mono PCM, 40 MFCC + 64 Log-Mel | `backend/app/ml/preprocess/audio.py` | **YES** | FFmpeg 16 kHz mono extraction, pre-emphasis ($0.97$), Librosa 40 MFCC + 64 Log-Mel, 1.0s segmentation into $(104, 100)$ patches. Verified. |
| **Audio-Visual Alignment** | Window alignment & missing audio mask | `backend/app/ml/preprocess/align.py` | **YES** | Temporal midpoint alignment between video windows and audio patches. Generates `audio_mask` ($1.0$ valid, $0.0$ missing/silent). Verified. |
| **Dataset Loader** | PyTorch `Dataset` with collation | `backend/training/dataset.py` | **PARTIAL** | Reads manifest CSV, performs on-the-fly video/audio extraction. **Issue Identified**: Relies on column `clip_path` while CSV has `relative_path`; split string mismatch (`val` vs `validation`). Preprocessing cache lookup missing. |
| **Loss & Optimization** | Weighted Cross-Entropy + AdamW + CosineAnnealing | `backend/training/train.py` | **YES** | Cross-entropy loss, AdamW ($lr=10^{-4}$), Cosine Annealing, early stopping on validation AUC. Verified. |
| **Evaluation Pipeline** | Held-out test evaluation + Confusion Matrix + DWF Ablation | `backend/training/evaluate.py`, `scripts/evaluate_model.py` | **YES** | Full metrics (Accuracy, Precision, Recall, F1, ROC-AUC), $2 \times 2$ Confusion Matrix, static 50/50 fusion ablation study. Verified. |
| **Model Checkpoint Management** | Safe candidate saving, archiving, promotion gate | `checkpoints/`, `scripts/promote_model.py` | **YES** | Candidate checkpoints saved to `checkpoints/candidate/`. Production weights `backend/app/ml/weights/avt2_dwf.pt` protected. Verified. |
| **Inference API** | Async background task processing | `backend/app/api/v1/analyses.py`, `backend/app/services/analysis_service.py` | **YES** | Magic byte verification, decodes media, invokes `AVT2DWF` forward, commits binary `REAL`/`FAKE` verdict to SQLite database. Verified. |

---

## 3. Key Issues & Training Blockers Identified

### 3.1 Manifest Column and Split String Inconsistency
- Manifest CSV files generated at `datasets/manifests/` have columns:
  `['sample_id', 'filename', 'relative_path', 'label', 'class_name', 'split']`.
- `backend/training/dataset.py` loaded `row["clip_path"]` and checked `row_split == self.split`.
- In `validation.csv`, `split` is `"validation"`, but `train.py` requested `split="val"`.
- **Remediation**: Update `MultimodalDataset` to resolve paths via `row.get("clip_path") or row.get("relative_path")` against `REPO_ROOT`, and map `val` $\leftrightarrow$ `validation` symmetrically.

### 3.2 On-the-Fly Video Decoding Bottleneck
- Currently, `MultimodalDataset.__getitem__` opens the raw video with OpenCV, runs Haar Cascade, and extracts audio with FFmpeg on every sample access.
- For 350 training videos across 20 epochs ($7,000$ decodings), this would bottleneck the CPU for hours.
- **Remediation**: In Phase 5/6, run batch preprocessing with `scripts/prepare_dataset.py --stage preprocess` to cache precomputed `video_windows` ($(K, 16, 3, 224, 224)$) and `audio_segments` ($(K, 104, 100)$) as compressed `.pt` or `.npy` tensors in `datasets/processed/`. `MultimodalDataset` will prioritize reading the cache.

### 3.3 Hardware Constraints & Compute Profile
- Target Host: **Intel Core i5-10300H (4 cores, 8 threads)**, **16 GB RAM**, **NVIDIA GeForce GTX 1650 (4 GB VRAM)**.
- `backend/.venv` currently has `torch==2.15.0+cpu`.
- The GTX 1650 with 4 GB VRAM is constrained for full ViT-B/16 end-to-end backpropagation. However, because the ViT-B/16 spatial encoder is **frozen** (`for p in self.vit.parameters(): p.requires_grad = False`), only the temporal Transformer, projection, and audio Transformer require gradients.
- Batch size must be kept small ($B=2$ to $4$) with gradient accumulation if necessary to stay within memory limits.
- On CPU, training with precomputed feature caches will execute in minutes instead of hours.

---

## 4. Verification Checkpoint Compatibility
- Checkpoint `backend/app/ml/weights/avt2_dwf.pt` (383 MB) uses standard `torch.save(model.state_dict())` matching `AVT2DWF(pretrained_vit=True)`.
- Candidate training will save directly to `checkpoints/candidate/avt2_dwf_candidate.pt` without altering production weights.

---

## 5. Audit Conclusion: READY FOR VALIDATION & PREPROCESSING

All model definitions, encoders, fusion networks, and evaluation modules are architecturally verified. Once `dataset.py` is patched for manifest path resolution and feature cache loading, Phase 2 (Validation) and Phase 3–6 (Preprocessing) can proceed safely.
