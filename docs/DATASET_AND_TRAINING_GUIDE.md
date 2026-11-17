# PARALLAX — Dataset Management & Model Training Guide

This guide describes how to ingest, validate, partition, preprocess, train, evaluate, and promote deep learning checkpoints for the **AVT²-DWF** (Audio-Visual Transformer with Dynamic Weight Fusion) architecture.

---

## 1. Directory Structure

Training data is managed under the root `datasets/` hierarchy:

```
PARALLAX/
├── datasets/
│   ├── raw/
│   │   ├── real/                   <-- Place original / authentic videos here
│   │   └── fake/                   <-- Place manipulated / deepfake videos here
│   ├── manifests/
│   │   ├── dataset_manifest.csv    <-- Master manifest generated from raw/
│   │   ├── train.csv               <-- Identity-disjoint training split
│   │   ├── validation.csv          <-- Identity-disjoint validation split
│   │   └── test.csv                <-- Identity-disjoint test split
│   ├── processed/
│   │   ├── video/                  <-- Cached face-window tensors (*_video.pt)
│   │   ├── audio/                  <-- Cached audio spectrogram tensors (*_audio.pt)
│   │   └── metadata/               <-- Sample-level preprocessing metadata (*.json)
│   └── reports/
│       ├── dataset_summary.json    <-- Quality control validation report
│       ├── split_report.json       <-- Partition statistics and leakage check
│       ├── preprocessing_report.json <-- Feature cache summary
│       └── evaluation_report.json  <-- Held-out test set metrics
├── checkpoints/
│   ├── candidate/                  <-- Freshly trained models
│   ├── best/                       <-- Checkpoint with highest validation AUC
│   └── archived/                   <-- Previous production checkpoints
└── scripts/
    ├── validate_dataset.py
    ├── prepare_dataset.py
    ├── train_model.py
    ├── evaluate_model.py
    └── promote_model.py
```

---

## 2. Ingesting Raw Samples

### Step 2.1: Add Media
Place video files into the respective category folders:

- **Authentic original clips**: `datasets/raw/real/` (Label `0`)
- **Deepfake / manipulated clips**: `datasets/raw/fake/` (Label `1`)

**Supported Video Containers**: `.mp4`, `.mov`, `.avi`, `.webm`, `.mkv`.  
**Recommended Naming**: `id<subject>_<clip_id>.mp4` (e.g. `id0042_clip01.mp4`). This allows the splitter to partition strictly by subject identity.

---

## 3. Dataset Validation & Quality Control

Verify file readability, container duration, resolution, audio track presence, and duplicate detection:

```bash
# Validate raw dataset
python scripts/validate_dataset.py

# Inspect JSON quality summary
cat datasets/reports/dataset_summary.json
```

**Quality Checks Performed**:
1. Zero-byte and corrupt file detection.
2. File duration checks ($\le 60\text{ seconds}$).
3. FFprobe container stream integrity.
4. SHA-256 duplicate detection across directories.
5. Class imbalance ratio calculation.

---

## 4. Manifest Generation & Identity-Disjoint Splitting

To prevent facial identity leakage (where frames from the same individual appear in both training and test sets), the splitter groups clips by subject identity and partitions them into 70% train, 15% validation, and 15% test:

```bash
# Generate master manifest
python scripts/prepare_dataset.py --stage manifest

# Generate identity-disjoint splits
python scripts/prepare_dataset.py --stage split --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42

# Verify split statistics and zero-leakage check
cat datasets/reports/split_report.json
```

---

## 5. Offline Feature Preprocessing

Pre-extract facial crops and audio spectrograms to accelerate training epochs:

```bash
# Run batch preprocessing
python scripts/prepare_dataset.py --stage preprocess

# Validate processed tensor caches
python scripts/validate_dataset.py --processed
```

Each video produces:
- `datasets/processed/video/<sample_id>_video.pt`: Tensor $[K, 16, 3, 224, 224]$
- `datasets/processed/audio/<sample_id>_audio.pt`: Tensor $[K, 104, 100]$ + audio mask
- `datasets/processed/metadata/<sample_id>.json`: Face coverage, window timestamps, label

---

## 6. Training the Model

Execute offline training using PyTorch. Training operates completely independently of the user-facing inference server:

```bash
# Full training run using configuration
python scripts/train_model.py \
  --config backend/training/configs/default.yaml \
  --manifest datasets/manifests/train.csv \
  --output-checkpoint checkpoints/candidate/avt2_dwf_candidate.pt \
  --epochs 20 \
  --batch-size 8

# Fast 2-step pipeline verification smoke test (uses synthetic fixture)
python scripts/train_model.py --smoke
```

**Training Features**:
- **Optimizer**: AdamW ($\text{lr} = 1\times 10^{-4}$, weight decay $0.01$).
- **LR Schedule**: Cosine Annealing with warmup.
- **Data Augmentations**: Horizontal flipping, color jitter, frequency masking, cross-modal mismatch injection.
- **Class Balancing**: `WeightedRandomSampler` inversely proportional to class frequencies.
- **Early Stopping**: Monitors validation AUC with patience = 5 epochs.

---

## 7. Model Evaluation & Ablation

Evaluate the candidate checkpoint on the held-out test split:

```bash
# Evaluate candidate checkpoint
python scripts/evaluate_model.py \
  --checkpoint checkpoints/candidate/avt2_dwf_candidate.pt \
  --manifest datasets/manifests/test.csv \
  --report datasets/reports/evaluation_report.json

# Synthetic smoke evaluation
python scripts/evaluate_model.py --smoke
```

**Metrics Computed**:
- Accuracy, Precision, Recall, F1-Score, AUC-ROC.
- Confusion Matrix (True Negative, False Positive, False Negative, True Positive).
- **Ablation Study**: DWF vs Static 50/50 Equal Weight Fusion to measure net gain.

---

## 8. Safe Model Promotion to Production

Never automatically overwrite the production checkpoint. Review evaluation reports and execute promotion explicitly:

```bash
# Review candidate metrics
cat datasets/reports/evaluation_report.json

# Promote candidate to active production
python scripts/promote_model.py \
  --checkpoint checkpoints/candidate/avt2_dwf_candidate.pt \
  --confirm
```

**Promotion Safeguards**:
1. Checks that the candidate checkpoint is loadable with `weights_only=True`.
2. Archives the existing production checkpoint to `checkpoints/archived/avt2_dwf_archived_<timestamp>_<hash>.pt`.
3. Atomically copies the candidate to `backend/app/ml/weights/avt2_dwf.pt`.
4. Computes SHA-256 and updates `backend/app/ml/weights/model_card.json`.
