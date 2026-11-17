# PARALLAX — Complete Machine Learning Pipeline Documentation
## Multimodal Deepfake Forensics: AVT²-DWF Lifecycle, Architecture, and Engineering

---

## 1. System Architecture

PARALLAX implements **AVT²-DWF** (Audio-Visual Transformer² with Dynamic Weight Fusion) for forensic detection of deepfake media. The system operates on two distinct execution paths:
- **Inference Path**: Real-time evaluation of user-submitted video clips via the FastAPI backend and bounded job runner.
- **Training Path**: Offline dataset ingestion, validation, identity-disjoint splitting, feature caching, and model training.

```
                    ┌───────────────────────────────────┐
                    │       User Media Upload           │
                    │   (POST /api/v1/analyses)         │
                    └─────────────────┬─────────────────┘
                                      ▼
                    ┌───────────────────────────────────┐
                    │      Input Validation & Probe     │
                    │      (FFprobe non-shell child)    │
                    └─────────────────┬─────────────────┘
                                      ▼
                    ┌───────────────────────────────────┐
                    │      Multimodal Preprocessing     │
                    │ • Face Detection (Haar + IoU)     │
                    │ • Spectrogram (MFCC-40 + Mel-64)  │
                    │ • 16-frame & 1-sec Alignment      │
                    └─────────────────┬─────────────────┘
                                      ▼
                    ┌───────────────────────────────────┐
                    │        AVT²-DWF Inference         │
                    │ • Face Transformer (ViT-B/16)     │
                    │ • Audio Transformer               │
                    │ • Dynamic Weight Fusion (DWF)     │
                    │ • Temperature-Scaled Logits       │
                    └─────────────────┬─────────────────┘
                                      ▼
                    ┌───────────────────────────────────┐
                    │     Forensic Artifact Engine      │
                    │ • ViT Attention Rollout Heatmaps  │
                    │ • Spectrogram Anomaly Overlay     │
                    │ • Timeline Anomaly Trajectory     │
                    └─────────────────┬─────────────────┘
                                      ▼
                    ┌───────────────────────────────────┐
                    │   Persistence & Result Delivery   │
                    │ • SQLite DB (WAL Mode)            │
                    │ • Server-Sent Events (SSE Stream) │
                    │ • Binary REAL / FAKE Verdict      │
                    └───────────────────────────────────┘
```

---

## 2. Actual Inference Sequence

When an authenticated analyst uploads media through the web frontend or API:
1. `POST /api/v1/analyses`: `upload_service.py` receives the multipart stream, computes SHA-256, validates magic bytes, and writes to `/storage/uploads/<uuid>.<ext>`.
2. `job_runner.py`: Enqueues an asynchronous background job and allocates a worker thread.
3. `pipeline.py`: Generator executes 10 sequential pipeline stages, yielding `PipelineEvent` messages:
   - Stage 1 (`probe`, 5%): Container duration, stream codecs, fps, and resolution probed via `probe_media()`.
   - Stage 2 (`frames`, 15%): 32 frames extracted uniformly at 25 fps via OpenCV.
   - Stage 3 (`faces`, 30%): Frontal faces tracked via Haar Cascade + IoU tracker; cropped with $1.3\times$ margin and resized to $224 \times 224$.
   - Stage 4 (`audio`, 45%): FFmpeg decodes 16 kHz mono WAV; Librosa extracts 40 MFCCs + 64 Log-Mel bands ($104$ channels total).
   - Stage 5 (`align`, 55%): 1-second audio segments matched to 16-frame video sliding windows by midpoint timestamp.
   - Stage 6 (`face_enc`, 65%): ViT-B/16 generates spatial embeddings; 4-frame token groupings pass through 4-layer temporal Transformer.
   - Stage 7 (`audio_enc`, 75%): 104x100 spectrogram patches projected and processed through 4-layer acoustic Transformer.
   - Stage 8 (`fusion`, 85%): Gating MLP computes $(w_a, w_v)$ and fuses representations into $F \in \mathbb{R}^{256}$. Classifier head evaluates binary logits.
   - Stage 9 (`explain`, 92%): Attention rollout generates heatmaps for top anomalous frames; spectrogram anomaly overlay and timeline JSON generated.
   - Stage 10 (`done`, 100%): Verdict mapped to `REAL` or `FAKE`, persisted to SQLite database, and pushed to client via SSE.

---

## 3. Video Preprocessing

- **Input**: Video file path (`.mp4`, `.mov`, `.avi`, `.webm`, `.mkv`).
- **Frame Sampling**: 32 frames uniformly sampled across video duration.
- **Face Localization**: Frontal face cascade classifier (`haarcascade_frontalface_default.xml`).
- **Tracking**: Greedy IoU tracker maintains temporal consistency of the primary subject across frames.
- **Crop Geometry**: Center point preserved; side length $= \max(w, h) \times 1.3$; bilinear resize to $224 \times 224 \times 3$.
- **Normalization**: Pixel values scaled to $[0, 1]$ and normalized by ImageNet channel means $[0.485, 0.456, 0.406]$ and standard deviations $[0.229, 0.224, 0.225]$.
- **Windowing**: Sliding windows of length 16 frames with stride 8:
  $$\mathbf{X}_{\text{windows}} \in \mathbb{R}^{K \times 16 \times 3 \times 224 \times 224}$$

---

## 4. Audio Preprocessing

- **Decoding**: FFmpeg extracts audio stream to 16,000 Hz, 16-bit mono PCM WAV.
- **Spectral Decomposition**:
  - FFT size: 512 samples.
  - Window length: 400 samples (25 ms).
  - Hop length: 160 samples (10 ms).
  - 40 Mel-Frequency Cepstral Coefficients (MFCC).
  - 64 Log-Mel filterbank energies.
- **Feature Concatenation**: Formatted as a 104-dimensional spectral feature vector per time step.
- **Patch Extraction**: Sliced into 1.0-second intervals (100 time frames at 10 ms hop):
  $$\mathbf{X}_{\text{segments}} \in \mathbb{R}^{K \times 104 \times 100}$$

---

## 5. Video Encoder Architecture

- **Backbone**: Vision Transformer (`vit_base_patch16_224`) from `timm`. Weights frozen during multimodal training.
- **$n$-Frame Grouping**: 4 consecutive frame embeddings $(4 \times 768 = 3072)$ grouped into single temporal tokens.
- **Token Projection**: Linear layer projects $3072 \to 256$.
- **Positional Encoding**: Learned embedding table for up to 64 window positions.
- **Temporal Transformer**: 4 encoder layers, 8 attention heads, hidden dim 256, feedforward dim 1024, dropout 0.10.
- **Output**:
  - Clip-level video embedding: $\mathbf{V}_{\text{clip}} \in \mathbb{R}^{256}$ (mean-pooled across windows).
  - Window-level video tokens: $\mathbf{v}_{\text{tokens}} \in \mathbb{R}^{K \times 256}$.

---

## 6. Audio Encoder Architecture

- **Patch Embedding**: Linear projection flattening each $104 \times 100$ patch $(10,400\text{ dims}) \to 256$.
- **Positional Encoding**: Learned embedding table indexing up to 64 segment positions.
- **Acoustic Transformer**: 4 encoder layers, 8 attention heads, hidden dim 256, feedforward dim 1024, dropout 0.10.
- **Output**:
  - Clip-level audio embedding: $\mathbf{A}_{\text{clip}} \in \mathbb{R}^{256}$.
  - Segment-level audio tokens: $\mathbf{a}_{\text{tokens}} \in \mathbb{R}^{K \times 256}$.

---

## 7. Dynamic Weight Fusion (DWF)

- **Input Vector**:
  $$\mathbf{z} = [\mathbf{A} \,\|\, \mathbf{V} \,\|\, |\mathbf{A} - \mathbf{V}| \,\|\, \text{cos\_sim}(\mathbf{A}, \mathbf{V})] \in \mathbb{R}^{769}$$
- **Gating Network**:
  $$\mathbf{h} = \text{ReLU}(\mathbf{z}\mathbf{W}_1 + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{769 \times 256}$$
  $$\mathbf{g} = \mathbf{h}\mathbf{W}_2 + \mathbf{b}_2, \quad \mathbf{W}_2 \in \mathbb{R}^{256 \times 2}$$
  $$[w_a, w_v] = \text{softmax}(\mathbf{g})$$
- **Masking Safeguard**: When audio stream is absent or silent, $w_a$ is forcibly zeroed and $w_v = 1.0$.
- **Fused Embedding**:
  $$\mathbf{F} = w_a \mathbf{A} + w_v \mathbf{V} \in \mathbb{R}^{256}$$
- **Lip-Sync Score**: Average cosine similarity across aligned segment/window token pairs.

---

## 8. Classification Head & Calibrated Verdict

- **Classifier Architecture**:
  $$\text{Linear}(256 \to 128) \to \text{GELU} \to \text{Dropout}(0.10) \to \text{Linear}(128 \to 2)$$
- **Temperature Scaling**: Logits divided by learned temperature parameter $T \in [0.5, 5.0]$.
- **Probability Calibration**: Softmax yields $[p_{\text{authentic}}, p_{\text{fake}}]$.
- **Decision Thresholds**:
  - $p_{\text{fake}} \ge 0.65 \implies \text{FAKE}$
  - $p_{\text{fake}} \le 0.35 \implies \text{REAL}$
  - Otherwise $\implies \text{INCONCLUSIVE}$

---

## 9. Dataset Folder Structure

```
datasets/
├── raw/
│   ├── real/       <-- Authentic video files (label 0)
│   └── fake/       <-- Deepfake video files (label 1)
├── manifests/
│   ├── dataset_manifest.csv
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── processed/
│   ├── video/      <-- Tensor files (*_video.pt)
│   ├── audio/      <-- Tensor files (*_audio.pt)
│   └── metadata/   <-- Extraction metadata (*.json)
└── reports/
    ├── dataset_summary.json
    ├── split_report.json
    ├── preprocessing_report.json
    └── evaluation_report.json
```

---

## 10. Dataset Validation

Executed via `scripts/validate_dataset.py`. Validates:
- Non-empty directory contents.
- Container readability and non-zero byte size.
- Maximum duration cap ($\le 60\text{ seconds}$).
- FFprobe stream validation (video codec, audio presence, resolution).
- SHA-256 duplicate detection.
- Class balance calculation.

---

## 11. Manifest Generation

Executed via `scripts/prepare_dataset.py --stage manifest`.
Produces `dataset_manifest.csv` with fields:
`sample_id, clip_path, relative_path, label, label_id, identity, audio_present`.

---

## 12. Identity-Disjoint Dataset Splitting

Executed via `scripts/prepare_dataset.py --stage split`.
Partitions samples strictly by subject identity (default 70% train, 15% validation, 15% test).
Guarantees 0% identity leakage between training and evaluation partitions.

---

## 13. Training Pipeline

Executed via `scripts/train_model.py`.
- **Framework**: PyTorch 2.15+ (CPU/CUDA).
- **Optimizer**: AdamW (`lr=1e-4`, `weight_decay=0.01`).
- **Learning Rate Schedule**: Cosine Annealing with warmup.
- **Loss Function**: Cross-Entropy Loss with optional label smoothing.
- **Sampling**: `WeightedRandomSampler` inversely proportional to class frequency.
- **Early Stopping**: Monitored on validation AUC with patience = 5 epochs.

---

## 14. Model Evaluation & Ablation Studies

Executed via `scripts/evaluate_model.py`.
- Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- Formats Confusion Matrix ($2 \times 2$).
- Runs **Static Fusion Ablation** ($w_a = 0.5, w_v = 0.5$) vs **Dynamic DWF** to isolate and quantify the net contribution of learned dynamic weighting.

---

## 15. Checkpoint Management

Checkpoints are isolated into functional directories:
- `checkpoints/candidate/`: Destination for new training runs.
- `checkpoints/best/`: Best validation AUC checkpoint during an active training run.
- `checkpoints/archived/`: Preserved previous production checkpoints with timestamp and hash.

---

## 16. Model Promotion

Executed via `scripts/promote_model.py --checkpoint <path> --confirm`.
- Loads candidate checkpoint to verify architectural state dictionary compatibility.
- Archives current active production checkpoint.
- Copies candidate to `backend/app/ml/weights/avt2_dwf.pt`.
- Updates `model_card.json` with new SHA-256 hash and timestamp.

---

## 17. Inference Deployment

In production:
- Model loaded once at backend startup via `app.ml.inference.load_model()`.
- Checkpoint hash verified and logged.
- If checkpoint missing, runs in `untrained` benchmark mode and logs warnings.

---

## 18. Limitations

- Extreme video compression ($\text{CRF} > 28$) degrades fine facial artifact detection.
- Audio-only deepfakes without visible facial lip movements require visual context.
- Cross-language evaluation: acoustic encoder trained primarily on English speech corpora.

---

## 19. Reproducibility

- Fixed random seed (`seed = 42`) for manifest shuffling, augmentation, and model initialization.
- PyTorch deterministic operations enabled where supported.
- Training configuration preserved in `backend/training/configs/default.yaml`.

---

## 20. CLI Execution Commands

```bash
# 1. Quality Control & Dataset Validation
python scripts/validate_dataset.py

# 2. Master Manifest Generation
python scripts/prepare_dataset.py --stage manifest

# 3. Identity-Disjoint Partitioning
python scripts/prepare_dataset.py --stage split --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15

# 4. Feature Extraction & Preprocessing
python scripts/prepare_dataset.py --stage preprocess

# 5. Verify Processed Feature Tensors
python scripts/validate_dataset.py --processed

# 6. Model Training (Full Run or Smoke Test)
python scripts/train_model.py --config backend/training/configs/default.yaml
python scripts/train_model.py --smoke

# 7. Model Evaluation & Ablation
python scripts/evaluate_model.py --checkpoint checkpoints/candidate/avt2_dwf_candidate.pt
python scripts/evaluate_model.py --smoke

# 8. Model Promotion to Production
python scripts/promote_model.py --checkpoint checkpoints/candidate/avt2_dwf_candidate.pt --confirm
```
