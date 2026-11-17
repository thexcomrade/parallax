# AVT²-DWF Architecture: Deep Technical Specification
## Audio-Visual Transformer² with Dynamic Weight Fusion for Deepfake Detection

---

## 1. High-Level Architectural Overview

PARALLAX implements **AVT²-DWF**, a multimodal deep learning architecture designed to expose deepfake media by fusing spatial-temporal facial representations with acoustic spectral representations.

```
Input Media (Video + Audio)
        │
        ├──────────────────────────────────────┬──────────────────────────────────────┐
        ▼                                      ▼                                      ▼
 Video Stream                           Audio Stream                           Audio Stream Presence
 (32 sampled frames @ 25fps)            (16 kHz Mono WAV)                      (Binary Mask: 1.0 or 0.0)
        │                                      │                                      │
        ▼                                      ▼                                      │
 Face Detection & Tracking              Spectrogram Extraction                        │
 (Haar Cascade + IoU Tracker)           (MFCC: 40 + Log-Mel: 64)                      │
        │                                      │                                      │
        ▼                                      ▼                                      │
 16-frame Sliding Windows               1-second Audio Patches                         │
 Tensor: [B, K, 16, 3, 224, 224]        Tensor: [B, K, 104, 100]                      │
        │                                      │                                      │
        ▼                                      ▼                                      │
 ┌───────────────────────────┐          ┌───────────────────────────┐                 │
 │   FACE TRANSFORMER        │          │   AUDIO TRANSFORMER       │                 │
 │   ENCODER                 │          │   ENCODER                 │                 │
 │ • ViT-B/16 (d=768)        │          │ • Patch Embed (104x100)   │                 │
 │ • 4-frame grouping (n=4)  │          │ • Positional Embeddings   │                 │
 │ • Linear proj (3072->256) │          │ • 4-layer Transformer     │                 │
 │ • 4-layer Transformer     │          │   (8 heads, d=256)        │                 │
 │   (8 heads, d=256)        │          │                           │                 │
 └─────────────┬─────────────┘          └─────────────┬─────────────┘                 │
               │                                      │                               │
    Video Embeddings:                      Audio Embeddings:                          │
    Clip V ∈ ℝ²⁵⁶, Tokens ∈ ℝᴷˣ²⁵⁶         Clip A ∈ ℝ²⁵⁶, Tokens ∈ ℝᴷˣ²⁵⁶             │
               │                                      │                               │
               └──────────────────────┬───────────────┴───────────────────────────────┘
                                      ▼
                       ┌─────────────────────────────┐
                       │   DYNAMIC WEIGHT FUSION     │
                       │   (DWF Gating MLP)          │
                       │ • [A; V; |A-V|; cos(A,V)]   │
                       │ • MLP: 769 -> 256 -> 2      │
                       │ • Softmax -> (w_a, w_v)     │
                       │ • Modality Mask Override    │
                       │ • F = w_a·A + w_v·V         │
                       │ • Lip-Sync Cosine Score     │
                       └──────────────┬──────────────┘
                                      │
                               Fused Feature
                                  F ∈ ℝ²⁵⁶
                                      │
                       ┌──────────────┴──────────────┐
                       │   CLASSIFICATION HEAD       │
                       │ • Linear(256 -> 128)        │
                       │ • GELU + Dropout(0.1)       │
                       │ • Linear(128 -> 2)          │
                       │ • Temperature Scaling (T)   │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                       Calibrated Logits [B, 2]
                             (REAL / FAKE)
```

---

## 2. Modality 1: Video Branch (Face Transformer Encoder)

### 2.1 Preprocessing & Input Tensor
- **Raw Input**: Uniformly sampled $T=32$ frames extracted at 25 fps.
- **Face Localization**: OpenCV Haar Cascade detects frontal faces. An IoU-based tracking algorithm links detections across time.
- **Bounding Box Cropping**: $1.3\times$ margin around the detected face bounding box, capturing facial boundaries, jawline, and forehead artifacts.
- **Normalization**: Resized to $224 \times 224$ and normalized using standard ImageNet constants ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Temporal Windows**: Partitioned into sliding windows of $W=16$ frames with stride $S=8$, yielding $K$ temporal windows.
- **Input Shape**: $\mathbf{X}_{\text{video}} \in \mathbb{R}^{B \times K \times 16 \times 3 \times 224 \times 224}$.

### 2.2 ViT-B/16 Spatial Embedding
- Each frame is passed through a Vision Transformer (`vit_base_patch16_224`) with frozen weights:
  $$\mathbf{f}_{t} = \text{ViT}(\mathbf{x}_{t}) \in \mathbb{R}^{768}$$
- For a sequence of 16 frames in window $k$, this yields $[\mathbf{f}_{1}, \ldots, \mathbf{f}_{16}] \in \mathbb{R}^{16 \times 768}$.

### 2.3 $n$-Frame Token Grouping ($n=4$)
- Rather than feeding individual frame embeddings into temporal attention, 4 consecutive frame embeddings are concatenated:
  $$\mathbf{g}_{m} = [\mathbf{f}_{4m-3} \,\|\, \mathbf{f}_{4m-2} \,\|\, \mathbf{f}_{4m-1} \,\|\, \mathbf{f}_{4m}] \in \mathbb{R}^{4 \times 768 = 3072}$$
- Projected linearly into model dimension $d=256$:
  $$\mathbf{t}_{m} = \mathbf{g}_{m} \mathbf{W}_{\text{token}} + \mathbf{b}_{\text{token}} \in \mathbb{R}^{256}$$

### 2.4 Temporal Transformer Encoder
- **Learned Positional Encoding**: $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{\text{MAX\_TOKENS} \times 256}$ added to token sequence.
- **Transformer Layers**: 4 layers, 8 self-attention heads, $d_{\text{ff}} = 1024$, dropout $0.10$.
  $$\mathbf{Q} = \mathbf{T} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{T} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{T} \mathbf{W}_V$$
  $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
- **Output**:
  - Per-window token representation: $\mathbf{v}_{\text{tokens}} \in \mathbb{R}^{B \times K \times 256}$.
  - Clip-level video embedding (mean-pooled across windows): $\mathbf{V}_{\text{clip}} \in \mathbb{R}^{B \times 256}$.

---

## 3. Modality 2: Audio Branch (Audio Transformer Encoder)

### 3.1 Preprocessing & Acoustic Features
- **Audio Stream**: Decoded to 16 kHz mono 16-bit PCM WAV.
- **Acoustic Transform**:
  - 40-band Mel-Frequency Cepstral Coefficients (MFCC).
  - 64-band Log-Mel Spectrogram.
  - Window length: 400 samples (25 ms).
  - Hop length: 160 samples (10 ms).
  - Combined feature dimension: $40 + 64 = 104$ channels.
- **Segmentation**: Extracted into 1.0-second temporal patches ($100$ time frames at 10 ms hop):
  $$\mathbf{X}_{\text{audio}} \in \mathbb{R}^{B \times K \times 104 \times 100}$$

### 3.2 Patch Projection & Temporal Attention
- **Patch Embedding**: Linear projection flattening each 1-second spectrogram patch $(104 \times 100 = 10,400)$ into $d=256$:
  $$\mathbf{a}_{\text{token}, k} = \text{Flatten}(\mathbf{X}_{\text{audio}, k}) \mathbf{W}_{\text{patch}} + \mathbf{b}_{\text{patch}} \in \mathbb{R}^{256}$$
- **Learned Positional Embedding**: Segment order encoding added across temporal index $k$.
- **Acoustic Transformer Layers**: 4 layers, 8 self-attention heads, $d_{\text{ff}} = 1024$.
- **Output**:
  - Segment-level audio tokens: $\mathbf{a}_{\text{tokens}} \in \mathbb{R}^{B \times K \times 256}$.
  - Clip-level audio embedding: $\mathbf{A}_{\text{clip}} \in \mathbb{R}^{B \times 256}$.

---

## 4. Cross-Modal Fusion: Dynamic Weight Fusion (DWF)

### 4.1 Fusion Objective & Rationale
Deepfake generation often introduces discrepancies:
1. Audio might be authentic speech while facial video is synthetic (e.g. Face Swap).
2. Video might be authentic while audio is AI voice-cloned (e.g. Speech Synthesis).
3. Both modalities might be real, but temporal alignment is disrupted (e.g. Wav2Lip sync artifacts).

A static 50/50 weighted sum or concatenation cannot adapt when one modality exhibits severe corruption or is absent altogether. DWF dynamically estimates confidence weights based on modality interaction.

### 4.2 DWF Mathematical Walkthrough

1. **Cosine Similarity Proxy**:
   $$\text{cos\_sim}(\mathbf{A}, \mathbf{V}) = \frac{\mathbf{A} \cdot \mathbf{V}}{\|\mathbf{A}\|_2 \|\mathbf{V}\|_2} \in [-1, 1]$$

2. **Gating Vector Construction**:
   Concatenate the audio representation, video representation, elementwise absolute discrepancy, and global cosine similarity:
   $$\mathbf{z} = \left[ \mathbf{A} \,\|\, \mathbf{V} \,\|\, |\mathbf{A} - \mathbf{V}| \,\|\, \text{cos\_sim}(\mathbf{A}, \mathbf{V}) \right] \in \mathbb{R}^{256 + 256 + 256 + 1 = 769}$$

3. **Gating MLP**:
   $$\mathbf{h} = \text{ReLU}(\mathbf{z} \mathbf{W}_1 + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{769 \times 256}$$
   $$\mathbf{g} = \mathbf{h} \mathbf{W}_2 + \mathbf{b}_2, \quad \mathbf{W}_2 \in \mathbb{R}^{256 \times 2}$$

4. **Softmax Weight Normalization**:
   $$[w_a, w_v] = \text{softmax}(\mathbf{g}) = \left[ \frac{e^{g_1}}{e^{g_1} + e^{g_2}}, \, \frac{e^{g_2}}{e^{g_1} + e^{g_2}} \right]$$

5. **Modality Masking & Zero-Audio Safeguard**:
   If audio is missing or silent ($\text{mask} = 0$):
   $$w_a \leftarrow 0.0, \quad w_v \leftarrow 1.0$$
   Else, ensure partition of unity:
   $$w_a \leftarrow \frac{w_a}{w_a + w_v + \epsilon}, \quad w_v \leftarrow \frac{w_v}{w_a + w_v + \epsilon}$$

6. **Fused Embedding**:
   $$\mathbf{F} = w_a \mathbf{A} + w_v \mathbf{V} \in \mathbb{R}^{256}$$

7. **Cross-Modal Synchronization Score**:
   $$\text{sync\_score} = \frac{1}{K} \sum_{k=1}^K \frac{\mathbf{a}_{\text{token}, k} \cdot \mathbf{v}_{\text{token}, k}}{\|\mathbf{a}_{\text{token}, k}\|_2 \|\mathbf{v}_{\text{token}, k}\|_2}$$

---

## 5. Classification Head & Calibrated Output

### 5.1 Classifier MLP
$$\mathbf{z}_c = \text{Linear}_{256 \to 128}(\mathbf{F})$$
$$\mathbf{z}_c = \text{GELU}(\mathbf{z}_c)$$
$$\mathbf{z}_c = \text{Dropout}_{0.1}(\mathbf{z}_c)$$
$$\mathbf{logits}_{\text{raw}} = \text{Linear}_{128 \to 2}(\mathbf{z}_c) = [z_0, z_1]$$

### 5.2 Temperature Calibration
To prevent overconfident softmax probabilities, temperature scaling is applied using a learned parameter $T = \text{clamp}(\exp(\tau), 0.5, 5.0)$:
$$\mathbf{logits} = \frac{\mathbf{logits}_{\text{raw}}}{T}$$
$$\mathbf{p} = \text{softmax}(\mathbf{logits}) = [p_{\text{authentic}}, p_{\text{fake}}]$$

### 5.3 User-Facing Classification Contract
$$\text{Verdict} = \begin{cases}
\text{FAKE} & \text{if } p_{\text{fake}} \ge 0.65 \\
\text{AUTHENTIC} & \text{if } p_{\text{fake}} \le 0.35 \\
\text{INCONCLUSIVE} & \text{otherwise}
\end{cases}$$
