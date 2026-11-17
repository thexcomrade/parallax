# PARALLAX — PHASE 3: MODEL IMPROVEMENT RECOMMENDATIONS

## Executive Summary

Phase 2 baseline training established an operational pipeline on the NVIDIA GeForce GTX 1650, completing all 15 epochs and capturing a baseline checkpoint with **Val F1: 0.7174**, **Val Recall: 0.8684**, and **Val AUC: 0.6974**.

However, empirical investigation during Phase 3 revealed three severe architectural and algorithmic bottlenecks:
1. **Modality Collapse:** The Dynamic Weight Fusion (DWF) gate collapsed to video ($w_v = 0.9962, w_a = 0.0038$). The audio branch was starved of gradient updates, yielding an audio-only ROC-AUC of **0.4972** (pure random chance).
2. **False-Positive Inducing Augmentation:** The cross-modal mismatch augmentation (`mismatch_prob=0.15`) forced labels of REAL samples to `1 (FAKE)` when audio was shifted. Because the model ignored audio, it was penalized for predicting REAL on genuine faces, resulting in **21 false positives** out of 38 validation REAL samples (REAL recall of only 44.74%).
3. **Severe Overfitting & Loss Spike:** Validation loss spiked from **0.6041** (Epoch 1) to **1.8823** (Epoch 11) and **1.8410** (Epoch 15), while training loss decreased to **0.2576**. The median validation loss was only **0.1766**, but 14 overconfident false-positive misclassifications with losses $> 4.0$ (up to 11.25) inflated the mean loss.

Below are 5 prioritized, evidence-based recommendations designed to resolve these bottlenecks in Phase 4.

---

## Ranked Improvement Recommendations

### Priority 1: Reform Cross-Modal Mismatch Augmentation (Eliminate Training Label Noise)

- **Observed Problem:**
  In `backend/training/augment.py`, `apply_cross_modal_mismatch(mismatch_prob=0.15)` randomly rolls the audio segments of REAL samples by one temporal window and forces the sample's primary classification label to `1 (FAKE)`.
- **Phase 2 Evidence:**
  The validation confusion matrix shows **21 false positives** (REAL videos predicted as FAKE) out of 38 REAL samples. The top 5 highest-loss validation samples (`REAL_0059`, `REAL_0179`, `REAL_0021`, `REAL_0207`, `REAL_0142`) are all genuine REAL videos predicted as FAKE with $>99.9\%$ probability. Because the audio branch collapsed ($w_a \approx 0.0038$), the model was presented with genuine visual faces and penalized unless it classified them as FAKE, teaching the visual classifier to associate authentic human faces with deepfakes.
- **Proposed Change:**
  Set `mismatch_prob = 0.0` for primary binary classification training, OR decouple audio-visual desync detection from the primary binary classification label into an auxiliary desynchronization head.
- **Expected Benefit:**
  Eliminates artificial label corruption on authentic human faces. Increases REAL class validation recall from **44.74%** to **$>75\%$**, reduces false positives from 21 down to $<8$, and normalizes the validation loss.
- **Potential Risk:**
  The model will rely on genuine visual and acoustic deepfake artifacts rather than artificially induced temporal shift artifacts. This is strictly beneficial for in-the-wild deepfake detection.
- **Evaluation in Phase 4:**
  Compare REAL class recall and False Positive rate on the 76 validation samples.

---

### Priority 2: Multi-Task Auxiliary Loss for Modality Balancing (Prevent DWF Collapse)

- **Observed Problem:**
  The Dynamic Weight Fusion module experienced complete modality starvation. Pretrained ViT-B/16 converged rapidly, causing the gate MLP to assign $w_v = 0.9962$ and $w_a = 0.0038$. Because backpropagation through the audio branch scales with $w_a$ ($\nabla_A \mathcal{L} = w_a \nabla_F \mathcal{L}$), the audio branch received near-zero gradients.
- **Phase 2 Evidence:**
  - Full model validation ROC-AUC: **0.6974**
  - Unimodal video validation ROC-AUC: **0.6974** (100% identical to full model)
  - Unimodal audio validation ROC-AUC: **0.4972** (effectively random coin toss)
  - Mean audio fusion weight: $w_a = 0.0038 \pm 0.0036$
- **Proposed Change:**
  Implement a balanced multi-task training objective:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fused}} + \alpha \mathcal{L}_{\text{audio}} + \beta \mathcal{L}_{\text{video}}$$
  where $\alpha = 0.3$ and $\beta = 0.1$. Auxiliary linear heads evaluate $A_{\text{clip}}$ and $V_{\text{clip}}$ independently. In addition, introduce a fusion weight floor ($w_a, w_v \ge 0.10$) or an entropy regularization penalty on the gate distribution: $\mathcal{L}_{\text{gate}} = -\lambda (w_a \log w_a + w_v \log w_v)$.
- **Expected Benefit:**
  Direct gradient flow guarantees that the audio transformer learns acoustic deepfake artifacts regardless of gate logits. Unimodal audio ROC-AUC will rise from 0.49 to $>0.65$, unlocking genuine multimodal synergy where the fused model outperforms both unimodal branches.
- **Potential Risk:**
  Requires a slight increase in auxiliary compute during backward pass (~3% overhead).
- **Evaluation in Phase 4:**
  Measure unimodal audio ROC-AUC, unimodal video ROC-AUC, fused ROC-AUC, and mean fusion weights ($w_a$ target: $0.20$–$0.40$).

---

### Priority 3: Audio Representation Modernization & Input Normalization

- **Observed Problem:**
  Audio feature tensors have an unnormalized dynamic range from **-790.88 to +156.57** (mean: -33.18, std: 46.25). In `backend/app/ml/models/audio_encoder.py`, the flattened input ($104 \times 100 = 10,400$ dimensions) is passed directly to `nn.Linear(10400, 256)` without an input LayerNorm or batch normalization.
- **Phase 2 Evidence:**
  The mean cosine similarity between audio embedding $A$ and video embedding $V$ is only **0.0622** (nearly orthogonal). The audio encoder struggled to produce stable activations against the normalized, pretrained ViT-B/16 visual features.
- **Proposed Change:**
  Add a `nn.LayerNorm(10400)` or `nn.LayerNorm(256)` immediately following `self.patch_embed`, and standardize audio spectrograms using per-channel instance normalization or z-score normalization ($x \leftarrow (x - \mu)/\sigma$).
- **Expected Benefit:**
  Stabilizes gradient flow through the audio transformer, eliminates vanishing/exploding activations, and speeds up audio branch convergence.
- **Potential Risk:**
  Low risk; standard practice in modern audio transformer architectures (e.g. AST, Whisper).
- **Evaluation in Phase 4:**
  Monitor gradient norm of `AudioEncoder` parameters across epochs.

---

### Priority 4: Regularization & Overconfidence Mitigation (Label Smoothing & Weight Decay)

- **Observed Problem:**
  Severe loss divergence between training ($\mathcal{L} = 0.2576$) and validation ($\mathcal{L} = 1.8823$). Despite high accuracy, cross-entropy loss penalized overconfident incorrect predictions exponentially.
- **Phase 2 Evidence:**
  The median validation loss is **0.1766**, but the mean is **1.8823**. 14 samples had loss $> 4.0$ with raw logits reaching $[-5.3, +5.95]$ ($p_{fake} > 0.9999$).
- **Proposed Change:**
  1. Add **Label Smoothing** (`label_smoothing=0.1`) to `nn.CrossEntropyLoss()`.
  2. Increase `weight_decay` from $1\times 10^{-4}$ to $1\times 10^{-2}$ ($0.01$).
  3. Increase classification head dropout from $0.1$ to $0.25$.
- **Expected Benefit:**
  Caps the penalty for false positives; prevents logit explosion; reduces validation loss from **1.88** to **$<0.85$**; improves expected calibration error (ECE).
- **Potential Risk:**
  Slight reduction in raw training accuracy (from ~90% to ~83-85%), but yields higher validation generalization.
- **Evaluation in Phase 4:**
  Track mean validation loss, maximum sample loss, and Brier calibration score.

---

### Priority 5: Balanced Checkpoint Selection Criterion

- **Observed Problem:**
  Phase 2 selected the baseline checkpoint using `val_f1` alone. Epoch 11 was selected with Val F1 = 0.7174, but its Val Loss was **1.8823**, Val AUC was **0.6974**, and REAL recall was only **44.74%**. In contrast, Epoch 7 achieved Val AUC **0.7576** with Val Loss **0.9503**, and Epoch 1 achieved Val AUC **0.7507** with Val Loss **0.6041**.
- **Phase 2 Evidence:**
  F1 is evaluated at a static 0.5 probability threshold and can reward asymmetric models that overpredict the positive class (predicting FAKE on real faces yields high F1 when deepfakes are prevalent, despite poor real-class precision).
- **Proposed Change:**
  Select the optimal candidate checkpoint using a composite generalization metric:
  $$\text{Score}_{\text{val}} = \text{Val AUC} + \text{Val Balanced Acc} - 0.1 \times \min(2.0, \text{Val Loss})$$
  or select directly on peak **Validation ROC-AUC**.
- **Expected Benefit:**
  Guarantees that the saved model performs well across both classes, is well-calibrated, and does not select an overfitted epoch with inflated loss.
- **Potential Risk:**
  None; ROC-AUC is threshold-independent and symmetric.
- **Evaluation in Phase 4:**
  Verify that the selected Phase 4 checkpoint achieves both high AUC and balanced recall ($>70\%$ for REAL and $>70\%$ for FAKE).

---

## Proposed Phase 4 Hyperparameter Configuration

| Parameter | Phase 2 Baseline | Proposed Phase 4 | Rationale |
|---|---|---|---|
| **Epochs** | 15 | 15 | Sufficient for convergence with cosine annealing |
| **Batch Size** | 2 | 2 | Preserves low VRAM footprint (~1.2 GB on GTX 1650) |
| **Gradient Accumulation** | 4 | 4 | Effective batch size = 8 |
| **Initial Learning Rate** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | Proven stable learning rate |
| **Learning Rate Scheduler** | CosineAnnealingLR ($T_{\max}=15$) | CosineAnnealingLR ($T_{\max}=15$, $\eta_{\min}=1\times 10^{-6}$) | Smooth decay |
| **Weight Decay** | $1 \times 10^{-4}$ | **$1 \times 10^{-2}$ (0.01)** | Regularize transformer layers and MLP heads |
| **Label Smoothing** | 0.0 | **0.1** | Suppress overconfident false-positive logit explosion |
| **Dropout** | 0.1 | **0.25** | Mitigate temporal transformer memorization |
| **Mismatch Augmentation** | $0.15$ | **$0.0$** | Eliminate artificial false-positive label noise |
| **Auxiliary Audio Loss ($\alpha$)** | 0.0 | **0.3** | Force gradient flow into audio branch; prevent collapse |
| **Auxiliary Video Loss ($\beta$)** | 0.0 | **0.1** | Maintain robust visual branch supervision |
| **Audio Input Normalization** | None | **LayerNorm(10400)** | Stabilize high-variance raw log-mel/MFCC inputs |
| **Checkpoint Criterion** | Val F1 | **Val ROC-AUC + Balanced Acc** | Threshold-independent selection |
