# PARALLAX — PHASE 3: TRAINING PERFORMANCE ANALYSIS & MODEL IMPROVEMENT ARCHITECTURE

## 1. Executive Summary

| Parameter | Value |
|---|---|
| **Phase Scope** | Phase 3: In-Depth Diagnostic Analysis & Evidence-Based Improvement Planning |
| **Analyzed Baseline Checkpoint** | `checkpoints/baseline_avt2_dwf.pt` (Epoch 11, Size: 442.29 MB) |
| **Analyzed Rolling Checkpoint** | `checkpoints/latest_avt2_dwf.pt` (Epoch 15, Size: 442.29 MB) |
| **Restoration Reproducibility** | **100% Exact Match** across all recorded metrics |
| **Overfitting Status** | **CONFIRMED** (Loss gap: +1.5834; Accuracy gap: +23.35%; F1 gap: +0.2037) |
| **Modality Health** | **COLLAPSED TO VIDEO** ($w_v = 0.9962 \pm 0.004$, $w_a = 0.0038 \pm 0.004$; Audio AUC: 0.4972) |
| **Primary Error Mode** | **Asymmetric False Positives** (21/38 REAL videos predicted as FAKE due to mismatch augmentation noise) |
| **Production Weights Status** | `backend/app/ml/weights/avt2_dwf.pt` **UNTOUCHED** (SHA-256 verified) |
| **Test Set Status** | `datasets/manifests/test.csv` **UNTOUCHED / UNACCESSED** |

---

## 2. Phase 2 Baseline Performance Audit

### 2.1 Complete 15-Epoch Trajectory Comparison

```
Epoch   Train Loss   Train Acc    Train F1    Val Loss    Val Acc     Val F1     Val AUC      Learning Rate
-------------------------------------------------------------------------------------------------------
  01      0.7659      58.57%       0.6234      0.6041     68.42%      0.6842      0.7507       1.00e-04
  02      0.6036      67.43%       0.6935      1.2575     56.58%      0.6598      0.6648       9.89e-05
  03      0.4902      76.57%       0.7980      0.8177     64.47%      0.6897      0.7119       9.57e-05
  04      0.4476      79.43%       0.8209      1.2508     63.16%      0.7021      0.7161       9.05e-05
  05      0.5241      73.71%       0.7723      0.7187     64.47%      0.6197      0.7126       8.35e-05
  06      0.4990      79.43%       0.8105      1.1885     57.89%      0.6735      0.6898       7.50e-05
  07      0.3392      85.71%       0.8737      0.9503     67.11%      0.6835      0.7576 (★)   6.55e-05
  08      0.3749      84.00%       0.8534      1.3834     61.84%      0.6742      0.7022       5.52e-05
  09      0.3938      82.86%       0.8276      1.2839     68.42%      0.7073      0.6918       4.48e-05
  10      0.3595      88.57%       0.8964      1.4744     64.47%      0.6667      0.6967       3.45e-05
  11(🏆)  0.3590      86.00%       0.8721      1.8823     65.79%      0.7174 (★)  0.6974       2.50e-05
  12      0.3032      90.29%       0.9124      1.9245     63.16%      0.6818      0.6676       1.65e-05
  13      0.3238      88.86%       0.9003      1.6916     63.16%      0.6744      0.6759       9.55e-06
  14      0.2475      90.00%       0.9176      1.8310     67.11%      0.7126      0.6717       4.32e-06
  15      0.2576      89.14%       0.9082      1.8410     65.79%      0.7045      0.6745       1.09e-06
```

### 2.2 Key Epoch Milestones
- **Epoch 1 (Optimal Calibration):** Val loss was at its global minimum (**0.6041**) with balanced accuracy (**68.42%**) and high AUC (**0.7507**).
- **Epoch 7 (Peak Discrimination):** Val AUC achieved its highest value of the training run (**0.7576**), while val loss remained restrained at **0.9503** and training accuracy reached **85.71%**.
- **Epoch 11 (Baseline Checkpoint):** Val F1 peaked at **0.7174**, with fake recall reaching **86.84%**, but val loss had escalated to **1.8823** and val AUC dropped to **0.6974**.
- **Epochs 12–15 (Overfitting Plateau):** Training loss plummeted to **0.2475** (train accuracy reached **90.29%**), but validation accuracy remained frozen around 63–67% and val loss hovered between **1.69** and **1.92**.

---

## 3. Diagnostic Findings & Root Cause Analysis

### 3.1 Overfitting Diagnosis: CONFIRMED
Overfitting is confirmed by five distinct quantitative indicators:
1. **Loss Divergence:** $\mathcal{L}_{\text{train}}$ declined by 66.4% ($0.7659 \to 0.2576$), while $\mathcal{L}_{\text{val}}$ expanded by +204.8% ($0.6041 \to 1.8410$). Generalization loss gap: $\Delta \mathcal{L} = +1.5834$.
2. **Accuracy Divergence:** Train accuracy improved by +30.57% ($58.57\% \to 89.14\%$), whereas validation accuracy plateaued ($68.42\% \to 65.79\%$). Generalization accuracy gap: $\Delta \text{Acc} = 23.35\%$.
3. **F1 Divergence:** Train F1 increased from 0.6234 to 0.9082, while validation F1 stagnated between 0.67 and 0.71.
4. **AUC Degradation:** Validation ROC-AUC decayed from a peak of **0.7576** at Epoch 7 down to **0.6745** at Epoch 15 (-8.31 percentage points).
5. **Loss Skew & Outlier Penalty:** 
   - Median validation sample loss is **0.1766**, but mean validation loss is **1.8823** ($10.7\times$ higher).
   - 21 out of 76 samples had loss $> 2.0$, and 14 samples had loss $> 4.0$ (up to 11.25).
   - Overconfident, unregularized logits on misclassified samples exponentially inflated the cross-entropy loss.

---

### 3.2 Modality Collapse & Dynamic Weight Fusion Breakdown

A diagnostic unimodal ablation was executed on the baseline checkpoint (`checkpoints/baseline_avt2_dwf.pt`) evaluated on the 76 validation samples:

| Model Configuration | Accuracy | F1-Score | Recall (Fake) | Precision (Fake) | ROC-AUC |
|---|---|---|---|---|---|
| **Full Multimodal Model** | **65.79%** | **0.7174** | **86.84%** | **61.11%** | **0.6974** |
| **Unimodal Video Branch ($w_v=1, w_a=0$)** | **65.79%** | **0.7174** | **86.84%** | **61.11%** | **0.6974** |
| **Unimodal Audio Branch ($w_v=0, w_a=1$)** | **51.32%** | **0.6726** | **94.74%** | **50.70%** | **0.4972** |

#### Empirical Conclusions:
1. **The Full Model is 100% Identical to the Video Branch:**
   The full multimodal model's validation predictions and performance metrics are exact clones of the unimodal video branch.
2. **The Audio Branch Learned Nothing:**
   Unimodal audio ROC-AUC is **0.4972**, which is indistinguishable from a random coin toss.
3. **Fusion Weight Collapse:**
   The learned gate weights across the validation set are:
   - $w_v = 0.9962 \pm 0.0036$ (99.62% video weight)
   - $w_a = 0.0038 \pm 0.0036$ (0.38% audio weight)
4. **Architectural Cause:**
   - Visual branch starts with a strong, pretrained ViT-B/16 backbone (86M frozen parameters).
   - Audio branch has a randomly initialized, dense `nn.Linear(10400, 256)` projection on unnormalized raw log-mel/MFCC inputs (range: -790 to +156).
   - The gate MLP discovered in early iterations that routing 99.6% weight to the video branch yielded an immediate loss reduction.
   - Once $w_a \to 0$, backpropagation gradients into the audio encoder were extinguished ($\nabla_A \mathcal{L} = w_a \nabla_F \mathcal{L} \approx 0$), permanently freezing the audio branch in an untrained state.

---

### 3.3 The False-Positive Mechanism: Cross-Modal Mismatch Label Noise

The validation confusion matrix for the baseline checkpoint reveals severe class asymmetry:

```
                  Predicted REAL (0)    Predicted FAKE (1)     Class Recall
Actual REAL (0):          17                    21                44.74%
Actual FAKE (1):           5                    33                86.84%
```

- **False Positives:** 21 out of 38 genuine REAL videos were misclassified as deepfakes.
- **Root Cause Identified in Code:**
  In `backend/training/augment.py`, lines 109–112:
  ```python
  if random.random() < self.mismatch_prob and audio_segments.shape[0] > 1:
      rolled_audio = torch.roll(audio_segments, shifts=1, dims=0)
      return video_windows, rolled_audio, 1, True
  ```
  In every epoch, 15% of REAL training samples had their audio rolled, and their ground-truth labels were **forcibly converted to 1 (FAKE)**.
  Because the model's fusion gate had collapsed to video ($w_v \approx 1$), the model could not detect that the audio was desynchronized. All the model observed were **100% authentic human facial features** paired with the target label `1 (FAKE)`.
  The model was penalized during training unless it flagged genuine human faces as deepfakes. This directly created the 21 false positives on the validation set.

---

## 4. Prioritized Improvement Plan for Phase 4

```
Priority   Improvement Title                   Root Problem Addressed                      Expected Impact
--------------------------------------------------------------------------------------------------------------------
   1       Disable Mismatch Label Flipping     15% label noise on authentic faces          REAL Recall: 44% -> >75%
   2       Auxiliary Multi-Task Audio Loss     DWF gate starvation; audio AUC=0.497        Audio AUC: 0.49 -> >0.65
   3       Audio LayerNorm & Normalization     Raw [-790, +156] feature variance           Stable audio gradient flow
   4       Label Smoothing & Weight Decay      Logit explosion (loss up to 11.25)          Val Loss: 1.88 -> <0.85
   5       Balanced Checkpoint Metric          Val F1 overpredicting positive class        Selects calibrated peak AUC
```

### Detailed Improvement Specifications

#### Improvement 1: Disable Cross-Modal Mismatch Label Flipping
- **Implementation:** Set `mismatch_prob = 0.0` in `MultimodalAugmenter` for primary classification training.
- **Mathematical Justification:** In standard binary classification, $\mathcal{L}_{\text{BCE}} = -y \log \hat{y} - (1-y) \log (1-\hat{y})$. Flipping $y \leftarrow 1$ for samples where $V$ is completely real introduces label noise $\eta = 0.15 \times 0.5 = 0.075$, which is well above the robustness threshold of a standard cross-entropy loss without noise-robust estimators.
- **Validation Target:** Reduce validation False Positives from 21 down to $<8$.

#### Improvement 2: Multi-Task Auxiliary Supervision for Audio
- **Implementation:** Add auxiliary classification heads for both modalities and train with a combined objective:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fused}} + \alpha \mathcal{L}_{\text{audio}} + \beta \mathcal{L}_{\text{video}}$$
  where $\alpha = 0.3$ and $\beta = 0.1$.
- **Mathematical Justification:** With auxiliary loss, the gradient received by the audio encoder is:
  $$\nabla_{\theta_A} \mathcal{L}_{\text{total}} = w_a \cdot \frac{\partial \mathcal{L}_{\text{fused}}}{\partial A} \frac{\partial A}{\partial \theta_A} + \alpha \cdot \frac{\partial \mathcal{L}_{\text{audio}}}{\partial A} \frac{\partial A}{\partial \theta_A}$$
  Even if $w_a \to 0$, the second term guarantees a strong, uninterrupted gradient signal of magnitude $\alpha$, completely preventing modality starvation.
- **Validation Target:** Unimodal audio ROC-AUC $> 0.65$; learned fusion weight $w_a$ in $[0.20, 0.40]$.

#### Improvement 3: Audio Input LayerNorm
- **Implementation:** Insert `nn.LayerNorm(256)` immediately following the `nn.Linear(10400, 256)` patch projection in `AudioEncoder`, ensuring that audio token representations match the normalized scale of the ViT-B/16 visual tokens.
- **Validation Target:** Audio embedding norm stabilizes near visual embedding norm ($\|A\| \approx \|V\| \approx 15.0$).

#### Improvement 4: Regularization via Label Smoothing & Weight Decay
- **Implementation:** 
  - CrossEntropyLoss with `label_smoothing=0.1`.
  - AdamW weight decay increased from $1\times 10^{-4}$ to $1\times 10^{-2}$ ($0.01$).
  - Classifier dropout increased from 0.1 to 0.25.
- **Mathematical Justification:** Label smoothing replaces one-hot targets with $y_k = (1 - \epsilon) y_k + \epsilon / K$. This bounds the maximum loss of any incorrect prediction to $-\log(\epsilon / 2) = -\log(0.05) \approx 2.99$, completely eliminating the 9.0–11.25 loss penalties that inflated Phase 2 validation loss.
- **Validation Target:** Mean validation loss $< 0.85$.

#### Improvement 5: Composite Balanced Checkpoint Selection
- **Implementation:** Select the candidate checkpoint using:
  $$\text{Score}_{\text{val}} = \text{Val AUC} + \text{Val Balanced Acc} - 0.1 \times \min(2.0, \text{Val Loss})$$
- **Validation Target:** Selects an epoch that balances REAL recall ($>70\%$) and FAKE recall ($>70\%$) rather than overfitting on asymmetric positive-class F1.

---

## 5. Proposed Phase 4 Hyperparameter Configuration

```yaml
model:
  d_model: 256
  pretrained_vit: true
  freeze_vit: true
  audio_layernorm: true
  classifier_dropout: 0.25
  auxiliary_heads: true

training:
  epochs: 15
  batch_size: 2
  grad_accum_steps: 4
  effective_batch_size: 8
  initial_lr: 0.0001
  min_lr: 0.000001
  weight_decay: 0.01
  label_smoothing: 0.1
  aux_audio_weight: 0.3
  aux_video_weight: 0.1
  scheduler: CosineAnnealingLR
  grad_clipping: 1.0
  device: cuda:0

data:
  mismatch_prob: 0.0
  flip_prob: 0.5
  color_jitter_prob: 0.3
  audio_noise_prob: 0.3
  audio_masking_prob: 0.3
  split_train: datasets/manifests/train.csv
  split_validation: datasets/manifests/validation.csv
```

---

## 6. Deliverables & Checkpoint Safety Confirmation

1. **Analysis JSON:** `reports/phase3_training_analysis.json`
2. **Improvement Markdown:** `reports/phase3_improvement_recommendations.md`
3. **Master Technical Documentation:** `docs/PHASE3_TRAINING_ANALYSIS.md`
4. **Checkpoint Integrity:**
   - `checkpoints/baseline_avt2_dwf.pt` (Epoch 11, untouched, validated)
   - `checkpoints/latest_avt2_dwf.pt` (Epoch 15, untouched, validated)
   - `backend/app/ml/weights/avt2_dwf.pt` (SHA-256: `6de2e663799ee8687c2ca06bbb4309034937c8fbf7a18ba3166ce745052ef354`, 100% verified untouched)
   - `datasets/manifests/test.csv` (74 samples, strictly unaccessed)

*Phase 3 is complete. Execution has stopped and is waiting for review before Phase 4.*
