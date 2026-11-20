"""Final verification script for Phase 4.4 clean AVT²-DWF trained checkpoint."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.models.avt2_dwf import AVT2DWF

exp_dir = REPO_ROOT / "experiments" / "phase4_4_clean_avt2_dwf"
ckpt_path = exp_dir / "best_checkpoint.pt"

print(f"=== Verifying Best Checkpoint: {ckpt_path.name} ===")

# 1. SHA-256
h = hashlib.sha256()
with open(ckpt_path, "rb") as f:
    while chunk := f.read(1024 * 1024):
        h.update(chunk)
sha256 = h.hexdigest()
size_bytes = ckpt_path.stat().st_size
print(f"File Size: {size_bytes:,} bytes")
print(f"SHA-256:   {sha256}")

# 2. Load Checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
epoch = ckpt.get("epoch")
val_auc = ckpt.get("val_auc")
val_bacc = ckpt.get("val_balanced_acc")
val_f1 = ckpt.get("val_f1")
val_loss = ckpt.get("val_loss")
print(f"Best Epoch:           {epoch}")
print(f"Validation ROC-AUC:   {val_auc:.4f}")
print(f"Validation BAcc:      {val_bacc:.4f}")
print(f"Validation F1:        {val_f1:.4f}")
print(f"Validation Loss:      {val_loss:.4f}")

# 3. Model Architecture & Weight Loading
cfg = ckpt.get("config", {})
training_cfg = cfg.get("training", {})
model = AVT2DWF(
    pretrained_vit=False,
    audio_layernorm=training_cfg.get("audio_layernorm", True),
    audio_input_norm=training_cfg.get("audio_input_norm", True),
    auxiliary_heads=True,
    classifier_dropout=training_cfg.get("classifier_dropout", 0.25),
)
missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
print(f"State Dict Loaded:    Strict match (missing={len(missing)}, unexpected={len(unexpected)})")

# 4. Check for NaN or Inf in all parameters
total_params = 0
nan_count = 0
inf_count = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if torch.isnan(param).any():
        nan_count += 1
        print(f"  [ERROR] NaN detected in parameter: {name}")
    if torch.isinf(param).any():
        inf_count += 1
        print(f"  [ERROR] Inf detected in parameter: {name}")

assert nan_count == 0, f"Found {nan_count} parameters with NaN"
assert inf_count == 0, f"Found {inf_count} parameters with Inf"
print(f"Parameter Sanity:     PASS — {total_params:,} parameters checked (0 NaN, 0 Inf)")

# 5. Inference Smoke Test
model.eval()
with torch.no_grad():
    # Synthetic batch: B=1, K=3 windows
    dummy_video = torch.randn(1, 3, 16, 3, 224, 224)
    dummy_audio = torch.randn(1, 3, 104, 100)
    dummy_mask = torch.ones(3)
    
    out = model(dummy_video, dummy_audio, dummy_mask)
    probs = torch.softmax(out.logits, dim=1)
    
    print("Inference Smoke Test: PASS")
    print(f"  Logits shape:      {out.logits.shape}")
    print(f"  Logits:            {out.logits.tolist()}")
    print(f"  Probabilities:     {probs.tolist()}")
    print(f"  P(fake):           {probs[0, 1].item():.4f}")
    print(f"  w_a:               {out.w_a.item():.4f}")
    print(f"  w_v:               {out.w_v.item():.4f}")
    print(f"  sync_score:        {out.sync_score.item():.4f}")
    
assert torch.isfinite(out.logits).all()
assert torch.isfinite(probs).all()
assert (probs.sum(dim=1) - 1.0).abs() < 1e-4

print("=== ALL VERIFICATION CHECKS PASSED ===")
