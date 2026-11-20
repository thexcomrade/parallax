"""Verify training resume setup without training."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

exp_dir = REPO_ROOT / "experiments" / "phase4_4_clean_avt2_dwf"
resume_ckpt_path = exp_dir / "latest_checkpoint.pt"
output_ckpt_path = exp_dir / "best_checkpoint.pt"

ckpt = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
completed_epoch = int(ckpt.get("epoch", 0))
start_epoch = completed_epoch + 1

best_ckpt = torch.load(output_ckpt_path, map_location="cpu", weights_only=False)
best_val_auc = float(best_ckpt.get("val_auc", -1.0))
best_epoch = int(best_ckpt.get("epoch", 1))
stagnant_epochs = max(0, completed_epoch - best_epoch)

history = []
exp_hist = exp_dir / "training_history.csv"
if exp_hist.exists():
    with open(exp_hist, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if int(r["epoch"]) <= completed_epoch:
                history.append({k: (float(v) if k != "epoch" else int(v)) for k, v in r.items()})

print(f"Resume Checkpoint: {resume_ckpt_path.name}")
print(f"Completed Epoch: {completed_epoch}")
print(f"Start Epoch: {start_epoch}")
print(f"Best Epoch: {best_epoch}")
print(f"Best Val AUC: {best_val_auc:.4f}")
print(f"Stagnant Epochs: {stagnant_epochs}")
print(f"Preloaded History Entries: {len(history)}")
assert start_epoch == 4, f"Expected start_epoch=4, got {start_epoch}"
assert best_epoch == 2, f"Expected best_epoch=2, got {best_epoch}"
assert stagnant_epochs == 1, f"Expected stagnant_epochs=1, got {stagnant_epochs}"
assert len(history) == 3, f"Expected history len=3, got {len(history)}"
print("VERIFICATION SUCCESSFUL: Resume setup is 100% consistent!")
