"""Audit Phase 4.4 clean AVT²-DWF training experiment checkpoints."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.models.avt2_dwf import AVT2DWF

exp_dir = REPO_ROOT / "experiments" / "phase4_4_clean_avt2_dwf"

def audit_file(p: Path) -> dict:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    sha256 = h.hexdigest()
    
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    keys = list(ckpt.keys())
    
    # Model architecture verification
    cfg = ckpt.get("config", {})
    training_cfg = cfg.get("training", {})
    
    aux_audio = training_cfg.get("aux_audio_weight", 0.3)
    aux_video = training_cfg.get("aux_video_weight", 0.1)
    auxiliary_heads = (aux_audio > 0.0 or aux_video > 0.0)
    
    model = AVT2DWF(
        pretrained_vit=False,
        audio_layernorm=training_cfg.get("audio_layernorm", True),
        audio_input_norm=training_cfg.get("audio_input_norm", True),
        auxiliary_heads=auxiliary_heads,
        classifier_dropout=training_cfg.get("classifier_dropout", 0.25),
    )
    
    state_dict = ckpt.get("model_state_dict", {})
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    opt_state = ckpt.get("optimizer_state_dict")
    sched_state = ckpt.get("scheduler_state_dict")
    scaler_state = ckpt.get("scaler_state_dict")
    
    return {
        "filename": p.name,
        "size_bytes": p.stat().st_size,
        "sha256": sha256,
        "keys": keys,
        "epoch": ckpt.get("epoch"),
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_auc": ckpt.get("best_val_auc"),
        "val_auc": ckpt.get("val_auc"),
        "val_loss": ckpt.get("val_loss"),
        "has_optimizer": opt_state is not None and len(opt_state) > 0,
        "optimizer_param_groups": len(opt_state.get("param_groups", [])) if opt_state else 0,
        "has_scheduler": sched_state is not None,
        "scheduler_last_epoch": sched_state.get("last_epoch") if sched_state else None,
        "has_scaler": scaler_state is not None,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "training_cfg": training_cfg,
    }

print("=== Phase 4.4 Checkpoint Audit ===")
for name in ["best_checkpoint.pt", "latest_checkpoint.pt"]:
    p = exp_dir / name
    if p.exists():
        info = audit_file(p)
        print(json.dumps(info, indent=2))
    else:
        print(f"{name}: NOT FOUND")
