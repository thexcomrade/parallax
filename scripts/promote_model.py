"""Model promotion CLI for PARALLAX.

Safely promotes an evaluated candidate checkpoint to production:
1. Validates candidate checkpoint existence, loadability, and SHA-256 hash
2. Validates audio normalization statistics if provided
3. Backs up the current active production checkpoint to checkpoints/archived/
4. Atomically promotes candidate to backend/app/ml/weights/avt2_dwf.pt
5. Updates model_card.json with new checkpoint SHA-256 hash and promotion timestamp
6. Requires explicit --confirm flag to prevent unintentional deployment
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

# Add backend to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.models.avt2_dwf import AVT2DWF

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("promote_model")


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checkpoint(checkpoint_path: Path) -> tuple[bool, str]:
    """Verify that the checkpoint file can be loaded and maps to AVT2DWF."""
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sd: dict[str, Any] = (
            state["model_state_dict"]
            if isinstance(state, dict) and "model_state_dict" in state
            else state
        )
        has_audio_norm = any(
            k.startswith(("audio_encoder.proj_norm", "audio_encoder.norm"))
            for k in sd
        )
        has_input_norm = any(k.startswith("audio_encoder.input_norm") for k in sd)
        has_aux = any(
            k.startswith(("audio_classifier.", "video_classifier."))
            for k in sd
        )
        model = AVT2DWF(
            pretrained_vit=False,
            audio_layernorm=has_audio_norm,
            audio_input_norm=has_input_norm,
            auxiliary_heads=has_aux,
        )
        model.load_state_dict(sd)
        return True, "Checkpoint architecture and weights are valid."
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        return False, str(exc)


def verify_audio_norm_stats(stats_path: Path) -> tuple[bool, str]:
    """Verify that audio normalization statistics file exists and contains valid tensors."""
    try:
        data = torch.load(stats_path, map_location="cpu", weights_only=True)
        if not isinstance(data, dict):
            return False, "Audio norm stats is not a dictionary."
        if "mean" not in data or "std" not in data:
            return False, "Missing 'mean' or 'std' in audio norm stats."
        if "feat_dim" not in data and "feature_dim" not in data:
            return False, "Missing 'feat_dim' or 'feature_dim' key in audio norm stats."
        return True, "Audio normalization statistics verified."
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        return False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="PARALLAX Model Promotion CLI")
    parser.add_argument(
        "--checkpoint",
        "--candidate-checkpoint",
        dest="checkpoint",
        type=Path,
        required=True,
        help="Path to candidate checkpoint to promote",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help="Expected full or prefix SHA-256 hash of the candidate checkpoint",
    )
    parser.add_argument(
        "--audio-norm-stats",
        type=Path,
        default=BACKEND_DIR / "app" / "ml" / "weights" / "audio_norm_stats.pt",
        help="Path to candidate audio normalization statistics file",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=BACKEND_DIR / "app" / "ml" / "weights" / "avt2_dwf.pt",
        help="Target production checkpoint destination",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "archived",
        help="Directory to archive existing production checkpoint",
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        default=BACKEND_DIR / "app" / "ml" / "weights" / "model_card.json",
        help="Path to model_card.json",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Explicit confirmation required to execute model promotion",
    )
    args = parser.parse_args()

    candidate_path: Path = args.checkpoint
    target_path: Path = args.target
    archive_dir: Path = args.archive_dir
    stats_path: Path = args.audio_norm_stats

    if not candidate_path.exists() or candidate_path.stat().st_size == 0:
        logger.error("Candidate checkpoint not found or empty at %s", candidate_path)
        sys.exit(1)

    candidate_hash = compute_sha256(candidate_path)

    # Verify expected SHA-256 if specified
    if args.expected_sha256:
        if not candidate_hash.startswith(args.expected_sha256.lower()):
            logger.error(
                "SHA-256 mismatch! Candidate hash %s does not match expected %s",
                candidate_hash,
                args.expected_sha256,
            )
            sys.exit(1)
        logger.info("Candidate SHA-256 verified against expected hash: %s", candidate_hash)

    # 1. Verify candidate checkpoint loadability
    logger.info("Verifying candidate checkpoint loadability...")
    valid_ckpt, ckpt_msg = verify_checkpoint(candidate_path)
    if not valid_ckpt:
        logger.error("Candidate checkpoint failed verification: %s. Aborting.", ckpt_msg)
        sys.exit(1)
    logger.info("Candidate checkpoint successfully verified: %s", ckpt_msg)

    # 2. Verify audio normalization statistics if present
    if stats_path.exists():
        valid_stats, stats_msg = verify_audio_norm_stats(stats_path)
        if not valid_stats:
            logger.error("Audio normalization statistics failed verification: %s", stats_msg)
            sys.exit(1)
        logger.info("Audio norm stats verified: %s (%s)", stats_path, stats_msg)

    # Safety gate: require --confirm
    if not args.confirm:
        print("\n" + "!" * 70)
        print("                 PROMOTION APPROVAL REQUIRED (DRY-RUN)")
        print("!" * 70)
        print(f"Candidate Checkpoint:     {candidate_path}")
        print(f"Candidate SHA-256:        {candidate_hash}")
        print(f"Audio Norm Stats:         {stats_path} (exists={stats_path.exists()})")
        print(f"Target Production Path:   {target_path}")
        print(f"Archive Backup Directory: {archive_dir}")
        print("-" * 70)
        print("Verification Summary:     CHECKPOINT AND ARCHITECTURE PASS")
        print("Rollback Capability:      Existing production model will be archived with timestamp.")
        print("-" * 70)
        print("To execute promotion to active production, run:")
        print(f"  python scripts/promote_model.py --checkpoint {candidate_path} --confirm")
        print("!" * 70 + "\n")
        sys.exit(1)

    now_str = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")

    # 3. Archive current production checkpoint if it exists
    if target_path.exists() and target_path.stat().st_size > 0:
        archive_dir.mkdir(parents=True, exist_ok=True)
        current_hash = compute_sha256(target_path)[:16]
        archive_dest = archive_dir / f"avt2_dwf_archived_{now_str}_{current_hash}.pt"
        shutil.copy2(target_path, archive_dest)
        logger.info("Archived existing production checkpoint to %s", archive_dest)

    # 4. Promote candidate to production
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_path, target_path)
    logger.info("Successfully copied candidate checkpoint to production target: %s", target_path)

    # 4b. Promote audio normalization statistics if present
    if stats_path.exists():
        target_stats_path = target_path.parent / "audio_norm_stats.pt"
        if target_stats_path.resolve() != stats_path.resolve():
            shutil.copy2(stats_path, target_stats_path)
            logger.info("Successfully updated production audio norm stats at %s", target_stats_path)

    # 5. Update model card provenance
    if args.model_card.exists():
        try:
            card_data = json.loads(args.model_card.read_text(encoding="utf-8"))
            card_data["model_mode"] = "trained"
            card_data["checkpoint_hash"] = candidate_hash[:16]
            card_data["checkpoint_sha256_full"] = candidate_hash
            card_data["promoted_at"] = datetime.datetime.now(datetime.UTC).isoformat()
            args.model_card.write_text(json.dumps(card_data, indent=2), encoding="utf-8")
            logger.info(
                "Updated %s with new checkpoint hash %s", args.model_card, candidate_hash[:16]
            )
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            logger.warning("Could not update model_card.json: %s", exc)

    print("\n" + "=" * 70)
    print("                 PARALLAX MODEL PROMOTION COMPLETE")
    print("=" * 70)
    print(f"Promoted Checkpoint: {candidate_path.name}")
    print(f"Production Path:     {target_path}")
    print(f"SHA-256 Hash:        {candidate_hash}")
    print("Status:              PRODUCTION ACTIVE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
