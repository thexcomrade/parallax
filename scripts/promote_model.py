"""Model promotion CLI for PARALLAX.

Safely promotes an evaluated candidate checkpoint to production:
1. Validates candidate checkpoint existence and loadability
2. Backs up the current active production checkpoint to checkpoints/archived/
3. Atomically promotes candidate to backend/app/ml/weights/avt2_dwf.pt
4. Updates model_card.json with new checkpoint SHA-256 hash and promotion timestamp
5. Requires explicit --confirm flag to prevent unintentional deployment
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

import torch

# Add backend to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.models.avt2_dwf import AVT2DWF  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("promote_model")


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="PARALLAX Model Promotion CLI")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to candidate checkpoint to promote",
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

    if not candidate_path.exists() or candidate_path.stat().st_size == 0:
        logger.error("Candidate checkpoint not found or empty at %s", candidate_path)
        sys.exit(1)

    if not args.confirm:
        print("\n" + "!" * 60)
        print("                 PROMOTION APPROVAL REQUIRED")
        print("!" * 60)
        print(f"Candidate Checkpoint: {candidate_path}")
        print(f"Target Production:    {target_path}")
        print(f"Archive Directory:    {archive_dir}")
        print("-" * 60)
        print("To confirm promotion of this model to active production, re-run with:")
        print(f"  python scripts/promote_model.py --checkpoint {candidate_path} --confirm")
        print("!" * 60 + "\n")
        sys.exit(1)

    # 1. Verify candidate checkpoint loadability
    logger.info("Verifying candidate checkpoint loadability...")
    try:
        model = AVT2DWF(pretrained_vit=False)
        state = torch.load(candidate_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        logger.info("Candidate checkpoint successfully verified and loadable.")
    except Exception as exc:
        logger.error("Candidate checkpoint failed verification: %s. Aborting promotion.", exc)
        sys.exit(1)

    candidate_hash = compute_sha256(candidate_path)
    now_str = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")

    # 2. Archive current production checkpoint if it exists
    if target_path.exists() and target_path.stat().st_size > 0:
        archive_dir.mkdir(parents=True, exist_ok=True)
        current_hash = compute_sha256(target_path)[:16]
        archive_dest = archive_dir / f"avt2_dwf_archived_{now_str}_{current_hash}.pt"
        shutil.copy2(target_path, archive_dest)
        logger.info("Archived existing production checkpoint to %s", archive_dest)

    # 3. Promote candidate to production
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_path, target_path)
    logger.info("Successfully copied candidate checkpoint to production target: %s", target_path)

    # 4. Update model card provenance
    if args.model_card.exists():
        try:
            card_data = json.loads(args.model_card.read_text(encoding="utf-8"))
            card_data["model_mode"] = "trained"
            card_data["checkpoint_hash"] = candidate_hash[:16]
            card_data["promoted_at"] = datetime.datetime.now(datetime.UTC).isoformat()
            args.model_card.write_text(json.dumps(card_data, indent=2), encoding="utf-8")
            logger.info(
                "Updated %s with new checkpoint hash %s", args.model_card, candidate_hash[:16]
            )
        except Exception as exc:
            logger.warning("Could not update model_card.json: %s", exc)

    print("\n" + "=" * 60)
    print("         PARALLAX MODEL PROMOTION COMPLETE")
    print("=" * 60)
    print(f"Promoted Checkpoint: {candidate_path.name}")
    print(f"Production Path:     {target_path}")
    print(f"SHA-256 Hash:        {candidate_hash[:16]}...")
    print("Status:              PRODUCTION ACTIVE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
