import os
import sys
import argparse
import json
from pathlib import Path

import torch

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.dataloader import create_train_val_dataloaders
from app.full_model import IntentAwareTrajectoryModel
from app.loss import WTALoss
from app.metrics import (
    compute_ade,
    compute_fde,
    compute_min_ade_k,
    compute_min_fde_k,
    compute_collision_prob,
)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def select_top_mode_trajectory(
    trajectories: torch.Tensor,
    mode_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Selects the predicted trajectory corresponding to the mode with max logit.
    trajectories: [B, K, T_future, 2]
    mode_logits:   [B, K]

    Returns:
        best_pred: [B, T_future, 2]
    """
    batch_size = trajectories.size(0)
    best_mode = torch.argmax(mode_logits, dim=1)  # [B]
    batch_idx = torch.arange(batch_size, device=trajectories.device)
    best_pred = trajectories[batch_idx, best_mode]  # [B, T_future, 2]
    return best_pred


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained trajectory prediction model"
    )
    parser.add_argument(
        "--dataroot", type=str, required=True, help="Path to nuScenes root"
    )
    parser.add_argument(
        "--version", type=str, default="v1.0-mini", help="nuScenes version"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Checkpoint file path",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save the evaluation summary as JSON",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")

    # Architecture config (explicit defaults matching standardized project settings)
    parser.add_argument(
        "--embed_dim", type=int, default=128, help="Shared embedding dimension"
    )
    parser.add_argument(
        "--num_modes", type=int, default=3, help="Number of trajectory modes"
    )
    parser.add_argument("--future_steps", type=int, default=6, help="Future timesteps")
    parser.add_argument("--t_past", type=int, default=4, help="Past timesteps")
    parser.add_argument(
        "--max_neighbors", type=int, default=10, help="Max neighbors per agent"
    )

    args = parser.parse_args()

    device = get_device()

    # --- Startup Summary ---
    print("=" * 60)
    print("      Evaluating Trajectory Prediction Model      ")
    print("=" * 60)
    print(f"Device:         {device}")
    if torch.cuda.is_available():
        print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Embed Dim:      {args.embed_dim}")
    print(f"Num Modes:      {args.num_modes}")
    print(f"Future Steps:   {args.future_steps}")
    print("=" * 60)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint file not found at '{args.checkpoint}'.")
        print("Please provide a valid path using --checkpoint.")
        sys.exit(1)

    # --- Dataloader ---
    print("\nLoading dataset (using validation split)...")
    _, val_loader = create_train_val_dataloaders(
        dataroot=args.dataroot,
        version=args.version,
        train_split="train",  # Not effectively used but required for the builder signature
        val_split="val",
        t_past=args.t_past,
        t_future=args.future_steps,
        max_neighbors=args.max_neighbors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Validation samples: {len(val_loader.dataset)}")

    # --- Model Setup ---
    model = IntentAwareTrajectoryModel(
        embed_dim=args.embed_dim,
        num_modes=args.num_modes,
        future_steps=args.future_steps,
    ).to(device)

    print(f"\nLoading weights from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    # Eval mode is CRITICAL for inference
    model.eval()

    criterion = WTALoss(alpha=1.0)

    # --- Evaluation Loop ---
    print("\nRunning evaluation on validation set...")
    total_loss = 0.0
    num_batches = 0

    ade_values = []
    fde_values = []
    minade_values = []
    minfde_values = []
    collision_values = []

    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch_to_device(batch, device)

            agent = batch["agent"]
            neighbors = batch["neighbors"]
            map_img = batch["map"]
            target = batch["target"]

            # Forward pass
            trajectories, mode_logits = model(
                agent=agent,
                neighbors=neighbors,
                map_img=map_img,
            )

            # Compute validation WTALoss
            loss, _ = criterion(
                pred_trajectories=trajectories,
                mode_logits=mode_logits,
                target_trajectories=target,
            )
            total_loss += loss.item()
            num_batches += 1

            # Extract top mode for deterministic ADE/FDE metrics
            top_pred = select_top_mode_trajectory(trajectories, mode_logits)

            ade_values.append(compute_ade(top_pred, target))
            fde_values.append(compute_fde(top_pred, target))
            minade_values.append(compute_min_ade_k(trajectories, target))
            minfde_values.append(compute_min_fde_k(trajectories, target))

            # Collision probability check
            try:
                collision = compute_collision_prob(top_pred, neighbors)
                collision_values.append(collision)
            except Exception:
                # Silently catch dimension or calculation bounds errors to avoid failing full evaluation
                pass

    # --- Metrics Aggregation ---
    avg_loss = total_loss / max(num_batches, 1)
    avg_ade = sum(ade_values) / max(len(ade_values), 1)
    avg_fde = sum(fde_values) / max(len(fde_values), 1)
    avg_minade = sum(minade_values) / max(len(minade_values), 1)
    avg_minfde = sum(minfde_values) / max(len(minfde_values), 1)

    avg_collision = None
    if len(collision_values) > 0:
        avg_collision = sum(collision_values) / len(collision_values)

    # --- Final Results Summary ---
    print("\n" + "=" * 40)
    print("           Evaluation Results           ")
    print("=" * 40)
    print(f"Val Loss       : {float(avg_loss):.4f}")
    print(f"ADE            : {float(avg_ade):.4f}")
    print(f"FDE            : {float(avg_fde):.4f}")
    print(f"MinADE@K       : {float(avg_minade):.4f}")
    print(f"MinFDE@K       : {float(avg_minfde):.4f}")

    if avg_collision is not None:
        print(f"Collision Prob : {float(avg_collision):.4f}")
    else:
        print("Collision Prob : N/A (Skipped)")
    print("=" * 40)

    if args.output_json:
        output_data = {
            "model_config": {
                "embed_dim": args.embed_dim,
                "num_modes": args.num_modes,
                "future_steps": args.future_steps,
            },
            "checkpoint": args.checkpoint,
            "dataset": {
                "dataroot": args.dataroot,
                "version": args.version,
            },
            "evaluation": {
                "val_loss": float(avg_loss),
                "ade": float(avg_ade),
                "fde": float(avg_fde),
                "minade_k": float(avg_minade),
                "minfde_k": float(avg_minfde),
                "collision_prob": (
                    float(avg_collision) if avg_collision is not None else None
                ),
            },
        }

        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Evaluations successfully saved to {json_path}")


if __name__ == "__main__":
    main()
