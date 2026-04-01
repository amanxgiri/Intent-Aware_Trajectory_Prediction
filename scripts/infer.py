import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

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


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single sample")
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

    # Optional JSON output
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Path to save inference output to JSON",
    )

    # Architecture config (explicit defaults matching standardized project settings)
    parser.add_argument(
        "--embed_dim", type=int, default=128, help="Shared embedding dimension"
    )
    parser.add_argument(
        "--num_modes", type=int, default=3, help="Number of trajectory modes"
    )
    parser.add_argument("--future_steps", type=int, default=6, help="Future timesteps")

    # Dataloader specifics
    parser.add_argument("--t_past", type=int, default=4, help="Past timesteps")
    parser.add_argument(
        "--max_neighbors", type=int, default=10, help="Max neighbors per agent"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference (default 1)"
    )
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers")

    args = parser.parse_args()

    device = get_device()

    # --- Startup Summary ---
    print("=" * 60)
    print("      Trajectory Prediction Inference Mode        ")
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
        train_split="train",  # Dataloader requirement
        val_split="val",
        t_past=args.t_past,
        t_future=args.future_steps,
        max_neighbors=args.max_neighbors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

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

    # --- Inference Execution ---
    print("\nExtracting single batch for inference...")

    # Grab the very first batch
    try:
        batch = next(iter(val_loader))
    except StopIteration:
        print(
            "\nERROR: The validation dataloader is empty. No samples to run inference on."
        )
        sys.exit(1)

    batch = move_batch_to_device(batch, device)

    agent = batch["agent"]
    neighbors = batch["neighbors"]
    map_img = batch["map"]

    print("\n--- Input Shapes ---")
    print(f"Agent:       {list(agent.shape)}")
    print(f"Neighbors:   {list(neighbors.shape)}")
    print(f"Map Image:   {list(map_img.shape)}")

    with torch.no_grad():
        trajectories, mode_logits = model(
            agent=agent,
            neighbors=neighbors,
            map_img=map_img,
        )

        # Softmax over the last dimension (modes)
        mode_probabilities = F.softmax(mode_logits, dim=-1)

    print("\n--- Output Shapes ---")
    print(f"Trajectories: {list(trajectories.shape)} -> [B, K, T, 2]")
    print(f"Mode Probs:   {list(mode_probabilities.shape)} -> [B, K]")

    # Identify the best trajectory mode (per item in batch)
    best_mode_idx = torch.argmax(mode_probabilities, dim=-1)

    print("\n" + "=" * 40)
    print("           Inference Results            ")
    print("=" * 40)

    for i in range(agent.size(0)):
        print(f"Sample #{i+1}:")
        probs = mode_probabilities[i].cpu().tolist()
        formatted_probs = [f"{p:.4f}" for p in probs]
        print(f"  Probabilities: {formatted_probs}")
        print(f"  Top Mode Index: {best_mode_idx[i].item()}")

    print("=" * 40)

    # --- JSON Output Saving ---
    if args.output_json:
        output_data = {
            "trajectories": trajectories.cpu().tolist(),
            "mode_probabilities": mode_probabilities.cpu().tolist(),
            "best_modes": best_mode_idx.cpu().tolist(),
        }

        try:
            with open(args.output_json, "w") as f:
                json.dump(output_data, f, indent=4)
            print(
                f"\nSuccessfully stored inference payload inside '{args.output_json}'."
            )
        except Exception as e:
            print(f"\nError: Could not save output JSON: {e}")


if __name__ == "__main__":
    main()
