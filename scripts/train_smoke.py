import os
import sys
import argparse
from pathlib import Path

import torch
import torch.optim as optim

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
# This helps both:
# 1. imports like "from app.full_model import ..."
# 2. older teammate imports like "from dataset import ..."
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
    trajectories: [B, K, T_future, 2]
    mode_logits:   [B, K]

    Returns:
        best_pred: [B, T_future, 2] using argmax(mode_logits)
    """
    batch_size = trajectories.size(0)
    best_mode = torch.argmax(mode_logits, dim=1)  # [B]
    batch_idx = torch.arange(batch_size, device=trajectories.device)
    best_pred = trajectories[batch_idx, best_mode]  # [B, T_future, 2]
    return best_pred


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        agent = batch["agent"]
        neighbors = batch["neighbors"]
        map_img = batch["map"]
        target = batch["target"]

        optimizer.zero_grad()

        trajectories, mode_logits = model(
            agent=agent,
            neighbors=neighbors,
            map_img=map_img,
        )

        loss, _ = criterion(
            pred_trajectories=trajectories,
            mode_logits=mode_logits,
            target_trajectories=target,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate_one_epoch(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    ade_values = []
    fde_values = []
    minade_values = []
    minfde_values = []
    collision_values = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        agent = batch["agent"]
        neighbors = batch["neighbors"]
        map_img = batch["map"]
        target = batch["target"]

        trajectories, mode_logits = model(
            agent=agent,
            neighbors=neighbors,
            map_img=map_img,
        )

        loss, _ = criterion(
            pred_trajectories=trajectories,
            mode_logits=mode_logits,
            target_trajectories=target,
        )

        total_loss += loss.item()
        num_batches += 1

        # Top-probability trajectory for ADE/FDE
        top_pred = select_top_mode_trajectory(trajectories, mode_logits)

        ade = compute_ade(top_pred, target)
        fde = compute_fde(top_pred, target)
        minade = compute_min_ade_k(trajectories, target)
        minfde = compute_min_fde_k(trajectories, target)

        ade_values.append(ade)
        fde_values.append(fde)
        minade_values.append(minade)
        minfde_values.append(minfde)

        # Collision probability against neighbors
        # Works with neighbors shaped [B, N, T_past, 4] or [B, N, 4]
        try:
            collision = compute_collision_prob(top_pred, neighbors)
            collision_values.append(collision)
        except Exception:
            # Do not fail the whole evaluation if collision metric mismatches
            pass

    results = {
        "val_loss": total_loss / max(num_batches, 1),
        "ADE": sum(ade_values) / max(len(ade_values), 1),
        "FDE": sum(fde_values) / max(len(fde_values), 1),
        "MinADE@K": sum(minade_values) / max(len(minade_values), 1),
        "MinFDE@K": sum(minfde_values) / max(len(minfde_values), 1),
    }

    if len(collision_values) > 0:
        results["CollisionProb"] = sum(collision_values) / len(collision_values)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="5-epoch smoke training for full trajectory pipeline"
    )
    parser.add_argument(
        "--dataroot", type=str, required=True, help="Path to nuScenes root"
    )
    parser.add_argument(
        "--version", type=str, default="v1.0-mini", help="nuScenes version"
    )
    parser.add_argument("--train_split", type=str, default="train", help="Train split")
    parser.add_argument("--val_split", type=str, default="val", help="Validation split")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--embed_dim", type=int, default=128, help="Shared embedding dimension"
    )
    parser.add_argument(
        "--num_modes", type=int, default=6, help="Number of trajectory modes"
    )
    parser.add_argument("--future_steps", type=int, default=6, help="Future timesteps")
    parser.add_argument("--t_past", type=int, default=4, help="Past timesteps")
    parser.add_argument("--max_neighbors", type=int, default=10, help="Max neighbors")
    parser.add_argument(
        "--checkpoint_out",
        type=str,
        default="checkpoints/smoke_best_model.pt",
        help="Where to save best model",
    )
    args = parser.parse_args()

    device = get_device()
    print("=" * 80)
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("=" * 80)

    os.makedirs("checkpoints", exist_ok=True)

    # -------------------------------------------------------------------------
    # Dataloaders
    # -------------------------------------------------------------------------
    train_loader, val_loader = create_train_val_dataloaders(
        dataroot=args.dataroot,
        version=args.version,
        train_split=args.train_split,
        val_split=args.val_split,
        t_past=args.t_past,
        t_future=args.future_steps,
        max_neighbors=args.max_neighbors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    # -------------------------------------------------------------------------
    # Model / loss / optimizer
    # -------------------------------------------------------------------------
    model = IntentAwareTrajectoryModel(
        embed_dim=args.embed_dim,
        num_modes=args.num_modes,
        future_steps=args.future_steps,
    ).to(device)

    criterion = WTALoss(alpha=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_results = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print(f"Train Loss   : {train_loss:.4f}")
        print(f"Val Loss     : {val_results['val_loss']:.4f}")
        print(f"ADE          : {val_results['ADE']:.4f}")
        print(f"FDE          : {val_results['FDE']:.4f}")
        print(f"MinADE@K     : {val_results['MinADE@K']:.4f}")
        print(f"MinFDE@K     : {val_results['MinFDE@K']:.4f}")
        if "CollisionProb" in val_results:
            print(f"CollisionProb: {val_results['CollisionProb']:.4f}")

        if val_results["val_loss"] < best_val_loss:
            best_val_loss = val_results["val_loss"]
            torch.save(model.state_dict(), args.checkpoint_out)
            print(f"Saved best model to: {args.checkpoint_out}")

    print("\nTraining smoke test complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
