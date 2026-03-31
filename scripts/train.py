import os
import sys
import argparse
from pathlib import Path
import itertools

import torch
import torch.optim as optim

# -----------------------------------------------------------------------------
# Path setup (same as your smoke test)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from app.dataloader import create_train_val_dataloaders
from app.full_model import IntentAwareTrajectoryModel
from app.loss import WTALoss
from app.metrics import (
    compute_ade,
    compute_fde,
    compute_min_ade_k,
    compute_min_fde_k,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def select_top_mode(trajectories, logits):
    idx = torch.argmax(logits, dim=1)
    return trajectories[torch.arange(len(idx)), idx]


# -----------------------------------------------------------------------------
# Training / Eval
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = move_batch(batch, device)

        pred, logits = model(
            agent=batch["agent"],
            neighbors=batch["neighbors"],
            map_img=batch["map"],
        )

        loss, _ = criterion(pred, logits, batch["target"])

        optimizer.zero_grad()
        loss.backward()

        # 🔥 CRITICAL: gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    ade, fde, minade, minfde = [], [], [], []

    for batch in loader:
        batch = move_batch(batch, device)

        pred, logits = model(
            agent=batch["agent"],
            neighbors=batch["neighbors"],
            map_img=batch["map"],
        )

        loss, _ = criterion(pred, logits, batch["target"])
        total_loss += loss.item()

        top = select_top_mode(pred, logits)

        ade.append(compute_ade(top, batch["target"]))
        fde.append(compute_fde(top, batch["target"]))
        minade.append(compute_min_ade_k(pred, batch["target"]))
        minfde.append(compute_min_fde_k(pred, batch["target"]))

    return {
        "val_loss": total_loss / len(loader),
        "ADE": sum(ade) / len(ade),
        "FDE": sum(fde) / len(fde),
        "MinADE@K": sum(minade) / len(minade),
        "MinFDE@K": sum(minfde) / len(minfde),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-mini")

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_modes", type=int, default=6)
    parser.add_argument("--future_steps", type=int, default=6)

    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    # -----------------------------------------------------------------------------
    # Hyperparameter Grid (TUNING)
    # -----------------------------------------------------------------------------
    lr_list = [5e-5]
    weight_decay_list = [0.0]

    best_global_loss = float("inf")

    for lr, wd in itertools.product(lr_list, weight_decay_list):
        print("\n" + "=" * 60)
        print(f"Running config: LR={lr}, WD={wd}")
        print("=" * 60)

        # -----------------------------------------------------------------------------
        # Data
        # -----------------------------------------------------------------------------
        train_loader, val_loader = create_train_val_dataloaders(
            dataroot=args.dataroot,
            version=args.version,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # -----------------------------------------------------------------------------
        # Model
        # -----------------------------------------------------------------------------
        model = IntentAwareTrajectoryModel(
            embed_dim=args.embed_dim,
            num_modes=args.num_modes,
            future_steps=args.future_steps,
        ).to(device)

        criterion = WTALoss(alpha=1.0)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # 🔥 LR Scheduler (IMPORTANT)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        # -----------------------------------------------------------------------------
        # Training Loop
        # -----------------------------------------------------------------------------
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val = evaluate(model, val_loader, criterion, device)

            scheduler.step(val["val_loss"])

            print(f"\nEpoch {epoch}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss  : {val['val_loss']:.4f}")
            print(f"ADE       : {val['ADE']:.4f}")
            print(f"FDE       : {val['FDE']:.4f}")
            print(f"MinADE@K  : {val['MinADE@K']:.4f}")

            # Early stopping
            if val["val_loss"] < best_val_loss:
                best_val_loss = val["val_loss"]
                patience_counter = 0

                torch.save(model.state_dict(), f"checkpoints/best_model.pt")
                print("Saved best model")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        # Track best config
        if best_val_loss < best_global_loss:
            best_global_loss = best_val_loss

    print("\nBest overall validation loss:", best_global_loss)


if __name__ == "__main__":
    main()
