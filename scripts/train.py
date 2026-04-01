import os
import sys
import argparse
from pathlib import Path
import itertools

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

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
    return trajectories[torch.arange(len(idx), device=trajectories.device), idx]


def setup_live_plot():
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Dashboard", fontsize=16)

    return fig, axes


def update_live_plot(fig, axes, history, config_name):
    epochs = history["epochs"]

    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    for ax in [ax1, ax2, ax3, ax4]:
        ax.clear()

    # Loss plot
    ax1.plot(epochs, history["train_loss"], marker="o", linewidth=2, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], marker="o", linewidth=2, label="Val Loss")
    ax1.set_title(f"Loss Curves ({config_name})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ADE / FDE
    ax2.plot(epochs, history["ADE"], marker="o", linewidth=2, label="ADE")
    ax2.plot(epochs, history["FDE"], marker="o", linewidth=2, label="FDE")
    ax2.set_title("Trajectory Error Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Distance Error")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # MinADE / MinFDE
    ax3.plot(epochs, history["MinADE@K"], marker="o", linewidth=2, label="MinADE@K")
    ax3.plot(epochs, history["MinFDE@K"], marker="o", linewidth=2, label="MinFDE@K")
    ax3.set_title("Best-of-K Metrics")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Distance Error")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Learning rate
    ax4.plot(epochs, history["lr"], marker="o", linewidth=2, label="Learning Rate")
    ax4.set_title("Learning Rate Schedule")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("LR")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.draw()
    plt.pause(0.1)


def save_final_plot(fig, output_path):
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved training plot to: {output_path}")


# -----------------------------------------------------------------------------
# Training / Eval
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

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

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

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
        "val_loss": total_loss / max(len(loader), 1),
        "ADE": sum(ade) / max(len(ade), 1),
        "FDE": sum(fde) / max(len(fde), 1),
        "MinADE@K": sum(minade) / max(len(minade), 1),
        "MinFDE@K": sum(minfde) / max(len(minfde), 1),
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
    parser.add_argument("--num_modes", type=int, default=3)
    parser.add_argument("--future_steps", type=int, default=6)

    parser.add_argument(
        "--plot_dir",
        type=str,
        default="checkpoints",
        help="Directory to save training plots",
    )
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    # -----------------------------------------------------------------------------
    # Hyperparameter Grid
    # -----------------------------------------------------------------------------
    lr_list = [5e-5]
    weight_decay_list = [0.0]

    best_global_loss = float("inf")

    for lr, wd in itertools.product(lr_list, weight_decay_list):
        config_name = f"lr={lr}_wd={wd}"
        print("\n" + "=" * 60)
        print(f"Running config: {config_name}")
        print("=" * 60)

        train_loader, val_loader = create_train_val_dataloaders(
            dataroot=args.dataroot,
            version=args.version,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        model = IntentAwareTrajectoryModel(
            embed_dim=args.embed_dim,
            num_modes=args.num_modes,
            future_steps=args.future_steps,
        ).to(device)

        criterion = WTALoss(alpha=1.0)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        history = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "ADE": [],
            "FDE": [],
            "MinADE@K": [],
            "MinFDE@K": [],
            "lr": [],
        }

        fig, axes = setup_live_plot()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val = evaluate(model, val_loader, criterion, device)

            scheduler.step(val["val_loss"])
            current_lr = optimizer.param_groups[0]["lr"]

            history["epochs"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val["val_loss"])
            history["ADE"].append(val["ADE"])
            history["FDE"].append(val["FDE"])
            history["MinADE@K"].append(val["MinADE@K"])
            history["MinFDE@K"].append(val["MinFDE@K"])
            history["lr"].append(current_lr)

            print(f"\nEpoch {epoch}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss  : {val['val_loss']:.4f}")
            print(f"ADE       : {val['ADE']:.4f}")
            print(f"FDE       : {val['FDE']:.4f}")
            print(f"MinADE@K  : {val['MinADE@K']:.4f}")
            print(f"MinFDE@K  : {val['MinFDE@K']:.4f}")
            print(f"LR        : {current_lr:.6f}")

            update_live_plot(fig, axes, history, config_name)

            if val["val_loss"] < best_val_loss:
                best_val_loss = val["val_loss"]
                patience_counter = 0

                ckpt_path = "checkpoints/best_model.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved best model to: {ckpt_path}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        plot_path = os.path.join(
            args.plot_dir,
            f"training_plot_{config_name.replace('=', '_').replace('.', 'p')}.png",
        )
        save_final_plot(fig, plot_path)
        plt.close(fig)

        if best_val_loss < best_global_loss:
            best_global_loss = best_val_loss

    plt.ioff()
    print("\nBest overall validation loss:", best_global_loss)


if __name__ == "__main__":
    main()
