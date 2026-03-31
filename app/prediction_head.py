import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_modes: int = 3,
        future_steps: int = 6,
        out_dim: int = 2,
        hidden_dim: int = 256,
    ):
        """
        Multi-modal trajectory prediction head.

        Args:
            in_features: Dimension of fused input embedding, usually 3 * D.
            num_modes: Number of trajectory modes (K).
            future_steps: Number of future timesteps to predict (T_future).
            out_dim: Coordinate dimension, usually 2 for (x, y).
            hidden_dim: Hidden layer size for the MLP.
        """
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.out_dim = out_dim

        self.traj_branch = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modes * future_steps * out_dim),
        )

        self.mode_branch = nn.Sequential(
            nn.Linear(in_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_modes),
        )

    def forward(self, fused_embed: torch.Tensor):
        """
        Args:
            fused_embed: Tensor of shape [B, F]

        Returns:
            trajectories: Tensor of shape [B, K, T_future, 2]
            mode_logits: Tensor of shape [B, K]
        """
        if fused_embed.dim() != 2:
            raise ValueError(
                f"Expected fused_embed shape [B, F], got {tuple(fused_embed.shape)}"
            )

        batch_size = fused_embed.size(0)

        raw_trajectories = self.traj_branch(fused_embed)
        trajectories = raw_trajectories.reshape(
            batch_size, self.num_modes, self.future_steps, self.out_dim
        )

        mode_logits = self.mode_branch(fused_embed)

        return trajectories, mode_logits
