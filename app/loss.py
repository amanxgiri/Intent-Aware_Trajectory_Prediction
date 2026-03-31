import torch
import torch.nn as nn
import torch.nn.functional as F


class WTALoss(nn.Module):
    def __init__(self, alpha: float = 1.0):
        """
        Winner-Takes-All (WTA) Loss for multi-modal trajectory prediction.

        Args:
            alpha: Weighting factor for the classification (mode probability) loss.
        """
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        pred_trajectories: torch.Tensor,
        mode_logits: torch.Tensor,
        target_trajectories: torch.Tensor,
    ):
        """
        Args:
            pred_trajectories: Tensor of shape [B, K, T_future, 2]
            mode_logits: Tensor of shape [B, K]
            target_trajectories: Tensor of shape [B, T_future, 2]

        Returns:
            total_loss: Scalar loss for backpropagation.
            best_mode_indices: Tensor of shape [B] with the index of the best matching mode.
        """
        # --- 1. Input Shape Validation ---
        if pred_trajectories.dim() != 4:
            raise ValueError(
                f"Expected pred_trajectories shape [B, K, T_future, 2], got {tuple(pred_trajectories.shape)}"
            )
        if mode_logits.dim() != 2:
            raise ValueError(
                f"Expected mode_logits shape [B, K], got {tuple(mode_logits.shape)}"
            )
        if target_trajectories.dim() != 3:
            raise ValueError(
                f"Expected target_trajectories shape [B, T_future, 2], got {tuple(target_trajectories.shape)}"
            )

        B, K, T_future, D = pred_trajectories.shape

        if mode_logits.shape != (B, K):
            raise ValueError(
                f"mode_logits shape {tuple(mode_logits.shape)} does not match B={B}, K={K}"
            )
        if target_trajectories.shape != (B, T_future, D):
            raise ValueError(
                f"target_trajectories shape {tuple(target_trajectories.shape)} does not match B={B}, T_future={T_future}, D={D}"
            )

        # --- 2. WTA Logic ---
        # Reshape target to broadcast over K modes: [B, 1, T_future, 2]
        target_expanded = target_trajectories.unsqueeze(1)

        # Calculate Mean Squared Error distance across all modes
        # Shape: [B, K, T_future, 2]
        mse_distances = F.mse_loss(
            pred_trajectories, target_expanded.expand(-1, K, -1, -1), reduction="none"
        )

        # Sum over time and coordinate dimensions to get total distance per mode
        # Shape: [B, K]
        mode_distances = mse_distances.sum(dim=(2, 3))

        # Find the index of the best matching prediction mode (Winner)
        # Shape: [B]
        best_mode_indices = torch.argmin(mode_distances, dim=1)

        # Select the best predicted trajectories based on the indices
        batch_indices = torch.arange(B, device=pred_trajectories.device)
        best_trajectories = pred_trajectories[batch_indices, best_mode_indices]

        # 3. Regression Loss: MSE of only the best mode against the target
        reg_loss = F.mse_loss(best_trajectories, target_trajectories, reduction="mean")

        # 4. Classification/Intent Loss: CrossEntropy to train mode_logits
        # to assign the highest probability to the best_mode_indices
        cls_loss = F.cross_entropy(mode_logits, best_mode_indices, reduction="mean")

        # Combine into a single scalar loss
        total_loss = reg_loss + self.alpha * cls_loss

        return total_loss, best_mode_indices
