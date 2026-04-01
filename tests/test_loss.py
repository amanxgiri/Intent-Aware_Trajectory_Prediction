import sys
from pathlib import Path

import pytest
import torch

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.loss import WTALoss


def test_wta_loss_returns_scalar_and_best_mode_indices():
    """
    Test that supplying valid standard shapes yields a 0D scalar loss
    and returns best_mode_indices with shape [B].
    """
    criterion = WTALoss()
    batch_size = 4
    num_modes = 3
    future_steps = 6

    # Create dummy tensors mapped to our project standards
    pred_trajectories = torch.randn(batch_size, num_modes, future_steps, 2)
    mode_logits = torch.randn(batch_size, num_modes)
    target_trajectories = torch.randn(batch_size, future_steps, 2)

    loss, best_mode_indices = criterion(
        pred_trajectories=pred_trajectories,
        mode_logits=mode_logits,
        target_trajectories=target_trajectories,
    )

    assert isinstance(loss, torch.Tensor), "Returned loss must be a torch.Tensor."
    assert loss.dim() == 0, f"Loss must be a scalar (0D), got dimension {loss.dim()}."

    expected_indices_shape = (batch_size,)
    assert (
        best_mode_indices.shape == expected_indices_shape
    ), f"Expected best_mode_indices shape {expected_indices_shape}, got {best_mode_indices.shape}."


def test_wta_loss_invalid_pred_rank():
    """
    Test that WTALoss correctly rejects invalid trajectory prediction ranks.
    """
    criterion = WTALoss()
    batch_size = 4
    num_modes = 3
    future_steps = 6

    # Invalid rank [B, T_future, 2] instead of [B, K, T_future, 2]
    invalid_pred_trajectories = torch.randn(batch_size, future_steps, 2)
    mode_logits = torch.randn(batch_size, num_modes)
    target_trajectories = torch.randn(batch_size, future_steps, 2)

    with pytest.raises(ValueError):
        criterion(
            pred_trajectories=invalid_pred_trajectories,
            mode_logits=mode_logits,
            target_trajectories=target_trajectories,
        )


def test_wta_loss_invalid_logits_shape():
    """
    Test that WTALoss catches a mismatch between the number of modes predicted
    and the number of logits provided.
    """
    criterion = WTALoss()
    batch_size = 4
    num_modes = 3
    future_steps = 6

    pred_trajectories = torch.randn(batch_size, num_modes, future_steps, 2)

    # Invalid mode dimension: [B, 5] instead of [B, 3]
    mismatched_mode_logits = torch.randn(batch_size, 5)

    target_trajectories = torch.randn(batch_size, future_steps, 2)

    with pytest.raises(ValueError):
        criterion(
            pred_trajectories=pred_trajectories,
            mode_logits=mismatched_mode_logits,
            target_trajectories=target_trajectories,
        )


def test_wta_loss_invalid_target_shape():
    """
    Test that WTALoss catches improperly structured ground truth tensors.
    """
    criterion = WTALoss()
    batch_size = 4
    num_modes = 3
    future_steps = 6

    pred_trajectories = torch.randn(batch_size, num_modes, future_steps, 2)
    mode_logits = torch.randn(batch_size, num_modes)

    # Invalid target shape [B, T_future] instead of [B, T_future, 2]
    invalid_target_trajectories = torch.randn(batch_size, future_steps)

    with pytest.raises(ValueError):
        criterion(
            pred_trajectories=pred_trajectories,
            mode_logits=mode_logits,
            target_trajectories=invalid_target_trajectories,
        )
