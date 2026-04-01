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

from app.full_model import IntentAwareTrajectoryModel


def test_full_model_output_shapes_with_4d_neighbors():
    """
    Test the model forward pass with 4D neighbors [B, N, T_past, 4]
    as outputted by the dataloader.
    """
    batch_size = 2
    t_past = 4
    num_neighbors = 5
    embed_dim = 128
    num_modes = 3
    future_steps = 6

    model = IntentAwareTrajectoryModel(
        embed_dim=embed_dim, num_modes=num_modes, future_steps=future_steps
    )

    agent = torch.randn(batch_size, t_past, 4)
    neighbors = torch.randn(batch_size, num_neighbors, t_past, 4)
    map_img = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        trajectories, mode_logits = model(
            agent=agent, neighbors=neighbors, map_img=map_img
        )

    expected_traj_shape = (batch_size, num_modes, future_steps, 2)
    expected_logits_shape = (batch_size, num_modes)

    assert (
        trajectories.shape == expected_traj_shape
    ), f"Expected trajectories shape {expected_traj_shape}, got {trajectories.shape}"

    assert (
        mode_logits.shape == expected_logits_shape
    ), f"Expected mode logits shape {expected_logits_shape}, got {mode_logits.shape}"


def test_full_model_output_shapes_with_3d_neighbors():
    """
    Test the model forward pass natively handling 3D neighbors [B, N, 4].
    """
    batch_size = 2
    t_past = 4
    num_neighbors = 5
    embed_dim = 128
    num_modes = 3
    future_steps = 6

    model = IntentAwareTrajectoryModel(
        embed_dim=embed_dim, num_modes=num_modes, future_steps=future_steps
    )

    agent = torch.randn(batch_size, t_past, 4)
    neighbors = torch.randn(batch_size, num_neighbors, 4)
    map_img = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        trajectories, mode_logits = model(
            agent=agent, neighbors=neighbors, map_img=map_img
        )

    expected_traj_shape = (batch_size, num_modes, future_steps, 2)
    expected_logits_shape = (batch_size, num_modes)

    assert (
        trajectories.shape == expected_traj_shape
    ), f"Expected trajectories shape {expected_traj_shape}, got {trajectories.shape}"

    assert (
        mode_logits.shape == expected_logits_shape
    ), f"Expected mode logits shape {expected_logits_shape}, got {mode_logits.shape}"


def test_full_model_invalid_agent_rank():
    """
    Test that the model validation catches improperly structured agent arrays.
    """
    model = IntentAwareTrajectoryModel(num_modes=3, future_steps=6)

    batch_size = 2
    num_neighbors = 5

    # Invalid agent format: [B, 4] instead of [B, T_past, 4]
    invalid_agent = torch.randn(batch_size, 4)
    neighbors = torch.randn(batch_size, num_neighbors, 4)
    map_img = torch.randn(batch_size, 3, 224, 224)

    with pytest.raises(ValueError):
        model(agent=invalid_agent, neighbors=neighbors, map_img=map_img)


def test_full_model_batch_size_mismatch():
    """
    Test that the model enforces batch size consistency across inputs.
    """
    model = IntentAwareTrajectoryModel(num_modes=3, future_steps=6)

    t_past = 4
    num_neighbors = 5

    # Batch size 2
    agent = torch.randn(2, t_past, 4)

    # Mismatch: Batch size 3
    mismatched_neighbors = torch.randn(3, num_neighbors, 4)

    # Batch size 2
    map_img = torch.randn(2, 3, 224, 224)

    with pytest.raises(ValueError):
        model(agent=agent, neighbors=mismatched_neighbors, map_img=map_img)
