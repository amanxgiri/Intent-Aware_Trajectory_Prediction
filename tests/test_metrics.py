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

# Note: We import all mentioned metric functions directly from app.metrics
from app.metrics import (
    compute_ade,
    compute_fde,
    compute_min_ade_k,
    compute_min_fde_k,
    compute_intent_accuracy,
    compute_collision_prob,
)


def test_compute_ade_zero_for_identical_trajectories():
    """
    Test Average Displacement Error (ADE) evaluates to 0.0
    when predictions exactly match the ground truth.
    """
    batch_size = 4
    future_steps = 6

    # Create dummy trajectories [B, T_future, 2]
    pred = torch.randn(batch_size, future_steps, 2)
    target = pred.clone()

    ade = compute_ade(pred, target)

    # Allow slight floating point discrepancies
    assert ade == pytest.approx(
        0.0, abs=1e-5
    ), f"ADE should be 0.0 for identical inputs, got {ade}"


def test_compute_fde_zero_for_identical_trajectories():
    """
    Test Final Displacement Error (FDE) evaluates to 0.0
    when predictions exactly match the ground truth.
    """
    batch_size = 4
    future_steps = 6

    # Create dummy trajectories [B, T_future, 2]
    pred = torch.randn(batch_size, future_steps, 2)
    target = pred.clone()

    fde = compute_fde(pred, target)

    assert fde == pytest.approx(
        0.0, abs=1e-5
    ), f"FDE should be 0.0 for identical inputs, got {fde}"


def test_compute_min_ade_k_returns_non_negative():
    """
    Test MinADE@K returns a valid, non-negative scalar float.
    """
    batch_size = 4
    num_modes = 3
    future_steps = 6

    # [B, K, T_future, 2] for predictions
    pred_trajectories = torch.randn(batch_size, num_modes, future_steps, 2)
    # [B, T_future, 2] for ground truth
    target_trajectory = torch.randn(batch_size, future_steps, 2)

    min_ade = compute_min_ade_k(pred_trajectories, target_trajectory)

    assert isinstance(min_ade, float), "Expected min_ade to return a python float"
    assert (
        min_ade >= 0.0
    ), f"MinADE@K should be fundamentally non-negative, got {min_ade}"


def test_compute_min_fde_k_returns_non_negative():
    """
    Test MinFDE@K returns a valid, non-negative scalar float.
    """
    batch_size = 4
    num_modes = 3
    future_steps = 6

    pred_trajectories = torch.randn(batch_size, num_modes, future_steps, 2)
    target_trajectory = torch.randn(batch_size, future_steps, 2)

    min_fde = compute_min_fde_k(pred_trajectories, target_trajectory)

    assert isinstance(min_fde, float), "Expected min_fde to return a python float"
    assert (
        min_fde >= 0.0
    ), f"MinFDE@K should be fundamentally non-negative, got {min_fde}"


def test_compute_intent_accuracy_perfect_prediction():
    """
    Verify intent accuracy is correctly 1.0 when logits cleanly match standard labels.
    """
    # Logits heavily predicting specific indices [B, K]
    logits = torch.tensor(
        [
            [10.0, -10.0, -10.0],  # argmax is 0
            [-10.0, 10.0, -10.0],  # argmax is 1
            [-10.0, -10.0, 10.0],  # argmax is 2
        ]
    )

    # Ground truth labels [B]
    labels = torch.tensor([0, 1, 2])

    accuracy = compute_intent_accuracy(logits, labels)

    assert accuracy == pytest.approx(
        1.0, abs=1e-5
    ), f"Expected perfect accuracy 1.0, got {accuracy}"


def test_compute_collision_prob_returns_valid_probability():
    """
    Ensure the collision probability is always bounded safely between 0.0 and 1.0.
    """
    batch_size = 4
    future_steps = 6
    max_neighbors = 10
    t_past = 4

    # Top prediction: [B, T_future, 2]
    pred_trajectory = torch.randn(batch_size, future_steps, 2)

    # Standard neighbor input matching model expectations: [B, N, 4]
    neighbors = torch.randn(batch_size, max_neighbors, 4)

    prob = compute_collision_prob(pred_trajectory, neighbors)

    assert isinstance(prob, float), "Expected compute_collision_prob to yield a float."
    assert (
        0.0 <= prob <= 1.0
    ), f"Collision probability must be in range [0, 1]. Got {prob}"
