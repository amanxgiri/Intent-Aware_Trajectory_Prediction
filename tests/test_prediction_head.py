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

from app.prediction_head import PredictionHead


def test_prediction_head_output_shapes():
    """
    Test that the default configured parameters properly emit
    trajectories structured as [B, K, T, 2] and logits [B, K].
    """
    batch_size = 4
    embed_dim = 128
    in_features = embed_dim * 3  # Fused Temporal + Scene + Social = 384
    num_modes = 3
    future_steps = 6

    model = PredictionHead(
        in_features=in_features, num_modes=num_modes, future_steps=future_steps
    )

    # Input tensor [B, in_features]
    fused_embeddings = torch.randn(batch_size, in_features)

    trajectories, mode_logits = model(fused_embeddings)

    expected_trajectories_shape = (batch_size, num_modes, future_steps, 2)
    expected_logits_shape = (batch_size, num_modes)

    assert (
        trajectories.shape == expected_trajectories_shape
    ), f"Expected trajectories shape {expected_trajectories_shape}, got {trajectories.shape}"

    assert (
        mode_logits.shape == expected_logits_shape
    ), f"Expected mode_logits shape {expected_logits_shape}, got {mode_logits.shape}"


def test_prediction_head_invalid_input_rank():
    """
    Test that supplying an unsqueezed or unexpected multdimensional
    tensor (e.g. sequence-like [B, 1, 384] instead of [B, 384]) safely fails.
    """
    in_features = 384
    model = PredictionHead(in_features=in_features, num_modes=3, future_steps=6)

    # Invalid rank [B, seq_len, features]
    invalid_input = torch.randn(4, 1, in_features)

    with pytest.raises(ValueError):
        model(invalid_input)


def test_prediction_head_configurable_dimensions():
    """
    Test that modifying constructor dimensions actively adjusts the outputs.
    """
    batch_size = 2
    in_features = 256
    custom_modes = 3
    custom_steps = 10

    model = PredictionHead(
        in_features=in_features, num_modes=custom_modes, future_steps=custom_steps
    )

    fused_embeddings = torch.randn(batch_size, in_features)
    trajectories, mode_logits = model(fused_embeddings)

    assert trajectories.shape == (
        batch_size,
        custom_modes,
        custom_steps,
        2,
    ), "Tensors did not abide by explicitly provided non-standard constructor modes."

    assert mode_logits.shape == (
        batch_size,
        custom_modes,
    ), "Logits did not abide by explicitly provided non-standard constructor modes."
