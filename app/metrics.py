import torch


def compute_ade(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Average Displacement Error (ADE).
    Computes mean L2 distance over the entire future trajectory.

    Args:
        pred: Shape [B, T, 2]
        target: Shape [B, T, 2]

    Returns:
        float: ADE value
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

    # L2 distance: [B, T]
    l2_dist = torch.linalg.norm(pred - target, dim=-1)

    # Mean over time and batch
    return l2_dist.mean().item()


def compute_fde(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Final Displacement Error (FDE).
    Computes mean L2 distance ONLY on the final future timestep.

    Args:
        pred: Shape [B, T, 2]
        target: Shape [B, T, 2]

    Returns:
        float: FDE value
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

    # Distance at final timestep T: [B]
    final_l2_dist = torch.linalg.norm(pred[:, -1, :] - target[:, -1, :], dim=-1)

    return final_l2_dist.mean().item()


def compute_min_ade_k(pred_multi: torch.Tensor, target: torch.Tensor) -> float:
    """
    MinADE@K: Best-matching trajectory's Average Displacement Error.

    Args:
        pred_multi: Shape [B, K, T, 2]
        target: Shape [B, T, 2]

    Returns:
        float: minADE@K value
    """
    if pred_multi.dim() != 4 or target.dim() != 3:
        raise ValueError(f"Expected pred [B,K,T,2] and target [B,T,2]")

    B, K, T, D = pred_multi.shape

    target_exp = target.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, T, 2]

    # Compute ADE for all K modes
    # L2 dist: [B, K, T] -> Mean over T: [B, K]
    l2_dist = torch.linalg.norm(pred_multi - target_exp, dim=-1)
    all_ades = l2_dist.mean(dim=2)

    # Min over K modes -> [B]
    min_ades, _ = all_ades.min(dim=1)

    return min_ades.mean().item()


def compute_min_fde_k(pred_multi: torch.Tensor, target: torch.Tensor) -> float:
    """
    MinFDE@K: Best-matching trajectory's Final Displacement Error.

    Args:
        pred_multi: Shape [B, K, T, 2]
        target: Shape [B, T, 2]

    Returns:
        float: minFDE@K value
    """
    if pred_multi.dim() != 4 or target.dim() != 3:
        raise ValueError(f"Expected pred [B,K,T,2] and target [B,T,2]")

    # Get just final points
    pred_final = pred_multi[:, :, -1, :]  # [B, K, 2]
    target_final = target[:, -1, :]  # [B, 2]

    target_exp = target_final.unsqueeze(1).expand_as(pred_final)

    # Distances at final step across all K modes: [B, K]
    fdes = torch.linalg.norm(pred_final - target_exp, dim=-1)

    # Min over K modes -> [B]
    min_fdes, _ = fdes.min(dim=1)

    return min_fdes.mean().item()


def compute_collision_prob(
    pred: torch.Tensor, neighbors: torch.Tensor, radius: float = 2.0
) -> float:
    """
    Estimates collision probability between the predicted ego trajectory and neighbor agents.

    Args:
        pred: Predicted ego trajectory, shape [B, T, 2]
        neighbors: Neighbor coordinates, shape [B, N, T, 4] or [B, N, 4].
                   Missing time dimension will be broadcasted to T.
                   Only the first 2 feature channels (x, y) are used.
        radius: Distance threshold in meters indicating a collision

    Returns:
        float: Fraction of batch samples containing at least one collision
    """
    if pred.dim() != 3:
        raise ValueError(f"Expected pred shape [B, T, 2], got {tuple(pred.shape)}")

    B, T, _ = pred.shape

    if neighbors.dim() not in [3, 4]:
        raise ValueError(
            f"Expected neighbors shape [B, N, 4] or [B, N, T, 4], got {tuple(neighbors.shape)}"
        )

    # Extract only x, y positions
    if neighbors.dim() == 3:  # [B, N, 4] -> [B, N, 2]
        neigh_pos = neighbors[..., :2]
        # Expand time dimension to match prediction: [B, N, T, 2]
        neigh_pos = neigh_pos.unsqueeze(2).expand(-1, -1, T, -1)
    else:  # [B, N, T, 4] -> [B, N, T, 2]
        neigh_pos = neighbors[..., :2]

    if neigh_pos.size(2) != T:
        raise ValueError(
            f"Neighbor time steps {neigh_pos.size(2)} do not match prediction steps {T}"
        )

    # Reshape prediction to broadcast against neighbors: [B, 1, T, 2]
    pred_exp = pred.unsqueeze(1)

    # Calculate pairwise distances over all timesteps
    # Shape: [B, N, T]
    distances = torch.linalg.norm(pred_exp - neigh_pos, dim=-1)

    # Create boolean mask for collisions: [B, N, T]
    is_collision = distances < radius

    # Check if ANY neighbor at ANY timestep collided for each batch item
    # Shape: [B]
    batch_has_collision = is_collision.any(dim=-1).any(dim=-1)

    # Return percentage of batch that had a collision
    return batch_has_collision.float().mean().item()


def compute_intent_accuracy(
    pred_logits: torch.Tensor, target_labels: torch.Tensor
) -> float:
    """
    Simple categorical intent accuracy.

    Args:
        pred_logits: Predicted logits or probabilities [B, num_classes]
        target_labels: Ground truth intent integer indices [B]

    Returns:
        float: Accuracy score in range [0.0, 1.0]
    """
    if pred_logits.dim() != 2 or target_labels.dim() != 1:
        raise ValueError(f"Expected logits [B, C], target [B]")

    # Argmax yields identical shape [B]
    predicted_labels = torch.argmax(pred_logits, dim=-1)
    correct = (predicted_labels == target_labels).float()

    return correct.mean().item()
