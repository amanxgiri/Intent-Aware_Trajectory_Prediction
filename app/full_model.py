import torch
import torch.nn as nn

from app.temporal_encoder import TemporalEncoder
from app.scene_encoder import SceneEncoder
from app.social_encoder import SocialEncoder
from app.prediction_head import PredictionHead


class IntentAwareTrajectoryModel(nn.Module):
    def __init__(self, embed_dim: int = 128, num_modes: int = 3, future_steps: int = 6):
        """
        Integrates Temporal, Scene, and Social encoders into a single prediction pipeline.

        Args:
            embed_dim: Common feature dimension D used across encoders.
            num_modes: Number of trajectory modes to predict.
            future_steps: Number of future timesteps to output.
        """
        super().__init__()

        # Instantiate encoders with their specific argument names
        self.temporal_encoder = TemporalEncoder(d_model=embed_dim)
        self.scene_encoder = SceneEncoder(embedding_dim=embed_dim)

        # SocialEncoder expects [B, N, 4] for adapted neighbors
        self.social_encoder = SocialEncoder(
            neighbor_dim=4, embed_dim=embed_dim, temporal_dim=embed_dim
        )

        # Fused embedding dimension is 3 * D (Temporal + Scene + Social)
        self.prediction_head = PredictionHead(
            in_features=embed_dim * 3, num_modes=num_modes, future_steps=future_steps
        )

    def forward(
        self, agent: torch.Tensor, neighbors: torch.Tensor, map_img: torch.Tensor
    ):
        """
        Args:
            agent: Shape [B, T_past, 4]
            neighbors: Shape [B, N, T_past, 4] (Dataloader format) or [B, N, 4]
            map_img: Shape [B, 3, H, W]

        Returns:
            trajectories: Shape [B, K, T_future, 2]
            mode_logits: Shape [B, K]
        """
        # --- 1. Shape validations ---
        if agent.dim() != 3:
            raise ValueError(f"Expected agent [B, T_past, 4], got {tuple(agent.shape)}")
        if map_img.dim() != 4:
            raise ValueError(f"Expected map [B, 3, H, W], got {tuple(map_img.shape)}")
        if neighbors.dim() not in [3, 4]:
            raise ValueError(
                f"Expected neighbors [B, N, 4] or [B, N, T_past, 4], got {tuple(neighbors.shape)}"
            )

        B = agent.size(0)
        if neighbors.size(0) != B or map_img.size(0) != B:
            raise ValueError(
                f"Batch size mismatch: agent B={B}, neighbors B={neighbors.size(0)}, map B={map_img.size(0)}"
            )

        # --- 2. Temporal Feature Extraction ---
        # Output: [B, T_past, D]
        agent_embed = self.temporal_encoder(agent)

        # --- 3. Scene Feature Extraction ---
        # Output: [B, D]
        scene_embed = self.scene_encoder(map_img)

        # --- 4. Social Integration and adaptation ---
        # Handle dataloader mismatch safely: [B, N, T_past, 4] -> [B, N, 4]
        # Taking the last timestep of neighbors for the spatial layout
        if neighbors.dim() == 4:
            adapted_neighbors = neighbors[:, :, -1, :]
        else:
            adapted_neighbors = neighbors

        # Social encoder contract expectations
        # Output: [B, D]
        social_embed = self.social_encoder(adapted_neighbors, agent_embed)

        # --- 5. Fusion ---
        # Extract the final hidden state from the temporal encoder's sequence
        # agent_embed[:, -1] has shape [B, D]
        fused_features = torch.cat(
            [agent_embed[:, -1, :], scene_embed, social_embed], dim=-1
        )

        # --- 6. Prediction Head ---
        # Output: Trajectories [B, K, T_future, 2] and Modalities [B, K]
        trajectories, mode_logits = self.prediction_head(fused_features)

        return trajectories, mode_logits
