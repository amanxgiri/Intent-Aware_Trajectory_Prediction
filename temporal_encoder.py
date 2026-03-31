import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1)].to(x.device)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1. Input projection (x,y,vx,vy → embedding)
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # IMPORTANT → [B, T, D]
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4. LayerNorm for stability
        self.norm = nn.LayerNorm(d_model)

    def forward(self, agent: torch.Tensor) -> torch.Tensor:
        """
        agent: [B, T_past, 4]

        returns:
        agent_embed: [B, T_past, D]
        """

        # Step 1: Project to embedding space
        x = self.input_proj(agent)  # [B, T, D]

        # Step 2: Add temporal positional encoding
        x = self.pos_encoder(x)

        # Step 3: Transformer encoding
        x = self.transformer(x)  # [B, T, D]

        # Step 4: Normalize
        x = self.norm(x)

        return x


if __name__ == "__main__":
    model = TemporalEncoder()

    dummy = torch.randn(2, 4, 4)  # [B=2, T=4, features=4]
    out = model(dummy)

    print("Input shape:", dummy.shape)
    print("Output shape:", out.shape)