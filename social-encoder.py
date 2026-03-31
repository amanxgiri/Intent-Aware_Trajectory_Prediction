import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Simple Spatial Graph Convolution Layer
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: [B, N, C], adj: [B, N, N]
        # Aggregate neighbor features using the adjacency matrix
        support = self.linear(x)
        output = torch.bmm(adj, support)
        return F.relu(output)

class SocialEncoder(nn.Module):
    """
    Module 3: Social Interaction (STGCN)
    Models multi-agent interactions and collision avoidance.
    """
    def __init__(self, neighbor_dim=4, embed_dim=256, temporal_dim=256):
        super(SocialEncoder, self).__init__()
        self.embed_dim = embed_dim
        
        # Feature encoding for neighbor nodes (position, velocity) -> [N, 4]
        self.node_encoder = nn.Sequential(
            nn.Linear(neighbor_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        
        # Spatial Graph Convolution
        self.gcn1 = GraphConvolution(embed_dim, embed_dim)
        self.gcn2 = GraphConvolution(embed_dim, embed_dim)
        
        # Fusion layer to combine agent's temporal embedding with spatial graph features
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + temporal_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def build_adjacency_matrix(self, neighbors):
        """
        Construct graph representation where edges represent relative position/velocity.
        Input: neighbors [B, N, 4] (x, y, vx, vy)
        Output: adj [B, N, N]
        """
        B, N, _ = neighbors.shape
        # Extract spatial coordinates (x, y) for distance calculation
        positions = neighbors[..., :2] 
        
        # Compute pairwise distance matrix [B, N, N]
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        
        # Create adjacency matrix: connections exist if agents are within a threshold (e.g., 5 meters)
        # Adds self-loops (identity matrix) to retain individual node features
        threshold = 5.0
        adj = (dist < threshold).float()
        
        # Normalize adjacency matrix
        rowsum = adj.sum(dim=-1, keepdim=True)
        adj = adj / torch.clamp(rowsum, min=1e-9)
        
        return adj

    def forward(self, neighbors, agent_embed):
        """
        Forward pass adhering strictly to the defined Module 3 contract.
        
        Inputs:
            neighbors: Tensor [B, N, 4] (pre-filtered neighbors' features)
            agent_embed: Tensor [B, T_past, D] (temporal embedding of ego agent)
            
        Output:
            social_embed: Tensor [B, D]
        """
        B, N, _ = neighbors.shape
        _, T_past, D = agent_embed.shape
        
        # 1. Construct graph representation (Adjacency Matrix)
        adj = self.build_adjacency_matrix(neighbors)
        
        # 2. Encode neighbor node features
        # neighbors: [B, N, 4] -> node_features: [B, N, D]
        node_features = self.node_encoder(neighbors)
        
        # 3. Apply Graph Convolutions to model interactions (Collision avoidance / Crowd navigation)
        gcn_out = self.gcn1(node_features, adj)
        gcn_out = self.gcn2(gcn_out, adj)
        
        # 4. Generate global social context (Max pooling across neighbor nodes)
        # [B, N, D] -> [B, D]
        social_context, _ = torch.max(gcn_out, dim=1)
        
        # 5. Extract latest temporal feature of the ego agent
        # agent_embed: [B, T_past, D] -> current_ego_state: [B, D]
        current_ego_state = agent_embed[:, -1, :]
        
        # 6. Fuse ego agent state with social context to generate final social embedding
        fused_features = torch.cat([current_ego_state, social_context], dim=-1)
        social_embed = self.fusion(fused_features)
        
        return social_embed

# --- Contract Testing / Execution Block ---
if __name__ == "__main__":
    # Define contract shapes
    B = 8          # Batch size
    N = 10         # Number of neighbors
    T_past = 4     # Past timesteps (2 seconds @ 2Hz)
    D = 128        # Embedding dimension
    
    # Mock inputs matching the contract
    mock_neighbors = torch.randn(B, N, 4)      # [B, N, 4]
    mock_agent_embed = torch.randn(B, T_past, D) # [B, T_past, D]
    
    # Initialize module
    social_module = SocialEncoder(neighbor_dim=4, embed_dim=D, temporal_dim=D)
    
    # Forward pass
    social_embed = social_module(mock_neighbors, mock_agent_embed)
    
    print(f"Input neighbors shape: {mock_neighbors.shape}")
    print(f"Input agent_embed shape: {mock_agent_embed.shape}")
    print(f"Output social_embed shape: {social_embed.shape}") # Should be [B, D]