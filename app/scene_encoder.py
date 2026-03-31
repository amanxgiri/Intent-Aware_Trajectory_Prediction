import torch
import torch.nn as nn
import torchvision.models as models

class SceneEncoder(nn.Module):
    """
    Module 2 - Spatial Feature Encoding [cite: 5]
    Implements a CNN encoder to extract environmental constraints from 
    rasterized HD maps (nuScenes)[cite: 4].
    """
    def __init__(self, embedding_dim=128):
        super(SceneEncoder, self).__init__()
        
        # Based on MAHE-Mobility-Proposed-Solution: Use ResNet-18 for speed [cite: 4]
        # and efficient feature extraction from BEV images[cite: 2].
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Input Contract: Tensor [3, H, W] representing a 20m x 20m region [cite: 5]
        # ResNet18 typically outputs 512 features from the last convolutional layer.
        num_filters = self.backbone.fc.in_features
        
        # Remove the original classification head
        self.backbone.fc = nn.Identity()
        
        # Projection head to reach the required latent embedding dimension (D) [cite: 5]
        self.projection = nn.Sequential(
            nn.Linear(num_filters, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        """
        Input: x (Tensor [B, 3, H, W]) - Rasterized local map [cite: 5]
        Output: scene_embed (Tensor [B, D]) - Latent embedding of the environment [cite: 5]
        """
        # Extract features using ResNet backbone
        features = self.backbone(x) # [B, 512]
        
        # Map to latent space D
        scene_embed = self.projection(features) # [B, D]
        
        return scene_embed

# Example usage aligning with WorkAllocation interface [cite: 5]
if __name__ == "__main__":
    # Parameters from project docs
    BATCH_SIZE = 8
    D_MODEL = 128 # embedding_dim
    MAP_CHANNELS = 3
    IMAGE_SIZE = 224 # Standard ResNet input size
    
    # Initialize Encoder
    encoder = SceneEncoder(embedding_dim=D_MODEL)
    
    # Mock input representing a batch of rasterized maps [B, 3, H, W]
    mock_map = torch.randn(BATCH_SIZE, MAP_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    
    # Forward Pass
    scene_embedding = encoder(mock_map)
    
    print(f"Input Map Shape: {mock_map.shape}")
    print(f"Output Scene Embedding Shape: {scene_embedding.shape}") # Should be [8, 128]