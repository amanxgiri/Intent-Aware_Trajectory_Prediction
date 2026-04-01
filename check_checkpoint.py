import torch
from app.full_model import IntentAwareTrajectoryModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = IntentAwareTrajectoryModel(embed_dim=128, num_modes=3, future_steps=6)
state_dict = torch.load("checkpoints/best_model.pt", map_location=device)
model.load_state_dict(state_dict)

print("Checkpoint loaded successfully with standardized config")
