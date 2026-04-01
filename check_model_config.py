import torch
from app.full_model import IntentAwareTrajectoryModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = IntentAwareTrajectoryModel().to(device)
model.eval()

print("Model created successfully")
print("Prediction head num_modes:", model.prediction_head.num_modes)
print("Prediction head future_steps:", model.prediction_head.future_steps)

B = 1
T_past = 4
N = 3
H = W = 224

agent = torch.randn(B, T_past, 4, device=device)
neighbors = torch.randn(B, N, T_past, 4, device=device)
map_img = torch.randn(B, 3, H, W, device=device)

with torch.no_grad():
    trajectories, mode_logits = model(agent, neighbors, map_img)

print("trajectories shape:", trajectories.shape)
print("mode_logits shape:", mode_logits.shape)
