import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.schemas import InferenceRequest, InferenceResponse
from app.full_model import IntentAwareTrajectoryModel

# Globals to hold our loaded model securely
MODEL_STATE = {}
CHECKPOINT_PATH = "checkpoints/best_model.pt"


def _validate_inference_shapes(
    agent_tensor: torch.Tensor,
    neighbors_tensor: torch.Tensor,
    map_tensor: torch.Tensor,
) -> None:
    """Validate request tensor shapes against model contract before forward pass."""
    if agent_tensor.dim() != 3:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid agent shape {tuple(agent_tensor.shape)}. Expected [B, T_past, 4].",
        )

    if agent_tensor.size(-1) != 4:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid agent feature dimension {agent_tensor.size(-1)}. "
                "Expected last dimension = 4 (x, y, vx, vy)."
            ),
        )

    if neighbors_tensor.dim() not in (3, 4):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid neighbors shape {tuple(neighbors_tensor.shape)}. "
                "Expected [B, N, 4] or [B, N, T_past, 4]."
            ),
        )

    if neighbors_tensor.size(-1) != 4:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid neighbors feature dimension {neighbors_tensor.size(-1)}. "
                "Expected last dimension = 4 (x, y, vx, vy)."
            ),
        )

    if map_tensor.dim() != 4:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid map_img shape {tuple(map_tensor.shape)}. Expected [B, 3, H, W].",
        )

    if map_tensor.size(1) != 3:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid map_img channel dimension {map_tensor.size(1)}. "
                "Expected 3 channels (RGB-like raster)."
            ),
        )

    b_agent = agent_tensor.size(0)
    b_neighbors = neighbors_tensor.size(0)
    b_map = map_tensor.size(0)
    if b_agent != b_neighbors or b_agent != b_map:
        raise HTTPException(
            status_code=422,
            detail=(
                "Batch size mismatch across inputs: "
                f"agent={b_agent}, neighbors={b_neighbors}, map_img={b_map}."
            ),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event to load the model checkpoint precisely once upon server startup.
    Hackathon friendly: Fails gracefully if the file isn't found allowing demo mode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading inference API on device: {device}")

    model = IntentAwareTrajectoryModel(embed_dim=128, num_modes=6, future_steps=6)

    checkpoint_loaded = False
    if os.path.exists(CHECKPOINT_PATH):
        try:
            # Map location safely loads GPU checkpoints onto CPUs for Docker/Judges
            state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(state_dict)
            checkpoint_loaded = True
            print("Checkpoint loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint weights: {e}")
    else:
        print(
            f"Warning: No checkpoint found at {CHECKPOINT_PATH}. Using untrained parameters for Demo Mode!"
        )

    model.to(device)
    model.eval()

    MODEL_STATE["model"] = model
    MODEL_STATE["device"] = device
    MODEL_STATE["checkpoint_loaded"] = checkpoint_loaded

    yield

    # Cleanup memory on shutdown
    MODEL_STATE.clear()


# Initialize FastAPI with the lifespan handler
app = FastAPI(
    title="Intent-Aware Trajectory Prediction API",
    description="Hackathon module for multi-agent trajectory forecasting",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "device": str(MODEL_STATE.get("device", "unknown")),
        "checkpoint_loaded": MODEL_STATE.get("checkpoint_loaded", False),
        "message": "Trajectory Prediction API is running!",
    }


@app.post("/predict", response_model=InferenceResponse)
def predict_trajectory(request: InferenceRequest):
    """
    Runs the forward pass on incoming agent, neighbor, and map data.
    """
    model: IntentAwareTrajectoryModel = MODEL_STATE.get("model")
    device: torch.device = MODEL_STATE.get("device")

    if model is None:
        raise HTTPException(status_code=500, detail="Server model is not initialized.")

    try:
        # Convert nested lists to strict float32 tensors and map to the active device
        agent_tensor = torch.tensor(request.agent, dtype=torch.float32, device=device)
        neighbors_tensor = torch.tensor(
            request.neighbors, dtype=torch.float32, device=device
        )
        map_tensor = torch.tensor(request.map_img, dtype=torch.float32, device=device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid tensor data formats: {e}")

    _validate_inference_shapes(agent_tensor, neighbors_tensor, map_tensor)

    # Forward pass safely without gradient tracking
    with torch.no_grad():
        try:
            # Note: The model's forward function inherently handles batch-size
            # and tensor-dimension sanity checks, avoiding duplicating code here.
            trajectories, mode_logits = model(
                agent=agent_tensor, neighbors=neighbors_tensor, map_img=map_tensor
            )

            # Apply softmax to output proper probabilities
            mode_probabilities = F.softmax(mode_logits, dim=-1)

        except ValueError as ve:
            # Catch internal validation errors from full_model.py
            raise HTTPException(
                status_code=422, detail=f"Model shape validation failed: {ve}"
            )
        except RuntimeError as re:
            # Shape/runtime issues from deep PyTorch ops should map to client errors
            raise HTTPException(
                status_code=422, detail=f"Invalid input shape for model: {re}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # Convert back to standard Python lists for standard JSON serialization returning
    return InferenceResponse(
        trajectories=trajectories.cpu().tolist(),
        mode_probabilities=mode_probabilities.cpu().tolist(),
    )
