## Project Overview

Autonomous vehicles operating in urban environments must not only detect pedestrians and cyclists but also anticipate where they are likely to move next. Simply reacting to their current position is not enough for safe navigation—systems need to predict future movement in advance.

This project focuses on **intent and trajectory prediction**, where the goal is to forecast the future path of pedestrians and cyclists based on their recent motion. Given **2 seconds of past movement data (positions/velocity)**, the model predicts their **future positions over the next 3 seconds**.

The challenge lies in the fact that human movement is not always predictable:

- **Multiple possible futures**: A person can turn, stop, or continue straight
- **Social behavior**: People adjust their movement based on others around them
- **Temporal patterns**: Motion changes over time and must be understood sequentially

To address this, the system is designed to:

- Learn movement patterns from **past trajectory data**
- Consider **interactions between nearby agents**
- Generate **multiple possible future paths (multi-modal prediction)**
- Infer likely behavior (**intent**) from observed motion and rasterized maps

The final outcome is a model that takes past movement as input and predicts several realistic future trajectories, helping autonomous systems make **safer and more proactive decisions**.

---

## Model Architecture

The system is organized as a modular prediction pipeline:

**Data Pipeline → Feature Encoding → Social Interaction → Prediction & Deployment**. 

### 1. Data Pipeline

The dataset layer prepares model-ready tensors from nuScenes samples. For each valid target agent, the pipeline extracts:

- past trajectory history in agent-centric coordinates
- neighboring agents within a spatial radius
- a local rasterized map crop
- future target trajectory for supervision

The current dataset contract is:

- `agent`: `[T_past, 4]`, where each step is `[x, y, vx, vy]`
- `neighbors`: `[N, T_past, 4]`, padded to a fixed maximum neighbor count
- `map`: `[3, H, W]`
- `target`: `[T_future, 2]`, where each step is `[x, y]`

### 2. Temporal Encoder

The temporal branch processes the agent’s 2s motion history using a Transformer encoder.

Input:
- `agent`: `[B, T_past, 4]`

Output:
- `agent_embed`: `[B, T_past, D]` 

This branch learns motion dynamics over time and captures sequential movement patterns.

### 3. Scene Encoder

The scene branch processes the rasterized local map using a CNN-based encoder built on ResNet-18 and captures scene context and long term intent.

Input:
- `map`: `[B, 3, H, W]`

Output:
- `scene_embed`: `[B, D]` 

This branch captures environmental context such as drivable area, walkway structure, and nearby map constraints and long term content. The design uses a **20m × 20m local crop**, centered on the target agent, to focus computation on the most relevant spatial region. :contentReference[oaicite:8]{index=8}

### 4. Social Encoder

The social interaction branch models neighboring agents and local crowd behavior using an STGCN-inspired graph encoder.
 We have approached to a **two-layer, 10-meter Graph Convolutional Network (GCN)** utilizing **Gaussian soft-adjacency**.
* **Beyond Isolated Objects:**  Our dual-layer approach captures the "social hops" of urban movement.
*  **The Ripple Effect:** The first layer models how neighbors interact with one another. The second layer passes that collective interaction—the "ripple effect"—to the target agent.
* **Collective Negotiation:** While maintaining a total 20-meter awareness, this architecture allows the model to predict complex group behaviors and path negotiations that a single-layer logic would overlook.

Inputs:
- `neighbors`: `[B, N, 4]`
- `agent_embed`: `[B, T_past, D]`

Output:
- `social_embed`: `[B, D]`

In the current implementation, the dataloader may provide neighbors with a time dimension, and the integration layer adapts them before passing them into the social encoder.

### 5. Fusion and Prediction Head

The final prediction module combines temporal, scene, and social context into one fused representation:

```python
fused = concat(
    agent_embed[:, -1],
    scene_embed,
    social_embed
)
```

This produces a fused feature of shape `[B, 3D]`, which is passed to the multi-modal prediction head. 

The prediction head outputs:

- `trajectories`: `[B, K, T_future, 2]`
- `probabilities`: `[B, K]` 

This allows the system to predict multiple possible future paths rather than forcing a single deterministic outcome.

### 6. Training Objective

The model is trained with a **Winner-Takes-All (WTA)** style objective. The model predicts multiple trajectory modes, and the loss is computed using the mode that best matches the ground-truth future trajectory. This supports multi-modal forecasting and reduces collapse toward a single average path. 

### 7. Metrics Used

The current evaluation setup focuses on:

- **ADE** — Average Displacement Error
- **FDE** — Final Displacement Error
- **MinADE@K**
- **MinFDE@K**

### 8. Performance and Results

The model achieves high-fidelity predictions with stable convergence. Best performance was achieved at **Epoch 3** with a **Validation Loss of 0.3256**.

| Metric (metres)   | Score  |
| :---              | :---   |
| **MinADE@K**      | 0.2358 |
| **MinFDE@K**      | 0.4330 |
| **ADE (Overall)** | 0.2558 |
| **FDE (Overall)** | 0.4775 |


---

## Dataset Used

### Primary Dataset

The project uses the **nuScenes** dataset as the primary data source. It is well-suited for this task because it provides:

- temporal tracking of agents
- multi-agent urban scenes
- HD map information
- realistic motion behavior in complex traffic environments 


### Current Input Representation

For each training sample, the current dataset implementation returns:

- `agent`: past motion history of the target agent in agent-centric coordinates
- `neighbors`: nearby agents represented by position and velocity features
- `map`: a local rasterized semantic map crop
- `target`: future trajectory of the target agent 

### Temporal Configuration

The project currently uses:

- **2 seconds of past motion**
- **3 seconds of future prediction** 

### Agent Categories

The current dataset loader filters categories relevant to motion forecasting, including:

- pedestrians
- bicycles
- motorcycles, depending on the category configuration in the loader implementation 

### Map Representation

The map encoder uses a **20m × 20m** crop around the target agent and rasterizes semantic map layers such as:

- drivable area
- walkway
- pedestrian crossing 

---

## Setup & Installation Instructions

### 1. Create and activate a virtual environment

From the project root:

```bash
python -m venv venv
```

Activate it:

**PowerShell**
```bash
.\venv\Scripts\Activate.ps1
```

**CMD**
```bash
venv\Scripts\activate
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch for your device setup

For local GPU training, install a CUDA-enabled PyTorch build in the same environment.

Example:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 5. Verify PyTorch and CUDA

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 6. Place the nuScenes dataset correctly

The dataset loader expects the dataset root to contain the nuScenes version folder. For example:

```txt
MAHE_MOBILITY/
└── v1.0-mini/
```

When running training or data loading code, pass:
- the dataset root directory as `dataroot`
- the version separately as `v1.0-mini`

---

## How to Run the Code

### 1. Train the model

A smoke-training script can be used to run a short end-to-end training cycle on the full architecture.

Example:

```bash
python scripts/train_smoke.py --dataroot "C:\Users\giria\Documents\MAHE_MOBILITY" --version v1.0-mini
```

This runs the full training path:

- dataset loading
- dataloader batching
- temporal encoder
- scene encoder
- social encoder
- prediction head
- WTA loss
- validation metrics
- checkpoint saving

The best checkpoint is saved to:

```txt
checkpoints/smoke_best_model.pt
```

### 2. Start the API server

Run the FastAPI server from the project root:

```bash
uvicorn app.api:app --reload
```

Once the server starts, open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

### 3. How to use the API

The `/predict` endpoint accepts JSON input with the following shapes:

- `agent`: `[B, T_past, 4]`
- `neighbors`: `[B, N, 4]` or `[B, N, T_past, 4]`
- `map_img`: `[B, 3, H, W]`

For a single sample, a typical structure is:

- `agent`: `[1, 4, 4]`
- `neighbors`: `[1, N, 4, 4]` or `[1, N, 4]`
- `map_img`: `[1, 3, 224, 224]`

The agent feature layout is:

- `[x, y, vx, vy]`

The neighbors follow the same feature layout.

The map input is the rasterized semantic map crop used by the scene encoder.

### 4. Recommended way to test the API

The most reliable way to test the API is to send a real sample from the dataloader rather than manually typing large tensors into Swagger.

You can also use the interactive docs at:

```txt
http://127.0.0.1:8000/docs
```

### 5. API output format

The API returns:

- `trajectories`: `[B, K, T_future, 2]`
- `mode_probabilities`: `[B, K]`

where:
- `K` is the number of predicted future modes
- `T_future` is the number of future timesteps

---

## Example Outputs / Results

### Example Training Results

A successful smoke-training run on CUDA showed stable end-to-end training and checkpoint saving. Among the compared settings, the best-performing configuration was:

- **Learning Rate:** `5e-5`
- **Weight Decay:** `0.0`
- **Best Validation Loss:** `0.3277`
- achieved at **epoch 3** :contentReference[oaicite:25]{index=25}

This indicates that:

- the complete architecture trains end-to-end
- the WTA multimodal setup is functioning correctly
- the model can generate plausible trajectory candidates
- the lower learning rate was more stable than the more aggressive configuration that improved quickly at first but did not perform as well overall on validation. :contentReference[oaicite:26]{index=26}

### Example API Prediction Output

A successful API inference call returned a response containing predicted trajectories and mode probabilities. Example structure:

```json
{
  "trajectories": [
    [
      [
        [-0.0845, 0.0079],
        [0.0537, 0.0539],
        [0.0866, -0.0115],
        [-0.0339, -0.0069],
        [0.1331, -0.0082],
        [-0.1287, -0.0444]
      ],
      [
        [0.0032, -0.0596],
        [-0.1524, 0.0448],
        [0.0599, -0.0428],
        [0.0277, 0.0230],
        [-0.0098, 0.0602],
        [0.1062, -0.0209]
      ],
      [
        [0.0397, 0.0979],
        [0.1246, -0.1246],
        [-0.0668, -0.2123],
        [-0.0048, 0.1250],
        [0.0443, -0.0605],
        [0.0996, -0.0610]
      ]
    ]
  ],
  "mode_probabilities": [
    [0.3078, 0.4570, 0.2352]
  ]
}
```

This output means:

- batch size = `1`
- multiple future trajectory hypotheses were returned
- each hypothesis consists of a sequence of predicted future `(x, y)` positions
- `mode_probabilities` gives the confidence assigned to each predicted mode

### Current Status

At the current stage, the project supports:

- end-to-end training on nuScenes
- multimodal trajectory prediction
- FastAPI-based inference
- checkpoint loading and serving through the API
- CUDA-based local training

The current system is suitable for:

- architecture validation
- training and checkpoint generation
- API-based prediction and testing
- further tuning and extension