FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV (required by opencv-python)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-specific PyTorch as requested
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY app/ ./app/
COPY checkpoints/ ./checkpoints/

EXPOSE 8000

CMD echo "--- Checkpoint Directory Contents ---" && \
    ls -l checkpoints/ && \
    if [ -f checkpoints/eval_summary.json ]; then \
    echo "" && echo "--- Evaluation Summary ---" && cat checkpoints/eval_summary.json; \
    fi && \
    if [ -f checkpoints/training_plot_lr_5e-05_wd_0p0.png ]; then \
    echo "" && echo "--- Training Plot Found ---" && echo "Path: checkpoints/training_plot_lr_5e-05_wd_0p0.png"; \
    fi && \
    echo "" && \
    echo "==========================================================" && \
    echo "Running this container starts the FastAPI application automatically." && \
    echo "" && \
    echo "To access the API from your local machine, run the container with:" && \
    echo "  docker run -p 8000:8000 <image_name>" && \
    echo "" && \
    echo "Then open your browser at:" && \
    echo "  Server: http://127.0.0.1:8000" && \
    echo "  Swagger UI Docs: http://127.0.0.1:8000/docs" && \
    echo "==========================================================" && \
    echo "" && \
    echo "--- Starting FastAPI ---" && \
    uvicorn app.api:app --host 0.0.0.0 --port 8000
