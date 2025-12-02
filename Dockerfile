# =========================
#  Base Image: CUDA + Ubuntu
# =========================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 is default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# -------------------------
# Create working directory
# -------------------------
WORKDIR /app

# Copy project files
COPY . /app

# -------------------------
# Install Python packages
# -------------------------
RUN pip install --upgrade pip

# Install base dependencies
RUN pip install \
    fastapi==0.115.0 \
    uvicorn[standard]==0.32.0 \
    pydantic==2.11.7 \
    redis==5.2.1 \
    aiofiles==24.1.0 \
    python-multipart==0.0.12 \
    pymongo motor \
    einops \
    mutagen \
    mediapipe \
    pyngrok \
    opencv-python \
    sageattention \
    numpy==1.26.4

# Install ONNX dependencies
RUN pip install onnx onnxruntime

# Install PyTorch 2.8 (CUDA 12.8)
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install remaining project dependencies
RUN pip install -r requirements.txt || true
RUN pip install -r requirements0.txt || true
RUN pip install -r requirements1.txt || true

# -------------------------
# Model Downloads
# -------------------------
RUN chmod +x download_model.sh && ./download_model.sh
RUN chmod +x download_model_image.sh && ./download_model_image.sh
RUN chmod +x install_custom_nodes.sh && ./install_custom_nodes.sh

# -------------------------
# Expose API port
# -------------------------
EXPOSE 8003

# -------------------------
# Start API Server
# -------------------------
CMD ["python", "run.py"]
