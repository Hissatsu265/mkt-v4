#!/bin/bash

# Script to install all requirements for Video Generation API
# Created: 2025-12-05

set -e  # Exit on error

echo "=========================================="
echo "Video Generation API - Install Requirements"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD=$(command -v python3 || command -v python)
echo "Using Python: $PYTHON_CMD"
echo "Python version: $($PYTHON_CMD --version)"
echo ""

# Upgrade pip
echo "Step 1: Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip
echo ""

# Install PyTorch FIRST (required by sageattention and other packages)
echo "Step 2: Installing PyTorch with CUDA 12.8 support..."
echo "This is required before installing other dependencies."
$PYTHON_CMD -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
echo ""

# Install Gradio first with pinned version (to set dependency constraints)
echo "Step 3: Installing Gradio with pinned version..."
$PYTHON_CMD -m pip install gradio==4.44.1 gradio_client==1.3.0
echo ""

# Install core dependencies that depend on PyTorch
echo "Step 4: Installing core dependencies..."
$PYTHON_CMD -m pip install opencv-python einops
# Install sageattention without build isolation (it needs torch during build)
# Note: sageattention requires CUDA/GPU support, skip if not available
echo "Attempting to install sageattention (GPU-only, may fail on CPU)..."
$PYTHON_CMD -m pip install --no-build-isolation sageattention || echo "Warning: sageattention installation failed (GPU required). Continuing without it..."
echo ""

# Install specific versions of critical packages (compatible with Gradio 4.44.1)
echo "Step 5: Installing FastAPI and Uvicorn..."
$PYTHON_CMD -m pip install "fastapi>=0.115.2" "uvicorn[standard]==0.32.0" "pydantic>=2.11.10,<=2.12.4"
echo ""

# Install database and queue
echo "Step 6: Installing database and queue dependencies..."
$PYTHON_CMD -m pip install pymongo motor redis==5.2.1
echo ""

# Install utilities
echo "Step 7: Installing utilities..."
$PYTHON_CMD -m pip install "aiofiles>=22.0,<24.0" "python-multipart>=0.0.18" onnx onnxruntime mutagen mediapipe pyngrok
echo ""

# Install all remaining requirements from consolidated file
echo "Step 8: Installing remaining requirements from requirements_all.txt..."
$PYTHON_CMD -m pip install -r requirements_all.txt
echo ""

echo "=========================================="
echo "Installation completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Configure .env file: cp example.env .env"
echo "2. Download models: bash download_model.sh && bash download_model_image.sh"
echo "3. Install custom nodes: bash install_custom_nodes.sh"
echo "4. Start the API: python run.py"
echo ""
