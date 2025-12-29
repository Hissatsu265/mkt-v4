#!/bin/bash
# ==============================================
# Script: install_all_libraries.sh
# Purpose: Install all libraries and custom nodes
# ==============================================

set -e  # Stop script if error occurs

echo "🚀 Starting installation of all libraries..."
echo "================================================"

# ==============================================
# STEP 1: INSTALL PYTORCH FIRST (REQUIRED BY OTHER LIBRARIES)
# ==============================================
echo ""
echo "📦 STEP 1: Installing PyTorch with CUDA 12.8 (Required first)"
echo "------------------------------------------------"

echo "⬇️  Installing PyTorch, TorchVision, TorchAudio..."
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

echo "✅ Step 1 completed!"

# ==============================================
# STEP 2: INSTALL BASIC LIBRARIES
# ==============================================
echo ""
echo "📦 STEP 2: Installing basic libraries"
echo "------------------------------------------------"

echo "⬇️  Installing opencv-python and sageattention..."
pip install opencv-python 
echo "⬇️  Installing einops..."
pip install einops

echo "⬇️  Installing pymongo..."
pip install pymongo

echo "⬇️  Installing motor..."
pip install motor

echo "⬇️  Upgrading pip..."
pip install --upgrade pip

echo "⬇️  Installing FastAPI and web libraries..."
pip install fastapi==0.115.0
pip install uvicorn[standard]==0.32.0
pip install pydantic==2.11.7

echo "⬇️  Installing Redis..."
pip install redis==5.2.1

echo "⬇️  Installing aiofiles..."
pip install aiofiles==24.1.0

echo "⬇️  Installing python-multipart..."
pip install python-multipart==0.0.12

echo "⬇️  Installing ONNX..."
pip install onnx onnxruntime

echo "⬇️  Installing mutagen..."
pip install mutagen

echo "⬇️  Installing mediapipe..."
pip install mediapipe

echo "⬇️  Installing pyngrok..."
pip install pyngrok

echo "⬇️  Installing from requirements files..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found"
fi

if [ -f "requirements0.txt" ]; then
    pip install -r requirements0.txt
else
    echo "⚠️  requirements0.txt not found"
fi

if [ -f "requirements1.txt" ]; then
    pip install -r requirements1.txt
else
    echo "⚠️  requirements1.txt not found"
fi

echo "✅ Step 2 completed!"

# ==============================================
# STEP 3: INSTALL CUSTOM NODES
# ==============================================
echo ""
echo "📦 STEP 3: Installing Custom Nodes"
echo "------------------------------------------------"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_NODES_DIR="$SCRIPT_DIR/ComfyUI/custom_nodes"

if [ -d "$CUSTOM_NODES_DIR" ]; then
    echo "📂 Custom nodes path: $CUSTOM_NODES_DIR"
    cd "$CUSTOM_NODES_DIR" || { echo "❌ Cannot access custom_nodes directory!"; exit 1; }
    
    # List of custom nodes to install
    NODES=(
        "ComfyUI-WanVideoWrapper"
        "InfiniteTalk"
        "audio-separation-nodes-comfyui"
        "comfyui-kjnodes"
        "comfyui-videohelpersuite"
        "ComfyUI-MelBandRoFormer"
    )
    
    # Loop through each node
    for NODE in "${NODES[@]}"; do
        NODE_PATH="$CUSTOM_NODES_DIR/$NODE"
        REQ_FILE="$NODE_PATH/requirements.txt"
        
        if [ -d "$NODE_PATH" ]; then
            echo "-------------------------------------------------"
            echo "📦 Processing: $NODE"
            cd "$NODE_PATH" || continue
            
            if [ -f "$REQ_FILE" ]; then
                echo "📘 Installing libraries from $REQ_FILE..."
                pip install -r requirements.txt --no-cache-dir
            else
                echo "⚠️  No requirements.txt file in $NODE"
            fi
            
            cd "$CUSTOM_NODES_DIR" || exit
        else
            echo "⚠️  Skipping: $NODE (directory does not exist)"
        fi
    done
    
    echo "✅ Step 3 completed!"
else
    echo "⚠️  ComfyUI/custom_nodes directory not found, skipping this step"
fi

# ==============================================
# COMPLETED
# ==============================================
echo ""
echo "================================================"
echo "🎉 ALL LIBRARIES INSTALLED SUCCESSFULLY!"
echo "================================================"
echo ""
echo "📋 Summary:"
echo "  ✓ PyTorch with CUDA 12.8 installed"
echo "  ✓ Basic libraries installed"
echo "  ✓ Custom nodes installed (if available)"
echo ""
echo "💡 Tip: Verify installation with: python -c 'import torch; print(torch.cuda.is_available())'"
echo ""