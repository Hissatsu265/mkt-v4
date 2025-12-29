#!/bin/bash

cd /root/anymateme-visualengine/ComfyUI || exit 1

mkdir -p models/vae
mkdir -p models/text_encoders
mkdir -p models/loras
mkdir -p models/diffusion_models
mkdir -p models/controlnet

echo "Checking and downloading models..."

# Function to download only if file does not exist
download_if_missing () {
    local file_path=$1
    local url=$2

    if [ -f "$file_path" ]; then
        echo "✔ Already exists: $file_path"
    else
        echo "⬇ Downloading: $file_path"
        wget -O "$file_path" "$url"
    fi
}


# NEW: Diffusion (additional)
download_if_missing "models/diffusion_models/qwen_image_fp8_e4m3fn.safetensors" \
  "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"

# NEW: ControlNet model
download_if_missing "models/controlnet/Qwen-Image-InstantX-ControlNet-Inpainting.safetensors" \
  "https://huggingface.co/Comfy-Org/Qwen-Image-InstantX-ControlNets/resolve/main/split_files/controlnet/Qwen-Image-InstantX-ControlNet-Inpainting.safetensors"
download_if_missing "models/loras/Qwen-Image-Lightning-4steps-V2.0.safetensors" \
  "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V2.0.safetensors"

echo "✅ All models are ready!"
