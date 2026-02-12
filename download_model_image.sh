#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/anymateme-visualengine/ComfyUI" || { echo "Không tìm thấy thư mục ComfyUI"; exit 1; }
# cd /workspace/anymateme-visualengine/ComfyUI || exit 1

mkdir -p models/vae
mkdir -p models/text_encoders
mkdir -p models/loras
mkdir -p models/diffusion_models

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

# 1. VAE
download_if_missing "models/vae/qwen_image_vae.safetensors" \
  "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"

download_if_missing "models/loras/Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors" \
  "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors"

# 2. Text Encoder
download_if_missing "models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
  "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"

# 3. LoRA
download_if_missing "models/loras/Qwen-Image-Lightning-4steps-V1.0.safetensors" \
  "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors"

download_if_missing "models/loras/qwen-edit-skin_1.1_000002000.safetensors" \
  "https://huggingface.co/tlennon-ie/qwen-edit-skin/resolve/main/qwen-edit-skin_1.1_000002000.safetensors"

download_if_missing "models/loras/Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors" \
  "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"


# 4. Diffusion Model
download_if_missing "models/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors" \
  "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"

echo "✅ All models are ready!"