#!/bin/bash

# Xác định thư mục chứa script này
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "📦 Model Download Script"
echo "=========================================="

# ============================================
# PHẦN 1: Download models cho WanVideo
# ============================================
echo ""
echo "🎬 Part 1: Downloading WanVideo models..."
echo "------------------------------------------"

BASE_DIR="$SCRIPT_DIR/ComfyUI/models"

# Các thư mục con cần tạo
DIFFUSION_DIR="$BASE_DIR/diffusion_models"
TEXT_ENCODER_DIR="$BASE_DIR/text_encoders"
CLIP_VISION_DIR="$BASE_DIR/clip_vision"
VAE_DIR="$BASE_DIR/vae"
LORA_DIR="$BASE_DIR/loras"

echo "📁 Base directory: $BASE_DIR"

# # Tạo thư mục nếu chưa tồn tại
# echo "🔧 Creating directories if not exist..."
# mkdir -p "$DIFFUSION_DIR" "$TEXT_ENCODER_DIR" "$CLIP_VISION_DIR" "$VAE_DIR" "$LORA_DIR"

# echo "📥 Downloading WanVideo models..."

# # Download Wan 2.1 I2V 14B 480P Q8
# wget -nc -O "$DIFFUSION_DIR/wan2.1-i2v-14b-480p-Q8_0.gguf" \
# "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q8_0.gguf?download=true"

# # Download MelBandRoFormer
# wget -nc -O "$DIFFUSION_DIR/MelBandRoformer_fp16.safetensors" \
# "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors?download=true"

# # Download UMT5 Text Encoder
# wget -nc -O "$TEXT_ENCODER_DIR/umt5-xxl-enc-bf16.safetensors" \
# "https://huggingface.co/Serenak/chilloutmix/resolve/main/umt5-xxl-enc-bf16.safetensors"

# # Download CLIP Vision
# wget -nc -O "$CLIP_VISION_DIR/clip_vision_h.safetensors" \
# "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"

# # Download VAE
# wget -nc -O "$VAE_DIR/Wan2_1_VAE_bf16.safetensors" \
# "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"

# # Download InfiniteTalk
# wget -nc -O "$DIFFUSION_DIR/Wan2_1-InfiniteTalk_Single_Q8.gguf" \
# "https://huggingface.co/Kijai/WanVideo_comfy_GGUF/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf"

# # Download Lightx2v LoRA
# wget -nc -O "$LORA_DIR/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
# "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

# # Download wav2vec2
# mkdir -p "$BASE_DIR/wav2vec2"
# wget -nc -O "$BASE_DIR/wav2vec2/wav2vec2-chinese-base_fp16.safetensors" \
# "https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/main/wav2vec2-chinese-base_fp16.safetensors"

echo "✅ WanVideo models downloaded successfully!"
echo "=========================================="
echo "📦 Download anymateme-visualengine"
echo "=========================================="

# Đường dẫn file zip
ZIP_FILE="$SCRIPT_DIR/anymateme-visualengine.zip"
ZIP_URL="https://huggingface.co/datasets/Hissatsu265/mkt/resolve/main/anymateme-visualengine.zip"

echo "📁 Current directory: $SCRIPT_DIR"

# Kiểm tra xem file zip đã tồn tại chưa
if [ -f "$ZIP_FILE" ]; then
    echo "✔ File zip already exists: $ZIP_FILE"
    echo "🗑️  Removing old zip file..."
    rm -f "$ZIP_FILE"
fi

# Download file zip
echo "⬇ Downloading anymateme-visualengine.zip..."
wget -O "$ZIP_FILE" "$ZIP_URL"

if [ $? -eq 0 ]; then
    echo "✅ Download completed!"
    
    # Giải nén file vào thư mục hiện tại
    echo "📦 Extracting zip file to current directory..."
    cd "$SCRIPT_DIR" || exit 1
    unzip -o "$ZIP_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✅ Extraction completed!"
        
        # Xóa file zip sau khi giải nén
        echo "🗑️  Cleaning up zip file..."
        rm -f "$ZIP_FILE"
        echo "✅ Cleanup completed!"
    else
        echo "❌ Extraction failed!"
        exit 1
    fi
else
    echo "❌ Download failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 Done!"
echo "=========================================="
# ============================================
# PHẦN 2: Download models cho Qwen-Image
# ============================================
echo ""
echo "🖼️  Part 2: Downloading Qwen-Image models..."
echo "------------------------------------------"

QWEN_BASE_DIR="$SCRIPT_DIR/anymateme-visualengine/ComfyUI"

if [ ! -d "$QWEN_BASE_DIR" ]; then
    echo "❌ Không tìm thấy thư mục: $QWEN_BASE_DIR"
    echo "⏩ Skipping Qwen-Image models..."
else
    cd "$QWEN_BASE_DIR" || exit 1
    
    mkdir -p models/vae
    mkdir -p models/text_encoders
    mkdir -p models/loras
    mkdir -p models/diffusion_models

    echo "Checking and downloading Qwen-Image models..."

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

    # 2. Text Encoder
    download_if_missing "models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
      "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"

    # 3. LoRAs
    download_if_missing "models/loras/Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors" \
      "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors"

    download_if_missing "models/loras/Qwen-Image-Lightning-4steps-V1.0.safetensors" \
      "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors"

    download_if_missing "models/loras/qwen-edit-skin_1.1_000002000.safetensors" \
      "https://huggingface.co/tlennon-ie/qwen-edit-skin/resolve/main/qwen-edit-skin_1.1_000002000.safetensors"

    download_if_missing "models/loras/Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors" \
      "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"

    # 4. Diffusion Model
    download_if_missing "models/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors" \
      "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"

    echo "✅ Qwen-Image models are ready!"
fi



# ============================================
# KẾT THÚC
# ============================================
echo ""
echo "=========================================="
echo "🎉 All downloads completed!"
echo "=========================================="