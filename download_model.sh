#!/bin/bash

# Xác định thư mục chứa script này
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR/ComfyUI/models"

# Các thư mục con cần tạo
DIFFUSION_DIR="$BASE_DIR/diffusion_models"
TEXT_ENCODER_DIR="$BASE_DIR/text_encoders"
CLIP_VISION_DIR="$BASE_DIR/clip_vision"
VAE_DIR="$BASE_DIR/vae"
LORA_DIR="$BASE_DIR/loras"

echo "📁 Base directory: $BASE_DIR"

# Tạo thư mục nếu chưa tồn tại
echo "🔧 Creating directories if not exist..."
mkdir -p "$DIFFUSION_DIR" "$TEXT_ENCODER_DIR" "$CLIP_VISION_DIR" "$VAE_DIR" "$LORA_DIR"

echo "📥 Downloading models..."

# Download Wan 2.1 I2V 14B 480P Q8
wget -nc -O "$DIFFUSION_DIR/wan2.1-i2v-14b-480p-Q8_0.gguf" \
"https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q8_0.gguf?download=true"

# Download Wan 2.1 I2V 14B 480P Q4
# wget -nc -O "$DIFFUSION_DIR/wan2.1-i2v-14b-480p-Q4_0.gguf" \
# "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q4_0.gguf"

# Download MelBandRoFormer
wget -nc -O "$DIFFUSION_DIR/MelBandRoformer_fp16.safetensors" \
"https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors?download=true"

# Download UMT5 Text Encoder
wget -nc -O "$TEXT_ENCODER_DIR/umt5-xxl-enc-bf16.safetensors" \
"https://huggingface.co/Serenak/chilloutmix/resolve/main/umt5-xxl-enc-bf16.safetensors"

# Download CLIP Vision
wget -nc -O "$CLIP_VISION_DIR/clip_vision_h.safetensors" \
"https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"

# Download VAE
wget -nc -O "$VAE_DIR/Wan2_1_VAE_bf16.safetensors" \
"https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"

# Download InfiniteTalk
wget -nc -O "$DIFFUSION_DIR/Wan2_1-InfiniteTalk_Single_Q8.gguf" \
"https://huggingface.co/Kijai/WanVideo_comfy_GGUF/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf"

# Download Lightx2v LoRA
wget -nc -O "$LORA_DIR/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
"https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"


mkdir -p "$BASE_DIR/wav2vec2"
# Tải file wav2vec2 từ Hugging Face
wget -O "$BASE_DIR/wav2vec2/wav2vec2-chinese-base_fp16.safetensors" \
"https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/main/wav2vec2-chinese-base_fp16.safetensors"


echo "✅ All models downloaded successfully!"
