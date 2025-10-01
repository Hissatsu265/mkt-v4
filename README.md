# Marketing Video AI - API Documentation

## System Requirements

- Python 3.10+
- PyTorch >= 2.7.0.dev
- CUDA 12.4
- FFmpeg
- MongoDB

## Installation

### 1. Install Basic Dependencies

```bash
pip install --upgrade pip
pip install pyngrok
pip install pymongo
pip install motor
pip install fastapi==0.115.0
pip install uvicorn[standard]==0.32.0  
pip install pydantic==2.11.7
pip install redis==5.2.1
pip install aiofiles==24.1.0
pip install python-multipart==0.0.12
pip install -r requirements.txt
pip install mutagen
```

### 2. Install PyTorch

**Important Note:** PyTorch >= 2.7.0.dev is required for system stability.

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
```

### 3. Install FFmpeg

```bash
apt-get update && apt-get install -y ffmpeg
```

### 4. Configure Ngrok

```bash
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
```

### 5. Install ComfyUI Custom Nodes

Navigate to the custom_nodes directory:

```bash
cd marketing-video-ai/ComfyUI/custom_nodes
```

Install each node:

```bash
# ComfyUI-WanVideoWrapper
cd ComfyUI-WanVideoWrapper
pip install -r requirements.txt
cd ..

# InfiniteTalk
cd InfiniteTalk
pip install -r requirements.txt
cd ..

# audio-separation-nodes-comfyui
cd audio-separation-nodes-comfyui
pip install -r requirements.txt
cd ..

# comfyui-kjnodes
cd comfyui-kjnodes
pip install -r requirements.txt
cd ..

# comfyui-videohelpersuite
cd comfyui-videohelpersuite
pip install -r requirements.txt
cd ..

# ComfyUI-MelBandRoFormer
cd ComfyUI-MelBandRoFormer
pip install -r requirements.txt
cd ..
```

## Download Models

### 1. Create Model Directories

```bash
mkdir -p marketing-video-ai/ComfyUI/models/diffusion_models
mkdir -p marketing-video-ai/ComfyUI/models/text_encoders
mkdir -p marketing-video-ai/ComfyUI/models/clip_vision
mkdir -p marketing-video-ai/ComfyUI/models/vae
mkdir -p marketing-video-ai/ComfyUI/models/loras
```

### 2. Download Required Models

**Note:** The paths in the wget commands below are relative. Adjust them according to your actual environment path.

```bash
# Download Wan 2.1 I2V 14B 480P Q8
wget -O marketing-video-ai/ComfyUI/models/diffusion_models/wan2.1-i2v-14b-480p-Q8_0.gguf \
"https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q8_0.gguf?download=true"

# Download Wan 2.1 I2V 14B 480P Q4
wget -O marketing-video-ai/ComfyUI/models/diffusion_models/wan2.1-i2v-14b-480p-Q4_0.gguf \
"https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q4_0.gguf"

# Download MelBandRoFormer
wget -O marketing-video-ai/ComfyUI/models/diffusion_models/MelBandRoformer_fp16.safetensors \
"https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors?download=true"

# Download UMT5 Text Encoder
wget -O marketing-video-ai/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors \
"https://huggingface.co/Serenak/chilloutmix/resolve/main/umt5-xxl-enc-bf16.safetensors"

# Download CLIP Vision
wget -O marketing-video-ai/ComfyUI/models/clip_vision/clip_vision_h.safetensors \
"https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"

# Download VAE
wget -O marketing-video-ai/ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors \
"https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"

# Download InfiniteTalk
wget -O marketing-video-ai/ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk_Single_Q8.gguf \
"https://huggingface.co/Kijai/WanVideo_comfy_GGUF/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf"

# Download Lightx2v LoRA
wget -O marketing-video-ai/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
"https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
```

## API Usage Guide

### Starting the Server

```bash
python run.py
```

### API Endpoints

#### 1. Create Video

**Endpoint:** `POST /api/v1/videos/create`

Creates a new video from images and audio.

**Request Structure:**

```bash
curl -X POST "http://localhost:8000/api/v1/videos/create" \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": [
      "/path/to/image1.jpg",
      "/path/to/image2.jpg"
    ],
    "prompts": ["", ""],
    "audio_path": "/path/to/audio.wav",
    "resolution": "1280x720"
  }'
```

**Parameters:**

- `image_paths` (array, required): Array of image file paths
- `prompts` (array, required): Array of prompts corresponding to each image (can be empty strings "")
- `audio_path` (string, required): Path to audio file
- `resolution` (string, required): Output video resolution

**Supported Resolutions:**

- `1280x720` - HD Landscape
- `854x480` - SD Landscape
- `720x1280` - HD Portrait
- `480x854` - SD Portrait
- And more...

**Response:**

```json
{
  "job_id": "unique-job-id-here",
  "status": "processing",
  "message": "Video creation job started"
}
```

#### 2. Check Job Status

**Endpoint:** `GET /api/v1/jobs/{job_id}/status`

Check the progress of a video creation job.

**Request Structure:**

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}/status"
```

Replace `{job_id}` with the actual job ID returned from the create video endpoint.

**Response:**

```json
{
  "job_id": "unique-job-id-here",
  "status": "completed",
  "progress": 100,
  "video_url": "/path/to/output/video.mp4",
  "created_at": "2025-10-01T10:00:00Z",
  "completed_at": "2025-10-01T10:05:00Z"
}
```

**Possible Status Values:**

- `pending` - Waiting to be processed
- `processing` - Currently processing
- `completed` - Successfully completed
- `failed` - Failed to process

## Usage Examples

### Example 1: Create Simple Video

```bash
curl -X POST "http://localhost:8000/api/v1/videos/create" \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": [
      "/home/user/images/photo1.jpg"
    ],
    "prompts": [""],
    "audio_path": "/home/user/audio/voice.wav",
    "resolution": "1280x720"
  }'
```

### Example 2: Create Video with Multiple Images

```bash
curl -X POST "http://localhost:8000/api/v1/videos/create" \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": [
      "/home/user/images/scene1.jpg",
      "/home/user/images/scene2.jpg",
      "/home/user/images/scene3.jpg"
    ],
    "prompts": [
      "A person talking",
      "Smiling face",
      "Waving goodbye"
    ],
    "audio_path": "/home/user/audio/narration.wav",
    "resolution": "720x1280"
  }'
```

## Error Handling

The API returns standard HTTP error codes:

- `200` - Success
- `400` - Bad Request
- `404` - Not Found
- `500` - Internal Server Error

**Error Response Example:**

```json
{
  "error": "Invalid resolution format",
  "message": "Resolution must be in format WIDTHxHEIGHT",
  "status_code": 400
}
```

## Important Notes

- Ensure all file paths (image_paths, audio_path) exist and have read permissions
- Supported audio formats: WAV, MP3, OGG
- Supported image formats: JPG, JPEG, PNG
- Processing time depends on resolution and number of images
- GPU is recommended for faster processing
- Minimum recommended VRAM: 12GB

## Troubleshooting

If you encounter issues during installation or usage, please check:

1. Server logs for error messages
2. Installed library versions
3. Available disk space (models are very large)
4. GPU VRAM availability (minimum 12GB recommended)
5. PyTorch version compatibility (>= 2.7.0.dev required)

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions, please open an issue on the project repository.
