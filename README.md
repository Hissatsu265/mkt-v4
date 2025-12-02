# Video Generation API - User Guide

API for creating videos from images and audio with transition effects and dolly zoom capabilities.

## System Requirements

### Hardware
- **Storage**: 100GB free space

### Software
- **Python**: 3.10
- **PyTorch**: 2.8
- **NumPy**: 1.26.4
- **CUDA**: 12.8 (recommended)

---

## Installation

### Step 1: Environment Configuration

Update the `.env` file based on `example.env`:

```bash
cp example.env .env
# Edit .env file according to your configuration
```

### Step 2: Install Required Libraries

```bash

pip install opencv-python sageattention
pip install einops
pip install pymongo
pip install motor
pip install --upgrade pip
pip install fastapi==0.115.0
pip install uvicorn[standard]==0.32.0  
pip install pydantic==2.11.7
pip install redis==5.2.1
pip install aiofiles==24.1.0
pip install python-multipart==0.0.12
pip install onnx onnxruntime
pip install mutagen
pip install mediapipe
pip install pyngrok

pip install -r requirements.txt
pip install -r requirements0.txt
pip install -r requirements.txt
pip install -r requirements1.txt
pip install onnx onnxruntime
```

### Step 3: Download AI Models

```bash
bash download_model.sh
bash download_model_image.sh
```

### Step 4: Install Custom Nodes

```bash
bash install_custom_nodes.sh
```

### Step 5: Install PyTorch

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

### Step 6: Add ngok token 

```bash
ngrok config add-authtoken 33ooTAhfqqhfHoWrrkThLxy4niD_H8N9ENov1PVdpbL3yywF
```

---

## Start API Server

```bash
python run.py
```

API will run at: `https://hailee-unrepresentational-ronnie.ngrok-free.dev ` and ` http://localhost:8003`

---

## API Usage

### 1. Create Video

**Endpoint:** `POST /api/v1/videos/create`

Create a new video from images and audio.

#### Request:

```bash
curl -X POST "https://hailee-unrepresentational-ronnie.ngrok-free.dev/api/v1/videos/create_christmas_campain " \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": ["https://cms.anymateme.pro/assets/e149cb1f-5b37-44e8-b248-c15bb79e31b6",""],
    "prompts": ["",""],
    "audio_path": "https://cms.anymateme.pro/assets/fba5eb69-d9aa-4280-9d10-c8de363c0e50",
    "resolution": "9:16"
}'
```

#### Parameters:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_paths` | array | ✅ | List of local image file paths |
| `prompts` | array | ❌ | List of prompts for each image (optional, can be empty `""`) |
| `audio_path` | string | ✅ | Local audio file path |
| `resolution` | string | ✅ | Output video aspect ratio |

#### Supported Resolutions:

- `16:9` - Landscape (Horizontal)
- `9:16` - Portrait (Vertical)

#### Response:

```json
{
  "job_id": "unique-job-id-here",
  "status": "processing",
  "message": "Video creation job started"
}
```

---

### 2. Check Job Status

**Endpoint:** `GET /api/v1/jobs/{job_id}/status`

#### Request:

```bash
curl "https://hailee-unrepresentational-ronnie.ngrok-free.dev/api/v1/jobs/{job_id}/status"
```

```bash
curl "http://localhost:8003/api/v1/jobs/{job_id}/status" 
```

#### Response:

```json
{
  "job_id": "1b26b5ab-090a-4d73-9361-10f212062ac9",
  "status": "completed",
  "progress": 100,
  "video_path": "https://cms.anymateme.pro/assets/a78cb4a9-373d-42c4-838d-76fae1ee9962",
  "error_message": null,
  "created_at": "2025-10-06T13:50:57.328800",
  "completed_at": "2025-10-06T13:55:13.629765",
  "list_scene": [],
  "queue_position": 1,
  "estimated_wait_time": null,
  "is_processing": null,
  "current_processing_job": null
}
```

#### Status Values:

- `processing` - Job in progress
- `completed` - Job finished successfully
- `failed` - Job failed

#### Important Response Fields:

- **`video_path`**: Path to the generated video (available when `status = "completed"`)
- **`list_scene`**: List of scene transition timestamps (if video has multiple scenes)


4. Sufficient disk space available (50GB+)

---

## License

[Add your license information here]
