# Video Generation API - User Guide

API for creating videos from images and audio with transition effects and dolly zoom capabilities.

## System Requirements

### Hardware
- **Storage**: 50GB free space

### Software
- **Python**: 3.8+
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

### Step 2: Install PyTorch

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
```

### Step 3: Install Required Libraries

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
```

### Step 4: Download AI Models

```bash
bash download_model.sh
```

### Step 5: Install Custom Nodes

```bash
bash install_custom_nodes.sh
```

---

## Start API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

API will run at: `http://localhost:8000`

---

## API Usage

### 1. Create Video

**Endpoint:** `POST /api/v1/videos/create`

Create a new video from images and audio.

#### Request:

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
    "resolution": "16:9"
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
curl "http://localhost:8000/api/v1/jobs/{job_id}/status"
```

#### Response:

```json
{
  "job_id": "1b26b5ab-090a-4d73-9361-10f212062ac9",
  "status": "completed",
  "progress": 100,
  "video_path": "/workspace/marketing-video-ai/outputs/video_400833_f8b36ebd.mp4",
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

---

### 3. Add Video Effects

**Endpoint:** `POST /api/v1/videos/effects`

Add transition effects and dolly zoom to video.

#### Request:

```bash
curl -X POST "http://localhost:8000/api/v1/videos/effects" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/workspace/marketing-video-ai/outputs/video_400833.mp4",
    "transition_times": [12.33],
    "transition_effects": ["slide"],
    "transition_durations": [1.0],
    "dolly_effects": [
      {
        "scene_index": 0,
        "start_time": 1.5,
        "duration": 1.0,
        "zoom_percent": 50,
        "effect_type": "auto_zoom",
        "end_time": 5.0
      },
      {
        "scene_index": 1,
        "start_time": 3.0,
        "duration": 1.5,
        "zoom_percent": 50,
        "effect_type": "manual_zoom",
        "x_coordinate": 100,
        "y_coordinate": 100,
        "end_time": 6.0,
        "end_type": "smooth"
      }
    ]
  }'
```

#### Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_path` | string | Path to video file for adding effects |
| `transition_times` | array | Transition timestamps (in seconds) |
| `transition_effects` | array | Transition effect names |
| `transition_durations` | array | Effect durations (in seconds) |
| `dolly_effects` | array | Dolly zoom effect configurations |

#### Supported Transition Effects:

```
slide, rotate, circle_mask, fade_in, fade_out, fadeout_fadein, 
crossfade, rgb_split, flip_horizontal, flip_vertical, push_blur, 
squeeze_horizontal, wave_distortion, zoom_blur, spiral, pixelate, 
shatter, kaleidoscope, page_turn, television, film_burn, matrix_rain, 
old_film, mosaic_blur, lens_flare, digital_glitch, waterfall, 
honeycomb, none
```

#### Dolly Effect Configuration:

**Effect Types:**
- `auto_zoom` - Automatic zoom effect
- `manual_zoom` - Manual zoom with specific coordinates

**End Types:**
- `instant` - Instant transition
- `smooth` - Smooth transition

**Auto Zoom Example:**
```json
{
  "scene_index": 0,
  "start_time": 1.5,
  "duration": 1.0,
  "zoom_percent": 50,
  "effect_type": "auto_zoom",
  "end_time": 5.0
}
```

**Manual Zoom Example:**
```json
{
  "scene_index": 1,
  "start_time": 3.0,
  "duration": 1.5,
  "zoom_percent": 50,
  "effect_type": "manual_zoom",
  "x_coordinate": 100,
  "y_coordinate": 100,
  "end_time": 6.0,
  "end_type": "smooth"
}
```

---

### 4. Check Effects Job Status

**Endpoint:** `GET /api/v1/effects/{job_id}/status`

#### Request:

```bash
curl -X GET "http://localhost:8000/api/v1/effects/{job_id}/status"
```

#### Response:

```json
{
  "job_id": "effect-job-id",
  "status": "completed",
  "video_path": "/workspace/marketing-video-ai/outputs/effect_f670e1b6.mp4"
}
```

**⚠️ Important:** Only retrieve `video_path` when `status = "completed"`

---

## Usage Workflow

### 1. Create Basic Video:

```bash
# Step 1: Send create video request
curl -X POST "http://localhost:8000/api/v1/videos/create" ...

# Step 2: Get job_id from response
# Response: {"job_id": "abc-123", "status": "processing"}

# Step 3: Check status until completed
curl "http://localhost:8000/api/v1/jobs/abc-123/status"

# Step 4: Extract video_path and list_scene from response
```

### 2. Add Effects (Optional):

```bash
# Step 1: Use video_path and list_scene from previous step
curl -X POST "http://localhost:8000/api/v1/videos/effects" ...

# Step 2: Get effect_job_id from response
# Step 3: Check status until completed
curl "http://localhost:8000/api/v1/effects/{effect_job_id}/status"

# Step 4: Get final video_path when completed
```

---

## Important Notes

- ⚠️ **API currently accepts local file paths** (Directus not in use)
- ⚠️ **All paths must be absolute paths on the server**
- ⚠️ **Always check `status = "completed"` before retrieving `video_path`**
- ⚠️ **`list_scene` only contains values when video has multiple scenes**
- ⚠️ **`prompts` parameter is optional** - can be omitted or left as empty strings

---

## Troubleshooting

### PyTorch Installation Error
```bash
# Check CUDA version
nvidia-smi

# Install correct CUDA version
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
```

### Missing Models Error
```bash
# Re-run download script
bash download_model.sh
```

### Port Already in Use
```bash
# Change port number
uvicorn main:app --host 0.0.0.0 --port 8001
```

---

## Support

If you encounter issues, please check:
1. API server logs
2. File paths exist and are accessible
3. Image/audio file formats are correct
4. Sufficient disk space available (50GB+)

---

## License

[Add your license information here]
