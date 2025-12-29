# RunPod Serverless Setup Guide

## ✅ Implementation Complete

All files have been created and are ready to deploy! Here's what was added:

### New Files Created
1. **[runpod_handler.py](runpod_handler.py)** - Main RunPod entry point
2. **[app/services/runpod_job_service.py](app/services/runpod_job_service.py)** - Serverless job processor
3. **[Dockerfile.runpod](Dockerfile.runpod)** - RunPod-optimized container
4. **[config.py](config.py)** - Updated with RUNPOD_MODE variable

---

## 🚀 Quick Start: Deploy to RunPod

### Step 1: Build & Push Docker Image

```bash
# Build the image
docker build -f Dockerfile.runpod -t shohanuranymateme/marketing-video-ai:latest .

# Login to Docker Hub
docker login

# Push to Docker Hub
docker push shohanuranymateme/marketing-video-ai:latest
```

**Note**: This will take 20-30 minutes due to model downloads (~30GB).

---

### Step 2: Setup RunPod Network Volume (Optional but Recommended)

This step dramatically reduces cold start time from ~5 min to ~30 sec.

1. **Create Network Volume**:
   - Go to [RunPod Console](https://runpod.io) → Storage → Network Volumes
   - Click "New Network Volume"
   - Name: `comfyui-models-v1`
   - Size: 100GB
   - Region: Select closest to your users

2. **Pre-load Models** (one-time setup):
   ```bash
   # Deploy a temporary GPU Pod with the network volume attached
   # Mount it to /workspace/app/ComfyUI/models
   # Then SSH into the pod and run:

   cd /workspace/app
   bash download_model.sh
   bash download_model_image.sh
   bash install_custom_nodes.sh

   # Verify models are downloaded
   ls -lh ComfyUI/models/diffusion_models/
   # Should see: wan2.1-i2v-14b-480p-Q8_0.gguf (~14GB)
   ```

3. **Stop the temporary Pod** - Models are now cached in the Network Volume

---

### Step 3: Create Serverless Endpoint

1. **Go to RunPod Console** → Serverless → Templates

2. **Create New Template**:
   - **Name**: `Marketing Video AI`
   - **Container Image**: `shohanuranymateme/marketing-video-ai:latest`
   - **Container Disk**: 80GB
   - **Docker Command**: `python -u runpod_handler.py`

   **Environment Variables**:
   ```
   MONGODB_URI=mongodb://mongo:mojfsnokvu85qbbd@87.106.214.210:27017
   MONGODB_DB_NAME=anymateme_eduhub_prod
   MONGODB_JOBS_COLLECTION=video_jobs
   DIRECTUS_URL=https://cms.anymateme.pro
   DIRECTUS_ACCESS_TOKEN=BMhKSyuDTE9mLntjfJGVO0HMWyoue6Xg
   DIRECTUS_FOLDER_ID=ab2cdaba-91f3-42d1-8227-5697aac7ee22
   RUNPOD_MODE=true
   ```

3. **Create Serverless Endpoint**:
   - Go to Serverless → Endpoints → New Endpoint
   - **Select Template**: Marketing Video AI
   - **GPU Type**: RTX 4090 (recommended) or A100
   - **Workers**:
     - Min: 0 (scale to zero when idle)
     - Max: 10 (adjust based on traffic)
   - **GPUs per Worker**: 1
   - **Timeouts**:
     - Idle Timeout: 10 seconds
     - Execution Timeout: 600 seconds (10 min)
   - **Network Volume**: Select `comfyui-models-v1` (if created)
     - Mount path: `/workspace/app/ComfyUI/models`

4. **Save and Deploy**

5. **Note your Endpoint Details**:
   - Endpoint ID: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
   - API URL: `https://api.runpod.ai/v2/<endpoint-id>/runsync`
   - API Key: (from RunPod Settings → API Keys)

---

### Step 4: Test Your Endpoint

```bash
# Set your credentials
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="your-api-key"

# Test creating a video job
curl -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "create",
      "image_paths": ["https://cms.anymateme.pro/assets/e149cb1f-5b37-44e8-b248-c15bb79e31b6",""],
      "prompts": ["",""],
      "audio_path": "https://cms.anymateme.pro/assets/fba5eb69-d9aa-4280-9d10-c8de363c0e50",
      "resolution": "9:16"
    }
  }'

# Expected response:
# {
#   "status": "success",
#   "job": {
#     "job_id": "...",
#     "status": "completed",
#     "directus_url": "https://cms.anymateme.pro/assets/...",
#     ...
#   }
# }
```

---

### Step 5: Update Frontend

Update your frontend to call the RunPod endpoint:

**Before** (ngrok):
```javascript
const response = await fetch('https://hailee-unrepresentational-ronnie.ngrok-free.dev/api/v1/videos/create_christmas_campain', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ image_paths, prompts, audio_path, resolution })
});
```

**After** (RunPod):
```javascript
const RUNPOD_ENDPOINT_ID = 'your-endpoint-id';
const RUNPOD_API_KEY = 'your-api-key';

const response = await fetch(`https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${RUNPOD_API_KEY}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    input: {
      action: 'create',
      image_paths,
      prompts,
      audio_path,
      resolution
    }
  })
});

// Response structure is the same:
const data = await response.json();
if (data.status === 'success') {
  const videoUrl = data.job.directus_url;
  // Use videoUrl
}
```

---

## 📊 Cost Comparison

### Current Setup (Always-On GPU Server)
- GPU Server: **$1,095/month**
- Redis: **$10/month**
- **Total: $1,105/month**

### RunPod Serverless (1,000 videos/month)
- GPU Compute (5 min/video @ $0.44/hour): **$37/month**
- **Total: $37/month**

**💰 Savings: $1,068/month (97% reduction!)**

---

## 🔍 Monitoring & Debugging

### View Logs in RunPod Console
- Go to Serverless → Your Endpoint → Logs
- Real-time logs show:
  - Job creation
  - Video processing progress
  - Upload status
  - Errors

### Check Job Status
```bash
curl -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "status",
      "job_id": "<job-id>"
    }
  }'
```

### MongoDB Direct Query
Since MongoDB is external, you can also query jobs directly:
```bash
mongosh "mongodb://mongo:mojfsnokvu85qbbd@87.106.214.210:27017/anymateme_eduhub_prod"

# Query recent jobs
db.video_jobs.find().sort({created_at: -1}).limit(10)
```

---

## ⚠️ Troubleshooting

### Issue: Cold start timeout
**Solution**: Make sure Network Volume with pre-loaded models is attached

### Issue: MongoDB connection failed
**Solution**: Verify MongoDB is accessible from RunPod IPs (currently accessible from anywhere)

### Issue: Directus upload failed
**Solution**: Check DIRECTUS_ACCESS_TOKEN is valid and has upload permissions

### Issue: ComfyUI workflow error
**Solution**: Check logs in RunPod console, verify models are downloaded

---

## 🎯 Next Steps

1. ✅ **Build and push Docker image** (~30 min)
2. ✅ **Create Network Volume** and pre-load models (~1 hour)
3. ✅ **Deploy serverless endpoint** (~15 min)
4. ✅ **Test with sample job** (~10 min)
5. ✅ **Update frontend** (~30 min)
6. ✅ **Monitor and optimize** (ongoing)

---

## 📚 Additional Resources

- [RunPod Documentation](https://docs.runpod.io/)
- [RunPod Serverless Guide](https://docs.runpod.io/serverless/overview)
- [Network Volumes Guide](https://docs.runpod.io/pods/storage/create-network-volumes)

---

## 🆘 Need Help?

If you encounter any issues:
1. Check RunPod console logs
2. Verify environment variables are set correctly
3. Test MongoDB connection from RunPod
4. Review the plan file: [/root/.claude/plans/wild-cuddling-comet.md](/.claude/plans/wild-cuddling-comet.md)

Good luck with your deployment! 🚀
