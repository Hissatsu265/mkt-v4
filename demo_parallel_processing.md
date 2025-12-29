# Multi-Worker Parallel Processing Demo

## What Happens When You Submit 2 Requests Simultaneously?

### 🔄 **OLD BEHAVIOR (Serial Processing - 1 Worker)**

```
Time: 0min
┌─────────────────┐
│  Request #1     │ ──► Worker 0 (PROCESSING) 🔴
└─────────────────┘
┌─────────────────┐
│  Request #2     │ ──► Queue (WAITING) ⏳
└─────────────────┘

Time: 25min
┌─────────────────┐
│  Request #1     │ ──► ✅ COMPLETED
└─────────────────┘
┌─────────────────┐
│  Request #2     │ ──► Worker 0 (PROCESSING) 🔴
└─────────────────┘

Time: 50min
┌─────────────────┐
│  Request #1     │ ──► ✅ COMPLETED
└─────────────────┘
┌─────────────────┐
│  Request #2     │ ──► ✅ COMPLETED
└─────────────────┘

Total Time: 50 minutes (25 min × 2 jobs)
```

---

### ⚡ **NEW BEHAVIOR (Parallel Processing - 2 Workers)**

```
Time: 0min
┌─────────────────┐
│  Request #1     │ ──► Worker 0 (PROCESSING) 🔴
└─────────────────┘
┌─────────────────┐
│  Request #2     │ ──► Worker 1 (PROCESSING) 🔴  ← RUNS SIMULTANEOUSLY!
└─────────────────┘

Time: 25min
┌─────────────────┐
│  Request #1     │ ──► ✅ COMPLETED
└─────────────────┘
┌─────────────────┐
│  Request #2     │ ──► ✅ COMPLETED
└─────────────────┘

Total Time: 25 minutes (both jobs processed in parallel)
⚡ 2x faster!
```

---

## 📊 What Happens with 5 Requests?

### OLD (1 Worker):
```
Job 1: 0-25min   ████████████
Job 2: 25-50min            ████████████
Job 3: 50-75min                      ████████████
Job 4: 75-100min                               ████████████
Job 5: 100-125min                                        ████████████

Total: 125 minutes
```

### NEW (2 Workers):
```
Job 1: 0-25min   ████████████  (Worker 0)
Job 2: 0-25min   ████████████  (Worker 1) ← Parallel!
Job 3: 25-50min            ████████████  (Worker 0)
Job 4: 25-50min            ████████████  (Worker 1) ← Parallel!
Job 5: 50-75min                      ████████████  (Worker 0)

Total: 75 minutes
⚡ 1.67x faster!
```

---

## 🎯 Real-World Example with Your Test Data

When you send this request **twice**:

```json
{
  "image_paths": ["https://cms.anymateme.shop/assets/93c5abdd..."],
  "prompts": ["english"],
  "audio_path": "https://cms.anymateme.shop/assets/508cd1eb...",
  "resolution": "9:16",
  "character": "santa",
  "background": "indoor"
}
```

### Timeline:

**T = 0 seconds:**
```
Request 1 → API → job_queue → Worker 0 starts processing
Request 2 → API → job_queue → Worker 1 starts processing
```

**Worker 0 logs:**
```
[Worker 0] 🎯 Processing job: abc-123
[Worker 0] Downloading assets...
[Worker 0] 🎥 Creating video for job: abc-123
```

**Worker 1 logs (at the SAME time):**
```
[Worker 1] 🎯 Processing job: def-456
[Worker 1] Downloading assets...
[Worker 1] 🎥 Creating video for job: def-456
```

**T = ~25 minutes:**
```
[Worker 0] ✅ Job completed: abc-123
[Worker 1] ✅ Job completed: def-456
```

Both videos are ready **at the same time**!

---

## 🔍 Under the Hood

### Code Flow for 2 Simultaneous Requests:

```python
# Request 1 arrives
job_service.create_job(...)  # Creates job_id="abc-123"
  └─► job_queue.put(job_data)
  └─► start_video_workers()  # Starts Worker 0

# Request 2 arrives (milliseconds later)
job_service.create_job(...)  # Creates job_id="def-456"
  └─► job_queue.put(job_data)
  └─► start_video_workers()  # Worker 0 already running, starts Worker 1

# Worker 0 (running independently)
async def process_video_jobs(worker_id=0):
    job_data = await job_queue.get()  # Gets "abc-123"
    async with video_processing_locks[0]:  # ✅ Acquires lock 0
        video_workers_busy[0] = True
        create_video(...)  # Processing...

# Worker 1 (running at the SAME TIME)
async def process_video_jobs(worker_id=1):
    job_data = await job_queue.get()  # Gets "def-456"
    async with video_processing_locks[1]:  # ✅ Acquires lock 1 (different lock!)
        video_workers_busy[1] = True
        create_video(...)  # Processing in parallel!
```

**Key Point:** Each worker has its **own lock**, so they don't block each other!

---

## 📈 GPU Memory Usage

### With 2 Jobs Running:

```
GPU Memory (40GB total):
┌──────────────────────────────────────┐
│  Worker 0: Job 1     │ 15GB          │
│  Worker 1: Job 2     │ 15GB          │
│  System Reserve      │ 5GB           │
│  Free                │ 5GB           │
└──────────────────────────────────────┘
Total Used: 30GB / 40GB (75% utilization)
```

### If You Send a 3rd Request:

```
Time: 0min
┌─────────────────┐
│  Request #1     │ ──► Worker 0 (PROCESSING) 🔴
└─────────────────┘
┌─────────────────┐
│  Request #2     │ ──► Worker 1 (PROCESSING) 🔴
└─────────────────┘
┌─────────────────┐
│  Request #3     │ ──► Queue (WAITING) ⏳  ← Waits for first available worker
└─────────────────┘

Time: 25min (Worker 0 finishes first)
┌─────────────────┐
│  Request #1     │ ──► ✅ COMPLETED
└─────────────────┘
┌─────────────────┐
│  Request #2     │ ──► Worker 1 (still processing) 🔴
└─────────────────┘
┌─────────────────┐
│  Request #3     │ ──► Worker 0 (PROCESSING) 🔴  ← Immediately starts!
└─────────────────┘
```

---

## 🎛️ Monitoring in Real-Time

You can check worker status at any time:

```bash
GET /api/v1/queue/info
```

**Response when 2 jobs are processing:**
```json
{
  "jobs_in_queue": 2,
  "waiting_jobs": 0,
  "max_workers": 2,
  "video_workers": {
    "total_workers": 2,
    "available_workers": 0,
    "busy_workers": 2,
    "pending_jobs": 0,
    "workers_status": [
      {"worker_id": 0, "is_busy": true, "is_running": true},
      {"worker_id": 1, "is_busy": true, "is_running": true}
    ]
  }
}
```

**Response when 1 job is processing, 1 is waiting:**
```json
{
  "jobs_in_queue": 1,
  "waiting_jobs": 1,
  "max_workers": 2,
  "video_workers": {
    "total_workers": 2,
    "available_workers": 1,
    "busy_workers": 1,
    "pending_jobs": 1,
    "workers_status": [
      {"worker_id": 0, "is_busy": true, "is_running": true},
      {"worker_id": 1, "is_busy": false, "is_running": true}
    ]
  }
}
```

---

## 🚀 Performance Summary

| Scenario | Old (1 Worker) | New (2 Workers) | Speedup |
|----------|----------------|-----------------|---------|
| 2 requests | 50 min | 25 min | **2x faster** |
| 3 requests | 75 min | 50 min | **1.5x faster** |
| 4 requests | 100 min | 50 min | **2x faster** |
| 5 requests | 125 min | 75 min | **1.67x faster** |
| 10 requests | 250 min | 125 min | **2x faster** |

---

## ✨ Key Benefits

1. **No Queue Buildup**: Second request doesn't wait for first to finish
2. **Better GPU Utilization**: 75% instead of 50%
3. **Fair Scheduling**: Jobs are processed FIFO, but 2 at a time
4. **Fault Isolation**: If Worker 0 fails, Worker 1 keeps running
5. **Graceful Degradation**: On 24GB GPU, automatically falls back to 1 worker

---

## 🔧 Configuration Options

Want more workers? Edit `.env`:

```bash
# For 80GB GPU - allow up to 5 workers
MAX_PARALLEL_WORKERS=5

# Or disable parallel processing
ENABLE_PARALLEL_PROCESSING=false  # Falls back to 1 worker
```

The system will **auto-calculate** the optimal number based on your GPU!
