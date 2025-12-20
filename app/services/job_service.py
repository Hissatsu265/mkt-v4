
import asyncio
import json
import uuid
import os
import psutil
from directus.function_downloadfile import download_image,download_audio
from typing import List
from datetime import datetime, timedelta
from typing import Dict, Any
from pathlib import Path
from utilities.audio_duration import get_audio_duration

# Make redis optional for serverless environments
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

from app.models.schemas import JobStatus
from typing import Dict, Any, List
from app.models.mongodb import mongodb
from app.services.job_repository import job_repository

import random
import re

# ✅ AUDIO TO PROCESSING TIME RATIO
# Change this value to adjust the relationship between audio duration and processing time
# Example: 1.2 means 1 second of audio = 1.2 minutes of processing time
# So a 5-second audio file will take 6 minutes to process (5 * 1.2 = 6)
AUDIO_TO_PROCESSING_RATIO = 1.2

URL_REGEX = re.compile(
    r'^(https?://)'
    r'(([A-Za-z0-9-]+\.)+[A-Za-z]{2,6})'
    r'(/[A-Za-z0-9._~:/?#\[\]@!$&\'()*+,;=-]*)?$'
)
def is_valid_url(url: str) -> bool:
    return isinstance(url, str) and bool(URL_REGEX.match(url))
async def download_assets(job_data):
 
    image_tasks = [
        asyncio.to_thread(download_image, img_url)
        for img_url in job_data["image_paths"]
        if is_valid_url(img_url)
    ]
    audio_task = asyncio.to_thread(download_audio, job_data["audio_path"])

    results = await asyncio.gather(*image_tasks, audio_task)

    images_pathdown = results[:-1]
    audio_path_down = results[-1]
    print(images_pathdown," ==============")
    return images_pathdown, audio_path_down

class JobService:
    def __init__(self):
        self.redis_client = None
        self.job_queue = asyncio.Queue()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.processing = False
        self.cleanup_task = None
        self.video_processing_lock = asyncio.Lock()  # Lock cho video processing
        self.current_processing_job = None  # Track job đang được xử lý
        self.worker_task = None  # Task của worker

        self.waiting_job_ids: List[str] = []
        self.queue_lock = asyncio.Lock()
        
        self.recovery_completed = False 

        self.job_queue = asyncio.Queue()
        self.effect_job_queue = asyncio.Queue()
        # NEW: Effect jobs system
        self.effect_jobs: Dict[str, Dict[str, Any]] = {}
        self.effect_job_queue = asyncio.Queue()
        self.effect_workers: List[asyncio.Task] = []
        self.effect_processing_locks: List[asyncio.Lock] = []
        self.effect_workers_busy: List[bool] = []
        self.max_effect_workers = self._calculate_effect_workers()
        self._init_effect_workers()

        # NEW: Multi-worker system for video jobs
        self.video_workers: List[asyncio.Task] = []
        self.video_processing_locks: List[asyncio.Lock] = []
        self.video_workers_busy: List[bool] = []
        self.max_video_workers = self._calculate_video_workers()
        self._init_video_workers()
    # =============================================================
    async def _reconnect_mongodb_if_needed(self, max_retries: int = 3, retry_delay: float = 2.0):
        """Thử reconnect MongoDB nếu connection bị mất"""
        for attempt in range(max_retries):
            try:
                # Test connection
                await mongodb.client.admin.command('ping')
                print(f"MongoDB connection verified (attempt {attempt + 1})")
                return True
            except Exception as e:
                print(f"MongoDB connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # Disconnect và reconnect
                    try:
                        await mongodb.disconnect()
                        await asyncio.sleep(retry_delay)
                        await mongodb.connect()
                    except Exception as reconnect_error:
                        print(f"Reconnection attempt failed: {reconnect_error}")
                        await asyncio.sleep(retry_delay)
                else:
                    print("All MongoDB reconnection attempts failed")
                    return False
        return False

    async def _execute_with_retry(self, operation, operation_name: str, max_retries: int = 3, retry_delay: float = 2.0):
        """Execute MongoDB operation với retry logic"""
        for attempt in range(max_retries):
            try:
                result = await operation()
                return result
            except Exception as e:
                error_msg = str(e).lower()
                is_timeout = any(keyword in error_msg for keyword in [
                    'timed out', 'timeout', 'networktimeout', 
                    'connectionfailure', 'server selection timeout'
                ])
                
                print(f"{operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if is_timeout and attempt < max_retries - 1:
                    print(f"Timeout detected, attempting MongoDB reconnection...")
                    reconnected = await self._reconnect_mongodb_if_needed()
                    if reconnected:
                        await asyncio.sleep(retry_delay)
                        continue
                
                # Nếu là attempt cuối hoặc không phải timeout error
                if attempt == max_retries - 1:
                    print(f"{operation_name} failed after {max_retries} attempts")
                    raise
                
                await asyncio.sleep(retry_delay)
        
        raise Exception(f"{operation_name} failed after all retries")
    # ================================================================
    async def init_mongodb(self):
        """Khởi tạo MongoDB connection"""
        try:
            await mongodb.connect()
            print("MongoDB initialized successfully")
            # ✅ THÊM: Recovery jobs sau khi connect MongoDB
            await self.recover_pending_jobs()
        except Exception as e:
            print(f"MongoDB initialization failed: {e}")
            raise
    def _calculate_effect_workers(self) -> int:
        """Tính số effect workers = số CPU cores / 2, tối thiểu 1"""
        cpu_count = psutil.cpu_count(logical=False) or 2
        workers = max(1, cpu_count // 2)
        print(f"Initializing {workers} effect workers (CPU cores: {cpu_count})")
        return workers

    def _init_effect_workers(self):
        """Khởi tạo effect workers và locks"""
        self.effect_processing_locks = [asyncio.Lock() for _ in range(self.max_effect_workers)]
        self.effect_workers_busy = [False for _ in range(self.max_effect_workers)]

    def _calculate_video_workers(self) -> int:
        """Calculate number of parallel video workers based on GPU memory"""
        from config import MAX_PARALLEL_WORKERS, ENABLE_PARALLEL_PROCESSING, GPU_MEMORY_PER_JOB_GB, GPU_MEMORY_RESERVE_GB

        if not ENABLE_PARALLEL_PROCESSING:
            print("Parallel processing disabled, using 1 worker")
            self.gpu_count = 0
            return 1  # Fallback to serial processing

        # Auto-detect GPU memory across all GPUs
        try:
            import torch
            if torch.cuda.is_available():
                # Detect all GPUs
                gpu_count = torch.cuda.device_count()
                total_gpu_memory = 0

                print(f"🔍 Detecting GPUs...")
                for i in range(gpu_count):
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_memory_gb = gpu_props.total_memory / (1024**3)
                    gpu_name = gpu_props.name
                    total_gpu_memory += gpu_memory_gb
                    print(f"   GPU {i}: {gpu_name} ({gpu_memory_gb:.1f}GB)")

                # Calculate workers based on TOTAL GPU memory
                available_for_jobs = total_gpu_memory - GPU_MEMORY_RESERVE_GB
                auto_workers = max(1, int(available_for_jobs / GPU_MEMORY_PER_JOB_GB))

                # Respect max limit from config
                workers = min(auto_workers, MAX_PARALLEL_WORKERS)

                # Store GPU count for worker assignment
                self.gpu_count = gpu_count

                print(f"🚀 Multi-GPU Detection: {gpu_count} GPU(s), {total_gpu_memory:.1f}GB total")
                print(f"📊 Calculated workers: {auto_workers} (using {workers} after config limit of {MAX_PARALLEL_WORKERS})")
                print(f"🎯 Parallel video processing enabled with {workers} worker(s) across {gpu_count} GPU(s)")

                return workers
            else:
                print("⚠️  No GPU detected, using 1 worker")
                self.gpu_count = 0
                return 1
        except Exception as e:
            print(f"❌ GPU detection failed: {e}, using config value of {MAX_PARALLEL_WORKERS}")
            self.gpu_count = 0
            return max(1, MAX_PARALLEL_WORKERS)

    def _init_video_workers(self):
        """Initialize video workers and locks"""
        self.video_processing_locks = [asyncio.Lock() for _ in range(self.max_video_workers)]
        self.video_workers_busy = [False for _ in range(self.max_video_workers)]

    def _get_gpu_for_worker(self, worker_id: int) -> int:
        """Assign GPU device to worker using round-robin distribution"""
        if not hasattr(self, 'gpu_count') or self.gpu_count <= 0:
            return 0  # Fallback to GPU 0 if no GPU count or single GPU

        # Round-robin: Worker 0→GPU 0, Worker 1→GPU 1, Worker 2→GPU 0, etc.
        return worker_id % self.gpu_count

    async def init_redis(self):
        """Initialize Redis connection"""
        try:
            from config import REDIS_URL
            self.redis_client = redis.from_url(REDIS_URL)
            await self.redis_client.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self.redis_client = None

    async def start_cleanup_task(self):
        """Bắt đầu task cleanup tự động"""
        if not self.cleanup_task or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self.periodic_cleanup())

    async def periodic_cleanup(self):
        """Task cleanup chạy định kỳ"""
        from config import CLEANUP_INTERVAL_MINUTES
        
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL_MINUTES * 60)  # Convert to seconds
                await self.cleanup_old_jobs()
                await self.cleanup_old_videos()
                print(f"Cleanup completed at {datetime.now()}")
            except Exception as e:
                print(f"Cleanup error: {e}")
    async def recover_pending_jobs(self):
        """Khôi phục các job PENDING và PROCESSING từ MongoDB sau khi server restart"""
        
        if self.recovery_completed:
            print("Job recovery already completed, skipping...")
            return
        
        try:
            print("🔄 Starting job recovery process...")
            
            # Lấy tất cả job PENDING và PROCESSING từ MongoDB
            pending_jobs = await job_repository.find_jobs_by_status(JobStatus.PENDING)
            processing_jobs = await job_repository.find_jobs_by_status(JobStatus.PROCESSING)
            
            all_jobs = pending_jobs + processing_jobs
            
            if not all_jobs:
                print("✅ No jobs to recover")
                self.recovery_completed = True
                return
            
            # Sắp xếp theo thời gian tạo (job cũ nhất lên đầu)
            all_jobs.sort(key=lambda x: x.get("created_at", ""))
            
            print(f"📋 Found {len(all_jobs)} jobs to recover:")
            print(f"   - PENDING: {len(pending_jobs)}")
            print(f"   - PROCESSING: {len(processing_jobs)}")
            
            recovered_count = 0
            
            async with self.queue_lock:
                for job in all_jobs:
                    job_id = job["job_id"]
                    
                    # Reset job về trạng thái PENDING nếu đang PROCESSING
                    if job["status"] == JobStatus.PROCESSING:
                        print(f"   🔄 Resetting PROCESSING job to PENDING: {job_id}")
                        await job_repository.update_job(job_id, {
                            "status": JobStatus.PENDING,
                            "progress": 0,
                            "error_message": "Job was interrupted by server restart"
                        })
                        job["status"] = JobStatus.PENDING
                    
                    # Thêm vào queue và tracking list
                    self.waiting_job_ids.append(job_id)
                    await self.job_queue.put(job.copy())
                    recovered_count += 1
                    
                    print(f"   ✅ Recovered job: {job_id} (position: {len(self.waiting_job_ids)})")
            
            print(f"✅ Job recovery completed: {recovered_count} jobs recovered")
            self.recovery_completed = True

            # Khởi động workers để xử lý các job đã recovery
            await self.start_video_workers()
            
        except Exception as e:
            print(f"❌ Error during job recovery: {e}")
            # Không raise error để server vẫn khởi động được
            self.recovery_completed = True 
    async def cleanup_old_jobs(self):
        """Xóa job cũ khỏi memory và Redis"""
        from config import JOB_RETENTION_HOURS
        
        cutoff_time = datetime.now() - timedelta(hours=JOB_RETENTION_HOURS)
        jobs_to_remove = []
        
        # Cleanup memory jobs
        for job_id, job_data in self.jobs.items():
            try:
                created_at = datetime.fromisoformat(job_data["created_at"])
                if created_at < cutoff_time:
                    jobs_to_remove.append(job_id)
            except (KeyError, ValueError):
                # Invalid job data, mark for removal
                jobs_to_remove.append(job_id)
        
        # Remove from memory
        for job_id in jobs_to_remove:
            self.jobs.pop(job_id, None)
            print(f"Removed job from memory: {job_id}")
        
        # Cleanup Redis jobs
        if self.redis_client:
            try:
                # Scan for job keys
                async for key in self.redis_client.scan_iter(match="job:*"):
                    job_data = await self.redis_client.get(key)
                    if job_data:
                        try:
                            data = json.loads(job_data)
                            created_at = datetime.fromisoformat(data["created_at"])
                            if created_at < cutoff_time:
                                await self.redis_client.delete(key)
                                print(f"Removed job from Redis: {key}")
                        except (json.JSONDecodeError, KeyError, ValueError):
                            # Invalid job data, remove it
                            await self.redis_client.delete(key)
                            print(f"Removed invalid job from Redis: {key}")
            except Exception as e:
                print(f"Redis cleanup error: {e}")

    async def cleanup_old_videos(self):
        """Xóa video files cũ"""
        from config import VIDEO_RETENTION_HOURS, OUTPUT_DIR
        
        cutoff_time = datetime.now() - timedelta(hours=VIDEO_RETENTION_HOURS)
        output_path = Path(OUTPUT_DIR)
        
        if not output_path.exists():
            return
        
        files_removed = 0
        for video_file in output_path.glob("*.mp4"):
            try:
                # Lấy thời gian tạo file
                file_time = datetime.fromtimestamp(video_file.stat().st_ctime)
                if file_time < cutoff_time:
                    video_file.unlink()
                    files_removed += 1
                    print(f"Removed old video: {video_file.name}")
            except Exception as e:
                print(f"Error removing video {video_file.name}: {e}")
        
        if files_removed > 0:
            print(f"Cleanup completed: {files_removed} video files removed")



    async def manual_cleanup(self):
        """Cleanup thủ công"""
        await self.cleanup_old_jobs()
        await self.cleanup_old_videos()
        return {"message": "Manual cleanup completed"}

    async def create_job(self, image_paths: list, prompts: list, audio_path: str, resolution: str = "1920x1080",background:str="",character:str="") -> str:

        if not self.recovery_completed:
            print("⏳ Waiting for job recovery to complete...")
            # Đợi tối đa 10 giây
            for _ in range(100):
                if self.recovery_completed:
                    break
                await asyncio.sleep(0.1)

        job_id = str(uuid.uuid4())

        # ✅ Extract audio duration for accurate time estimation (1s audio = 1min processing)
        audio_duration = None
        try:
            # Download audio file if URL, or use local path
            if audio_path.startswith("http://") or audio_path.startswith("https://"):
                audio_local_path = await asyncio.to_thread(download_audio, audio_path)
            else:
                audio_local_path = audio_path

            # Extract duration in seconds
            audio_duration = await asyncio.to_thread(get_audio_duration, audio_local_path)
            print(f"[Job {job_id}] Audio duration: {audio_duration}s → Estimated processing time: {audio_duration}min")
        except Exception as e:
            # Fallback to default if audio duration extraction fails
            from config import AVERAGE_JOB_TIME_MINUTES
            print(f"[Job {job_id}] Failed to extract audio duration: {e}. Using fallback: {AVERAGE_JOB_TIME_MINUTES}min")
            audio_duration = AVERAGE_JOB_TIME_MINUTES * 60  # Convert to seconds (1min = 60s)

        # Calculate accurate time estimation for multi-GPU parallel processing
        async with self.queue_lock:
            queue_length = len(self.waiting_job_ids)

        # Get worker availability
        workers_info = await self.get_video_workers_info()
        total_workers = workers_info.get("total_workers", 1)
        available_workers = workers_info.get("available_workers", total_workers)

        # Use actual workers, not zero
        active_workers = max(1, total_workers)

        # Calculate position in queue: waiting jobs + this new job
        # Example: 3 waiting + 1 new = position 4
        position_in_queue = queue_length + 1

        # Calculate which batch this job belongs to
        # Example: position 4 with 2 workers = ceil(4/2) = batch 2
        import math
        batch_number = math.ceil(position_in_queue / active_workers)

        # If there are available workers, current batch is 0 (starts immediately)
        # If all workers busy, we're in future batches
        if available_workers >= active_workers:
            # All workers available, job starts in first batch (batch 0)
            batches_to_wait = batch_number - 1
        else:
            # Some workers busy, count from current batch
            # Jobs already processing are in batch 1, so we wait batch_number batches
            batches_to_wait = batch_number

        # ✅ Use audio duration for job time estimation (1s audio = 1.2min processing)
        # audio_duration is in seconds, processing time in minutes (1:1.2 ratio seconds→minutes)
        # Example: 5s audio → 6 minutes processing time
        job_processing_time_minutes = audio_duration * AUDIO_TO_PROCESSING_RATIO

        # Total time = batches to wait * this job's processing time + queue wait
        # Queue wait assumes previous jobs also take similar time
        total_wait_minutes = batches_to_wait * job_processing_time_minutes

        # Store calculation metadata for debugging
        estimate_time_complete = (datetime.now() + timedelta(minutes=total_wait_minutes)).isoformat()
        
        job_data = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "image_paths": image_paths,
            "prompts": prompts,
            "audio_path": audio_path,
            "audio_duration": audio_duration,  # ✅ Store audio duration in seconds
            "resolution": resolution,
            "progress": 0,
            "video_path": None,
            "error_message": None,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "queue_position": None,
            "background":background,
            "character":character,
            "estimate_time_complete": estimate_time_complete,

            "retry_count": 0
        }
        
        async def _insert_job():
            return await job_repository.insert_job(job_data)
        
        try:
            await self._execute_with_retry(
                operation=_insert_job,
                operation_name=f"create_job[{job_id}]",
                max_retries=3,
                retry_delay=2.0
            )
            print(f"Job created successfully: {job_id}")
            
        except Exception as e:
            print(f"Failed to create job {job_id} after all retries: {e}")
            raise  # Re-raise để API trả về error
        async with self.queue_lock:
            self.waiting_job_ids.append(job_id)
            await self.job_queue.put(job_data.copy())

        # Start workers và cleanup
        await self.start_video_workers()
        asyncio.create_task(self.start_cleanup_task())
        
        return job_id

    async def _save_job_to_redis(self, job_id: str, job_data: dict):
        """Helper method để save job to Redis không block"""
        try:
            await self.redis_client.set(
                f"job:{job_id}", 
                json.dumps(job_data, default=str),
                ex=86400  # Expire after 24 hours
            )
        except Exception as e:
            print(f"Error saving job to Redis: {e}")

    async def start_video_workers(self):
        """Start video workers if not running"""
        # Clean up completed workers
        self.video_workers = [w for w in self.video_workers if not w.done()]

        # Start workers if needed
        workers_needed = self.max_video_workers - len(self.video_workers)

        for i in range(workers_needed):
            worker_id = len(self.video_workers)
            worker_task = asyncio.create_task(self.process_video_jobs(worker_id))
            self.video_workers.append(worker_task)
            print(f"✅ Started video worker {worker_id}/{self.max_video_workers}")

    # async def get_job_status(self, job_id: str) -> Dict[str, Any]:
    #     """Lấy job status từ MongoDB"""
        
    #     # LẤY TỪ MONGODB thay vì RAM
    #     job_data = await job_repository.find_job_by_id(job_id)
        
    #     if not job_data:
    #         return None
            
    #     # Cập nhật queue position cho pending jobs
    #     if job_data["status"] == JobStatus.PENDING:
    #         position = await self.get_queue_position(job_id)
    #         job_data["queue_position"] = position
        
    #     return job_data
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        async def _get_job():
            return await job_repository.find_job_by_id(job_id)
        
        try:
            job_data = await self._execute_with_retry(
                operation=_get_job,
                operation_name=f"get_job_status[{job_id}]",
                max_retries=3,
                retry_delay=2.0
            )
            
            if not job_data:
                return None
                
            # Cập nhật queue position cho pending jobs
            if job_data["status"] == JobStatus.PENDING or job_data["status"] == JobStatus.PROCESSING:
                position = await self.get_queue_position(job_id)
                job_data["queue_position"] = position

                # ✅ Tính estimate_waiting_time (phút còn lại)
                if job_data["status"] == JobStatus.PROCESSING:
                    # Job đang chạy - tính dựa trên audio duration và progress
                    if job_data.get("started_at"):
                        from config import AVERAGE_JOB_TIME_MINUTES
                        started_at = datetime.fromisoformat(job_data["started_at"])
                        elapsed_seconds = (datetime.now() - started_at).total_seconds()
                        progress = job_data.get("progress", 1)

                        # ✅ Use audio_duration if available (1s audio = 1.2min processing)
                        audio_duration = job_data.get("audio_duration")

                        if audio_duration and progress > 0 and progress < 100:
                            # Total estimated time = audio_duration in seconds → minutes (1:1.2 ratio)
                            # Example: 5s audio = 6 minutes processing
                            estimated_total_minutes = audio_duration * AUDIO_TO_PROCESSING_RATIO
                            estimated_total_seconds = estimated_total_minutes * 60

                            # Calculate remaining based on progress
                            completed_seconds = estimated_total_seconds * (progress / 100)
                            remaining_seconds = max(0, estimated_total_seconds - elapsed_seconds)
                            # Round up để luôn hiển thị ít nhất 1 phút nếu còn time
                            remaining_minutes = max(1, int((remaining_seconds + 59) / 60))
                        elif progress > 0 and progress < 100:
                            # Fallback: calculate based on elapsed time and progress
                            estimated_total_seconds = (elapsed_seconds / progress) * 100
                            remaining_seconds = max(0, estimated_total_seconds - elapsed_seconds)
                            remaining_minutes = max(1, int((remaining_seconds + 59) / 60))
                        else:
                            # Fallback nếu không có progress data
                            remaining_minutes = AVERAGE_JOB_TIME_MINUTES

                        job_data["estimate_waiting_time"] = remaining_minutes
                    else:
                        # Không có started_at, dùng estimate cũ
                        from config import AVERAGE_JOB_TIME_MINUTES
                        job_data["estimate_waiting_time"] = AVERAGE_JOB_TIME_MINUTES

                elif job_data["status"] == JobStatus.PENDING:
                    # ✅ IMPROVED: Calculate based on actual queue and audio durations
                    # Need to account for:
                    # 1. Currently processing jobs
                    # 2. Jobs ahead in queue with their audio durations

                    from config import AVERAGE_JOB_TIME_MINUTES

                    # Get this job's audio duration
                    this_job_duration = job_data.get("audio_duration", AVERAGE_JOB_TIME_MINUTES * 60)

                    # Calculate wait time from jobs ahead
                    wait_time_minutes = 0

                    # Get workers info to know how many jobs are processing
                    workers_info = await self.get_video_workers_info()
                    busy_workers = workers_info.get("busy_workers", 0)
                    total_workers = workers_info.get("total_workers", 1)

                    # If workers are busy, need to wait for them to finish
                    if busy_workers > 0:
                        # Estimate remaining time for processing jobs (use average)
                        # TODO: Could be improved by tracking actual processing jobs
                        wait_time_minutes += AVERAGE_JOB_TIME_MINUTES

                    # Get queue position
                    queue_pos = position  # Already calculated above

                    if queue_pos > 0:
                        # Get jobs ahead in queue and sum their durations
                        async with self.queue_lock:
                            jobs_ahead_count = queue_pos - 1

                            # Estimate time for jobs ahead
                            # If we have jobs in waiting_job_ids, calculate their durations
                            for idx in range(min(jobs_ahead_count, len(self.waiting_job_ids))):
                                ahead_job_id = self.waiting_job_ids[idx]

                                # Fetch that job's audio_duration from DB
                                try:
                                    ahead_job = await job_repository.find_job_by_id(ahead_job_id)
                                    if ahead_job:
                                        ahead_duration = ahead_job.get("audio_duration", AVERAGE_JOB_TIME_MINUTES * 60)
                                        wait_time_minutes += ahead_duration * AUDIO_TO_PROCESSING_RATIO  # 1s = 1.2min
                                except:
                                    # Fallback if can't fetch
                                    wait_time_minutes += AVERAGE_JOB_TIME_MINUTES

                        # Account for parallel workers (jobs can process in parallel)
                        if total_workers > 1:
                            wait_time_minutes = wait_time_minutes / total_workers

                    job_data["estimate_waiting_time"] = max(1, int(wait_time_minutes))

            # Add retry information to response
            from config import MAX_JOB_RETRIES
            job_data["retry_count"] = job_data.get("retry_count", 0)
            job_data["max_retries"] = MAX_JOB_RETRIES
            job_data["retries_remaining"] = MAX_JOB_RETRIES - job_data.get("retry_count", 0)

            return job_data
            
        except Exception as e:
            print(f"Error getting job status for {job_id} after all retries: {e}")
            return None
    # async def get_queue_position(self, job_id: str) -> int:
    #     """Lấy vị trí job trong queue"""
    #     position = 1
    #     temp_queue = []
    #     found = False
        
    #     # Tạm thời lấy items từ queue để tìm position
    #     try:
    #         while not self.job_queue.empty():
    #             item = await asyncio.wait_for(self.job_queue.get(), timeout=0.1)
    #             temp_queue.append(item)
    #             if item["job_id"] == job_id:
    #                 found = True
    #                 break
    #             position += 1
            
    #         # Đưa items trở lại queue
    #         for item in reversed(temp_queue):
    #             await self.job_queue.put(item)
            
    #         return position if found else 0
    #     except:
    #         # Nếu có lỗi, đưa items trở lại queue
    #         for item in reversed(temp_queue):
    #             await self.job_queue.put(item)
    #         return 0
    async def get_queue_position(self, job_id: str) -> int:
        """✅ Lấy vị trí job trong queue - THREAD-SAFE, KHÔNG RỦI RO"""
        async with self.queue_lock:
            try:
                # Tìm vị trí trong danh sách (index bắt đầu từ 0, nên +1)
                position = self.waiting_job_ids.index(job_id) + 1
                return position
            except ValueError:
                # Job không còn trong queue (đang processing hoặc đã xong)
                return 0
   
    async def update_job_status(self, job_id: str, status: JobStatus, **kwargs):
        print(f"[UPDATE_JOB_STATUS] Called for job {job_id} with status={status}, kwargs={kwargs}")
        update_data = {"status": status}
        update_data.update(kwargs)

        if status == JobStatus.PROCESSING:
            progress = kwargs.get("progress", 99)
            if progress <= 51:
                update_data["started_at"] = datetime.now().isoformat()
                print(f"[{datetime.now().isoformat()}] Set started_at for job {job_id}")

            

        # Tính tổng thời gian khi job hoàn thành hoặc thất bại
        if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            update_data["completed_at"] = datetime.now().isoformat()
            
            # Lấy thông tin job để tính thời gian
            try:
                job_data = await job_repository.find_job_by_id(job_id)
                if job_data and job_data.get("started_at"):
                    started_at = datetime.fromisoformat(job_data["started_at"])
                    completed_at = datetime.now()
                    total_time = int((completed_at - started_at).total_seconds())
                    update_data["total_generation_time"] = int(total_time / 60)
            except Exception as e:
                print(f"[update_job_status] Error calculating total_generation_time: {e}")

        try:
            print(f"[UPDATE_JOB_STATUS] Calling job_repository.update_job with data: {update_data}")
            result = await job_repository.update_job(job_id, update_data)
            print(f"[UPDATE_JOB_STATUS] ✅ SUCCESS - job_id={job_id} status={status} update_result={result} modified_count={result}")
            if not result:
                print(f"[UPDATE_JOB_STATUS] ⚠️ WARNING: update_job returned False - job may not exist or no changes made for job {job_id}")
        except Exception as e:
            print(f"[UPDATE_JOB_STATUS] ❌ EXCEPTION when updating job {job_id}: {e}")
            try:
                await job_repository.update_job(job_id, {"status": JobStatus.FAILED, "error_message": f"Update error: {e}", "completed_at": datetime.now().isoformat()})
            except Exception as e2:
                print(f"[update_job_status] Failed to mark job as FAILED after update exception: {e2}")
    async def _update_job_in_redis(self, job_id: str):
        """Helper method để update job in Redis không block"""
        print("Sử dụng redis trong file JobService")
        try:
            await self.redis_client.set(
                f"job:{job_id}",
                json.dumps(self.jobs[job_id], default=str),
                ex=86400
            )
        except Exception as e:
            print(f"Error updating job in Redis: {e}")

    # async def process_jobs(self):
    #     print("Job worker started")
        
    #     while True:
    #         try:
    #             job_data = await self.job_queue.get()
    #             job_id = job_data["job_id"]
                
    #             current_job_db = await job_repository.find_job_by_id(job_id)
    #             if current_job_db and current_job_db.get("status") == JobStatus.FAILED:
    #                 print(f"Skipping cancelled job: {job_id}")
    #                 self.job_queue.task_done()
    #                 continue
    #             # ✅ XÓA khỏi tracking list NGAY SAU KHI lấy từ queue
    #             async with self.queue_lock:
    #                 if job_id in self.waiting_job_ids:
    #                     self.waiting_job_ids.remove(job_id)
                
    #             print(f"Processing job: {job_id}")
    #             print(f"Remaining in queue: {self.waiting_job_ids}")
                
    #             print("download image=======================")
    #             images_pathdown, audio_path_down = await download_assets(job_data)
                
    #             async with self.video_processing_lock:
    #                 self.current_processing_job = job_id
                    
    #                 await self.update_job_status(job_id, JobStatus.PROCESSING, progress=10)
                    
    #                 from app.services.video_service import VideoService
    #                 video_service = VideoService()
                    
    #                 try:
    #                     print(f"Creating video for job: {job_id}")
    #                     video_path, list_scene = await video_service.create_video(
    #                         image_paths=images_pathdown,
    #                         prompts=job_data["prompts"],
    #                         audio_path=audio_path_down,
    #                         resolution=job_data["resolution"],
    #                         job_id=job_id,
    #                         character=job_data.get("character", ""),
    #                         background=job_data.get("background", "")
    #                     )
                        
    #                     await self.update_job_status(
    #                         job_id, 
    #                         JobStatus.COMPLETED, 
    #                         progress=100,
    #                         video_path=video_path,
    #                         list_scene=list_scene
    #                     )
    #                     print(f"Job completed: {job_id}")
                        
    #                 except Exception as e:
    #                     print(f"Job failed: {job_id}, Error: {e}")
    #                     await self.update_job_status(
    #                         job_id,
    #                         JobStatus.FAILED,
    #                         error_message=str(e)
    #                     )
                    
    #                 finally:
    #                     self.current_processing_job = None
                
    #             self.job_queue.task_done()
                
    #         except Exception as e:
    #             print(f"Error in job worker: {e}")
    #             await asyncio.sleep(1)
    async def process_video_jobs(self, worker_id: int):
        """Video job worker - processes jobs in parallel"""
        print(f"🎬 Video worker {worker_id} started")

        while True:
            try:
                job_data = await self.job_queue.get()
                job_id = job_data["job_id"]

                current_job_db = await job_repository.find_job_by_id(job_id)
                if current_job_db and current_job_db.get("status") == JobStatus.FAILED:
                    # ✅ THÊM: Kiểm tra nếu đã retry rồi thì skip
                    if current_job_db.get("retry_count", 0) >= 1:
                        print(f"[Worker {worker_id}] Skipping failed job (already retried): {job_id}")
                        self.job_queue.task_done()
                        continue

                # Xóa khỏi tracking list
                async with self.queue_lock:
                    if job_id in self.waiting_job_ids:
                        self.waiting_job_ids.remove(job_id)

                print(f"[Worker {worker_id}] 🎯 Processing job: {job_id}")
                print(f"[Worker {worker_id}] Remaining in queue: {self.waiting_job_ids}")

                print(f"[Worker {worker_id}] Downloading assets...")
                images_pathdown, audio_path_down = await download_assets(job_data)

                # Acquire ONLY this worker's lock (allows other workers to continue)
                async with self.video_processing_locks[worker_id]:
                    self.video_workers_busy[worker_id] = True
                    
                    await self.update_job_status(job_id, JobStatus.PROCESSING, progress=10)
                    
                    from app.services.video_service import VideoService
                    video_service = VideoService()

                    try:
                        # Calculate GPU assignment for this worker
                        gpu_id = self._get_gpu_for_worker(worker_id)
                        print(f"[Worker {worker_id}] 🎥 Creating video for job: {job_id} on GPU {gpu_id}")

                        video_path, list_scene = await video_service.create_video(
                            image_paths=images_pathdown,
                            prompts=job_data["prompts"],
                            audio_path=audio_path_down,
                            resolution=job_data["resolution"],
                            job_id=job_id,
                            character=job_data.get("character", ""),
                            background=job_data.get("background", ""),
                            worker_id=worker_id,
                            gpu_id=gpu_id  # NEW: Pass GPU assignment
                        )

                        await self.update_job_status(
                            job_id,
                            JobStatus.COMPLETED,
                            progress=100,
                            video_path=video_path,
                            list_scene=list_scene
                        )
                        print(f"[Worker {worker_id}] ✅ Job completed: {job_id}")
                        
                    except Exception as e:
                        print(f"[Worker {worker_id}] ❌ Job failed: {job_id}, Error: {e}")

                        # Enhanced retry logic with configurable retries and exponential backoff
                        from config import MAX_JOB_RETRIES, RETRY_DELAY_SECONDS, USE_EXPONENTIAL_BACKOFF

                        current_retry_count = job_data.get("retry_count", 0)

                        if current_retry_count < MAX_JOB_RETRIES:
                            print(f"[Worker {worker_id}] 🔄 Scheduling retry for job {job_id} (attempt {current_retry_count + 1}/{MAX_JOB_RETRIES})")

                            # Calculate delay with exponential backoff if enabled
                            if USE_EXPONENTIAL_BACKOFF:
                                # Exponential backoff: delay * (2 ^ retry_count)
                                # Example: 30s, 60s, 120s, 240s...
                                delay = RETRY_DELAY_SECONDS * (2 ** current_retry_count)
                                print(f"[Worker {worker_id}] ⏰ Will retry in {delay}s (exponential backoff)")
                            else:
                                # Fixed delay
                                delay = RETRY_DELAY_SECONDS
                                print(f"[Worker {worker_id}] ⏰ Will retry in {delay}s")

                            # Increment retry count
                            job_data["retry_count"] = current_retry_count + 1
                            job_data["status"] = JobStatus.PENDING
                            job_data["error_message"] = None

                            # Update in DB
                            await job_repository.update_job(job_id, {
                                "retry_count": current_retry_count + 1,
                                "status": JobStatus.PENDING,
                                "error_message": f"Retry {current_retry_count + 1}/{MAX_JOB_RETRIES} scheduled in {delay}s after error: {str(e)}"
                            })

                            # Schedule retry as background task (doesn't block worker)
                            asyncio.create_task(self._schedule_retry(job_id, job_data, delay))

                            print(f"[Worker {worker_id}] ✅ Retry scheduled for job {job_id} in {delay}s")
                        else:
                            # Max retries reached, mark as FAILED permanently
                            print(f"[Worker {worker_id}] 💀 Job {job_id} failed permanently after {MAX_JOB_RETRIES} retries")
                            await self.update_job_status(
                                job_id,
                                JobStatus.FAILED,
                                error_message=f"Failed after {MAX_JOB_RETRIES} retries: {str(e)}"
                            )

                    finally:
                        self.video_workers_busy[worker_id] = False

                self.job_queue.task_done()

            except Exception as e:
                print(f"[Worker {worker_id}] ⚠️  Error in worker: {e}")
                await asyncio.sleep(1)

    async def _schedule_retry(self, job_id: str, job_data: dict, delay: int):
        """Schedule a job retry after a delay (runs in background, doesn't block workers)"""
        try:
            print(f"[Retry Scheduler] ⏳ Waiting {delay}s before retrying job {job_id}")
            await asyncio.sleep(delay)

            # Add job back to queue after delay
            async with self.queue_lock:
                self.waiting_job_ids.append(job_id)
            await self.job_queue.put(job_data)

            print(f"[Retry Scheduler] ✅ Job {job_id} added back to queue (position {len(self.waiting_job_ids)})")

        except Exception as e:
            print(f"[Retry Scheduler] ❌ Error scheduling retry for {job_id}: {e}")

    async def get_video_workers_info(self):
        """Get information about video workers"""
        available_workers = sum(1 for busy in self.video_workers_busy if not busy)
        busy_workers = sum(1 for busy in self.video_workers_busy if busy)

        return {
            "total_workers": self.max_video_workers,
            "available_workers": available_workers,
            "busy_workers": busy_workers,
            "pending_jobs": self.job_queue.qsize(),
            "gpu_count": getattr(self, 'gpu_count', 1),  # Number of GPUs detected
            "workers_status": [
                {
                    "worker_id": i,
                    "is_busy": self.video_workers_busy[i],
                    "is_running": i < len(self.video_workers) and not self.video_workers[i].done(),
                    "assigned_gpu": self._get_gpu_for_worker(i)  # GPU assignment for this worker
                }
                for i in range(self.max_video_workers)
            ]
        }

    async def get_queue_info(self):
        """✅ Thêm thông tin về waiting jobs"""
        async with self.queue_lock:
            waiting_count = len(self.waiting_job_ids)
            waiting_ids = self.waiting_job_ids.copy()

        # Add worker info
        video_workers_info = await self.get_video_workers_info()

        return {
            "jobs_in_queue": self.job_queue.qsize(),
            "waiting_jobs": waiting_count,
            "waiting_job_ids": waiting_ids,
            "worker_running": len(self.video_workers) > 0,
            "video_workers": video_workers_info,
            "max_workers": self.max_video_workers,
            "pending_jobs": self.job_queue.qsize()  # Add for backward compatibility
        }

    # async def cancel_job(self, job_id: str) -> bool:
    #     """Hủy job nếu đang pending - NON-BLOCKING"""
    #     if job_id in self.jobs:
    #         job = self.jobs[job_id]
    #         if job["status"] == JobStatus.PENDING:
    #             await self.update_job_status(job_id, JobStatus.FAILED, error_message="Job cancelled by user")
    #             return True
    #         elif job["status"] == JobStatus.PROCESSING:
    #             return False  # Không thể hủy job đang processing
    #     return False
    async def cancel_job(self, job_id: str) -> bool:
        job_data = await job_repository.find_job_by_id(job_id)
        
        if not job_data:
            return False
            
        if job_data["status"] == JobStatus.PENDING:
            # Xóa khỏi tracking list
            async with self.queue_lock:
                if job_id in self.waiting_job_ids:
                    self.waiting_job_ids.remove(job_id)
            
            await self.update_job_status(
                job_id, 
                JobStatus.FAILED, 
                error_message="Job cancelled by user"
            )
            return True
        elif job_data["status"] == JobStatus.PROCESSING:
            return False
            
        return False
    # ============================VIDEO EFFECT JOBS SYSTEM==========================================================
    
    async def create_effect_job(self, 
                              video_path: str,
                              transition_times: List[float],
                              transition_effects: List[str],
                              transition_durations: List[float],
                              dolly_effects: List[Dict] = None) -> str:
        
        job_id = str(uuid.uuid4())
        
        job_data = {
            "job_id": job_id,
            "job_type": "effect",  # Phân biệt với video creation job
            "status": JobStatus.PENDING,
            "video_path": video_path,
            "transition_times": transition_times,
            "transition_effects": transition_effects,
            "transition_durations": transition_durations,
            "dolly_effects": dolly_effects or [],
            "progress": 0,
            "output_video_path": None,
            "error_message": None,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "queue_position": self.effect_job_queue.qsize() + 1,
            "worker_id": None
        }
        
        # Lưu vào memory và Redis
        self.effect_jobs[job_id] = job_data
        if self.redis_client:
            asyncio.create_task(self._save_effect_job_to_redis(job_id, job_data))
        
        # Thêm vào effect queue
        await self.effect_job_queue.put(job_data)
        
        # Đảm bảo effect workers đang chạy
        await self.start_effect_workers()
        
        return job_id

    async def _save_effect_job_to_redis(self, job_id: str, job_data: dict):
        """Save effect job to Redis"""
        try:
            await self.redis_client.set(
                f"effect_job:{job_id}",
                json.dumps(job_data, default=str),
                ex=86400  # 24h
            )
        except Exception as e:
            print(f"Error saving effect job to Redis: {e}")

    async def start_effect_workers(self):
        """Khởi động các effect workers nếu chưa chạy"""
        
        # Clean up completed workers
        self.effect_workers = [w for w in self.effect_workers if not w.done()]
        
        # Start workers if needed
        workers_needed = self.max_effect_workers - len(self.effect_workers)
        
        for i in range(workers_needed):
            worker_id = len(self.effect_workers)
            worker_task = asyncio.create_task(self.process_effect_jobs(worker_id))
            self.effect_workers.append(worker_task)
            print(f"Started effect worker {worker_id}")

    async def process_effect_jobs(self, worker_id: int):
        print(f"Effect worker {worker_id} started")
        
        while True:
            try:
                job_data = await self.effect_job_queue.get()
                job_id = job_data["job_id"]
                
                print(f"Effect worker {worker_id} processing job: {job_id}")
                
                async with self.effect_processing_locks[worker_id]:
                    self.effect_workers_busy[worker_id] = True
                    
                    await self.update_effect_job_status(
                        job_id, 
                        JobStatus.PROCESSING, 
                        progress=10,
                        worker_id=worker_id
                    )
                    try:
                        from app.services.video_effect_service import VideoEffectService
                        effect_service = VideoEffectService()
                        
                        loop = asyncio.get_event_loop()
                        video_duration = await loop.run_in_executor(
                            None, 
                            effect_service.get_video_duration_sync,
                            job_data["video_path"]
                        )
                        await self.update_effect_job_status(job_id, JobStatus.PROCESSING, progress=20)
                        
                        await loop.run_in_executor(
                            None,
                            effect_service.validate_effects_timing,
                            job_data["transition_times"],
                            job_data["dolly_effects"], 
                            video_duration
                        )
                        await self.update_effect_job_status(job_id, JobStatus.PROCESSING, progress=30)
                        
                        import concurrent.futures
                        
                        def process_video_sync():
                            return effect_service.apply_effects_sync(
                                video_path=job_data["video_path"],
                                transition_times=job_data["transition_times"],
                                transition_effects=job_data["transition_effects"],
                                transition_durations=job_data["transition_durations"],
                                dolly_effects=job_data["dolly_effects"],
                                job_id=job_id
                            )
                        
                        loop = asyncio.get_event_loop()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            output_path = await loop.run_in_executor(executor, process_video_sync)
                        
                        # ✅ QUAN TRỌNG: Cập nhật cả output_video_path VÀ video_path
                        await self.update_effect_job_status(
                            job_id,
                            JobStatus.COMPLETED,
                            progress=100,
                            output_video_path=output_path,
                            video_path=output_path  # ✅ Thêm dòng này
                        )
                        
                        print(f"Effect job completed: {job_id} by worker {worker_id}")
                        
                    except Exception as e:
                        print(f"Effect job failed: {job_id}, Error: {e}")
                        await self.update_effect_job_status(
                            job_id,
                            JobStatus.FAILED,
                            error_message=str(e)
                        )
                    
                    finally:
                        self.effect_workers_busy[worker_id] = False
                
                self.effect_job_queue.task_done()
                
            except Exception as e:
                print(f"Error in effect worker {worker_id}: {e}")
                self.effect_workers_busy[worker_id] = False
                await asyncio.sleep(1)

    async def get_effect_job_status(self, job_id: str) -> Dict[str, Any]:
        """Lấy trạng thái effect job"""
        
        # Check memory first
        if job_id in self.effect_jobs:
            job_data = self.effect_jobs[job_id].copy()
            
            # Update queue position for pending jobs
            if job_data["status"] == JobStatus.PENDING:
                position = await self.get_effect_queue_position(job_id)
                job_data["queue_position"] = position
            
            return job_data
        
        # Check Redis
        if self.redis_client:
            try:
                job_data = await self.redis_client.get(f"effect_job:{job_id}")
                if job_data:
                    return json.loads(job_data)
            except Exception as e:
                print(f"Error getting effect job from Redis: {e}")
        
        return None

    async def get_effect_queue_position(self, job_id: str) -> int:
        """Lấy vị trí trong effect queue"""
        position = 1
        temp_queue = []
        found = False
        
        try:
            while not self.effect_job_queue.empty():
                item = await asyncio.wait_for(self.effect_job_queue.get(), timeout=0.1)
                temp_queue.append(item)
                if item["job_id"] == job_id:
                    found = True
                    break
                position += 1
            
            # Put items back
            for item in reversed(temp_queue):
                await self.effect_job_queue.put(item)
            
            return position if found else 0
        except:
            for item in reversed(temp_queue):
                await self.effect_job_queue.put(item)
            return 0

    async def update_effect_job_status(self, job_id: str, status: JobStatus, **kwargs):
        """Cập nhật effect job status"""
        
        if job_id in self.effect_jobs:
            self.effect_jobs[job_id]["status"] = status
            for key, value in kwargs.items():
                self.effect_jobs[job_id][key] = value
            
            if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                self.effect_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
            # Update Redis
            if self.redis_client:
                asyncio.create_task(self._update_effect_job_in_redis(job_id))

    async def _update_effect_job_in_redis(self, job_id: str):
        """Update effect job in Redis"""
        try:
            await self.redis_client.set(
                f"effect_job:{job_id}",
                json.dumps(self.effect_jobs[job_id], default=str),
                ex=86400
            )
        except Exception as e:
            print(f"Error updating effect job in Redis: {e}")

    async def get_effect_workers_info(self):
        """Lấy thông tin về effect workers"""
        
        available_workers = sum(1 for busy in self.effect_workers_busy if not busy)
        busy_workers = sum(1 for busy in self.effect_workers_busy if busy)
        
        return {
            "total_workers": self.max_effect_workers,
            "available_workers": available_workers,
            "busy_workers": busy_workers,
            "pending_jobs": self.effect_job_queue.qsize(),
            "workers_status": [
                {
                    "worker_id": i,
                    "is_busy": self.effect_workers_busy[i],
                    "is_running": i < len(self.effect_workers) and not self.effect_workers[i].done()
                }
                for i in range(self.max_effect_workers)
            ]
        }

    async def cancel_effect_job(self, job_id: str) -> bool:
        """Hủy effect job nếu đang pending"""
        
        if job_id in self.effect_jobs:
            job = self.effect_jobs[job_id]
            if job["status"] == JobStatus.PENDING:
                await self.update_effect_job_status(
                    job_id, 
                    JobStatus.FAILED, 
                    error_message="Job cancelled by user"
                )
                return True
            elif job["status"] == JobStatus.PROCESSING:
                return False  # Cannot cancel processing job
        return False

    # === UPDATED STATS METHOD ===
    async def get_stats(self):
        """Lấy thống kê bao gồm cả effect jobs"""
        from config import OUTPUT_DIR
        
        # Existing stats
        memory_jobs = len(self.jobs)
        status_count = {}
        for job_data in self.jobs.values():
            status = job_data.get("status", "unknown")
            status_count[status] = status_count.get(status, 0) + 1
        
        # Effect jobs stats  
        effect_memory_jobs = len(self.effect_jobs)
        effect_status_count = {}
        for job_data in self.effect_jobs.values():
            status = job_data.get("status", "unknown")
            effect_status_count[status] = effect_status_count.get(status, 0) + 1
        
        # Redis stats
        redis_jobs = 0
        redis_effect_jobs = 0
        if self.redis_client:
            try:
                redis_jobs = len([key async for key in self.redis_client.scan_iter(match="job:*")])
                redis_effect_jobs = len([key async for key in self.redis_client.scan_iter(match="effect_job:*")])
            except:
                redis_jobs = redis_effect_jobs = -1
        
        # Video files stats
        output_path = Path(OUTPUT_DIR)
        video_files = len(list(output_path.glob("*.mp4"))) if output_path.exists() else 0
        total_size = 0
        if output_path.exists():
            for video_file in output_path.glob("*.mp4"):
                try:
                    total_size += video_file.stat().st_size
                except:
                    pass
        
        # Workers info
        effect_workers_info = await self.get_effect_workers_info()
        video_workers_info = await self.get_video_workers_info()

        return {
            "video_creation_jobs": {
                "memory": memory_jobs,
                "redis": redis_jobs,
                "status_breakdown": status_count,
                "queue_pending": self.job_queue.qsize(),
                "workers": video_workers_info
            },
            "effect_jobs": {
                "memory": effect_memory_jobs,
                "redis": redis_effect_jobs,
                "status_breakdown": effect_status_count,
                "queue_pending": self.effect_job_queue.qsize(),
                "workers": effect_workers_info
            },
            "videos": {
                "count": video_files,
                "total_size_mb": round(total_size / 1024 / 1024, 2)
            },
            "system": {
                "cpu_cores": psutil.cpu_count(logical=False),
                "max_video_workers": self.max_video_workers,
                "max_effect_workers": self.max_effect_workers
            }
        }

# Global instance
job_service = JobService()