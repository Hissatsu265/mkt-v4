"""
RunPod Serverless Job Service

This service handles video generation jobs in serverless mode.
Key differences from the stateful job_service.py:
- No in-memory queue - processes jobs immediately (blocking)
- Uses MongoDB for job state persistence
- Reuses existing VideoService for video generation
- Reuses existing Directus upload logic
"""

import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any
from app.models.schemas import JobStatus
from app.services.job_repository import job_repository
from app.services.video_service import VideoService


class RunPodJobService:
    """Serverless-compatible job service"""

    def __init__(self):
        """Initialize service - no stateful components"""
        self.video_service = VideoService()
        print("[RunPodJobService] Initialized")

    async def create_and_process_job(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create job in MongoDB and process immediately (synchronous blocking)
        This is the main video generation function

        Args:
            input_data: {
                "image_paths": ["url1", "url2", ...],
                "prompts": ["prompt1", "prompt2", ...],
                "audio_path": "url",
                "resolution": "9:16" or "16:9"
            }

        Returns:
            {
                "status": "success" | "error",
                "job": {...job_data...},
                "error": "error_message" (if failed)
            }
        """
        job_id = str(uuid.uuid4())

        print(f"[RunPodJobService] Creating job {job_id}")

        # Validate input
        if not input_data.get("image_paths") or not input_data.get("audio_path"):
            return {
                "status": "error",
                "error": "image_paths and audio_path are required",
                "job_id": job_id
            }

        # Create job record in MongoDB
        job_data = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "image_paths": input_data["image_paths"],
            "prompts": input_data.get("prompts", [""]),
            "audio_path": input_data["audio_path"],
            "resolution": input_data.get("resolution", "9:16"),
            "progress": 0,
            "video_path": None,
            "directus_url": None,
            "list_scene": None,
            "error_message": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "completed_at": None
        }

        try:
            # Save to MongoDB
            await job_repository.insert_job(job_data)
            print(f"[RunPodJobService] Job {job_id} created in MongoDB")

            # Process immediately (blocking - this is intentional in serverless)
            await self._process_video_job(job_id, job_data)

            # Retrieve final job data from MongoDB
            final_job_data = await job_repository.find_job_by_id(job_id)

            if final_job_data and final_job_data["status"] == JobStatus.COMPLETED:
                print(f"[RunPodJobService] Job {job_id} completed successfully")
                return {"status": "success", "job": final_job_data}
            else:
                error_msg = final_job_data.get("error_message", "Unknown error") if final_job_data else "Job not found after processing"
                print(f"[RunPodJobService] Job {job_id} failed: {error_msg}")
                return {"status": "error", "error": error_msg, "job": final_job_data}

        except Exception as e:
            error_msg = str(e)
            print(f"[RunPodJobService] Job {job_id} failed with exception: {error_msg}")

            # Mark as failed in MongoDB
            try:
                await job_repository.update_job(job_id, {
                    "status": JobStatus.FAILED,
                    "error_message": error_msg,
                    "updated_at": datetime.now().isoformat(),
                    "completed_at": datetime.now().isoformat()
                })
            except:
                pass

            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id
            }

    async def _process_video_job(self, job_id: str, job_data: Dict[str, Any]):
        """
        Process video generation (ComfyUI + Directus upload)
        This is the heavy lifting function - reuses existing VideoService
        """
        try:
            print(f"[RunPodJobService] Processing job {job_id}")

            # Update to processing
            await job_repository.update_job(job_id, {
                "status": JobStatus.PROCESSING,
                "progress": 10,
                "updated_at": datetime.now().isoformat()
            })

            # Generate video using existing VideoService
            # This handles:
            # - Downloading assets from Directus
            # - Running ComfyUI workflow
            # - Uploading to Directus
            # - Returning video path and scene transitions
            print(f"[RunPodJobService] Calling VideoService.create_video()")

            video_path, list_scene = await self.video_service.create_video(
                image_paths=job_data["image_paths"],
                prompts=job_data["prompts"],
                audio_path=job_data["audio_path"],
                resolution=job_data["resolution"],
                job_id=job_id
            )

            print(f"[RunPodJobService] Video generated: {video_path}")

            # Update progress before final completion
            await job_repository.update_job(job_id, {
                "progress": 95,
                "updated_at": datetime.now().isoformat()
            })

            # Mark as completed
            await job_repository.update_job(job_id, {
                "status": JobStatus.COMPLETED,
                "progress": 100,
                "video_path": video_path,
                "directus_url": video_path,  # video_path is already Directus URL
                "list_scene": list_scene,
                "updated_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat()
            })

            print(f"[RunPodJobService] Job {job_id} completed successfully")

        except Exception as e:
            error_msg = str(e)
            print(f"[RunPodJobService] Job {job_id} failed during processing: {error_msg}")

            # Mark as failed
            await job_repository.update_job(job_id, {
                "status": JobStatus.FAILED,
                "error_message": error_msg,
                "updated_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat()
            })

            raise  # Re-raise to be caught by create_and_process_job

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status from MongoDB

        Args:
            job_id: Job ID to query

        Returns:
            {
                "status": "success" | "error",
                "job": {...job_data...}
            }
        """
        try:
            print(f"[RunPodJobService] Getting status for job {job_id}")

            job_data = await job_repository.find_job_by_id(job_id)

            if not job_data:
                return {
                    "status": "error",
                    "error": f"Job not found: {job_id}"
                }

            return {
                "status": "success",
                "job": job_data
            }

        except Exception as e:
            print(f"[RunPodJobService] Error getting job status: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }


# Global instance
runpod_job_service = RunPodJobService()
