"""
RunPod Serverless Handler for Video Generation API

This handler processes video generation requests in RunPod serverless environment.
Supports two actions:
- create: Create and process video job synchronously
- status: Get job status from MongoDB
"""

import runpod
import asyncio
import sys
import os
from typing import Dict, Any

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.runpod_job_service import runpod_job_service


async def handler_async(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async handler for RunPod serverless requests

    Input schema:
    {
        "input": {
            "action": "create" | "status",
            "job_id": "optional-for-status",
            "image_paths": [...],  # for create
            "prompts": [...],      # for create
            "audio_path": "...",   # for create
            "resolution": "9:16"   # for create
        }
    }

    Returns:
    {
        "status": "success" | "error",
        "job": {...},          # job data
        "error": "..."         # error message if failed
    }
    """
    try:
        input_data = job.get("input", {})
        action = input_data.get("action", "create")

        print(f"[RunPod Handler] Received action: {action}")

        if action == "create":
            # Create and process job synchronously
            print("[RunPod Handler] Creating and processing video job...")
            result = await runpod_job_service.create_and_process_job(input_data)
            return result

        elif action == "status":
            # Get job status from MongoDB
            job_id = input_data.get("job_id")
            if not job_id:
                return {"status": "error", "error": "job_id is required for status action"}

            print(f"[RunPod Handler] Getting status for job: {job_id}")
            result = await runpod_job_service.get_job_status(job_id)
            return result

        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}. Supported actions: create, status"
            }

    except Exception as e:
        print(f"[RunPod Handler] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sync wrapper for RunPod - required by RunPod SDK
    RunPod expects a synchronous function, so we wrap the async handler
    """
    return asyncio.run(handler_async(job))


async def initialize():
    """Initialize MongoDB connection before starting handler"""
    from app.models.mongodb import mongodb
    try:
        await mongodb.connect()
        print("[RunPod Handler] MongoDB connected")
    except Exception as e:
        print(f"[RunPod Handler] MongoDB connection failed: {e}")
        # Continue anyway - some tests might work without MongoDB

if __name__ == "__main__":
    print("[RunPod Handler] Starting RunPod serverless handler...")
    print(f"[RunPod Handler] Python version: {sys.version}")
    print(f"[RunPod Handler] Working directory: {os.getcwd()}")

    # Initialize MongoDB
    asyncio.run(initialize())

    # Start RunPod serverless
    runpod.serverless.start({"handler": handler})
