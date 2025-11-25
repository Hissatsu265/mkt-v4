from app.core.database import DBManager
from datetime import datetime
from typing import Optional, Dict, Any
import traceback

# Import logger
try:
    from app.core.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

async def log_static_slide_image_to_db(
    job_id: str,
    slide_number: int,
    slide_type: int,
    image_prompt: str,
    status: str = "pending",
    file_path: Optional[str] = None,
    error_message: Optional[str] = None,
    created_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None
):
    """
    Log or update image generation job status in MongoDB
    """
    try:
        # Get the current timestamp
        timestamp = datetime.now()
        
        # Create a log entry
        log_entry = {
            "job_id": job_id,
            "slide_number": slide_number,
            "slide_type": slide_type,
            "image_prompt": image_prompt,
            "status": status,
            "file_path": file_path,
            "error_message": error_message,
            "created_at": created_at or timestamp,
            "updated_at": updated_at or timestamp,
            "timestamp": timestamp,  # Keep your existing timestamp field
        }
        
        if completed_at:
            log_entry["completed_at"] = completed_at
        
        # Use upsert to insert or update existing record
        result = await DBManager.db.static_slide_images.update_one(
            {"job_id": job_id},
            {"$set": log_entry},
            upsert=True
        )
        
        if result.upserted_id:
            logger.debug(f"✅ Inserted new image job {job_id} with status: {status} - ID: {result.upserted_id}")
        else:
            logger.debug(f"✅ Updated image job {job_id} with status: {status}")
        
        return result.upserted_id or result.matched_count
        
    except Exception as e:
        logger.error(f"❌ Failed to log image job to MongoDB: {str(e)}")
        logger.error(traceback.format_exc())
        # Don't raise exception - this is non-critical functionality
        return None