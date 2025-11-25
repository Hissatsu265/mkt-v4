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


async def update_image_job_status_in_db(job_id: str, update_data: Dict[str, Any]):
    """
    Update image job status in MongoDB
    """
    try:
        # Add updated timestamp
        update_data["updated_at"] = datetime.now()
        
        # Update the document
        result = await DBManager.db.static_slide_images.update_one(
            {"job_id": job_id},
            {"$set": update_data}
        )
        
        if result.matched_count > 0:
            logger.debug(f"✅ Updated image job {job_id} in DB - modified: {result.modified_count}")
        else:
            logger.warning(f"⚠️ Image job {job_id} not found for update")
        
        return result.modified_count
        
    except Exception as e:
        logger.error(f"❌ Error updating image job in DB: {str(e)}")
        logger.error(traceback.format_exc())
        # Don't raise exception - this is non-critical functionality
        return 0