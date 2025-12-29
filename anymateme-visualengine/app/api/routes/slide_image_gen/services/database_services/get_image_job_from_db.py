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

async def get_image_job_from_db(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get image job status from MongoDB by job_id
    """
    try:
        # Find the document
        result = await DBManager.db.static_slide_images.find_one({"job_id": job_id})
        
        if result:
            # Convert ObjectId to string if present
            if '_id' in result:
                result['_id'] = str(result['_id'])
            logger.debug(f"✅ Found image job {job_id} with status: {result.get('status', 'unknown')}")
        else:
            logger.debug(f"❌ Image job {job_id} not found in database")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error fetching image job from DB: {str(e)}")
        logger.error(traceback.format_exc())
        return None