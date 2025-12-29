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


async def get_images_by_status(status: str) -> list:
    """
    Get all image jobs with a specific status
    """
    try:
        # Find all documents with the given status
        cursor = DBManager.db.static_slide_images.find({"status": status})
        results = await cursor.to_list(length=None)
        
        # Convert ObjectIds to strings
        for result in results:
            if '_id' in result:
                result['_id'] = str(result['_id'])
        
        logger.debug(f"✅ Found {len(results)} image jobs with status: {status}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Error fetching images by status: {str(e)}")
        logger.error(traceback.format_exc())
        return []