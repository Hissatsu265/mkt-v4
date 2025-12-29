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


async def get_images_by_slide(slide_number: int, slide_type: int) -> list:
    """
    Get all image jobs for a specific slide
    """
    try:
        # Find all documents for this slide
        cursor = DBManager.db.static_slide_images.find({
            "slide_number": slide_number,
            "slide_type": slide_type
        })
        results = await cursor.to_list(length=None)
        
        # Convert ObjectIds to strings
        for result in results:
            if '_id' in result:
                result['_id'] = str(result['_id'])
        
        logger.debug(f"✅ Found {len(results)} image jobs for slide {slide_number} type {slide_type}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Error fetching images by slide: {str(e)}")
        logger.error(traceback.format_exc())
        return []