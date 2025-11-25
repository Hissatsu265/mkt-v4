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


async def get_image_jobs_stats() -> Dict[str, Any]:
    """
    Get statistics about image jobs
    """
    try:
        # Aggregate statistics
        pipeline = [
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1}
                }
            }
        ]
        
        cursor = DBManager.db.static_slide_images.aggregate(pipeline)
        status_counts = await cursor.to_list(length=None)
        
        # Convert to dict
        stats = {item["_id"]: item["count"] for item in status_counts}
        
        # Get total count
        total_count = await DBManager.db.static_slide_images.count_documents({})
        stats["total"] = total_count
        
        logger.debug(f"✅ Image jobs stats: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error getting image jobs stats: {str(e)}")
        logger.error(traceback.format_exc())
        return {"total": 0}