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


async def cleanup_old_image_jobs(days_old: int = 7):
    """
    Clean up old image jobs from database
    """
    try:
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # Delete old documents
        result = await DBManager.db.static_slide_images.delete_many({
            "timestamp": {"$lt": cutoff_date}
        })
        
        logger.info(f"✅ Cleaned up {result.deleted_count} old image jobs (older than {days_old} days)")
        return result.deleted_count
        
    except Exception as e:
        logger.error(f"❌ Error cleaning up old image jobs: {str(e)}")
        logger.error(traceback.format_exc())
        return 0