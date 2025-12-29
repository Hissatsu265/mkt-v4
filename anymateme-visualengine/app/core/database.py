from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import asyncio
from app.core.config import settings
from app.core.mongodb import MongoDB

# Import the logger from our new logger module
# First we try to import, but if the module doesn't exist yet, we use standard logging
try:
    from app.core.logger import get_logger
    logger = get_logger("database")
except ImportError:
    import logging
    logger = logging.getLogger("database")


class DBCollection:
    """Proxy class for MongoDB collection access"""
    def __init__(self, collection_name):
        self.collection_name = collection_name
    
    async def insert_one(self, document):
        collection = MongoDB.get_collection(self.collection_name)
        return collection.insert_one(document)
    
    async def find(self, *args, **kwargs):
        collection = MongoDB.get_collection(self.collection_name)
        return collection.find(*args, **kwargs)
    
    async def find_one(self, *args, **kwargs):
        collection = MongoDB.get_collection(self.collection_name)
        return collection.find_one(*args, **kwargs)
    
    async def update_one(self, *args, **kwargs):
        collection = MongoDB.get_collection(self.collection_name)
        return collection.update_one(*args, **kwargs)
    
    async def update_many(self, *args, **kwargs):
        collection = MongoDB.get_collection(self.collection_name)
        return collection.update_many(*args, **kwargs)
    
    async def delete_one(self, *args, **kwargs):
        collection = MongoDB.get_collection(self.collection_name)
        return collection.delete_one(*args, **kwargs)
    
    async def delete_many(self, *args, **kwargs):
        collection = MongoDB.get_collection(self.collection_name)
        return collection.delete_many(*args, **kwargs)
    
    async def count_documents(self, *args, **kwargs):
        collection = MongoDB.get_collection(self.collection_name)
        return collection.count_documents(*args, **kwargs)
    
    async def aggregate(self, pipeline, **kwargs):
        collection = MongoDB.get_collection(self.collection_name)
        return collection.aggregate(pipeline, **kwargs)


class DBProxy:
    """Proxy class to mimic MongoDB database with attributes as collections"""
    def __getattr__(self, collection_name):
        return DBCollection(collection_name)


class DBManager:
    """Database manager for MongoDB operations"""
    is_connected = False
    client = None
    db_name = settings.MONGODB_DB_NAME
    db = DBProxy()  # Static db property that mimics MongoDB database structure
    
    @classmethod
    async def connect_to_mongo(cls):
        """Connect to MongoDB database"""
        try:
            # Get MongoDB client
            mongo_client = MongoDB.get_client()
            cls.client = mongo_client
            # Get database
            mongo_db = MongoDB.get_database()
            cls.is_connected = True
            logger.info(f"Connected to MongoDB database: {settings.MONGODB_DB_NAME}")
            return mongo_db
        except Exception as e:
            cls.is_connected = False
            logger.error(f"Failed to connect to MongoDB: {e}")
            return None
    
    @classmethod
    def get_collection(cls, collection_name: str):
        """Get a specific collection from MongoDB"""
        return MongoDB.get_collection(collection_name)
    
    @classmethod
    async def close_mongo_connection(cls):
        """Close MongoDB connection"""
        try:
            MongoDB.close()
            cls.is_connected = False
            cls.client = None
            logger.info("Closed connection to MongoDB")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")
    
    @classmethod
    async def log_api_request(cls, log_data: Dict[str, Any]) -> None:
        """
        Log API request data to MongoDB
        
        Args:
            log_data (dict): Data about the API request
            
        Returns:
            None
        """
        if not hasattr(settings, 'ENABLE_MONGODB_LOGGING') or not settings.ENABLE_MONGODB_LOGGING:
            return
            
        try:
            # Ensure we're connected
            if not cls.is_connected or not cls.client:
                await cls.connect_to_mongo()
                
            # Add log ID and timestamp if not already present
            if "log_id" not in log_data:
                log_data["log_id"] = str(uuid.uuid4())
            if "timestamp" not in log_data or isinstance(log_data["timestamp"], float):
                log_data["timestamp"] = datetime.utcnow()
            
            # Get the logs collection
            collection_name = getattr(settings, 'MONGODB_LOGS_COLLECTION', 'api_logs')
            collection = MongoDB.get_collection(collection_name)
            
            # Insert the log
            result = collection.insert_one(log_data)
            
            # Only log at debug level to prevent noise
            logger.debug(f"Logged API request with ID: {log_data.get('log_id') or result.inserted_id}")
            
        except Exception as e:
            # Log error but don't propagate
            logger.warning(f"Failed to log API request to MongoDB: {e}")
    
    @classmethod
    async def get_logs(cls, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent API logs"""
        try:
            collection_name = getattr(settings, 'MONGODB_LOGS_COLLECTION', 'api_logs')
            collection = MongoDB.get_collection(collection_name)
            cursor = collection.find().sort("timestamp", -1).limit(limit)
            logs = list(cursor)
            return logs
        except Exception as e:
            logger.error(f"Error retrieving API logs: {e}")
            return []