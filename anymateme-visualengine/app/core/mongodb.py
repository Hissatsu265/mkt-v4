import os
import logging
from pymongo import MongoClient
from urllib.parse import quote_plus
from app.core.config import settings

logger = logging.getLogger("mongodb")

def get_mongodb_client():
    """
    Create and return a MongoDB client using connection details from settings
    
    Returns:
        MongoClient: A MongoDB client instance
    """
    try:
        # URL encode the username and password to handle special characters
        encoded_username = quote_plus(settings.MONGO_USERNAME)
        encoded_password = quote_plus(settings.MONGO_PASSWORD)
        
        # Build the connection string
        connection_string = f"mongodb://{encoded_username}:{encoded_password}@{settings.MONGO_HOST}:{settings.MONGO_PORT}"
        
        # Create the client
        client = MongoClient(connection_string)
        
        # Test the connection by listing database names
        client.list_database_names()
        
        # Log successful connection
        logger.info('Connected to MongoDB successfully')
        logger.debug(f'Connection String: mongodb://{encoded_username}:****@{settings.MONGO_HOST}:{settings.MONGO_PORT}')
        
        return client
    except Exception as error:
        # Log connection error
        logger.error(f'Failed to connect to MongoDB: {error}')
        raise

class MongoDB:
    """MongoDB client singleton class"""
    _instance = None
    client = None
    db = None
    
    @classmethod
    def get_instance(cls):
        """Get or create MongoDB client instance (singleton pattern)"""
        if cls._instance is None:
            cls._instance = cls()
            cls.client = get_mongodb_client()
            cls.db = cls.client[settings.MONGODB_DB_NAME]
        return cls._instance
    
    @classmethod
    def get_client(cls):
        """Get the MongoDB client"""
        instance = cls.get_instance()
        return cls.client
    
    @classmethod
    def get_database(cls):
        """Get the MongoDB database"""
        instance = cls.get_instance()
        return cls.db
    
    @classmethod
    def get_collection(cls, collection_name):
        """Get a specific collection from the database"""
        instance = cls.get_instance()
        return cls.db[collection_name]
    
    @classmethod
    def close(cls):
        """Close the MongoDB connection"""
        if cls.client:
            cls.client.close()
            logger.info('MongoDB connection closed')
            cls._instance = None
            cls.client = None
            cls.db = None
            
    @classmethod
    async def log_api_request(cls, log_data):
        """
        Log API request data to MongoDB
        
        Args:
            log_data (dict): Data about the API request
            
        Returns:
            None
        """
        try:
            # Ensure we're connected
            if not cls.client:
                await cls.connect_to_mongo()
                
            # Get the logs collection (create if it doesn't exist)
            db = cls.client[cls.db_name]
            logs_collection = db.api_logs
            
            # Insert the log
            await logs_collection.insert_one(log_data)
            
        except Exception as e:
            # Use a different logger to avoid circular logging issues
            import logging
            db_logger = logging.getLogger("database")
            db_logger.warning(f"Failed to log API request to MongoDB: {e}")