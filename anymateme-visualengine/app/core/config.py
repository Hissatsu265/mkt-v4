from pydantic_settings import BaseSettings
from typing import List, Union, Optional
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings management class that loads configuration from environment variables
    based on the specified environment (LOCAL, DEV, STAGING, PROD).
    """
    # Environment setting
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "LOCAL")
    # Docker flag to determine service configuration
    IN_DOCKER: bool = os.getenv("IN_DOCKER", "False").lower() == "true"
    
    # API and project configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Text Color Detection API"
    PROJECT_VERSION: str = "1.0.0-beta"
    PROJECT_DESCRIPTION: str = "AnyMate EduHub is a platform that provides educational resources and tools for students and educators."
    
    # CORS Configuration - allows "*" for all origins or specific list of origins
    BACKEND_CORS_ORIGINS: Union[str, List[str]] = "*"
    
    # Upload folder for images
    UPLOAD_FOLDER: str = "uploads"
    
    # Set the base directory - used for managing relative paths
    BASE_DIR: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).parent
    
    # External API Keys
    ICONFINDER_API_KEY: str = os.getenv("ICONFINDER_API_KEY", "")
    
    # AI Model Configuration - specifies which AI model to use
    AI_MODEL: str = os.getenv("AI_MODEL", "phi4-test:latest")
    
    # Model mappings for different service types
    MODEL_MAPPINGS: dict = {
        "plain_language": "phi4-extended",
        "exercise_generation": "phi4-extended",
        "speech_gen": "phi4-extended",
        "multilingual_translation": "phi4-extended",
        "slide_gen": "phi4-extended",
    }
    
    # MongoDB Configuration - Define all fields here with defaults
    MONGODB_URL: str = ""
    MONGODB_DB_NAME: str = ""
    MONGODB_COLLECTION_NAME: str = ""
    MONGODB_LOGS_COLLECTION: str = ""
    MONGO_HOST: str = ""
    MONGO_PORT: int = 27017
    MONGO_USERNAME: Optional[str] = None
    MONGO_PASSWORD: Optional[str] = None
    
    # Server Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    RELOAD: bool = True  # Auto-reload server on code changes (for development)
    LOG_LEVEL: str = "DEBUG"
    
    # Ollama service URLs - used for AI model inference
    OLLAMA_SERVICE_1_URL: str = ""
    OLLAMA_SERVICE_2_URL: str = ""
    OLLAMA_API_ENDPOINT: str = "http://localhost:11434"
    
    # Logging Configuration
    ENABLE_MONGODB_LOGGING: bool = True  # Store API logs in MongoDB
    
    # authentication settings
    DIRECTUS_AUTH_URL: str = os.getenv("DIRECTUS_AUTH_URL", "https://cms.anymateme.pro")
    
    # GitHub Integration
    GITHUB_WEBHOOK_SECRET: str = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    
    # Discord Integration
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    DISCORD_BOT_TOKEN: str = os.getenv("DISCORD_BOT_TOKEN", "")
    DISCORD_CHANNEL_ID: str = os.getenv("DISCORD_CHANNEL_ID", "")
    def model_post_init(self, __context):
        """
        Initialize environment-specific settings after the model is created.
        This method sets different configurations based on the ENVIRONMENT value.
        """
        # First, set up environment-specific configurations
        if self.ENVIRONMENT == "PROD":
            # Production DB config
            self.MONGODB_URL = self.get_required_env("PROD_MONGODB_URL")
            self.MONGODB_DB_NAME = self.get_required_env("PROD_MONGODB_DB_NAME")
            self.MONGODB_COLLECTION_NAME = self.get_required_env("PROD_MONGODB_COLLECTION_NAME")
            self.MONGODB_LOGS_COLLECTION = self.get_required_env("PROD_MONGODB_LOGS_COLLECTION")
            self.MONGO_HOST = self.get_required_env("PROD_MONGO_HOST")
            self.MONGO_PORT = int(self.get_required_env("PROD_MONGO_PORT"))
            self.MONGO_USERNAME = self.get_required_env("PROD_MONGO_USERNAME")
            self.MONGO_PASSWORD = self.get_required_env("PROD_MONGO_PASSWORD")
            
            # Production server config - more restrictive for security
            self.HOST = self.get_required_env("PROD_HOST")
            self.PORT = int(self.get_required_env("PROD_PORT"))
            self.RELOAD = False  # Disable auto-reload in production
            self.LOG_LEVEL = self.get_required_env("PROD_LOG_LEVEL").upper()
            
            # Production Ollama services
            self.OLLAMA_SERVICE_1_URL = os.getenv("PROD_OLLAMA_SERVICE_1_URL", "http://ollama-1:11434")
            self.OLLAMA_SERVICE_2_URL = os.getenv("PROD_OLLAMA_SERVICE_2_URL", "http://ollama-2:11434")
            
        elif self.ENVIRONMENT == "STAGING":
            # Staging DB config
            self.MONGODB_URL = self.get_required_env("STAGING_MONGODB_URL")
            self.MONGODB_DB_NAME = self.get_required_env("STAGING_MONGODB_DB_NAME")
            self.MONGODB_COLLECTION_NAME = self.get_required_env("STAGING_MONGODB_COLLECTION_NAME")
            self.MONGODB_LOGS_COLLECTION = self.get_required_env("STAGING_MONGODB_LOGS_COLLECTION")
            self.MONGO_HOST = self.get_required_env("STAGING_MONGO_HOST")
            self.MONGO_PORT = int(self.get_required_env("STAGING_MONGO_PORT"))
            self.MONGO_USERNAME = self.get_required_env("STAGING_MONGO_USERNAME")
            self.MONGO_PASSWORD = self.get_required_env("STAGING_MONGO_PASSWORD")
            
            # Staging server config
            self.HOST = self.get_required_env("STAGING_HOST")
            self.PORT = int(self.get_required_env("STAGING_PORT"))
            self.RELOAD = False  # Disable auto-reload in staging
            self.LOG_LEVEL = self.get_required_env("STAGING_LOG_LEVEL").upper()
            
            # Staging Ollama services
            self.OLLAMA_SERVICE_1_URL = os.getenv("STAGING_OLLAMA_SERVICE_1_URL", "http://ollama-1:11434")
            self.OLLAMA_SERVICE_2_URL = os.getenv("STAGING_OLLAMA_SERVICE_2_URL", "http://ollama-2:11434")
            
        elif self.ENVIRONMENT == "DEV":
            # Dev DB config
            self.MONGODB_URL = os.getenv("DEV_MONGODB_URL", "")
            self.MONGODB_DB_NAME = os.getenv("DEV_MONGODB_DB_NAME", "anymateDB_dev")
            self.MONGODB_COLLECTION_NAME = os.getenv("DEV_MONGODB_COLLECTION_NAME", "default")
            self.MONGODB_LOGS_COLLECTION = os.getenv("DEV_MONGODB_LOGS_COLLECTION", "api_logs")
            self.MONGO_HOST = os.getenv("DEV_MONGO_HOST", "localhost")
            self.MONGO_PORT = int(os.getenv("DEV_MONGO_PORT", 27017))
            self.MONGO_USERNAME = os.getenv("DEV_MONGO_USERNAME")
            self.MONGO_PASSWORD = os.getenv("DEV_MONGO_PASSWORD")
            
            # Dev server config - more permissive for development
            self.HOST = os.getenv("DEV_HOST", "0.0.0.0")
            self.PORT = int(os.getenv("DEV_PORT", 8002))
            self.RELOAD = False  # Enable auto-reload for development
            self.LOG_LEVEL = os.getenv("DEV_LOG_LEVEL", "DEBUG").upper()
            
            # Dev Ollama services
            self.OLLAMA_SERVICE_1_URL = os.getenv("DEV_OLLAMA_SERVICE_1_URL", "http://ollama-1:11434")
            self.OLLAMA_SERVICE_2_URL = os.getenv("DEV_OLLAMA_SERVICE_2_URL", "http://ollama-2:11434")
            
        else:  # Default to LOCAL
            # Local DB config
            self.MONGODB_URL = os.getenv("LOCAL_MONGODB_URL", "")
            self.MONGODB_DB_NAME = os.getenv("LOCAL_MONGODB_DB_NAME", "anymateme_eduhub")
            self.MONGODB_COLLECTION_NAME = os.getenv("LOCAL_MONGODB_COLLECTION_NAME", "default")
            self.MONGODB_LOGS_COLLECTION = os.getenv("LOCAL_MONGODB_LOGS_COLLECTION", "api_logs")
            self.MONGO_HOST = os.getenv("LOCAL_MONGO_HOST", "localhost")
            self.MONGO_PORT = int(os.getenv("LOCAL_MONGO_PORT", 27017))
            self.MONGO_USERNAME = os.getenv("LOCAL_MONGO_USERNAME")
            self.MONGO_PASSWORD = os.getenv("LOCAL_MONGO_PASSWORD")
            
            # Local server config - most permissive for local development
            self.HOST = os.getenv("LOCAL_HOST", "127.0.0.1")
            self.PORT = int(os.getenv("LOCAL_PORT", 8001))
            self.RELOAD = True  # Enable auto-reload for local development
            self.LOG_LEVEL = os.getenv("LOCAL_LOG_LEVEL", "DEBUG").upper()
            
            # Local Ollama - using a single endpoint for simplicity
            self.OLLAMA_API_ENDPOINT = os.getenv("LOCAL_OLLAMA_API_ENDPOINT", "http://localhost:11434")
        
        # Now handle Docker-specific overrides for Ollama URLs
        # Docker flag takes precedence over environment for service URLs
        if not self.IN_DOCKER:
            # When not in Docker, all environments use localhost
            self.OLLAMA_API_ENDPOINT = os.getenv("LOCAL_OLLAMA_API_ENDPOINT", "http://localhost:11434")
            self.OLLAMA_SERVICE_1_URL = os.getenv("LOCAL_OLLAMA_API_ENDPOINT", "http://localhost:11434")
            self.OLLAMA_SERVICE_2_URL = os.getenv("LOCAL_OLLAMA_API_ENDPOINT", "http://localhost:11434")
    
    def get_required_env(self, env_variable):
        """
        Get and validate required environment variables.
        Raises ValueError if the variable is not set.
        
        Args:
            env_variable: The name of the environment variable to retrieve
            
        Returns:
            The value of the environment variable
            
        Raises:
            ValueError: If the environment variable is not set
        """
        value = os.getenv(env_variable)
        if value is None:
            raise ValueError(f"Invalid or missing '{env_variable}' in the environment variables")
        return value
    
    def get_cors_origins(self):
        """
        Process CORS settings to handle both string and list formats.
        
        Returns:
            List of allowed origins for CORS
        """
        if isinstance(self.BACKEND_CORS_ORIGINS, str):
            if self.BACKEND_CORS_ORIGINS == "*":
                return ["*"]
            return [origin.strip() for origin in self.BACKEND_CORS_ORIGINS.split(",")]
        return self.BACKEND_CORS_ORIGINS
    
    def get_ollama_url(self):
        """
        Return the appropriate Ollama URL based on environment and Docker status.
        Used as a fallback or default when not routing by service type.
        
        Returns:
            The Ollama API endpoint URL for the current environment
        """
        # Docker availability is the primary factor in determining URLs
        if not self.IN_DOCKER:
            # When not in Docker, always use the local endpoint
            return self.OLLAMA_API_ENDPOINT
        
        # When in Docker, the environment determines which endpoint(s) to use
        if self.ENVIRONMENT == "LOCAL":
            # Local environment uses single endpoint even in Docker
            return self.OLLAMA_API_ENDPOINT
        else:
            # For non-local environments in Docker, default to service 1
            # (actual routing by service type happens in ollama_service.py)
            return self.OLLAMA_SERVICE_1_URL

    class Config:
        """
        Pydantic configuration for the Settings class.
        """
        case_sensitive = True  # Environment variable names are case-sensitive
        env_file = ".env"      # File to load environment variables from
        extra = "ignore"       # Ignore extra fields in environment variables

# Create a global settings instance
settings = Settings()