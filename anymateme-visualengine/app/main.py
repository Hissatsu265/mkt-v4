from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import requests
import threading
import os
import traceback
from datetime import datetime
from dotenv import load_dotenv
from app.middleware.rate_limiter import RateLimitMiddleware
from app.middleware.logging import RequestLoggingMiddleware
from app.register_routes import register_routes
from app.core.config import settings
from app.core.database import DBManager
from app.core.logger import setup_logging, get_logger
from app.modules.Discord.bot import AnymateMeBot  # Import the Discord bot class
from app.api.backend.services.image_service import ImageGenerationService
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# Load environment variables
load_dotenv()

# Setup logging configuration
loggers = setup_logging()
logger = get_logger("app.main")

# Use settings from config
API_VERSION = settings.PROJECT_VERSION
ENVIRONMENT = settings.ENVIRONMENT
API_TITLE = settings.PROJECT_NAME

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=API_VERSION
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# Initialize services
image_service = ImageGenerationService()


# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")
except:
    logger.warning("Static files directory not found. Custom UI elements may not work properly.")

register_routes(app)

# @app.on_event("startup")
# async def startup_db_client():
#     """Initialize database connection on startup"""
#     try:
#         await DBManager.connect_to_mongo()
#         logger.info("Connected to MongoDB database")
#     except Exception as e:
#         logger.error(f"Failed to connect to MongoDB: {e}")

# start image generation service
@app.on_event("startup")
async def startup_image_service():
    """Start the image generation service"""
    try:
        await image_service.initialize_models()
        logger.info("Image generation service started successfully")
    except Exception as e:
        logger.error(f"Failed to start image generation service: {e}")
        logger.error(traceback.format_exc())

@app.on_event("shutdown")
async def shutdown_db_client():
    """Close database connection on shutdown"""
    await DBManager.close_mongo_connection()
    logger.info("Disconnected from MongoDB database")

# Start Discord Bot on Startup
@app.on_event("startup")
def start_discord_bot():
    """Start the Discord bot in a separate thread"""
    try:
        bot_token = os.getenv("DISCORD_BOT_TOKEN")
        if not bot_token:
            logger.warning("No Discord bot token provided - Discord bot will not start")
            return
            
        # Create and start the bot in a separate thread
        # discord_bot = AnymateMeBot(bot_token)
        
        # # Run bot in a background thread so it doesn't block FastAPI
        # bot_thread = threading.Thread(target=discord_bot.run, daemon=True)
        # bot_thread.start()
        
        logger.info("Discord bot started in background thread")
    except Exception as e:
        logger.error(f"Failed to start Discord bot: {str(e)}")
        logger.error(traceback.format_exc())

@app.get("/")
async def root():
    return {
        "message": "Welcome to Image Generation API", 
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/ui/")
async def serve_ui():
    return FileResponse("frontend/index.html")

app.mount("/ui", StaticFiles(directory="frontend"), name="frontend")

# Health check endpoint
@app.get("/health")
async def health_check():
    model_status = "available"
    try:
        # Use the appropriate Ollama endpoint based on environment
        ollama_endpoint = settings.get_ollama_url() if settings.ENVIRONMENT == "LOCAL" else settings.OLLAMA_SERVICE_1_URL
        response = requests.get(ollama_endpoint)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        model_status = f"unavailable: {str(e)}"

    return {
        "status": "healthy",
        "api_version": API_VERSION,
        "environment": ENVIRONMENT,
        "timestamp": datetime.now().isoformat(),
        "dependencies": {
            "model_service": model_status,
            "database": "connected"
        }
    }


# if __name__ == "__main__":
#     logger.info(f"Starting {API_TITLE} server v{API_VERSION}")
#     # Use settings from config instead of environment variables directly
#     host = settings.HOST
#     port = settings.PORT
#     reload = settings.RELOAD
#     logger.info(f"Application running in {ENVIRONMENT} environment")
#     logger.info(f"Listening on {host}:{port}")
#     uvicorn.run("app.main:app", host=host, port=port, log_level=settings.LOG_LEVEL.lower(), reload=reload)