import time
from fastapi import Request
import asyncio
from app.core.logger import get_logger

# Create a dedicated logger for API requests
logger = get_logger("api")

class RequestLoggingMiddleware:
    """Middleware for logging API requests to MongoDB and console"""
    
    def __init__(self, app):
        """Initialize the middleware with the app
        Args:
            app: The ASGI application
        """
        self.app = app
        
    async def __call__(self, scope, receive, send):
        """Process the request and log it
        Args:
            scope: The ASGI scope
            receive: The ASGI receive function
            send: The ASGI send function
        """
        if scope["type"] != "http":
            # If it's not an HTTP request (e.g., WebSocket), skip logging
            return await self.app(scope, receive, send)
        
        # Create request object to extract information
        request = Request(scope)
        path = request.url.path
        method = request.method
        client_host = request.client.host if request.client else "unknown"
        
        # Record request start time
        start_time = time.time()
        
        # Create a wrapper for the send function to capture the status code
        status_code = 0
        
        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
        
        # Process the request
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            # Log the error but don't catch it so it can be handled by the error middleware
            logger.error(f"Error processing request: {e}")
            status_code = 500
            raise
        finally:
            # Calculate process time
            process_time = time.time() - start_time
            
            # Extract query parameters (safely)
            query_params = {}
            try:
                query_params = dict(request.query_params)
            except Exception:
                pass
            
            # Create log entry
            log_data = {
                "timestamp": time.time(),
                "path": path,
                "method": method,
                "status_code": status_code,
                "process_time_ms": round(process_time * 1000, 2),
                "client_host": client_host,
                "query_params": query_params
            }
            
            # Important: Force log to console even if level might be filtered
            # This ensures API routes are always logged
            if "/api/" in path:
                # This is an API route - ensure it's logged
                logger.info(f"{method} {path} - {status_code} - {log_data['process_time_ms']}ms")
            else:
                # For non-API routes, use debug level
                logger.debug(f"{method} {path} - {status_code} - {log_data['process_time_ms']}ms")
            
            # Log to MongoDB asynchronously to avoid blocking
            try:
                from app.core.database import DBManager
                # Use asyncio.create_task to run this in the background
                asyncio.create_task(DBManager.log_api_request(log_data))
            except Exception as e:
                # Log MongoDB errors but don't let them affect the response
                logger.warning(f"Failed to log request to MongoDB: {e}")