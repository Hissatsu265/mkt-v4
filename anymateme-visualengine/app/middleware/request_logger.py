from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import time
import logging
import uuid

logger = logging.getLogger(__name__)

class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        logger.info(f"Request started | ID: {request_id} | Method: {request.method} | Path: {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            # Log response
            logger.info(
                f"Request completed | ID: {request_id} | "
                f"Status: {response.status_code} | "
                f"Time: {process_time:.4f}s"
            )
            
            return response
        except Exception as e:
            logger.error(
                f"Request failed | ID: {request_id} | "
                f"Error: {str(e)}"
            )
            raise
