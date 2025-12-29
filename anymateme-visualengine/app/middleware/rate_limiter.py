from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time
import asyncio
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.request_counts = defaultdict(list)
        self.rate_limit = 100  # requests per minute
        
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = request.client.host
        
        # Skip rate limiting for certain paths
        if request.url.path in ["/health", "/", "/docs", "/redoc"]:
            return await call_next(request)
            
        # Check if rate limit is exceeded
        current_time = time.time()
        self.request_counts[client_ip].append(current_time)
        
        # Count requests in the last minute
        recent_requests = [ts for ts in self.request_counts[client_ip] 
                          if current_time - ts < 60]
        self.request_counts[client_ip] = recent_requests
        
        if len(recent_requests) > self.rate_limit:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return Response(
                content='{"error": {"code": 429, "message": "Too many requests", "type": "rate_limit_exceeded"}}',
                status_code=429,
                media_type="application/json"
            )
            
        response = await call_next(request)
        return response
