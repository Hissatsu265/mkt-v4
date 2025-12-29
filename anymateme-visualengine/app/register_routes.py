from fastapi import FastAPI, APIRouter
from app.api.routes.image_gen.image_gen_route import router as image_router
from app.api.routes.slide_image_gen.slide_image_gen_route import slide_image_gen_route as slide_image_gen_route
from app.middleware.rate_limiter import RateLimitMiddleware
from app.api.routes.auth import auth_routes
from app.api.routes.github.github_routes import router as github_router


def register_routes(app: FastAPI):
    """
    Register all routes for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    
    # Register API routers
    app.include_router(image_router, prefix="/api/v1/ai", tags=["Image Generation"])
    app.include_router(slide_image_gen_route, prefix="/api/v1/ai", tags=["Image Generation"])
    app.include_router(auth_routes.router, prefix="/api/v1/auth", tags=["Authentication"])
