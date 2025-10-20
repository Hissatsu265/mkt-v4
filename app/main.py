from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.services.job_service import job_service
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await job_service.init_mongodb()
    await job_service.init_redis()
    yield
    # Shutdown
    if job_service.redis_client:
        await job_service.redis_client.close()

app = FastAPI(
    title="Video Generator API",
    description="API để tạo video từ ảnh, prompt và audio",
    version="1.0.0",
    lifespan=lifespan
)

# ✅ CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Middleware xử lý OPTIONS request (Preflight)
@app.middleware("http")
async def add_options_support(request, call_next):
    if request.method == "OPTIONS":
        from fastapi.responses import Response
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    response = await call_next(request)
    return response

# Router
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Video Generator API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}