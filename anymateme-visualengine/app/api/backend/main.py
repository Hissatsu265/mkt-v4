from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas.requests import ImageGenerationRequest, ImageGenerationResponse
from backend.services.image_service import ImageGenerationService

app = FastAPI(
    title="Flux Image Generator API",
    description="API create image by FLUX and LLM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_service = ImageGenerationService()

@app.on_event("startup")
async def startup_event():
    await image_service.initialize_models()

@app.post("/generate-image", response_model=ImageGenerationResponse)
async def generate_image_endpoint(request: ImageGenerationRequest):
    print(f"Received request: {request}")
    try:
        response = await image_service.generate_image(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generate image: {str(e)}")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas.requests import ImageGenerationRequest, ImageGenerationResponse
from backend.services.image_service import ImageGenerationService
from pydantic import BaseModel

app = FastAPI(
    title="Flux Image Generator API",
    description="API create image by FLUX and LLM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_service = ImageGenerationService()

# Ví dụ schema cho endpoint mới
class TextRequest(BaseModel):
    text: str
    language: str = "vi"

class TextResponse(BaseModel):
    processed_text: str
    word_count: int
    status: str

@app.on_event("startup")
async def startup_event():
    await image_service.initialize_models()

@app.post("/generate-image", response_model=ImageGenerationResponse)
async def generate_image_endpoint(request: ImageGenerationRequest):
    print(f"Received request: {request}")
    try:
        response = await image_service.generate_image(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generate image: {str(e)}")


@app.post("/slide-image", response_model=ImageGenerationResponse)
async def generate_slide_image_endpoint(request: ImageGenerationRequest):
    print(f"Received request for endpoint slide image: {request}")
    try:
        response = await image_service.generate_imageslide(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generate image: {str(e)}")
