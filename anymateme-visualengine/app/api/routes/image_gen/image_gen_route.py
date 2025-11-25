import uuid
import os
import asyncio
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io

from app.config import BASE_DIR, UPLOAD_DIR
from app.api.backend.schemas.requests import ImageRequest, ImageGenerationRequest, ImageGenerationResponse
from app.api.backend.services.image_service import ImageGenerationService
from app.api.routes.slide_image_gen.services.database_services.log_static_slide_image_to_db import (
    log_static_slide_image_to_db
)
from app.api.routes.slide_image_gen.services.database_services.get_image_job_from_db import (
    get_image_job_from_db
)
from app.api.routes.slide_image_gen.services.database_services.update_image_job_status_in_db import (
    update_image_job_status_in_db
)
from app.api.routes.slide_image_gen.services.remote_storage.directus_utils import(
    upload_file_to_directus
)
import base64
from uuid import uuid4

router = APIRouter()
image_service = ImageGenerationService()

# Create directory for saving images
AI_IMAGES_DIR = Path("ai_generated_images")
AI_IMAGES_DIR.mkdir(exist_ok=True)

# ===== QUEUE SYSTEM =====
# Số lượng request có thể xử lý đồng thời (có thể thay đổi tùy server)
MAX_CONCURRENT_REQUESTS = 1

# Queue và semaphore để quản lý request
image_generation_queue = asyncio.Queue()
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
queue_processor_task = None


class QueuedJob(BaseModel):
    job_id: str
    request: ImageRequest
    timestamp: datetime


class AIGenerationStatusResponse(BaseModel):
    success: bool
    message: str
    job_id: str
    status: str = "processing"
    queue_position: Optional[int] = None


class AIImageStatusResponse(BaseModel):
    job_id: str
    status: str
    file_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    image_prompt: Optional[str] = None
    queue_position: Optional[int] = None


class Req:
    def __init__(self, aspect, quality):
        self.aspect = aspect
        self.quality = quality


def get_dimensions(request):
    aspect_map = {
        "1:1": (1, 1),
        "2:3": (2, 3),
        "16:9": (16, 9),
        "9:16": (9, 16),
    }

    quality_map = {
        "low": 512,
        "medium": 768,
        "high": 1024,
        "ultra": 2048,
    }

    aspect = request.aspect.strip()
    quality = request.quality.strip().lower()

    if aspect not in aspect_map or quality not in quality_map:
        raise ValueError("Aspect hoặc Quality không hợp lệ")

    w_ratio, h_ratio = aspect_map[aspect]
    base = quality_map[quality]

    if aspect == "1:1":
        width = height = base
    else:
        if w_ratio >= h_ratio:
            height = base
            width = int(base * w_ratio / h_ratio)
        else:
            width = base
            height = int(base * h_ratio / w_ratio)

        max_dim = 2048
        if width > max_dim or height > max_dim:
            scale = max_dim / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

    return width, height


def convert_aspect_to_dimensions(aspect_ratio: str, max_size: int = 1024) -> tuple[int, int]:
    """Convert aspect ratio string to width and height dimensions."""
    aspect_ratios = {
        '1:1': (max_size, max_size),
        '16:9': (max_size, round(max_size * 9/16)),
        '9:16': (round(max_size * 9/16), max_size),
        '4:3': (max_size, round(max_size * 3/4)),
        '3:4': (round(max_size * 3/4), max_size),
        '21:9': (max_size, round(max_size * 9/21))
    }
    
    return aspect_ratios.get(aspect_ratio, aspect_ratios['1:1'])


def get_available_styles():
    return {
        "Realistic": "Ultra-realistic, 8K UHD, highly detailed, photo-realistic lighting and textures, cinematic composition.",
        "Cartoon": "Cartoon style, bold black outlines, flat and vibrant colors, exaggerated features, anime and comic influence.",
        "DigitalArt": "Digital painting, fantasy or sci-fi themes, soft shading, dynamic lighting, high-resolution concept art feel.",
        "Sketch": "Hand-drawn pencil sketch, rough linework, monochrome or grayscale, raw and minimal artistic style.",
        "Cyberpunk": "Futuristic cyberpunk style, neon lights, dark cityscape, high-tech elements, dystopian atmosphere.",
        "Fantasy": "Fantasy art style, mythical creatures, magical landscapes, vibrant colors, epic and adventurous themes.",
        "LoRA": "Custom LoRA style",
        "Minimal": "Minimalist design, clean composition, soft neutral colors, simple shapes, lots of whitespace, modern aesthetic.",
        "Vintage": "Vintage style, warm tones, retro textures, film grain, nostalgic look, old-fashioned composition.",
        "Anime": "Anime art style, expressive characters, bold outlines, vibrant colors, dynamic action scenes, Japanese animation influence.",
        "Artistic": "Creative artistic style, expressive brushstrokes, abstract elements, bold colors, emotional and imaginative atmosphere."
    }


def get_available_aspects():
    """Get available aspect ratios"""
    return ['1:1', '16:9', '9:16', '4:3', '3:4', '21:9']


def validate_style(style: str) -> tuple[bool, str]:
    """Validate if style is supported"""
    available_styles = get_available_styles()
    if style not in available_styles:
        return False, f"Invalid style '{style}'. Available styles: {list(available_styles.keys())}"
    return True, ""


def validate_aspect(aspect: str) -> tuple[bool, str]:
    """Validate if aspect ratio is supported"""
    available_aspects = get_available_aspects()
    if aspect not in available_aspects:
        return False, f"Invalid aspect ratio '{aspect}'. Available aspects: {available_aspects}"
    return True, ""


def save_pil_image(image, filename: str, directory: Path = AI_IMAGES_DIR) -> str:
    """Save a PIL Image to a file."""
    try:
        file_path = directory / f"{filename}.png"
        image.save(file_path, "PNG")
        return str(file_path)
    except Exception as e:
        print(f"Error saving image {filename}: {str(e)}")
        return None


async def update_ai_image_status(job_id: str, status: str, file_path: Optional[str] = None, error_message: Optional[str] = None):
    """Update the status of an AI image generation job in the database"""
    try:
        update_data = {
            "status": status,
            "updated_at": datetime.now()
        }

        if status == "completed" and file_path:
            update_data["file_path"] = file_path
            update_data["completed_at"] = datetime.now()
        elif status == "failed" and error_message:
            update_data["error_message"] = error_message
            update_data["completed_at"] = datetime.now()

        await update_image_job_status_in_db(job_id, update_data)

    except Exception as e:
        print(f"Error updating AI image status in DB for job {job_id}: {str(e)}")
# ===================================================================================
import base64, io, uuid, aiohttp
from PIL import Image

async def download_image_async(url, upload_dir):
    """Tải ảnh từ URL và lưu lại bất đồng bộ"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to download image: {resp.status}")
            image_data = await resp.read()
            
            img = Image.open(io.BytesIO(image_data))
            img.verify()
            
            filename = f"{uuid.uuid4().hex}.png"
            image_path = upload_dir / filename
            with open(image_path, "wb") as f:
                f.write(image_data)
            print(f"✅ Downloaded image from {url} -> {image_path}")
            return image_path

# ====================================================================

async def process_single_image_generation(request: ImageRequest, job_id: str):
    """Process a single image generation request (used by queue processor)"""
    async with processing_semaphore:
        try:
            print(f"🎯 Processing image generation for job {job_id}")
            
            # Log to database
            await log_static_slide_image_to_db(
                job_id=job_id,
                slide_number=1,
                slide_type=1,
                image_prompt=request.user_prompt,
                status="processing"
            )
            
            await update_ai_image_status(job_id, "processing")
            
            image_path = None
            image_path2 = None
            if "[change pose]" in request.user_prompt:
                request.type_generation="change_pose"
            elif "[change clothes]" in request.user_prompt:
                request.type_generation="image_edit"
            elif "[inpaint]" in request.user_prompt:
                request.type_generation="inpaint"
            # Handle input_image
            if request.input_image and request.input_image != "none":
                if request.type_generation.lower() == "image_edit" or request.type_generation.lower() == "change_pose" or  request.type_generation.lower() == "inpaint":
                    image_path = await download_image_async(request.input_image, UPLOAD_DIR)
                else:
                    base64_string = request.input_image
                    if ',' in base64_string:
                        base64_string = base64_string.split(',')[1]
                    image_data = base64.b64decode(base64_string)
                    img = Image.open(io.BytesIO(image_data))
                    img.verify()
                    filename = f"{uuid.uuid4().hex}.png"
                    image_path = UPLOAD_DIR / filename
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    print(f"✅ Saved input image at {image_path}")

            # Handle input_image2

            if request.input_image2 and request.input_image2 != "none" :
                if request.type_generation.lower() == "image_edit" or request.type_generation.lower() == "change_pose" or  request.type_generation.lower() == "inpaint":
                    image_path2 = await download_image_async(request.input_image2, UPLOAD_DIR)
                else:
                    base64_string = request.input_image2
                    if ',' in base64_string:
                        base64_string = base64_string.split(',')[1]
                    image_data = base64.b64decode(base64_string)
                    img = Image.open(io.BytesIO(image_data))
                    img.verify()
                    filename = f"{uuid.uuid4().hex}.png"
                    image_path2 = UPLOAD_DIR / filename
                    with open(image_path2, "wb") as f:
                        f.write(image_data)
                    print(f"✅ Saved input image at {image_path2}")
            
            # Get dimensions
            width, height = get_dimensions(Req(request.aspect, request.quality))
            
            # Create service request
            service_request = ImageGenerationRequest(
                user_prompt=request.user_prompt,
                aspect=request.aspect,
                width=width,
                height=height,
                job_id=job_id,
                type_generation=request.type_generation if hasattr(request, 'type_generation') else "normal",
                input_image=str(image_path) if image_path else None,
                input_image2=str(image_path2) if image_path2 else None,
                negative_prompt=request.negative_prompt
            )
            
            print(f"⏳ Generating image for {job_id}...")
            response = await image_service.generate_image(service_request)
            
            if isinstance(response, dict):
                image_base64 = response.get('image1_base64') or response.get('image_base64')
            elif hasattr(response, 'image1_base64'):
                image_base64 = response.image1_base64
            elif hasattr(response, 'image_base64'):
                image_base64 = response.image_base64
            else:
                image_base64 = None
            
            if image_base64:
                if image_base64.startswith('data:image'):
                    image_data = image_base64.split(',')[1]
                else:
                    image_data = image_base64
                
                image_bytes = base64.b64decode(image_data)
                pil_image = Image.open(io.BytesIO(image_bytes))
                
                saved_path = save_pil_image(pil_image, job_id)
                
                if saved_path:
                    print(f"✅ Image saved locally: {saved_path}")
                    
                    # Upload to Directus
                    print(f"⬆️ Uploading to Directus...")
                    directus_response = await upload_file_to_directus(saved_path,request.folder_id)
                    
                    file_info = directus_response.get('data', {})
                    file_id = file_info.get('id')
                    file_url = f"{os.getenv('DIRECTUS_URL')}/assets/{file_id}" if file_id else None
                    
                    print(f"✅ Uploaded to Directus: {file_url}")
                    
                    await update_ai_image_status(job_id, "completed", file_path=file_url)
                    
                    # Clean up
                    if os.path.exists(saved_path):
                        os.remove(saved_path)
                        print(f"🗑️ Cleaned up local file: {saved_path}")
                else:
                    raise Exception("Failed to save image file locally")
            else:
                raise Exception("No image data received from generation service")
            
        except Exception as e:
            error_message = str(e)
            print(f"❌ Error processing job {job_id}: {error_message}")
            await update_ai_image_status(job_id, "failed", error_message=error_message)


async def queue_processor():
    """Background task that processes the queue"""
    print("🚀 Queue processor started")
    while True:
        try:
            # Get next job from queue
            queued_job: QueuedJob = await image_generation_queue.get()
            print(f"📤 Processing job {queued_job.job_id} from queue")
            
            # Process the job
            await process_single_image_generation(queued_job.request, queued_job.job_id)
            
            # Mark task as done
            image_generation_queue.task_done()
            
        except Exception as e:
            print(f"❌ Error in queue processor: {str(e)}")
            await asyncio.sleep(1)


def start_queue_processor():
    """Start the queue processor task"""
    global queue_processor_task
    if queue_processor_task is None or queue_processor_task.done():
        queue_processor_task = asyncio.create_task(queue_processor())
        print("✅ Queue processor task created")


def get_queue_position(job_id: str) -> Optional[int]:
    """Get the position of a job in the queue"""
    try:
        queue_list = list(image_generation_queue._queue)
        for i, queued_job in enumerate(queue_list):
            if queued_job.job_id == job_id:
                return i + 1
        return None
    except:
        return None


@router.post("/generate-image", response_model=AIGenerationStatusResponse)
async def generate_image_endpoint(request: ImageRequest):
    """
    Generate image with queue system.
    Returns immediately with job ID and queue position.
    """
    print(f"📥 Received image generation request")
    
    # Start queue processor if not running
    start_queue_processor()
    
    # Generate job ID
    job_id = request.job_id if hasattr(request, 'job_id') and request.job_id else str(uuid.uuid4())
    print(f"🆔 Job ID: {job_id}")
    
    try:
        # Create queued job
        queued_job = QueuedJob(
            job_id=job_id,
            request=request,
            timestamp=datetime.now()
        )
        
        # Add to queue
        await image_generation_queue.put(queued_job)
        queue_size = image_generation_queue.qsize()
        
        print(f"✅ Job {job_id} added to queue (position: {queue_size})")
        
        # Log initial status
        await log_static_slide_image_to_db(
            job_id=job_id,
            slide_number=1,
            slide_type=1,
            image_prompt=request.user_prompt,
            status="queued"
        )
        
        return AIGenerationStatusResponse(
            success=True,
            message=f"Image generation queued. Position in queue: {queue_size}",
            job_id=job_id,
            status="queued",
            queue_position=queue_size
        )
        
    except Exception as e:
        print(f"❌ Error queuing job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error queuing image generation: {str(e)}")


@router.post("/slide-image", response_model=ImageGenerationResponse)
async def generate_slide_image_endpoint(request: ImageRequest):
    """Generate slide image (legacy endpoint - kept for compatibility)"""
    print(f"Received request for slide image: {request}")
    
    job_id = request.job_id if hasattr(request, 'job_id') and request.job_id else str(uuid.uuid4())
    width, height = convert_aspect_to_dimensions("1:1")
    
    try:
        service_request = ImageGenerationRequest(
            user_prompt=request.user_prompt,
            negative_prompt=request.negative_prompt,
            style=request.style,
            aspect="1:1",
            width=width,
            height=height,
            job_id=job_id
        )
        
        response = await image_service.generate_imageslide(service_request)
        
        if isinstance(response, dict):
            return ImageGenerationResponse(
                success=response.get('success', False),
                message=response.get('message', 'Image generated'),
                job_id=job_id,
                image1_base64=response.get('image1_base64'),
                image2_base64=response.get('image2_base64'),
                final_prompt=response.get('final_prompt'),
                system_info=response.get('system_info'),
                warning=response.get('warning')
            )
        elif hasattr(response, 'job_id'):
            if not response.job_id:
                response.job_id = job_id
            return response
        else:
            return ImageGenerationResponse(
                success=False,
                message="Unexpected response format from service",
                job_id=job_id,
                image_base64=None,
                final_prompt=None,
                system_info=None,
                warning=None
            )
            
    except Exception as e:
        print(f"Error in generate_slide_image_endpoint: {str(e)}")
        return ImageGenerationResponse(
            success=False,
            message=f"Error generating image: {str(e)}",
            job_id=job_id,
            image1_base64=None,
            image2_base64=None,
            final_prompt=None,
            system_info=None,
            warning=None
        )


@router.post("/ai-generate", response_model=AIGenerationStatusResponse)
async def ai_generate_endpoint(request: ImageRequest, background_tasks: BackgroundTasks):
    """
    AI Generate with queue system (uses same queue as /generate-image)
    """
    try:
        # Validate style and aspect
        style_valid, style_error = validate_style(request.style)
        if not style_valid:
            raise HTTPException(status_code=400, detail=style_error)

        aspect_valid, aspect_error = validate_aspect(request.aspect)
        if not aspect_valid:
            raise HTTPException(status_code=400, detail=aspect_error)

        # Start queue processor
        start_queue_processor()
        
        job_id = request.job_id if hasattr(request, 'job_id') and request.job_id else str(uuid.uuid4())
        print(f"🚀 AI Generate for job {job_id}")

        # Create queued job
        queued_job = QueuedJob(
            job_id=job_id,
            request=request,
            timestamp=datetime.now()
        )
        
        # Add to queue
        await image_generation_queue.put(queued_job)
        queue_size = image_generation_queue.qsize()
        
        return AIGenerationStatusResponse(
            success=True,
            message=f"AI image generation queued. Position: {queue_size}",
            job_id=job_id,
            status="queued",
            queue_position=queue_size
        )

    except Exception as e:
        print(f"Error starting AI image generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/ai-generate-status/{job_id}", response_model=AIImageStatusResponse)
async def get_ai_generate_status(job_id: str):
    """Get status with queue position if still queued"""
    try:
        image_status = await get_image_job_from_db(job_id)

        if not image_status:
            raise HTTPException(status_code=404, detail="Job not found")

        # Get queue position if still queued
        queue_position = None
        if image_status.get('status') == 'queued':
            queue_position = get_queue_position(job_id)

        return AIImageStatusResponse(
            job_id=job_id,
            status=image_status.get('status', 'unknown'),
            file_path=image_status.get('file_path'),
            error_message=image_status.get('error_message'),
            created_at=image_status.get('created_at'),
            completed_at=image_status.get('completed_at'),
            image_prompt=image_status.get('image_prompt'),
            queue_position=queue_position
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/ai-generate-file/{job_id}")
async def get_ai_generate_file(job_id: str):
    """Get completed image file info"""
    try:
        image_status = await get_image_job_from_db(job_id)

        if not image_status:
            raise HTTPException(status_code=404, detail="Job not found")

        if image_status.get('status') != 'completed':
            raise HTTPException(
                status_code=400,
                detail=f"Image not ready. Status: {image_status.get('status')}"
            )

        file_path = image_status.get('file_path')
        if not file_path:
            raise HTTPException(status_code=404, detail="Image file not found")

        return {
            "job_id": job_id,
            "status": "completed",
            "file_path": file_path,
            "directus_url": file_path,
            "created_at": image_status.get('created_at'),
            "completed_at": image_status.get('completed_at'),
            "image_prompt": image_status.get('image_prompt')
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/queue-status")
async def get_queue_status():
    """Get current queue statistics"""
    return {
        "queue_size": image_generation_queue.qsize(),
        "max_concurrent": MAX_CONCURRENT_REQUESTS,
        "processor_running": queue_processor_task is not None and not queue_processor_task.done()
    }


@router.get("/styles")
async def get_styles():
    """Get available styles and aspects"""
    return {
        "styles": get_available_styles(),
        "aspects": get_available_aspects()
    }