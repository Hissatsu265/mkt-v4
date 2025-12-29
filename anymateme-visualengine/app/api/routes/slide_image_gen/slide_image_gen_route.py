import uuid
import base64
import os
import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from uuid import UUID
import json
from datetime import datetime

from app.api.backend.services.image_service import (
    ImageGenerationService,
    ImageGenerationRequest,
)
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


slide_image_gen_route = APIRouter()
image_service = ImageGenerationService()

# Create directory for saving images
IMAGES_DIR = Path("generated_images")
IMAGES_DIR.mkdir(exist_ok=True)

class ImageContent(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the image generation job")
    image_prompt: str = Field(..., description="Text prompt for image generation")
    layout: Optional[str] = Field(None, description="Layout configuration (only for certain slide types)")


class SlideImageRequest(BaseModel):
    slide_number: int = Field(..., description="The slide number", ge=1)
    type: int = Field(..., description="Type of slide content", ge=1)
    content: List[ImageContent] = Field(..., description="List of image content for the slide", min_items=1)

class ImageRequest(BaseModel):
    slides: List[SlideImageRequest] = Field(..., description="List of slides with image generation requests", min_items=1)

class ImageGenerationStatusResponse(BaseModel):
    success: bool
    message: str
    request_id: str
    total_images: int
    status: str = "processing"

class IndividualImageStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    file_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    slide_number: Optional[int] = None
    slide_type: Optional[int] = None
    image_prompt: Optional[str] = None

def save_base64_image(base64_string: str, filename: str, directory: Path = IMAGES_DIR) -> str:
    """Save a base64 encoded image to a file."""
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        file_path = directory / f"{filename}.png"
        
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        return str(file_path)
    
    except Exception as e:
        print(f"Error saving image {filename}: {str(e)}")
        return None

async def update_image_status_in_db(job_id: str, status: str, file_path: Optional[str] = None, error_message: Optional[str] = None):
    """Update the status of an individual image generation job in the database"""
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
        
        # Update using your existing database service
        await update_image_job_status_in_db(job_id, update_data)
        
    except Exception as e:
        print(f"Error updating image status in DB for job {job_id}: {str(e)}")

async def pre_log_all_images_to_db(request: ImageRequest):
    """Pre-log all image jobs to database with 'queued' status"""
    try:
        print("🔄 Pre-logging all images to database...")
        
        for slide in request.slides:
            slide_number = slide.slide_number
            slide_type = slide.type
            
            for content in slide.content:
                job_id = content.job_id
                image_prompt = content.image_prompt
                
                # Log each image with 'queued' status
                await log_static_slide_image_to_db(
                    job_id=job_id,
                    slide_number=slide_number,
                    slide_type=slide_type,
                    image_prompt=image_prompt,
                    status="queued" 
                )
                
                print(f"  📝 Logged job {job_id} with status 'queued'")
        
        print("✅ All images pre-logged to database")
        
    except Exception as e:
        print(f"❌ Error pre-logging images to database: {str(e)}")
        raise


def get_type_config(slide_type: int, layout: str = None) -> dict:
    """Get aspect ratio and dimensions based on slide type and layout."""

    type_configs = {
        1: {'aspect': '4:5', 'width': 324, 'height': 368},
        3: {
            'horizontal': {'aspect': '4:1', 'width': 674, 'height': 172},
            'vertical': {'aspect': '4:5', 'width': 310, 'height': 352}
        },
        4: {'aspect': '16:9', 'width': 310, 'height': 164}
    }



    if slide_type == 3:
        # For Type 3, use layout to determine configuration
        layout_key = layout if layout in ['horizontal', 'vertical'] else 'vertical'  # Default to horizontal
        return type_configs[3][layout_key]
    else:
        return type_configs.get(slide_type, {'aspect': '1:1', 'width': 1024, 'height': 1024})





async def process_images_sequentially(request: ImageRequest, request_id: str):
    """Process images one by one with database status updates"""
    try:
        total_images = sum(len(slide.content) for slide in request.slides)
        
        print(f"🎯 Starting SEQUENTIAL processing for {total_images} images")
        
        # First, pre-log all images to database with 'queued' status
        await pre_log_all_images_to_db(request)
        
        image_counter = 0
        
        # Process each slide
        for slide in request.slides:
            slide_number = slide.slide_number
            slide_type = slide.type
            layout = slide.content[0].layout if slide.content else None
            
            print(f"\n📋 Processing (Slide {slide_number}) (Type: {slide_type}) (Layout: {layout}) ")
            
            # Get the configuration for this slide type (pass layout for Type 3)
            type_config = get_type_config(slide_type, layout)
            aspect_ratio = type_config['aspect']
            width = type_config['width']
            height = type_config['height']
    
            print(f"   📐 Aspect Ratio: {aspect_ratio} | Dimensions: {width}x{height}")
            # Process each image in the slide ONE BY ONE
            for i, content in enumerate(slide.content, 1):
                image_counter += 1
                job_id = content.job_id
                image_prompt = content.image_prompt
                
                
                print(f"  🔄 Processing image {image_counter}/{total_images} - Job ID: {job_id}")
                print(f"  📝 Prompt: {image_prompt}")
                
                try:
                    # Update status from 'queued' to 'processing' in database
                    await update_image_status_in_db(job_id, "processing")
                    print(f"  ⏳ Updated status to 'processing' for {job_id}")
                    
                    # Create service request
                    service_request = ImageGenerationRequest(
                        user_prompt=image_prompt,
                        style="Realistic",
                        aspect=aspect_ratio,
                        width=width,
                        height=height,
                        job_id=job_id
                    )
                    
                    # Generate image (WAIT for this to complete before moving to next)
                    print(f"  ⏳ Sending generation request for {job_id}...")
                    response = await image_service.generate_image(service_request)
                    
                    # Check if it's a successful response and has image data
                    if response.success and response.image_base64:
                        # Save the image locally first
                        saved_path = save_base64_image(response.image_base64, job_id)
                        
                        if saved_path:
                            print(f"  ✅ Image saved locally: {saved_path}")
                            
                            # Upload to Directus asynchronously
                            print(f"  ⬆️ Uploading to Directus...")
                            directus_response = await upload_file_to_directus(saved_path)
                            
                            # Extract file info from Directus response
                            file_info = directus_response.get('data', {})
                            file_id = file_info.get('id')
                            file_url = f"{os.getenv('DIRECTUS_URL')}/assets/{file_id}" if file_id else None
                            
                            print(f"  ✅ Uploaded to Directus: {file_url}")
                            
                            # Update status to "completed" in database with both local path and remote URL
                            await update_image_status_in_db(
                                job_id, 
                                "completed", 
                                file_path=file_url,
                            )
                            
                            # Optional: Clean up local file after successful upload
                            os.remove(saved_path)
                            
                        else:
                            raise Exception("Failed to save image file locally")
                    else:
                        # Handle case where generation failed but didn't throw exception
                        error_msg = response.message if hasattr(response, 'message') else "No image data received from service"
                        if hasattr(response, 'warning') and response.warning:
                            error_msg += f" Warning: {response.warning}"
                        raise Exception(error_msg)
                        
                except Exception as e:
                    error_message = str(e)
                    print(f"  ❌ Error processing job {job_id}: {error_message}")
                    
                    # Update status to "failed" in database with error message
                    await update_image_status_in_db(job_id, "failed", error_message=error_message)
                
                # Small delay between images (optional)
                await asyncio.sleep(0.5)
            
            print(f"✅ Completed slide {slide_number}")
        
        print(f"🎉 Sequential processing completed for request {request_id}")
        
    except Exception as e:
        print(f"💥 Error in sequential processing: {str(e)}")
        
        # If pre-logging failed, update any remaining 'queued' jobs to 'failed'
        try:
            for slide in request.slides:
                for content in slide.content:
                    job_id = content.job_id
                    # Check if job is still queued and update to failed
                    job_status = await get_image_job_from_db(job_id)
                    if job_status and job_status.get('status') == 'queued':
                        await update_image_status_in_db(job_id, "failed", error_message=f"Processing failed: {str(e)}")
        except Exception as cleanup_error:
            print(f"Error during cleanup: {str(cleanup_error)}")

@slide_image_gen_route.post("/slide-image-gen-route", response_model=ImageGenerationStatusResponse)
async def generate_slide_image_gen_endpoint(request: ImageRequest, background_tasks: BackgroundTasks):
    """
    Immediately respond and process images one by one in background
    """
    try:
        request_id = str(uuid.uuid4())
        total_images = sum(len(slide.content) for slide in request.slides)

        # json pretty print of the request
        print(f"Received request for slide image generation: {json.dumps(request.dict(), indent=2)}")
        print(f"🚀 Starting sequential generation of {total_images} images")

        # save the reqeust body in a file
        

        
        # Add background task for SEQUENTIAL processing
        background_tasks.add_task(process_images_sequentially, request, request_id)
        
        return ImageGenerationStatusResponse(
            success=True,
            message=f"🚀 Started sequential generation of {total_images} images. All images queued and will be processed one by one.",
            request_id=request_id,
            total_images=total_images,
            status="processing"
        )
        
    except Exception as e:
        print(f"Error starting sequential image generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting image generation: {str(e)}")

@slide_image_gen_route.get("/image-status/{job_id}", response_model=IndividualImageStatusResponse)
async def get_individual_image_status(job_id: str):
    """Get the status of a specific image generation job from database"""
    try:
        # Get image status from database
        image_status = await get_image_job_from_db(job_id)
        
        if not image_status:
            raise HTTPException(status_code=404, detail="Image job not found")
        
        return IndividualImageStatusResponse(
            job_id=job_id,
            status=image_status.get('status', 'unknown'),
            file_path=image_status.get('file_path'),
            error_message=image_status.get('error_message'),
            created_at=image_status.get('created_at'),
            completed_at=image_status.get('completed_at'),
            slide_number=image_status.get('slide_number'),
            slide_type=image_status.get('slide_type'),
            image_prompt=image_status.get('image_prompt')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image status: {str(e)}")

@slide_image_gen_route.get("/image-file/{job_id}")
async def get_image_file(job_id: str):
    """Get the image file if generation is completed"""
    try:
        # Get status from database
        image_status = await get_image_job_from_db(job_id)
        
        if not image_status:
            raise HTTPException(status_code=404, detail="Image job not found")
        
        if image_status.get('status') != 'completed':
            raise HTTPException(status_code=400, detail=f"Image generation not completed. Status: {image_status.get('status')}")
        
        file_path = image_status.get('file_path')
        if not file_path or not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        
        return {
            "job_id": job_id,
            "status": "completed",
            "file_path": file_path,
            "download_url": f"/download-image/{job_id}"  # You can implement this endpoint
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image file: {str(e)}")