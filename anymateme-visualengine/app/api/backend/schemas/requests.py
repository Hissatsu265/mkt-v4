# schemas/requests.py
from pydantic import BaseModel, Field
from typing import Optional

# Client request model - job_id is optional, will be auto-generated if not provided
class ImageRequest(BaseModel):
    user_prompt: str = Field(..., description="Description of the image to generate")
    # style: str = Field(default="Realistic", description="Image style")
    aspect: str = Field(default="1:1", description="Aspect ratio")
    quality: str = Field(default="High", description="Quality setting")
    input_image:Optional[str] = Field(default="none", description="Input image base64")
    input_image2: Optional[str] = Field(default="none", description="Input image base64 2")
    type_generation: Optional[str] = Field(default="normal", description="Type of generation")
    # width: int = Field(default=1024, ge=0, le=3000, description="Width (0 = auto)")
    # height: int = Field(default=1024, ge=0, le=3000, description="Height (0 = auto)")
    job_id: Optional[str] = Field(None, description="Optional job ID for tracking")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt to avoid certain features")
    folder_id: Optional[str] = Field(default="526a7db6-cb1a-48ef-a0ea-a49bad045f1e", description="Folder ID for organizing jobs")

# Service request model with all generation parameters
class ImageGenerationRequest(BaseModel):
    user_prompt: str = Field(..., description="Description of the image to generate")
    method: str = Field(default="phi4", description="Prompt processing method")
    model_choice: str = Field(default="FLUX", description="Image generation model")
    style: str = Field(default="Realistic", description="Image style")
    aspect: str = Field(default="1:1", description="Aspect ratio")
    use_custom_lora: bool = Field(default=False, description="Use custom LoRA")
    lora_repo: Optional[str] = Field(default="", description="LoRA repository")
    lora_scale: float = Field(default=0.8, ge=0.1, le=2.0, description="LoRA strength")
    guidance_scale: float = Field(default=9.5, ge=1.0, le=20.0, description="Guidance scale")
    num_steps: int = Field(default=15, ge=5, le=50, description="Inference steps")
    width: int = Field(default=1024, ge=0, le=3000, description="Width (0 = auto)")
    height: int = Field(default=1024, ge=0, le=3000, description="Height (0 = auto)")
    theme: str = Field(default="", description="Additional theme")
    safetensors: str = Field(default="", description="Safetensors filename")
    job_id: str = Field(..., description="Job ID for tracking")
    input_image: Optional[str] = Field(default=None, description="Input image base64")
    input_image2: Optional[str] = Field(default=None, description="Input image base64 2")
    type_generation: Optional[str] = Field(default="normal", description="Type of generation")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt to avoid certain features")


class ImageGenerationResponse(BaseModel):
    success: bool
    message: str
    job_id: Optional[str] = None
    image1_base64: Optional[str] = None
    image2_base64: Optional[str] = None
    final_prompt: Optional[str] = None
    system_info: Optional[str] = None
    warning: Optional[str] = None