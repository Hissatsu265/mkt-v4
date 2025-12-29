# FILE: app/modules/ImageGeneration/service.py

# FORCE GPU 0 at OS level (like your reference code)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import asyncio
import time
from typing import Dict, Optional, Any
from diffusers import FluxPipeline
from PIL import Image
import logging

# Import your existing modules
from app.modules.ImageGeneration.config import ImageConfig

logger = logging.getLogger(__name__)

class ImageGenerationService:
    def __init__(self, checkpoint_folder: str = "./checkpoints/FLUX.1-dev"):
        # Simple device assignment (GPU 0 appears as cuda:0 due to CUDA_VISIBLE_DEVICES)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use a single model holder like working code (not a dict)
        self.pipe = None
        self.model_loaded = False
        self.config = ImageConfig()
        self.checkpoint_folder = checkpoint_folder
        self._model_loading_lock = asyncio.Lock()
        
        logger.info(f"✅ Image service initialized on device: {self.device}")
        logger.info(f"📁 Checkpoint folder: {self.checkpoint_folder}")
        logger.info(f"🔧 CUDA_VISIBLE_DEVICES set to GPU 0")
        
        # Load model at startup like working code
        logger.info("🚀 Loading model at startup...")
        self._load_model_sync()

    def _load_model_sync(self):
        """Load model synchronously at startup like working code"""
        try:
            if self.pipe is not None:
                logger.info("✅ Model already loaded")
                return True
                
            # Clear any existing cache first
            torch.cuda.empty_cache()
            logger.info("Loading FLUX model... (this may take a moment)")
            
            # Determine model path
            if os.path.exists(self.checkpoint_folder):
                model_path = self.checkpoint_folder
                logger.info(f"📂 Loading from checkpoint: {model_path}")
            else:
                model_path = "black-forest-labs/FLUX.1-dev"
                logger.info(f"🌐 Loading from HuggingFace: {model_path}")
            
            # Load model exactly like working code
            self.pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
            
            # Use CPU offloading exactly like working code
            self.pipe.enable_model_cpu_offload(gpu_id=0)
            
            self.model_loaded = True
            logger.info("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            self.pipe = None
            self.model_loaded = False
            torch.cuda.empty_cache()
            return False

    def check_gpu_memory(self, device_id: int = 0) -> Dict[str, float]:
        """Check GPU memory availability (device_id=0 because of CUDA_VISIBLE_DEVICES)"""
        if self.device == "cuda":
            try:
                total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
                allocated_memory = torch.cuda.memory_allocated(device_id) / (1024**3)
                free_memory = total_memory - allocated_memory
                
                return {
                    "total_gb": total_memory,
                    "allocated_gb": allocated_memory, 
                    "free_gb": free_memory,
                    "can_load_flux": free_memory >= 12,
                    "device_id": device_id
                }
            except Exception as e:
                logger.error(f"Error checking GPU memory: {e}")
                
        return {"total_gb": 0, "allocated_gb": 0, "free_gb": 0, "can_load_flux": False, "device_id": -1}
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("🧹 GPU cache cleared")
    
    def get_gpu_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory usage statistics"""
        if self.device == "cuda":
            try:
                device_id = 0  # Always 0 due to CUDA_VISIBLE_DEVICES
                return {
                    "used_mb": torch.cuda.memory_allocated(device=device_id) / 1024**2,
                    "peak_mb": torch.cuda.max_memory_allocated(device=device_id) / 1024**2,
                    "free_mb": (torch.cuda.get_device_properties(device_id).total_memory - 
                               torch.cuda.memory_allocated(device_id)) / 1024**2,
                    "total_mb": torch.cuda.get_device_properties(device_id).total_memory / 1024**2,
                    "device_id": device_id
                }
            except Exception as e:
                logger.error(f"Error getting GPU stats: {e}")
                
        return {"used_mb": 0, "peak_mb": 0, "free_mb": 0, "total_mb": 0, "device_id": -1}

    async def generate_image(
        self,
        prompt: str,
        style: str = "Photorealistic",
        aspect_ratio: str = "1:1",
        quality: str = "standard",
        model: str = "FLUX",
        enhance_prompt: bool = True,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate an image using the specified parameters"""
        try:
            # Check if model is loaded
            if self.pipe is None or not self.model_loaded:
                logger.warning("Model not loaded, attempting to load...")
                if not self._load_model_sync():
                    raise Exception(f"Failed to load model {model}")
            
            # Use the single pipeline like working code
            pipeline = self.pipe
            
            # Apply style if specified and valid
            final_prompt = prompt
            final_negative_prompt = negative_prompt or ""
            
            # Safe style application with fallbacks
            if style and hasattr(self.config, 'validate_style') and self.config.validate_style(style):
                if hasattr(self.config, 'get_style_prompt'):
                    style_prompt = self.config.get_style_prompt(style)
                    final_prompt = f"{prompt}, {style_prompt}"
                if hasattr(self.config, 'get_negative_prompt'):
                    final_negative_prompt = self.config.get_negative_prompt(style, negative_prompt)
                logger.info(f"🎨 Applied style '{style}' to prompt")
            
            # Get dimensions with fallback
            if hasattr(self.config, 'get_dimensions'):
                width, height = self.config.get_dimensions(aspect_ratio)
            else:
                # Fallback dimensions
                dimensions = {"1:1": (1024, 1024), "16:9": (1920, 1080), "9:16": (1080, 1920)}
                width, height = dimensions.get(aspect_ratio, (1024, 1024))
            
            # Get quality parameters with fallback
            if hasattr(self.config, 'get_quality_params'):
                quality_params = self.config.get_quality_params(quality)
            else:
                # Fallback quality params
                quality_presets = {
                    "draft": {"num_inference_steps": 15, "guidance_scale": 3.5},
                    "standard": {"num_inference_steps": 28, "guidance_scale": 3.5},
                    "high": {"num_inference_steps": 40, "guidance_scale": 7.0}
                }
                quality_params = quality_presets.get(quality, quality_presets["standard"])
            
            # Set seed if provided (use 0 like working code for consistency)
            if seed is None:
                seed = 0
            
            # Use CPU generator like working code for stability
            generator = torch.Generator("cpu").manual_seed(seed)
            
            # Generate image
            logger.info(f"🖼️  Generating {width}x{height} image with {quality} quality")
            logger.info(f"📝 Prompt: {final_prompt[:100]}...")
            
            # Use Python time for timing
            start_time = time.time()
            
            # Generate exactly like working code
            result = pipeline(
                final_prompt,
                height=height,
                width=width,
                guidance_scale=quality_params.get("guidance_scale", 3.5),
                num_inference_steps=quality_params.get("num_inference_steps", 28),
                max_sequence_length=512,
                generator=generator
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            image = result.images[0]
            logger.info(f"✅ Image generated successfully in {generation_time:.2f}s")
            
            return {
                "image": image,
                "enhanced_prompt": final_prompt,
                "original_prompt": prompt,
                "style": style,
                "generation_time": generation_time,
                "gpu_memory_usage": self.get_gpu_memory_stats(),
                "model_used": model,
                "dimensions": {"width": width, "height": height},
                "parameters": {
                    "quality": quality,
                    "guidance_scale": quality_params.get("guidance_scale", 3.5),
                    "num_inference_steps": quality_params.get("num_inference_steps", 28),
                    "seed": seed,
                    "aspect_ratio": aspect_ratio
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in image generation: {str(e)}")
            # Don't clear cache aggressively - keep model loaded
            raise Exception(f"Image generation failed: {str(e)}")
    
    # Configuration access methods
    def get_available_styles(self) -> dict:
        """Get dict of available image styles"""
        try:
            return self.config.STYLES
        except Exception as e:
            logger.error(f"Error getting available styles: {e}")
            return {
                "Photorealistic": "Ultra-realistic, lifelike images with natural lighting and textures",
                "Digital Art": "Modern digital artwork with vibrant colors and artistic flair",
                "Anime": "Japanese animation style with expressive characters",
                "Sketch": "Hand-drawn pencil or ink sketch style",
                "Cyberpunk": "Futuristic neon-lit cityscape with high-tech elements"
            }
    
    def get_available_models(self) -> dict:
        """Get dict of available models"""
        try:
            return self.config.MODELS
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return {"FLUX": {"model_path": "black-forest-labs/FLUX.1-dev", "torch_dtype": "bfloat16"}}
    
    def get_available_aspect_ratios(self) -> dict:
        """Get dict of available aspect ratios"""
        try:
            return self.config.ASPECT_RATIOS
        except Exception as e:
            logger.error(f"Error getting available aspect ratios: {e}")
            return {"1:1": (1024, 1024), "16:9": (1920, 1080), "9:16": (1080, 1920)}
    
    def get_quality_presets(self) -> Dict:
        """Get quality presets dictionary"""
        try:
            return self.config.QUALITY_PRESETS
        except Exception as e:
            logger.error(f"Error getting quality presets: {e}")
            return {
                "draft": {"num_inference_steps": 15, "guidance_scale": 3.5, "description": "Fast generation"},
                "standard": {"num_inference_steps": 28, "guidance_scale": 3.5, "description": "Balanced speed and quality"},
                "high": {"num_inference_steps": 40, "guidance_scale": 7.0, "description": "High quality"}
            }
    
    async def enhance_prompt_with_phi4(self, prompt: str, style: str = "Photorealistic") -> str:
        """Enhance prompt with AI (placeholder - implement with your preferred LLM)"""
        try:
            # Placeholder implementation - replace with your actual prompt enhancement logic
            style_hints = {
                "Photorealistic": "extremely detailed, professional photography, DSLR quality",
                "Digital Art": "digital masterpiece, trending on artstation, highly detailed",
                "Anime": "anime style, studio quality, vibrant colors, detailed character design",
                "Sketch": "detailed pencil sketch, artistic shading, fine line work",
                "Cyberpunk": "cyberpunk aesthetic, neon lights, futuristic cityscape"
            }
            
            hint = style_hints.get(style, "high quality, detailed")
            enhanced = f"{prompt}, {hint}"
            
            logger.info(f"Enhanced prompt: {prompt} -> {enhanced}")
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            return prompt  # Return original if enhancement fails
    
    def get_loaded_models(self) -> list:
        """Get list of currently loaded models"""
        return ["FLUX"] if self.model_loaded else []
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is loaded"""
        return self.model_loaded and model_name == "FLUX"
    
    async def load_model(self, model_name: str) -> bool:
        """Load model (legacy method for compatibility)"""
        if model_name == "FLUX":
            return self._load_model_sync()
        return False
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a specific model to free memory"""
        try:
            if self.model_loaded and model_name == "FLUX":
                self.pipe = None
                self.model_loaded = False
                self.clear_gpu_cache()
                logger.info(f"🗑️  Unloaded model {model_name}")
                return True
            else:
                logger.warning(f"Model {model_name} was not loaded")
                return False
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {str(e)}")
            return False
    
    def get_startup_status(self) -> dict:
        """Get startup status and readiness information"""
        try:
            gpu_stats = self.get_gpu_memory_stats()
            loaded_models = self.get_loaded_models()
            
            return {
                "ready_for_generation": self.model_loaded,
                "loaded_models": loaded_models,
                "available_models": list(self.get_available_models().keys()),
                "gpu_available": self.device == "cuda",
                "gpu_memory": gpu_stats,
                "checkpoint_folder": self.checkpoint_folder,
                "device": self.device
            }
        except Exception as e:
            logger.error(f"Error getting startup status: {e}")
            return {
                "ready_for_generation": False,
                "error": str(e),
                "device": self.device
            }
    
    async def unload_all_models(self):
        """Unload all models to free memory"""
        try:
            self.pipe = None
            self.model_loaded = False
            self.clear_gpu_cache()
            logger.info("🗑️  Unloaded all models")
        except Exception as e:
            logger.error(f"Error unloading all models: {str(e)}")


# Create the singleton instance (following reference pattern)
image_service = ImageGenerationService()