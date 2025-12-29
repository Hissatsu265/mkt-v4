# FILE: app/modules/ImageGeneration/config.py

"""
Image Generation Configuration
"""

from typing import Dict, Tuple

class ImageConfig:
    """Configuration class for image generation settings"""
    
    # High-quality aspect ratios 
    ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
        "1:1": (1024, 1024),      # Square format
        "16:9": (1344, 768),      # Widescreen 
        "9:16": (768, 1344),      # Mobile/Portrait
    }
    
    # Art style definitions
    STYLES: Dict[str, str] = {
        "Photorealistic": (
            "Ultra-realistic, 8K UHD, highly detailed, photo-realistic lighting and textures, "
            "cinematic composition, professional photography, sharp focus, natural lighting"
        ),
        "Digital Art": (
            "Digital painting, concept art, highly detailed, vibrant colors, fantasy themes, "
            "dramatic lighting, professional illustration, artstation quality"
        ),
        "Anime": (
            "Anime style, manga aesthetic, vibrant colors, clean lines, detailed shading, "
            "Japanese animation style, high quality anime art"
        ),
        "Sketch": (
            "Pencil sketch, hand-drawn, artistic linework, detailed shading, graphite drawing, "
            "traditional sketching technique"
        ),
        "Cyberpunk": (
            "Cyberpunk aesthetic, neon lights, futuristic cityscape, dark atmosphere, "
            "high-tech elements, dystopian future, sci-fi style"
        ),
    }
    
    # Negative prompts for each style
    NEGATIVE_PROMPTS: Dict[str, str] = {
        "Photorealistic": (
            "cartoon, anime, 3d render, painting, drawing, sketch, "
            "low quality, blurry, pixelated, distorted, ugly, deformed"
        ),
        "Digital Art": (
            "photography, realistic, blurry, low quality, pixelated, "
            "amateur, poorly drawn, distorted"
        ),
        "Anime": (
            "realistic, photography, 3d render, western cartoon, "
            "low quality, blurry, distorted, ugly"
        ),
        "Sketch": (
            "colored, digital art, photography, painting, "
            "clean lines, finished artwork"
        ),
        "Cyberpunk": (
            "medieval, nature, bright daylight, rural, "
            "low-tech, traditional, pastoral"
        ),
    }
    
    # Model configurations
    MODELS: Dict[str, Dict] = {
        "FLUX": {
            "model_path": "black-forest-labs/FLUX.1-dev",
            "torch_dtype": "bfloat16",
            "recommended_steps": 20,
            "recommended_guidance": 7.5,
        },
        "FLUX_SCHNELL": {
            "model_path": "black-forest-labs/FLUX.1-schnell",
            "torch_dtype": "bfloat16", 
            "recommended_steps": 8,
            "recommended_guidance": 3.5,
        }
    }
    
    # Quality presets
    QUALITY_PRESETS: Dict[str, Dict] = {
        "draft": {
            "num_inference_steps": 8,
            "guidance_scale": 3.5,
            "description": "Fast generation"
        },
        "standard": {
            "num_inference_steps": 15,
            "guidance_scale": 7.5,
            "description": "Balanced speed and quality"
        },
        "high": {
            "num_inference_steps": 25,
            "guidance_scale": 9.0,
            "description": "High quality"
        }
    }
    
    @classmethod
    def get_quality_params(cls, quality: str) -> Dict:
        """Get parameters for a quality preset"""
        return cls.QUALITY_PRESETS.get(quality, cls.QUALITY_PRESETS["standard"])
    
    @classmethod
    def get_model_config(cls, model: str) -> Dict:
        """Get configuration for a specific model"""
        return cls.MODELS.get(model, cls.MODELS["FLUX"])
    
    @classmethod
    def validate_aspect_ratio(cls, aspect_ratio: str) -> bool:
        """Validate if aspect ratio is supported"""
        return aspect_ratio in cls.ASPECT_RATIOS
    
    @classmethod
    def validate_style(cls, style: str) -> bool:
        """Validate if style is supported"""
        return style in cls.STYLES
    
    @classmethod
    def get_dimensions(cls, aspect_ratio: str) -> Tuple[int, int]:
        """Get width and height for an aspect ratio"""
        return cls.ASPECT_RATIOS.get(aspect_ratio, cls.ASPECT_RATIOS["1:1"])
    
    @classmethod
    def get_style_prompt(cls, style: str) -> str:
        """Get style enhancement prompt"""
        return cls.STYLES.get(style, cls.STYLES["Photorealistic"])
    
    @classmethod
    def get_negative_prompt(cls, style: str, custom_negative: str = None) -> str:
        """Get combined negative prompt for style"""
        base_negative = cls.NEGATIVE_PROMPTS.get(style, "")
        if custom_negative:
            return f"{base_negative}, {custom_negative}"
        return base_negative