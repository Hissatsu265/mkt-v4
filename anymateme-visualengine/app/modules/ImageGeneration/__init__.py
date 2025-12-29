# FILE: app/modules/ImageGeneration/__init__.py

"""
ImageGeneration Module

This module provides image generation capabilities.
"""

from .service import ImageGenerationService, image_service

__all__ = ["ImageGenerationService", "image_service"]