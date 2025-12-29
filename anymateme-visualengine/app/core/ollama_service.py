"""
Ollama Service Manager - Centralized management for multiple Ollama instances

This module provides functions to route requests to the appropriate Ollama
instance based on the service type and environment.
"""
from app.core.config import settings
from fastapi import HTTPException
import httpx
from typing import Dict, Any, Optional
import logging
from enum import Enum

# Setup logger
logger = logging.getLogger("app.core.ollama_service")

class ServiceType(str, Enum):
    PROMPT_ENHANCEMENT = "prompt_enhancement"
    PLAIN_LANGUAGE = "plain_language"


# Service routing configuration - maps services to Ollama instances
SERVICE_ROUTING = {
    "OLLAMA_1": [
        ServiceType.PROMPT_ENHANCEMENT,

        
    ],
    "OLLAMA_2": [
        ServiceType.PLAIN_LANGUAGE,
    ],
}

async def route_ollama_request(
    service_type: str, 
    payload: Dict[Any, Any], 
    endpoint: str = "generate",
    timeout: float = 120.0
) -> Dict:
    """
    Route a request to the appropriate Ollama instance based on service type
    
    Args:
        service_type: The type of service requesting Ollama (e.g., "plain_language")
        payload: The request payload to send to Ollama
        endpoint: The Ollama API endpoint (default: "generate")
        timeout: Request timeout in seconds
        
    Returns:
        The response from the Ollama API
        
    Raises:
        HTTPException: If service_type is unknown or if there's an error with the Ollama service
    """
    # Determine which Ollama instance to use
    ollama_url = get_ollama_url_for_service(service_type)
    
    if not ollama_url:
        raise HTTPException(status_code=400, detail=f"Unknown service type: {service_type}")
    
    # Set the model based on service type if not specified in payload
    if "model" not in payload and service_type in settings.MODEL_MAPPINGS:
        payload["model"] = settings.MODEL_MAPPINGS[service_type]
    
    logger.info(f"Routing {service_type} request to {ollama_url} with model {payload.get('model', 'default')}")
    
    try:
        # Send the request to the appropriate Ollama instance
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ollama_url}/api/{endpoint}",
                json=payload,
                timeout=timeout
            )
            
            # Check for success
            if response.status_code != 200:
                logger.error(f"Ollama service error: {response.text}")
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Ollama service error: {response.text}"
                )
                
            return response.json()
            
    except httpx.RequestError as e:
        logger.error(f"Request error to Ollama service: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error processing Ollama request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def get_ollama_url_for_service(service_type: str) -> Optional[str]:
    """
    Get the appropriate Ollama URL for a given service type
    
    Args:
        service_type: The service type (e.g., "plain_language")
        
    Returns:
        The URL for the appropriate Ollama instance, or None if not found
    """
    # Find which Ollama instance this service should use
    for instance, services in SERVICE_ROUTING.items():
        if service_type in services:
            # LOCAL environment uses a different Ollama configuration
            if settings.ENVIRONMENT == "LOCAL":
                return settings.OLLAMA_API_ENDPOINT
            
            # For DEV/STAGING/PROD environments
            if instance == "OLLAMA_1":
                return settings.OLLAMA_SERVICE_1_URL
            elif instance == "OLLAMA_2":
                return settings.OLLAMA_SERVICE_2_URL
    
    # Service type not found in routing configuration
    logger.warning(f"Unknown service type requested: {service_type}")
    return None