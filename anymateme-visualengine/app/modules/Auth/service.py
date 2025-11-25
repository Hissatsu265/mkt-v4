# app/modules/Auth/service.py
import httpx
import jwt
from typing import Dict, Any, Optional
from fastapi import HTTPException, status
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("app.modules.Auth.service")

class AuthService:
    """Service class for authentication with Directus CMS"""
    
    def __init__(self):
        """Initialize the auth service with configuration"""
        self.auth_base_url = settings.DIRECTUS_AUTH_URL
    
    async def login(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user with Directus
        
        Args:
            email: User email
            password: User password
            
        Returns:
            Dict containing access_token, refresh_token and expires
            
        Raises:
            HTTPException: If authentication fails
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.auth_base_url}/auth/login",
                    json={
                        "email": email,
                        "password": password,
                        "mode": "json"
                    }
                )
                
                if response.status_code != 200:
                    logger.warning(f"Authentication failed: {response.text}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid credentials"
                    )
                
                data = response.json()
                return {
                    "access_token": data.get("data", {}).get("access_token"),
                    "refresh_token": data.get("data", {}).get("refresh_token"),
                    "expires": data.get("data", {}).get("expires")
                }
                
        except httpx.RequestError as e:
            logger.error(f"Error connecting to auth service: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service unavailable"
            )
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify an access token
        
        Args:
            token: JWT access token
            
        Returns:
            Dict with decoded token payload
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            # This method decodes the JWT without verification
            # In production, you might want to verify with Directus or use a public key
            payload = jwt.decode(
                token, 
                options={"verify_signature": False}
            )
            
            # Check required fields
            if "id" not in payload:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token format"
                )
                
            return payload
            
        except jwt.PyJWTError as e:
            logger.warning(f"Token validation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format"
            )
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh an access token
        
        Args:
            refresh_token: Refresh token from previous authentication
            
        Returns:
            Dict with new access_token, refresh_token and expires
            
        Raises:
            HTTPException: If refresh fails
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.auth_base_url}/auth/refresh",
                    json={
                        "refresh_token": refresh_token,
                        "mode": "json"
                    }
                )
                
                if response.status_code != 200:
                    logger.warning(f"Token refresh failed: {response.text}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or expired refresh token"
                    )
                
                data = response.json()
                return {
                    "access_token": data.get("data", {}).get("access_token"),
                    "refresh_token": data.get("data", {}).get("refresh_token"),
                    "expires": data.get("data", {}).get("expires")
                }
                
        except httpx.RequestError as e:
            logger.error(f"Error connecting to auth service: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service unavailable"
            )
    
    async def logout(self, refresh_token: str) -> bool:
        """
        Logout the user by invalidating the refresh token
        
        Args:
            refresh_token: The refresh token to invalidate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.auth_base_url}/auth/logout",
                    json={
                        "refresh_token": refresh_token,
                        "mode": "json"
                    }
                )
                
                return response.status_code == 200
                
        except httpx.RequestError as e:
            logger.error(f"Error connecting to auth service: {str(e)}")
            return False

    async def get_user_info(self, token: str) -> Dict[str, Any]:
        """
        Get complete user information from Directus
        
        Args:
            token: JWT access token
            
        Returns:
            Dict with complete user information
        """
        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {token}"}
                response = await client.get(
                    f"{self.auth_base_url}/users/me",
                    headers=headers
                )
                
                if response.status_code != 200:
                    logger.warning(f"Failed to get user info: {response.text}")
                    return {}
                
                data = response.json()
                
                # Extract user information from the response
                user_data = data.get("user", {}).get("data", {})
                
                return {
                    "user_id": user_data.get("id"),
                    "email": user_data.get("email"),
                    "first_name": user_data.get("first_name"),
                    "last_name": user_data.get("last_name"),
                    "role": user_data.get("role"),
                    # Include any other fields you need
                }
                
        except httpx.RequestError as e:
            logger.error(f"Error connecting to auth service: {str(e)}")
            return {}
# Create a singleton instance
auth_service = AuthService()