# app/dependencies/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any
from app.modules.Auth.service import auth_service
from app.core.logger import get_logger

# Initialize logger
logger = get_logger("app.dependencies.auth")

# Security scheme for Bearer token
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Verify the authentication token and return the payload
    
    Args:
        credentials: The HTTP Authorization header credentials
        
    Returns:
        dict: The decoded token payload
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        token = credentials.credentials
        payload = await auth_service.verify_token(token)
        return payload
    except Exception as e:
        logger.warning(f"Token verification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

async def get_current_user(payload: Dict[str, Any] = Depends(verify_token)) -> Dict[str, Any]:
    """
    Get current user from token payload
    """
    # Log the full payload to debug
    logger.debug(f"Token payload: {payload}")
    
    # Try to handle multiple possible structures
    user_id = payload.get("id")
    
    # Email might be at the top level or nested
    email = payload.get("email")
    
    # If token has nested user structure, extract from there
    user_data = payload.get("user", {})
    if isinstance(user_data, dict):
        user_data_nested = user_data.get("data", {})
        if isinstance(user_data_nested, dict):
            # If top-level values weren't found, try to get from nested structure
            if not user_id:
                user_id = user_data_nested.get("id")
            if not email:
                email = user_data_nested.get("email")
    
    # Create user object with found values
    user_info = {
        "user_id": user_id,
        "email": email,
        "first_name": user_data.get("data", {}).get("first_name"),
        "last_name": user_data.get("data", {}).get("last_name"),
        "role": payload.get("role") or user_data.get("data", {}).get("role")
    }
    
    return user_info

# Optional: Create role-based access control
def require_role(required_role: str):
    """
    Create a dependency for role-based access control
    
    Args:
        required_role: The role required to access the endpoint
        
    Returns:
        function: A dependency function for role checking
    """
    async def role_checker(user = Depends(get_current_user)):
        user_role = user.get("role")
        
        # Role checking logic can be customized based on your Directus roles
        if user_role != required_role:
            logger.warning(f"Access denied: User {user.get('user_id')} with role {user_role} tried to access {required_role} endpoint")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: {required_role} role required"
            )
        return user
    
    return role_checker