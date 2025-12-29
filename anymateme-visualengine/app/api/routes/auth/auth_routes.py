# app/api/routes/auth/auth_routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import Optional
from app.modules.Auth.service import auth_service

router = APIRouter()

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires: int
    token_type: str = "bearer"

class RefreshRequest(BaseModel):
    refresh_token: str

@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Login with email and password"""
    tokens = await auth_service.login(request.email, request.password)
    
    return {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "expires": tokens["expires"],
        "token_type": "bearer"
    }

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest):
    """Refresh the access token"""
    tokens = await auth_service.refresh_token(request.refresh_token)
    
    return {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "expires": tokens["expires"],
        "token_type": "bearer"
    }

@router.post("/logout")
async def logout(request: RefreshRequest):
    """Logout and invalidate the refresh token"""
    success = await auth_service.logout(request.refresh_token)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to logout"
        )
    
    return {"message": "Successfully logged out"}