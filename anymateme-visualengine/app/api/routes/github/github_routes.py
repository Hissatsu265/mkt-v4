from fastapi import APIRouter, Request, HTTPException, Header, Depends
import hmac
import hashlib
import json
import os
from app.core.config import Settings
from app.modules.Discord.service import DiscordService
from app.core.logger import get_logger

router = APIRouter()
settings = Settings()
logger = get_logger(__name__)

# Create a Discord service instance
discord_service = DiscordService()

async def verify_github_signature(request: Request, x_hub_signature_256: str = Header(None)):
    """Verify the GitHub webhook signature"""
    github_secret = settings.GITHUB_WEBHOOK_SECRET
    
    if not github_secret:
        logger.warning("GitHub webhook secret not configured")
        # In development, you might want to bypass this check
        if os.getenv("ENVIRONMENT", "production").lower() == "development":
            payload_body = await request.body()
            return payload_body
        else:
            raise HTTPException(status_code=500, detail="GitHub webhook secret not configured")
    
    if not x_hub_signature_256:
        raise HTTPException(status_code=401, detail="Missing signature header")
    
    payload_body = await request.body()
    signature = hmac.new(
        github_secret.encode('utf-8'),
        payload_body,
        hashlib.sha256
    ).hexdigest()
    
    expected_signature = f"sha256={signature}"
    if not hmac.compare_digest(expected_signature, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    return payload_body

@router.post("/webhook", summary="GitHub webhook endpoint")
async def github_webhook(payload_body: bytes = Depends(verify_github_signature),
                         x_github_event: str = Header(None)):
    """Handle GitHub webhook events and send notifications to Discord"""
    try:
        payload = json.loads(payload_body)
        
        # Log the event type
        logger.info(f"Received GitHub webhook event: {x_github_event} - {payload.get('repository', {}).get('full_name', 'unknown')}")
        
        # Handle ping event (sent when webhook is created)
        if x_github_event == "ping":
            logger.info("Received ping event from GitHub - webhook connection successful")
            return {"status": "success", "message": "Ping received successfully"}
        
        # Check if it's a pull request event
        if x_github_event == "pull_request" and "pull_request" in payload and "action" in payload:
            logger.info(f"Processing pull request {payload['action']}")
            # Send the formatted message
            await discord_service.send_github_pr_notification(payload)
        
        # Check if it's an issue event
        elif x_github_event == "issues" and "issue" in payload and "action" in payload:
            # Process only certain actions
            action = payload["action"]
            if action in ["opened", "closed", "reopened", "assigned", "labeled"]:
                logger.info(f"Processing issue {action}")
                await discord_service.send_github_issue_notification(payload)
            else:
                logger.info(f"Ignoring issue action: {action}")
        
        # Add additional event types here in the future
        
        else:
            logger.info(f"Ignoring event: {x_github_event} - {payload.get('action', 'unknown')}")
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing GitHub webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")