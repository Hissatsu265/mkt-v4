# app/modules/Discord/service.py

import aiohttp
import os
import datetime
from app.core.logger import get_logger

logger = get_logger(__name__)

class DiscordService:
    def __init__(self, webhook_url=None, bot_token=None, channel_id=None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self.bot_token = bot_token or os.getenv("DISCORD_BOT_TOKEN")
        self.channel_id = channel_id or os.getenv("DISCORD_CHANNEL_ID")
        
        if not (self.webhook_url or (self.bot_token and self.channel_id)):
            logger.warning("Discord service initialized without webhook URL or bot credentials")
    
    async def send_github_pr_notification(self, payload):
        """Send a formatted GitHub PR notification to Discord"""
        # Extract PR details
        pr = payload["pull_request"]
        action = payload["action"]
        repo_name = payload["repository"]["full_name"]
        
        # Get the user who performed this action (sender), not just the PR author
        action_performer = payload["sender"]["login"]
        action_performer_avatar = payload["sender"]["avatar_url"]
        
        # Set color based on action
        colors = {
            "opened": 5025616,     # Green
            "reopened": 16776960,  # Yellow
            "closed": 15548997     # Red/Pink
        }
        
        # If PR was merged, use a purple color
        is_merged = pr.get("merged", False)
        if is_merged:
            color = 10181046  # Purple
            action_text = "Merged"
        else:
            color = colors.get(action, 3447003)  # Default to Discord blue
            action_text = action.capitalize()
            
        # Get PR details
        pr_number = pr["number"]
        pr_title = pr["title"]
        pr_url = pr["html_url"]
        
        # Format the timestamp
        timestamp = datetime.datetime.utcnow().isoformat()
        
        # Create embed
        embed = {
            "title": f"Pull Request #{pr_number} {action_text}",
            "url": pr_url,
            "color": color,
            "author": {
                "name": action_performer,
                "icon_url": action_performer_avatar
            },
            "fields": [
                {
                    "name": "Repository",
                    "value": repo_name,
                    "inline": True
                },
                {
                    "name": "Title",
                    "value": pr_title,
                    "inline": True
                }
            ],
            "thumbnail": {
                "url": "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
            },
            "footer": {
                "text": "GitHub",
                "icon_url": "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
            },
            "timestamp": timestamp
        }
        
        # Add description if available
        if pr.get("body") and pr["body"].strip():
            description = pr["body"].strip()
            # Truncate if too long
            if len(description) > 300:
                description = description[:297] + "..."
            
            embed["description"] = description
        
        # Add additional fields based on PR state
        if action == "closed":
            if is_merged:
                embed["fields"].append({
                    "name": "Status",
                    "value": "✅ Merged",
                    "inline": True
                })
            else:
                embed["fields"].append({
                    "name": "Status",
                    "value": "❌ Closed without merging",
                    "inline": True
                })
        
        # Add branch information
        embed["fields"].append({
            "name": "Branches",
            "value": f"From: `{pr['head']['ref']}` → To: `{pr['base']['ref']}`",
            "inline": False
        })
        
        # Add link to view changes
        embed["fields"].append({
            "name": "Actions",
            "value": f"[View Changes]({pr_url}/files)",
            "inline": False
        })
        
        # Create message payload with embed
        message = {"embeds": [embed]}
        
        # Send to Discord
        if self.webhook_url:
            return await self._send_via_webhook(message)
        elif self.bot_token and self.channel_id:
            return await self._send_via_bot(message)
        else:
            logger.error("Cannot send Discord message: No webhook URL or bot credentials")
            return False
    
    async def send_github_issue_notification(self, payload):
        """Send a formatted GitHub Issue notification to Discord"""
        # Extract issue details
        issue = payload["issue"]
        action = payload["action"]
        repo_name = payload["repository"]["full_name"]
        
        # Get the user who performed this action (sender), not just the issue author
        action_performer = payload["sender"]["login"]
        action_performer_avatar = payload["sender"]["avatar_url"]
        
        # Set color based on action
        colors = {
            "opened": 5025616,      # Green
            "reopened": 16776960,   # Yellow
            "closed": 15548997,     # Red/Pink
            "assigned": 3447003,    # Blue
            "labeled": 10181046     # Purple
        }
        
        color = colors.get(action, 3447003)  # Default to blue
        action_text = action.capitalize()
        
        # Get issue details
        issue_number = issue["number"]
        issue_title = issue["title"]
        issue_url = issue["html_url"]
        
        # Format the timestamp
        timestamp = datetime.datetime.utcnow().isoformat()
        
        # Create embed
        embed = {
            "title": f"Issue #{issue_number} {action_text}",
            "url": issue_url,
            "color": color,
            "author": {
                "name": action_performer,
                "icon_url": action_performer_avatar
            },
            "fields": [
                {
                    "name": "Repository",
                    "value": repo_name,
                    "inline": True
                },
                {
                    "name": "Title",
                    "value": issue_title,
                    "inline": True
                }
            ],
            "thumbnail": {
                "url": "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
            },
            "footer": {
                "text": "GitHub Issues",
                "icon_url": "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
            },
            "timestamp": timestamp
        }
        
        # Add description if available (only for opened issues to keep notification clean)
        if action == "opened" and issue.get("body") and issue["body"].strip():
            description = issue["body"].strip()
            # Truncate if too long
            if len(description) > 300:
                description = description[:297] + "..."
            
            embed["description"] = description
        
        # Add status for closed issues
        if action == "closed":
            embed["fields"].append({
                "name": "Status",
                "value": "✅ Resolved",
                "inline": True
            })
        
        # Add assignee information if applicable
        if action == "assigned" and "assignee" in payload:
            assignee = payload["assignee"]["login"]
            embed["fields"].append({
                "name": "Assigned to",
                "value": assignee,
                "inline": True
            })
        
        # Add label information if applicable
        if action == "labeled" and "label" in payload:
            label_name = payload["label"]["name"]
            embed["fields"].append({
                "name": "Label Added",
                "value": label_name,
                "inline": True
            })
        
        # Create message payload with embed
        message = {"embeds": [embed]}
        
        # Send to Discord
        if self.webhook_url:
            return await self._send_via_webhook(message)
        elif self.bot_token and self.channel_id:
            return await self._send_via_bot(message)
        else:
            logger.error("Cannot send Discord message: No webhook URL or bot credentials")
            return False
    
    async def _send_via_webhook(self, message):
        """Send message via Discord webhook"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=message) as response:
                    if response.status >= 400:
                        response_text = await response.text()
                        logger.error(f"Discord webhook error: {response.status} - {response_text}")
                        return False
                    return True
        except Exception as e:
            logger.error(f"Failed to send Discord webhook message: {str(e)}")
            return False
    
    async def _send_via_bot(self, message):
        """Send message via Discord bot API"""
        try:
            url = f"https://discord.com/api/v10/channels/{self.channel_id}/messages"
            headers = {
                "Authorization": f"Bot {self.bot_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=message) as response:
                    if response.status >= 400:
                        response_text = await response.text()
                        logger.error(f"Discord bot API error: {response.status} - {response_text}")
                        return False
                    return True
        except Exception as e:
            logger.error(f"Failed to send Discord bot message: {str(e)}")
            return False
    
    # Simple message sender (keep for other notifications)
    async def send_message(self, content):
        """Send a simple text message to Discord"""
        message = {"content": content}
        if self.webhook_url:
            return await self._send_via_webhook(message)
        elif self.bot_token and self.channel_id:
            return await self._send_via_bot(message)
        else:
            logger.error("Cannot send Discord message: No webhook URL or bot credentials")
            return False