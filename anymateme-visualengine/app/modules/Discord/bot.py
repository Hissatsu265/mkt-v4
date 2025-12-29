import discord
from discord import app_commands
from discord.ext import commands
import os
import traceback
import requests
import json
import asyncio
from app.core.logger import get_logger

logger = get_logger(__name__)

class AnymateMeBot:
    def __init__(self, token, ollama_url=None):
        self.token = token
        self.ollama_url = ollama_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        
        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True
        
        # Create bot with prefix
        self.bot = commands.Bot(command_prefix='/', intents=intents)
        
        # Set up commands
        self.setup_commands()
    
    def setup_commands(self):
        """Set up bot commands"""
        
        @self.bot.event
        async def on_ready():
            """Called when bot is connected and ready"""
            logger.info(f"Discord bot logged in as {self.bot.user}")
            try:
                # Sync slash commands
                synced = await self.bot.tree.sync()
                logger.info(f"Synced {len(synced)} slash command(s)")
            except Exception as e:
                logger.error(f"Failed to sync commands: {str(e)}")
                traceback.print_exc()
        
        # Add error handler
        @self.bot.event
        async def on_command_error(ctx, error):
            if isinstance(error, commands.CommandNotFound):
                return  # Ignore command not found errors
            logger.error(f"Command error: {str(error)}")
            traceback.print_exc()
        
        # Method 1: Traditional prefix command
        @self.bot.command(name="ask")
        async def ask_prefix(ctx, *, question=None):
            """Ask a question about Anymate Me using prefix command"""
            if not question:
                await ctx.send("Please provide a question after the `/ask` command.")
                return
            
            async with ctx.typing():
                try:
                    # Get AI response
                    response = await self.get_ai_response(question)
                    await ctx.send(response)
                except Exception as e:
                    logger.error(f"Error processing question: {str(e)}")
                    traceback.print_exc()
                    await ctx.send("Sorry, something went wrong while processing your question.")
        
        # Method 2: Modern slash command
        @self.bot.tree.command(name="ask", description="Ask a question about Anymate Me")
        async def ask_slash(interaction: discord.Interaction, question: str):
            """Ask a question about Anymate Me using slash command"""
            await interaction.response.defer(thinking=True)
            try:
                # Get AI response
                response = await self.get_ai_response(question)
                await interaction.followup.send(response)
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                traceback.print_exc()
                await interaction.followup.send("Sorry, something went wrong while processing your question.")
    
    async def get_ai_response(self, question):
        """Get response from Ollama for a user question"""
        try:
            # Enhanced company context with correct contact info
            company_context = """
    Anymate Me is our innovative AI video production platform that creates videos in 60+ languages.
    We're based in Cologne, Germany (founded in 2023) and help companies save 96% on production costs while speeding up video creation by 15x.
    Our photorealistic avatars turn any text into high-quality videos for marketing, sales, and training.
    We specialize in AI-powered lip-syncing across multiple languages.
    Our platform is designed to be user-friendly - clients just upload a script and get professional videos.
    Our website is anymateme.com and we can be reached at info@anymateme.com.
    """
            
            # Team-focused prompt that makes the bot feel like a colleague
            prompt = f"""You are a helpful team member at Anymate Me, our AI video production company. You're here to assist your colleagues and answer questions about our work.

    OUR COMPANY:
    {company_context}

    COLLEAGUE'S QUESTION:
    {question}

    HOW TO RESPOND:
    - Talk like you're part of our team - use "we", "our", "us"
    - Be friendly and casual, like talking to a colleague
    - Keep answers short and to the point (1-3 sentences usually)
    - If you're not sure about something specific, suggest they check with the team or email info@anymateme.com
    - Help them understand our services, benefits, and what makes us special
    - Be enthusiastic about our work and what we do
    - If someone welcomes you to Anymate Me, thank them warmly and express excitement about being part of the team

    Your helpful response as a team member:
    """
            
            # Call Ollama API
            ollama_api_url = f"{self.ollama_url}/api/generate"
            
            payload = {
                "model": os.getenv("OLLAMA_CHAT_MODEL", "phi4-extended:latest"),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8  # Increased for more natural, conversational tone
                }
            }
            
            # Use asyncio to run the request without blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(ollama_api_url, json=payload)
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "response" in result:
                return result["response"]
            else:
                return "Hey! I couldn't generate a response right now. Can you try asking again? 😊"
                
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            traceback.print_exc()
            return "Oops! I ran into a technical issue. Let me know if you need help with anything else! 🤖"
    
    def run(self):
        """Run the Discord bot"""
        try:
            self.bot.run(self.token)
        except Exception as e:
            logger.error(f"Error running Discord bot: {str(e)}")
            traceback.print_exc()