import os
import aiohttp
import aiofiles
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def upload_file_to_directus(file_path, folder_id1=None):
    """Upload a file to Directus asynchronously and return the file info"""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get configuration from environment variables
        # Get configuration from environment variables
    directus_url = os.getenv('DIRECTUS_URL')
    access_token = os.getenv('DIRECTUS_ACCESS_TOKEN')
    if folder_id1 is not None:
        print("====dùng folder id của mình====")
        folder_id = os.getenv('DIRECTUS_FOLDER_ID_STATIC_SLIDE')  # Optional
    else:
        print("No folder id provided, using default folder")
        folder_id=folder_id1    
    
    # Validate required environment variables
    if not directus_url:
        raise ValueError("DIRECTUS_URL environment variable is required")
    if not access_token:
        raise ValueError("DIRECTUS_ACCESS_TOKEN environment variable is required")
    
    upload_endpoint = f"{directus_url}/files"
    headers = {"Authorization": f"Bearer {access_token}"}
    file_name = os.path.basename(file_path)
    
    # Prepare form data
    data = aiohttp.FormData()
    if folder_id:
        data.add_field('folder', folder_id)
    
    # Read file asynchronously and add to form data
    async with aiofiles.open(file_path, 'rb') as file:
        file_content = await file.read()
        data.add_field('file', file_content, filename=file_name, content_type='application/octet-stream')
    
    # Upload file
    async with aiohttp.ClientSession() as session:
        async with session.post(upload_endpoint, headers=headers, data=data) as response:
            if response.status in [200, 201, 204]:
                response_data = await response.json()
                return response_data
            else:
                error_text = await response.text()
                raise Exception(f"Upload failed with status code {response.status}: {error_text}")