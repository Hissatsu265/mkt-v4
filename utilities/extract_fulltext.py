import requests
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class AudioTextExtractor:
    def __init__(self, auth_token: str):
        self.auth_token = auth_token
        self.subtitle_url = "https://dev.shohanursobuj.online/api/v1/marketing-video/generate-subtitles"

    def generate_subtitles(self, audio_path: str, language: str = "") -> Dict[str, Any]:
        params = {
            "format": "json",
            "language": language
        }

        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "accept": "application/json"
        }

        filename = audio_path.split("/")[-1]

        with open(audio_path, "rb") as f:
            files = {
                "audio_file": (filename, f, "audio/wav")
            }

            response = requests.post(
                self.subtitle_url,
                headers=headers,
                params=params,
                files=files
            )

        if response.status_code != 200:
            raise Exception(f"Lỗi tạo subtitles: {response.status_code} - {response.text}")

        return response.json()

    def extract_full_text(self, audio_path: str, language: str = "") -> str:
        # Gọi API để lấy segments
        subtitle_data = self.generate_subtitles(audio_path, language)
        segments = subtitle_data.get('segments', [])

        # Ghép toàn bộ text
        full_text = " ".join([seg['text'] for seg in segments])

        return full_text


def process_audio_to_fulltext(audio_path: str):
    # AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjNhODQ1MmMxLWVjNTYtNDhjMi05NGM0LTgyZTY0ZWRjOWE3MyIsInJvbGUiOiI2NzA5YmQ3Yi05ZjU2LTQ2NjEtYTJmYy04NzRmMDgzMTgxMWIiLCJhcHBfYWNjZXNzIjp0cnVlLCJhZG1pbl9hY2Nlc3MiOnRydWUsImlhdCI6MTc2MTkxNjQwNSwiZXhwIjoxNzYyMDg5MjA1LCJpc3MiOiJkaXJlY3R1cyJ9.mCj5ttsOTiEyXuiRCLgo1XEPD6Oc74mTZZe3q_MDW9Y"  
    extractor = AudioTextExtractor(os.getenv("AUTH_TOKEN"))
    # extractor = AudioTextExtractor(AUTH_TOKEN)

    try:
        full_text = extractor.extract_full_text(audio_path)
        return full_text
    except Exception as e:
        print(f"Error: {e}")
        return ""