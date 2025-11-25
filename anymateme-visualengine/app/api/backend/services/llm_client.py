import os
import requests
from dotenv import load_dotenv

load_dotenv() 

class PromptEnhancerClient:
    def __init__(self):
        self.base_url = os.getenv("API_PROMPT_ENHANCER_URL")
        self.enhance_endpoint = f"{self.base_url}/api/v1/image-prompts/enhance-prompt"
        self.classify_endpoint = f"{self.base_url}/api/v1/image-prompts/classify-prompt-type"
        self.enhance_slide_endpoint = f"{self.base_url}/api/v1/image-prompts/enhance-slide-prompt"
        self.check_safety_endpoint = f"{self.base_url}/api/v1/image-prompts/check-prompt-safety"

    def enhance_prompt(self, prompt_text, style="Realistic", context="general"):
        job_id=""
        try:
            payload = {
                "user_prompt": prompt_text,
                "style": style,
                "context": context,
                "job_id": job_id
            }
            headers = {
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded"
            }

            response = requests.post(self.enhance_endpoint, data=payload, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data.get("enhanced_prompt")

        except requests.RequestException as e:
            print(f"❌ Error calling enhance API: {e}")
            return None
    def classify_prompt_type(self, prompt_text):
        job_id=""
        try:
            payload = {
                "prompt": prompt_text,
                "job_id": job_id
            }
            headers = {
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded"
            }

            response = requests.post(self.classify_endpoint, data=payload, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data.get("classification")

        except requests.RequestException as e:
            print(f"❌ Error calling classify API: {e}")
            return None
    def enhance_slide_prompt(self, prompt_text):
        job_id=""
        try:
            payload = {
                "prompt": prompt_text,
                "job_id": job_id
            }
            headers = {
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded"
            }

            response = requests.post(self.enhance_slide_endpoint, data=payload, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data.get("enhanced_prompt")

        except requests.RequestException as e:
            print(f"❌ Error calling enhance slide API: {e}")
            return None
    def check_prompt_safety(self, prompt_text):
        job_id=""
        try:
            payload = {
                "prompt": prompt_text,
                "job_id": job_id
            }
            headers = {
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded"
            }

            response = requests.post(self.check_safety_endpoint, data=payload, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data.get("raw_response")

        except requests.RequestException as e:
            print(f"❌ Error calling check safety API: {e}")
            return None