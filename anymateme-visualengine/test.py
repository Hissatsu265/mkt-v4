# # import torch
# # from diffusers import FluxPipeline

# # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
# # # pipe.enable_model_cpu_offload()

# # prompt = "a tiny astronaut hatching from an egg on the moon"
# # out = pipe(
# #     prompt=prompt,
# #     guidance_scale=3.5,
# #     height=1024,
# #     width=1024,
# #     num_inference_steps=50,
# # ).images[0]
# # out.save("image.png")
# import os
# import requests
# from dotenv import load_dotenv

# load_dotenv()  # Load biến môi trường từ .env

# class PromptEnhancerClient:
#     def __init__(self):
#         self.base_url = os.getenv("API_PROMPT_ENHANCER_URL")
#         self.enhance_endpoint = f"{self.base_url}/api/v1/image-prompts/enhance-prompt"
#         self.classify_endpoint = f"{self.base_url}/api/v1/image-prompts/classify-prompt-type"
#         self.enhance_slide_endpoint = f"{self.base_url}/api/v1/image-prompts/enhance-slide-prompt"
#         self.check_safety_endpoint = f"{self.base_url}/api/v1/image-prompts/check-prompt-safety"

#     def enhance_prompt(self, prompt_text, style="Realistic", context="general", job_id=""):
#         try:
#             payload = {
#                 "user_prompt": prompt_text,
#                 "style": style,
#                 "context": context,
#                 "job_id": job_id
#             }
#             headers = {
#                 "accept": "application/json",
#                 "Content-Type": "application/x-www-form-urlencoded"
#             }

#             response = requests.post(self.enhance_endpoint, data=payload, headers=headers, timeout=10)
#             response.raise_for_status()

#             data = response.json()
#             return data.get("enhanced_prompt")

#         except requests.RequestException as e:
#             print(f"❌ Error calling enhance API: {e}")
#             return None
#     def classify_prompt_type(self, prompt_text, job_id=""):
#         try:
#             payload = {
#                 "prompt": prompt_text,
#                 "job_id": job_id
#             }
#             headers = {
#                 "accept": "application/json",
#                 "Content-Type": "application/x-www-form-urlencoded"
#             }

#             response = requests.post(self.classify_endpoint, data=payload, headers=headers, timeout=10)
#             response.raise_for_status()

#             data = response.json()
#             return data.get("classification")

#         except requests.RequestException as e:
#             print(f"❌ Error calling classify API: {e}")
#             return None
#     def enhance_slide_prompt(self, prompt_text, job_id=""):
#         try:
#             payload = {
#                 "prompt": prompt_text,
#                 "job_id": job_id
#             }
#             headers = {
#                 "accept": "application/json",
#                 "Content-Type": "application/x-www-form-urlencoded"
#             }

#             response = requests.post(self.enhance_slide_endpoint, data=payload, headers=headers, timeout=10)
#             response.raise_for_status()

#             data = response.json()
#             return data.get("enhanced_prompt")

#         except requests.RequestException as e:
#             print(f"❌ Error calling enhance slide API: {e}")
#             return None
#     def check_prompt_safety(self, prompt_text, job_id=""):
#         try:
#             payload = {
#                 "prompt": prompt_text,
#                 "job_id": job_id
#             }
#             headers = {
#                 "accept": "application/json",
#                 "Content-Type": "application/x-www-form-urlencoded"
#             }

#             response = requests.post(self.check_safety_endpoint, data=payload, headers=headers, timeout=10)
#             response.raise_for_status()

#             data = response.json()
#             return data.get("raw_response")

#         except requests.RequestException as e:
#             print(f"❌ Error calling check safety API: {e}")
#             return None
# if __name__ == "__main__":
#     client = PromptEnhancerClient()
#     original_prompt = "a man is reading book"

#     enhanced = client.check_prompt_safety(original_prompt)

#     print("⭐ Original prompt:", original_prompt)
#     print("✅ Enhanced prompt:", enhanced)


import asyncio

from app.api.backend.services.image_service import ImageGenerationService
from app.api.backend.schemas.requests import ImageGenerationRequest
import base64
from PIL import Image
import io

async def generate_image_example():
    """Ví dụ sử dụng ImageGenerationService để tạo ảnh"""
    
    # Tạo service instance
    image_service = ImageGenerationService()
    
    try:
        # Bước 1: Khởi tạo models (cần thiết trước khi sử dụng)
        print("Đang khởi tạo models...")
        await image_service.initialize_models()
        print("Khởi tạo models thành công!")
        
        # Bước 2: Tạo request với prompt của bạn
        request = ImageGenerationRequest(
            user_prompt="a beautiful sunset over mountains",  # Thay đổi prompt tại đây
            style="Realistic",  # Các option: "Realistic", "Cartoon", "Digital Art", "Sketch", "Cyberpunk", "Fantasy"
            aspect="1:1",  # Tỷ lệ khung hình: "1:1", "16:9", "9:16", "4:3", "3:4"
            width=512,
            height=512,
            guidance_scale=7.5,
            num_steps=20,
            use_custom_lora=False,
            lora_scale=0.8
        )
        
        # Bước 3: Gọi hàm tạo ảnh
        print("Đang tạo ảnh...")
        response = await image_service.generate_image(request)
        
        # Bước 4: Xử lý kết quả
        if response.success:
            print("Tạo ảnh thành công!")
            print(f"Final prompt: {response.final_prompt}")
            print(f"System info: {response.system_info}")
            
            # Lưu ảnh ra file
            image_data = base64.b64decode(response.image_base64)
            image = Image.open(io.BytesIO(image_data))
            image.save("generated_image.png")
            print("Ảnh đã được lưu thành 'generated_image.png'")
            
        else:
            print(f"Lỗi tạo ảnh: {response.message}")
            if response.warning:
                print(f"Cảnh báo: {response.warning}")
                
    except Exception as e:
        print(f"Lỗi: {e}")

def simple_generate_image(prompt, style="Realistic", save_path="output.png"):
    """Hàm đơn giản để tạo ảnh với prompt"""
    
    async def _generate():
        service = ImageGenerationService()
        await service.initialize_models()
        
        request = ImageGenerationRequest(
            user_prompt=prompt,
            style=style,
            aspect="1:1",
            width=512,
            height=512,
            guidance_scale=7.5,
            num_steps=20,
            use_custom_lora=False
        )
        
        response = await service.generate_image(request)
        
        if response.success:
            image_data = base64.b64decode(response.image_base64)
            image = Image.open(io.BytesIO(image_data))
            image.save(save_path)
            print(f"Ảnh đã được tạo và lưu tại: {save_path}")
            return True
        else:
            print(f"Lỗi: {response.message}")
            return False
    
    return asyncio.run(_generate())

# Cách sử dụng đơn giản nhất
if __name__ == "__main__":

    asyncio.run(generate_image_example())
