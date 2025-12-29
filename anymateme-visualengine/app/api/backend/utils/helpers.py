import re
import base64
from io import BytesIO
from PIL import Image

def clean_prompt(prompt):
    """Làm sạch prompt bằng cách loại bỏ ký tự không in được"""
    return re.sub(r'[^\x20-\x7E]|"', ' ', prompt)

def extract_answer(text,key: str ="answer:"):
    # key = "answer:"
    idx = text.lower().find(key)
    if idx == -1:
        return text.strip()
    return text[idx + len(key):].strip()
def extract_answer1(text):
    key = "prompt:"
    idx = text.lower().find(key)
    if idx == -1:
        return text.strip()
    return text[:idx].strip()

def image_to_base64(image: Image.Image) -> str:
    """Chuyển đổi PIL Image sang base64 string"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode()

def base64_to_image(base64_string: str) -> Image.Image:
    """Chuyển đổi base64 string sang PIL Image"""
    img_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_bytes))

def get_aspect_ratio_dimensions(aspect: str, width: int = 0, height: int = 0):
    """Lấy kích thước từ tỷ lệ khung hình"""
    aspect_ratio_map = {
        "1:1": (1024, 1024),
        "3:2": (576, 384),
        "2:3": (384, 576),
        "16:9": (1920, 1080),
        "9:16": (1080, 1920),

    }
    
    # Sử dụng kích thước tùy chỉnh nếu được cung cấp
    if width > 0 and height > 0:
        return width, height
    
    return aspect_ratio_map.get(aspect, (1024, 1024))

def warning_message(label: str) -> str:
    warnings = {
        "none": "⚠️",
        "main-human": "⚠️ Warning: The prompt describes a person as the main subject of the image or the prompt contains sensitive content.",
        "public-figure": "⚠️ Warning: Prompt involves a public figure or real-life person. May violate copyright or terms of service.",
        "avatar": "⚠️ Warning: Prompt requests an avatar or profile picture. Ask the user to provide a real photo if necessary.",
        "inappropriate": "🚫 ALERT: Prompt contains sensitive or inappropriate content. Blocked.",
    }
    
    return warnings.get(label, "❓ Unknown label type.") + " Please edit the prompt carefully to avoid violating policies."