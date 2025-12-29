# import base64
# import requests
# import json
# import os

# # ==============================
# # CONFIG
# # ==============================
# API_BASE_URL = "http://127.0.0.1:8001/api/v1/ai/generate-image"  # đổi thành URL API thật của bạn

# # ==============================
# # HÀM GỬI REQUEST
# # ==============================
# def send_image_request(image_path, prompt, aspect="9:16", quality="medium"):
#     # Đọc ảnh và chuyển sang base64
#     with open(image_path, "rb") as img_file:
#         image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

#     # Tạo payload gửi đi
#     request_data = {
#         "user_prompt": prompt,
#         "aspect": aspect,
#         "quality": quality,
#         "negative_prompt":"human, text, watermark, logo, extra objects, hands, people, human, low quality, blurry, distorted, messy background, overexposed, unrealistic shadows, poor lighting",
#         "type_generation":"normal1",
#         "input_image2": image_base64,
#         "input_image": image_base64
#     }

#     print("Sending request...")
#     response = requests.post(
#         API_BASE_URL,
#         headers={"Content-Type": "application/json"},
#         data=json.dumps(request_data)
#     )

#     # Kiểm tra phản hồi
#     if response.status_code != 200:
#         print("❌ Error:", response.status_code, response.text)
#         return None

#     result = response.json()

#     # Nếu server trả về ảnh base64, lưu lại
#     if "image1_base64" in result:
#         output_image_data = base64.b64decode(result["image1_base64"])
#         output_path = "output_image1233.png"
#         with open(output_path, "wb") as f:
#             f.write(output_image_data)
#         print(f"✅ Image saved to {output_path}")
#     else:
#         print("⚠️ No image1_base64 found in response.")
    
#     return result


# # ==============================
# # CHẠY DEMO
# # ==============================
# if __name__ == "__main__":
#     image_path = "/workspace/analyzed_video_task_object_replace_llm_81c7816dd1c04b0b9ff7f0576aa759fc_81c7816dd1c04b0b9ff7f0576aa759fc_131 (1).jpg"
#     if not os.path.exists(image_path):
#         print("❌ Image not found:", image_path)
#     else:
#         prompt = ""
#         result = send_image_request(image_path, prompt)
#         print("Response:", result)
import base64
import requests
import json
import os

# ==============================
# CONFIG
# ==============================
API_BASE_URL = "http://127.0.0.1:8001/api/v1/ai/generate-image"  # đổi thành URL API thật của bạn

# ==============================
# HÀM GỬI REQUEST
# ==============================
def send_image_request(image_path1, image_path2, prompt, aspect="16:9", quality="medium"):
    # Hàm chuyển ảnh sang base64 (nếu có)
    def encode_image(path):
        if path and os.path.exists(path):
            with open(path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        else:
            print(f"⚠️ Image not found or path empty: {path}")
            return None

    # Chuyển 2 ảnh sang base64
    image_base64_1 = encode_image(image_path1)
    image_base64_2 = encode_image(image_path2)

    # Tạo payload gửi đi
    request_data = {
        "user_prompt": prompt,
        "aspect": aspect,
        "quality": quality,
        "negative_prompt": "human, text, watermark, logo, extra objects, hands, people, human, low quality, blurry, distorted, messy background, overexposed, unrealistic shadows, poor lighting",
        "type_generation": "none",
        "input_image": image_base64_1,
        "input_image2": image_base64_2
    }
    print(request_data)

    print("🚀 Sending request...")
    response = requests.post(
        API_BASE_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(request_data)
    )

    # Kiểm tra phản hồi
    if response.status_code != 200:
        print("❌ Error:", response.status_code, response.text)
        return None

    result = response.json()

    # Nếu server trả về ảnh base64, lưu lại
    if "image1_base64" in result:
        output_image_data = base64.b64decode(result["image1_base64"])
        output_path = "output_image_result.png"
        with open(output_path, "wb") as f:
            f.write(output_image_data)
        print(f"✅ Image saved to {output_path}")
    else:
        print("⚠️ No image1_base64 found in response.")
    
    return result


# ==============================
# CHẠY DEMO
# ==============================
if __name__ == "__main__":
    image_path1 = "/home/toan/ComfyUI_00001_jggbn_1760443812-Photoroom (1).png"
    image_path2 = ""  # ảnh thứ 2
    prompt = ""

    result = send_image_request(image_path1, image_path2, prompt)
    print("Response:", result)
