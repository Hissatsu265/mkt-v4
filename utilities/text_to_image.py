# import requests
# import uuid
# import time
# import os
from config import SERVER_COMFYUI,WORKFLOW_INFINITETALK_PATH,BASE_DIR


# BASE_URL = "https://image.shohanursobuj.online/api/v1/ai"
# STATUS_INTERVAL = 7           # poll 7s/lần
# TIMEOUT_SECONDS = 120         # 2 phút timeout


# def generate_image(prompt: str, aspect_ratio: str, style="Realistic", quality="Medium"):
#     # Tạo job_id ngẫu nhiên
#     job_id = str(uuid.uuid4())

#     payload = {
#         "user_prompt": prompt,
#         "style": style,
#         "aspect": aspect_ratio,
#         "quality": quality,
#         "job_id": job_id
#     }

#     print(f"🔄 Sending generate request with job_id = {job_id} ...")

#     response = requests.post(
#         f"{BASE_URL}/ai-generate",
#         json=payload,
#         headers={"accept": "application/json", "Content-Type": "application/json"},
#         timeout=30
#     )
#     data = response.json()
#     print("📨 Response:", data)

#     if not data.get("success", False):
#         raise Exception("❌ Generate API failed: " + str(data))

#     print("\n⏳ Waiting for job to complete...")

#     start_time = time.time()

#     while True:
#         # Check timeout
#         if time.time() - start_time > TIMEOUT_SECONDS:
#             raise TimeoutError("❌ Job timed out after 2 minutes.")

#         time.sleep(STATUS_INTERVAL)

#         status_data = requests.get(
#             f"{BASE_URL}/ai-generate-status/{job_id}",
#             timeout=20
#         ).json()

#         current_status = status_data.get("status")
#         print(f"➡️ Status: {current_status}")

#         if current_status == "completed":
#             file_url = status_data["file_path"]
#             print("✅ Job Completed!")
#             print("📎 File URL:", file_url)
#             return download_file(file_url, job_id)

#         if current_status == "failed":
#             raise Exception("❌ Job failed: " + str(status_data))


# def download_file(url: str, job_id: str):
#     print("⬇️ Downloading file from:", url)
    
#     save_dir =str(BASE_DIR) + "/downloads_text2images"
#     print(save_dir,"  ============Text to image save dir==========")
#     os.makedirs(save_dir, exist_ok=True)

#     file_path = os.path.join(save_dir, f"{job_id}.png")

#     response = requests.get(url, stream=True)
#     response.raise_for_status()

#     with open(file_path, "wb") as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             if chunk:
#                 f.write(chunk)

#     print("📁 File saved to:", file_path)
#     return file_path


# # ============================
# # Example usage
# # ============================
# # if __name__ == "__main__":
# #     prompt = input("Prompt: ")
# #     ratio = input("Aspect ratio (ví dụ: 9:16, 16:9): ")

# #     output_path = generate_image(prompt, ratio)
# #     print("\n🎉 DONE! Final file saved at:", output_path)
import requests
import uuid
import time
import os

BASE_URL = "https://image.shohanursobuj.online/api/v1/ai"
STATUS_INTERVAL = 7
TIMEOUT_SECONDS = 180


def safe_request(method, url, **kwargs):
    try:
        return requests.request(method, url, **kwargs)
    except Exception:
        print("⚠️ Request error, retrying in 5 seconds...")
        time.sleep(5)
        try:
            return requests.request(method, url, **kwargs)
        except Exception:
            print("❌ Request failed twice")
            return None


def generate_image(prompt: str, aspect_ratio: str, style="Realistic", quality="Medium"):
    job_id = str(uuid.uuid4())

    payload = {
        "user_prompt": prompt,
        "style": style,
        "aspect": aspect_ratio,
        "quality": quality,
        "job_id": job_id
    }

    print(f"🔄 Sending generate request with job_id = {job_id} ...")

    response = safe_request(
        "POST",
        f"{BASE_URL}/ai-generate",
        json=payload,
        headers={"accept": "application/json", "Content-Type": "application/json"},
        timeout=30
    )

    if response is None:
        return {"success": False}

    data = response.json()
    print("📨 Response:", data)

    if not data.get("success", False):
        print("❌ Generate API failed.")
        return {"success": False}

    print("\n⏳ Waiting for job to complete...")
    start_time = time.time()

    while True:
        if time.time() - start_time > TIMEOUT_SECONDS:
            print("❌ Job timed out.")
            return {"success": False}

        time.sleep(STATUS_INTERVAL)

        status_response = safe_request(
            "GET",
            f"{BASE_URL}/ai-generate-status/{job_id}",
            timeout=20
        )

        if status_response is None:
            return {"success": False}

        status_data = status_response.json()
        current_status = status_data.get("status")
        print(f"➡️ Status: {current_status}")

        if current_status == "completed":
            file_url = status_data["file_path"]
            print("✅ Job Completed")

            saved_local_path = download_file(file_url, job_id)

            if saved_local_path is False:
                return {"success": False}

            return {
                "success": True,
                "status": current_status,
                "file_path": saved_local_path
            }

        if current_status == "failed":
            print("❌ Job failed:", status_data)
            return {"success": False}


def download_file(url: str, job_id: str):
    print("⬇️ Downloading file from:", url)
    save_dir =str(BASE_DIR) + "/downloads_text2images"
    # save_dir = "downloads"
    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, f"{job_id}.png")

    response = safe_request("GET", url, stream=True)
  
    if response is None:
        return False

    try:
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except Exception:
        print("❌ Failed to save file")
        return False

    print("📁 File saved to:", local_path)
    return local_path