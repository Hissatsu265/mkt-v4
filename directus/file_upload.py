# from directus.directus_utils import upload_file_to_directus
# from config import DirectusConfig

# directus_config = DirectusConfig()
# # import sys, os
# # sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# def Uploadfile_directus(path):
#     # path = "/home/toan/marketing-video-ai/temp_no_audio.mp4"
#     # print("=== DEBUG Directus ===")
#     # print(f"URL: {DirectusConfig.DIRECTUS_URL}")
#     # print(f"TOKEN: {DirectusConfig.ACCESS_TOKEN}")
#     # print("======================")
#     try:
#         directus_response = upload_file_to_directus(path)
#         print("Upload successful!")
#         print(f"Response: {directus_response}")
#         directus_url = f"{directus_config.DIRECTUS_URL}/assets/{directus_response['data']['id']}"
#         print(directus_url)
#         return directus_url
#     except Exception as e:
#         print(f"Upload failed: {e}")
#         return None

# # if __name__ == "__main__":
# #     Uploadfile_directus("")
import time
from directus.directus_utils import upload_file_to_directus
from config import DirectusConfig

directus_config = DirectusConfig()

def Uploadfile_directus(path):
    """
    Upload file lên Directus, nếu lần đầu thất bại sẽ tự động retry sau 5s.
    Trả về URL của file nếu thành công, None nếu thất bại.
    """
    max_retries = 2  # tổng số lần thử (lần đầu + 1 lần retry)
    delay_seconds = 5

    for attempt in range(1, max_retries + 1):
        try:
            print(f"🔹 Attempt {attempt}/{max_retries} uploading {path}...")
            directus_response = upload_file_to_directus(path)
            
            print("✅ Upload successful!")
            directus_url = f"{directus_config.DIRECTUS_URL}/assets/{directus_response['data']['id']}"
            print(f"Directus URL: {directus_url}")
            return directus_url

        except Exception as e:
            print(f"❌ Upload failed (attempt {attempt}): {e}")
            if attempt < max_retries:
                print(f"⏳ Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                print("🚫 All upload attempts failed.")
                return None
