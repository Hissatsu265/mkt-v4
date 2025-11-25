import requests

def generate_prompts(full_audio_text, target_segment_text):
    try:
        url = "https://dev.shohanursobuj.online/api/v1/media-prompt/generate-segment"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }

        payload = {
            "full_audio_text": full_audio_text,
            "target_segment_text": target_segment_text
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()  # nếu lỗi HTTP sẽ ném exception

        data = response.json()
        image_prompt = data.get("image_prompt", "")
        video_prompt = data.get("video_prompt", "")

        return image_prompt, video_prompt

    except Exception:
        return "", ""


# # Ví dụ gọi hàm
# if __name__ == "__main__":
#     full_audio = (
#         "The holiday season is finally here, and I hope you can feel the warmth "
#         "and magic in the air. Christmas isn’t just about lights, gifts, or "
#         "celebrations — it’s about sharing joy, kindness, and moments that bring "
#         "us closer together."
#     )
#     target_segment = "Christmas isn’t just about lights, gifts, or celebrations"

#     img_prompt, vid_prompt = generate_prompts(full_audio, target_segment)

#     print("Image prompt:", img_prompt)
#     print("Video prompt:", vid_prompt)
