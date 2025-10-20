# from moviepy.editor import ImageClip, concatenate_videoclips
# import numpy as np
# import cv2

# def ken_burns_effect(image_path, output_path="ken_burns_output.mp4", duration=6, scale=0.8, fps=30):
#     # Đọc ảnh
#     image = cv2.imread(image_path)
#     h, w, _ = image.shape

#     # Kích thước khung hình nhỏ hơn
#     crop_h, crop_w = int(h * scale), int(w * scale)
#     step_count = int(duration * fps)

#     frames = []

#     for i in range(step_count):
#         progress = i / step_count

#         # Giai đoạn đầu: từ trái qua phải (nửa thời gian đầu)
#         if progress <= 0.5:
#             x = int(progress * 2 * (w - crop_w))  # trái ➝ phải
#             y = 0
#         else:
#             # Giai đoạn sau: phải qua trái ở dưới
#             x = int((1 - (progress - 0.5) * 2) * (w - crop_w))  # phải ➝ trái
#             y = h - crop_h

#         # Cắt ảnh con
#         cropped = image[y:y+crop_h, x:x+crop_w]
#         resized = cv2.resize(cropped, (w, h))
#         frames.append(resized)

#     # Ghi video bằng MoviePy
#     def make_frame(t):
#         idx = min(int(t * fps), len(frames) - 1)
#         return frames[idx][:, :, ::-1]  # BGR ➝ RGB

#     clip = ImageClip(frames[0][:, :, ::-1], duration=duration)
#     video = clip.set_make_frame(make_frame).set_duration(duration)
#     video.write_videofile(output_path, fps=fps)

# # Ví dụ sử dụng
# ken_burns_effect("/content/phone.jpg", output_path="output_ken_burns.mp4", duration=6)
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
import numpy as np
import cv2

def ken_burns_effect(image_path, audio_path, output_path="ken_burns_output.mp4", scale=0.8, fps=30):
    """
    Tạo Ken Burns effect với audio
    
    Args:
        image_path: đường dẫn đến ảnh
        audio_path: đường dẫn đến file audio
        output_path: đường dẫn output video
        scale: tỷ lệ zoom (0.8 = zoom 80%)
        fps: frame per second
    """
    # Đọc audio và lấy duration
    audio = AudioFileClip(audio_path)
    duration = audio.duration
    
    print(f"Thời lượng audio: {duration:.2f} giây")
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Kích thước khung hình nhỏ hơn
    crop_h, crop_w = int(h * scale), int(w * scale)
    step_count = int(duration * fps)

    frames = []

    for i in range(step_count):
        progress = i / step_count

        # Giai đoạn đầu: từ trái qua phải (nửa thời gian đầu)
        if progress <= 0.5:
            x = int(progress * 2 * (w - crop_w))  # trái ➝ phải
            y = 0
        else:
            # Giai đoạn sau: phải qua trái ở dưới
            x = int((1 - (progress - 0.5) * 2) * (w - crop_w))  # phải ➝ trái
            y = h - crop_h

        # Cắt ảnh con
        cropped = image[y:y+crop_h, x:x+crop_w]
        resized = cv2.resize(cropped, (w, h))
        frames.append(resized)

    # Ghi video bằng MoviePy
    def make_frame(t):
        idx = min(int(t * fps), len(frames) - 1)
        return frames[idx][:, :, ::-1]  # BGR ➝ RGB

    clip = ImageClip(frames[0][:, :, ::-1], duration=duration)
    video = clip.set_make_frame(make_frame).set_duration(duration)
    
    # Gắn audio vào video
    video = video.set_audio(audio)
    
    # Xuất video
    video.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac')
    
    print(f"Video đã được tạo: {output_path}")

# # Ví dụ sử dụng
# ken_burns_effect(
#     image_path="/content/6422b239-da46-4fa6-9101-97d811ac74d9.png",
#     audio_path="/content/caroline_de.wav",  # Thay bằng đường dẫn audio của bạn
#     output_path="output_ken_burns.mp4"
# )