import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip

def zoom_and_light_effect(
    image_path,
    audio_path,
    output_path="zoom_light_output.mp4",
    zoom_factor=1.2,
    zoom_portion=0.3,
    light_portion=0.5,
    fps=30
):
    """
    Tạo zoom và light effect với audio
    
    Args:
        image_path: đường dẫn đến ảnh
        audio_path: đường dẫn đến file audio
        output_path: đường dẫn output video
        zoom_factor: độ phóng to (1.2 = zoom 120%)
        zoom_portion: tỷ lệ thời gian zoom (0.3 = zoom trong 30% đầu video)
        light_portion: tỷ lệ thời gian ánh sáng (0.5 = sáng trong 50% đầu video)
        fps: frame per second
    """
    # Đọc audio và lấy duration
    audio = AudioFileClip(audio_path)
    duration = audio.duration
    
    print(f"Thời lượng audio: {duration:.2f} giây")
    
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    step_count = int(duration * fps)

    frames = []
    zoom_end_step = int(step_count * zoom_portion)
    light_end_step = int(step_count * light_portion)

    print(f"Đang tạo {step_count} frames...")

    for i in range(step_count):
        t = i / step_count
        zoom_progress = min(i / zoom_end_step, 1.0)
        # dùng easing cubic để zoom mượt
        zoom_progress = zoom_progress ** 2 * (3 - 2 * zoom_progress)

        light_progress = min(i / light_end_step, 1.0)

        # --- Zoom OUT mượt bằng warpAffine ---
        # Bắt đầu từ zoom_factor, kết thúc ở 1.0 (kích thước gốc)
        scale = zoom_factor - (zoom_factor - 1) * zoom_progress
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        frame = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        # --- Hiệu ứng ánh sáng ---
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        dist = dist / dist.max()
        
        # Độ tối ban đầu: 0.05 = rất tối (5% sáng), tăng dần lên 1.0 (100% sáng)
        min_brightness = 0.05 + 0.95 * light_progress
        light = np.clip(1 - dist * (1 - light_progress), min_brightness, 1.0)
        light = cv2.merge([light, light, light])

        frame_light = np.clip(frame.astype(np.float32) * light, 0, 255).astype(np.uint8)
        frames.append(cv2.cvtColor(frame_light, cv2.COLOR_BGR2RGB))

    print("Đang tạo video...")
    clip = ImageSequenceClip(frames, fps=fps)
    
    # Gắn audio vào video
    clip = clip.set_audio(audio)
    
    # Xuất video với audio
    clip.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac')
    
    print(f"Video đã được tạo: {output_path}")


# # Ví dụ sử dụng
# zoom_and_light_effect(
#     image_path="/content/6422b239-da46-4fa6-9101-97d811ac74d9.png",
#     audio_path="/content/caroline_de.wav",  # Thay bằng đường dẫn audio của bạn
#     output_path="output_zoom_out_light_sang.mp4",
#     zoom_factor=1.3,      # zoom nhẹ
#     zoom_portion=0.7,     # zoom 50% đầu
#     light_portion=0.9     # ánh sáng 80% đầu
# )