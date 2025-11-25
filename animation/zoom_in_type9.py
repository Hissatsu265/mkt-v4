import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip

def zoom_and_light_effect_type9(
    image_path,
    duration,
    output_path="zoom_light_output.mp4",
    zoom_factor=1.2,
    zoom_portion=0.3,
    light_portion=0.5,
    fps=30
):
    """
    Tạo zoom và light effect với thời lượng tùy chỉnh
    
    Args:
        image_path: đường dẫn đến ảnh
        duration: thời lượng video (giây)
        output_path: đường dẫn output video
        zoom_factor: độ phóng to (1.2 = zoom 120%)
        zoom_portion: tỷ lệ thời gian zoom (0.3 = zoom trong 30% đầu video)
        light_portion: tỷ lệ thời gian ánh sáng (0.5 = sáng trong 50% đầu video)
        fps: frame per second
    """
    
    print(f"Thời lượng video: {duration:.2f} giây")
    
    # Đọc ảnh gốc
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    print(f"Kích thước ảnh gốc: {w}x{h}")
    
    # Xác định tỷ lệ ảnh
    ratio = w / h
    
    # Resize ảnh theo tỷ lệ
    if ratio > 1:  # Ảnh ngang (16:9)
        target_w, target_h = 1280, 720
        print(f"Ảnh ngang (16:9) - Resize về {target_w}x{target_h}")
    else:  # Ảnh dọc (9:16)
        target_w, target_h = 720, 1280
        print(f"Ảnh dọc (9:16) - Resize về {target_w}x{target_h}")
    
    # Resize ảnh với INTER_AREA cho chất lượng tốt khi thu nhỏ
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    h, w = target_h, target_w
    
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

        # --- Zoom mượt bằng warpAffine ---
        scale = 1 + (zoom_factor - 1) * zoom_progress
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
    
    # Xuất video không có audio
    clip.write_videofile(output_path, fps=fps, codec='libx264')
    
    print(f"Video đã được tạo: {output_path}")


# # Ví dụ sử dụng
# zoom_and_light_effect(
#     image_path="/content/downloads/326dbf07-dfe5-4ef5-acee-705e282ad65e.png",
#     duration=10.5,  # Thời lượng video 10.5 giây
#     output_path="output_zoom_light_sang.mp4",
#     zoom_factor=1.3,      # zoom nhẹ
#     zoom_portion=0.8,     # zoom 80% đầu
#     light_portion=0.9     # ánh sáng 90% đầu
# )