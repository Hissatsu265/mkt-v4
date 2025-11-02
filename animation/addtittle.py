from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random


def calculate_font_size(resolution):
    """Tự động tính font size phù hợp cho từng loại video"""
    w, h = resolution
    aspect_ratio = w / h
    
    if aspect_ratio > 1:  # 16:9 (landscape)
        base_size = min(w, h) // 7  # Tăng lên để chữ to hơn nữa
    else:  # 9:16 (portrait)
        base_size = w // 8
    
    return base_size


# Bảng màu đẹp cho video
COLOR_SCHEMES = [
    {"bg": (10, 20, 40), "text": "white"},           # Xanh đậm + trắng
    {"bg": (25, 25, 25), "text": "#FFD700"},         # Xám đen + vàng gold
    {"bg": (40, 10, 30), "text": "#FF69B4"},         # Tím đậm + hồng
    {"bg": (15, 35, 50), "text": "#00D9FF"},         # Xanh dương + cyan
    {"bg": (50, 20, 0), "text": "#FFAA00"},          # Nâu đậm + cam
    {"bg": (20, 40, 20), "text": "#7FFF00"},         # Xanh lá đậm + xanh neon
    {"bg": (45, 15, 45), "text": "#FF1493"},         # Tím hồng + đỏ hồng
    {"bg": (5, 25, 45), "text": "#FFFFFF"},          # Xanh navy + trắng
    {"bg": (30, 10, 10), "text": "#FF6B6B"},         # Đỏ đậm + đỏ pastel
    {"bg": (15, 30, 40), "text": "#FFE66D"},         # Xanh lam + vàng nhạt
]


def create_text_image(keyword, font_path, font_size, color, max_width, add_shadow=True, shadow_offset=3):
    """Tạo ảnh text với shadow cho hiệu ứng đẹp hơn"""
    font = ImageFont.truetype(font_path, font_size)
    lines = []
    words = keyword.split(" ")
    line = ""
    
    for word in words:
        test_line = line + word + " "
        bbox = font.getbbox(test_line)
        w = bbox[2] - bbox[0]
        if w <= max_width - 100:
            line = test_line
        else:
            if line:
                lines.append(line.strip())
            line = word + " "
    if line:
        lines.append(line.strip())

    # Tính kích thước ảnh
    line_height = int(font_size * 1.3)
    padding = 20
    img_height = len(lines) * line_height + padding * 2
    img_width = max_width
    
    # Tạo ảnh với shadow
    img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        bbox = font.getbbox(line)
        w = bbox[2] - bbox[0]
        x = (img_width - w) / 2
        
        if add_shadow:
            shadow_color = (0, 0, 0, 180)
            draw.text((x + shadow_offset, y + shadow_offset), line, font=font, fill=shadow_color)
        
        draw.text((x, y), line, font=font, fill=color)
        y += line_height

    return np.array(img)


def create_text_clip_typewriter(keyword, start, end, font, font_size, color, w, h):
    """
    Hiệu ứng typewriter: hiển thị từng chữ, KHÔNG chồng chữ cũ
    """
    duration = end - start
    if len(keyword.strip()) == 0:
        return []

    char_duration = (duration / len(keyword))*0.4
    clips = []

    for i in range(1, len(keyword) + 1):
        visible_text = keyword[:i]
        frame_start = start + (i - 1) * char_duration
        frame_end = start + i * char_duration

        # Tạo frame với text hiện tại
        img_array = create_text_image(visible_text, font, font_size, color, max_width=w - 100)
        frame_clip = (ImageClip(img_array)
                      .set_start(frame_start)
                      .set_duration(char_duration)
                      .set_position("center"))
        clips.append(frame_clip)

    return clips


def create_text_clip(keyword, start, end, font, font_size, color, w, h, effect_type):
    """Tạo text clip với các hiệu ứng được chọn"""
    duration = end - start

    # Hiệu ứng typewriter trả về list clips
    if effect_type == 4:
        return create_text_clip_typewriter(keyword, start, end, font, font_size, color, w, h)

    # Tạo ảnh text
    img_array = create_text_image(keyword, font, font_size, color, max_width=w - 100)
    txt_clip = ImageClip(img_array).set_duration(duration)

    # Áp dụng hiệu ứng
    if effect_type == 1:  # Fade in/out mượt mà
        txt_clip = (txt_clip
                    .set_start(start)
                    .crossfadein(0.6)
                    .crossfadeout(0.6))
    
    elif effect_type == 2:  # Zoom in + fade
        def zoom_effect(t):
            progress = min(t / 0.6, 1)
            return 0.3 + 0.7 * (1 - (1 - progress) ** 2)
        txt_clip = (txt_clip
                    .set_start(start)
                    .resize(zoom_effect)
                    .crossfadein(0.6)
                    .crossfadeout(0.6))
    
    elif effect_type == 3:  # Bounce in
        def bounce_in(t):
            progress = min(t / 0.7, 1)
            if progress < 0.7:
                scale = progress / 0.7
                scale = scale * (1 + 0.3 * np.sin(scale * np.pi * 3))
            else:
                scale = 1
            return max(0.1, min(scale, 1.2))
        txt_clip = (txt_clip
                    .set_start(start)
                    .resize(bounce_in)
                    .crossfadeout(0.5))
    
    else:  # Mặc định: fade đơn giản
        txt_clip = txt_clip.set_start(start).crossfadein(0.4).crossfadeout(0.4)

    txt_clip = txt_clip.set_position("center")
    return [txt_clip]


def create_keyword_video(
    keywords, start_times, end_times, duration,
    resolution=(1280, 720), font="MontserratBlack-3zOvZ.ttf",
    bg_color=None, font_color=None, font_size=None,
    effect_type=None, output_path="output.mp4"
):

    w, h = resolution
    
    # Random color scheme nếu không được cung cấp
    if bg_color is None or font_color is None:
        color_scheme = random.choice(COLOR_SCHEMES)
        bg_color = color_scheme["bg"]
        font_color = color_scheme["text"]
    
    # Tự động tính font size nếu không được cung cấp
    if font_size is None:
        font_size = calculate_font_size(resolution)
    
    # Random effect nếu không được cung cấp (chỉ 1 loại cho cả video)
    if effect_type is None:
        effect_type = random.randint(1, 4)
    
    # Tạo background
    bg_clip = ColorClip(size=resolution, color=bg_color, duration=duration)
    
    clips = []
    
    for i, (kw, st, et) in enumerate(zip(keywords, start_times, end_times)):
        # Tạo text clips (có thể là list cho typewriter effect)
        text_clips = create_text_clip(kw, st, et, font, font_size, font_color, w, h, effect_type)
        clips.extend(text_clips)
    
    # Composite tất cả clips
    final_clip = CompositeVideoClip([bg_clip] + clips)
    final_clip.write_videofile(output_path, fps=30, codec="libx264", audio=False, threads=4)
    
    effect_names = {1: "Fade", 2: "Zoom", 3: "Bounce", 4: "Typewriter"}
    print(f"✅ Video đã được tạo: {output_path}")
    print(f"🎨 Màu nền: {bg_color}, Màu chữ: {font_color}")
    print(f"✨ Hiệu ứng: {effect_names.get(effect_type, 'Unknown')}")


# ======================
# 🎬 VÍ DỤ SỬ DỤNG
# ======================

# Ví dụ 1: Video 16:9 với màu và hiệu ứng random
# keywords_16_9 = ["Xin chào", "Video 16:9", "Hiệu ứng đẹp mắt", "Cảm ơn!"]
# start_times = [0, 2.5, 5, 8]
# end_times = [2.5, 5, 8, 11]

# create_keyword_video(
#     keywords_16_9,
#     start_times,
#     end_times,
#     duration=12,
#     resolution=(1280, 720),  # 16:9
#     font="/content/PoppinsSemibold-8l8n.otf",
#     effect_type=4,
#     # bg_color và font_color để None để random màu
#     output_path="video_16_9.mp4"
# )

# # Ví dụ 2: Video 9:16 với hiệu ứng Typewriter
# keywords_9_16 = ["Chào mừng", "Video dọc 9:16", "Phù hợp TikTok", "Instagram Reels"]
# start_times = [1, 3, 6, 9]
# end_times = [3, 6, 9, 12]

# create_keyword_video(
#     keywords_9_16,
#     start_times,
#     end_times,
#     duration=13,
#     resolution=(720, 1280),  # 9:16
#     font="/content/PoppinsSemibold-8l8n.otf",
#     effect_type=2,  # 1: Fade, 2: Zoom, 3: Bounce, 4: Typewriter
#     output_path="video_9_16.mp4"
# )

# # Ví dụ 3: Tự chọn màu cụ thể
# create_keyword_video(
#     ["Text của bạn","sdfs","lfldfkgkfdmf"],
#     [0,5,8],
#     [5,8,9],
#     duration=10,
#     resolution=(1280, 720),
#     bg_color=(30, 58, 138),      # Xanh đậm
#     font_color="#FFFFFF",
#     font="/content/PoppinsSemibold-8l8n.otf",
#     effect_type=1,              # Bounce
#     output_path="custom_colors.mp4"
# )