from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random


def calculate_font_size(resolution):
    """Tự động tính font size phù hợp cho từng loại video"""
    w, h = resolution
    aspect_ratio = w / h
    
    if aspect_ratio > 1:  # 16:9 (landscape)
        base_size = min(w, h) // 7
    else:  # 9:16 (portrait)
        base_size = w // 8
    
    return base_size


# Bảng màu đẹp cho video
COLOR_SCHEMES = [
    {"bg": (10, 20, 40), "text": "white"},
    {"bg": (25, 25, 25), "text": "#FFD700"},
    {"bg": (40, 10, 30), "text": "#FF69B4"},
    {"bg": (15, 35, 50), "text": "#00D9FF"},
    {"bg": (50, 20, 0), "text": "#FFAA00"},
    {"bg": (20, 40, 20), "text": "#7FFF00"},
    {"bg": (45, 15, 45), "text": "#FF1493"},
    {"bg": (5, 25, 45), "text": "#FFFFFF"},
    {"bg": (30, 10, 10), "text": "#FF6B6B"},
    {"bg": (15, 30, 40), "text": "#FFE66D"},
]


def create_text_image(keyword, font_path, font_size, color, max_width, add_shadow=True, shadow_offset=3):
    """Tạo ảnh text với shadow và anti-aliasing cao cấp"""
    # Tăng độ phân giải gấp 2 để render mượt hơn
    scale = 2
    font = ImageFont.truetype(font_path, font_size * scale)
    lines = []
    words = keyword.split(" ")
    line = ""
    
    for word in words:
        test_line = line + word + " "
        bbox = font.getbbox(test_line)
        w = bbox[2] - bbox[0]
        if w <= (max_width - 100) * scale:
            line = test_line
        else:
            if line:
                lines.append(line.strip())
            line = word + " "
    if line:
        lines.append(line.strip())

    line_height = int(font_size * 1.3 * scale)
    padding = 20 * scale
    img_height = len(lines) * line_height + padding * 2
    img_width = max_width * scale
    
    # Tạo ảnh với độ phân giải cao hơn
    img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        bbox = font.getbbox(line)
        w = bbox[2] - bbox[0]
        x = (img_width - w) / 2
        
        if add_shadow:
            shadow_color = (0, 0, 0, 180)
            draw.text((x + shadow_offset * scale, y + shadow_offset * scale), line, font=font, fill=shadow_color)
        
        draw.text((x, y), line, font=font, fill=color)
        y += line_height

    # Resize về kích thước gốc với LANCZOS filter (chất lượng cao nhất)
    img = img.resize((max_width, img_height // scale), Image.Resampling.LANCZOS)
    return np.array(img)


def create_text_image_word_mask(full_text, visible_words_count, font_path, font_size, color, max_width, add_shadow=True, shadow_offset=3):
    """
    Tạo ảnh text với TOÀN BỘ câu, nhưng chỉ hiện một số từ đầu tiên
    Các từ còn lại hoàn toàn trong suốt (alpha=0)
    """
    font = ImageFont.truetype(font_path, font_size)
    words = full_text.split()
    
    # Chia thành dòng như cũ
    lines = []
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

    line_height = int(font_size * 1.3)
    padding = 20
    img_height = len(lines) * line_height + padding * 2
    img_width = max_width
    
    img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Đếm số từ đã vẽ
    word_count = 0
    y = padding
    
    for line in lines:
        bbox = font.getbbox(line)
        line_width = bbox[2] - bbox[0]
        x = (img_width - line_width) / 2
        
        # Vẽ từng từ trong dòng
        line_words = line.split()
        current_x = x
        
        for word in line_words:
            word_count += 1
            
            # Chỉ hiện từ nếu nằm trong số từ được phép hiển thị
            if word_count <= visible_words_count:
                if add_shadow:
                    shadow_color = (0, 0, 0, 180)
                    draw.text((current_x + shadow_offset, y + shadow_offset), word, font=font, fill=shadow_color)
                draw.text((current_x, y), word, font=font, fill=color)
            
            # Tính vị trí cho từ tiếp theo
            word_bbox = font.getbbox(word + " ")
            current_x += word_bbox[2] - word_bbox[0]
        
        y += line_height

    return np.array(img)


def create_text_image_char_mask(full_text, visible_chars_count, font_path, font_size, color, max_width, add_shadow=True, shadow_offset=3):
    """
    Tạo ảnh text với TOÀN BỘ câu, nhưng chỉ hiện một số ký tự đầu tiên
    Các ký tự còn lại hoàn toàn trong suốt (alpha=0)
    """
    font = ImageFont.truetype(font_path, font_size)
    
    # Chia thành dòng
    lines = []
    words = full_text.split(" ")
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

    line_height = int(font_size * 1.3)
    padding = 20
    img_height = len(lines) * line_height + padding * 2
    img_width = max_width
    
    img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Đếm số ký tự đã vẽ
    char_count = 0
    y = padding
    
    for line in lines:
        bbox = font.getbbox(line)
        line_width = bbox[2] - bbox[0]
        x = (img_width - line_width) / 2
        
        # Vẽ từng ký tự trong dòng
        current_x = x
        
        for char in line:
            char_count += 1
            
            # Chỉ hiện ký tự nếu nằm trong số ký tự được phép hiển thị
            if char_count <= visible_chars_count:
                if add_shadow:
                    shadow_color = (0, 0, 0, 180)
                    draw.text((current_x + shadow_offset, y + shadow_offset), char, font=font, fill=shadow_color)
                draw.text((current_x, y), char, font=font, fill=color)
            
            # Tính vị trí cho ký tự tiếp theo
            char_bbox = font.getbbox(char)
            current_x += char_bbox[2] - char_bbox[0]
        
        y += line_height

    return np.array(img)


def create_text_clip_word_reveal(keyword, start, end, font, font_size, color, w, h):
    """
    Hiệu ứng word reveal: text CỐ ĐỊNH vị trí, từ từ hiển thị từng từ
    """
    duration = end - start
    words = keyword.split()
    
    if len(words) == 0:
        return []
    
    # Tốc độ nhanh hơn: 0.15 thay vì 0.3
    word_duration = max(0.08, (duration / len(words)) * 0.15)
    clips = []
    
    # Tạo frame cho mỗi trạng thái hiển thị (1 từ, 2 từ, 3 từ...)
    for i in range(1, len(words) + 1):
        frame_start = start + (i - 1) * word_duration
        
        # Tạo ảnh với i từ đầu tiên hiển thị, các từ còn lại ẩn
        img_array = create_text_image_word_mask(keyword, i, font, font_size, color, max_width=w - 100)
        frame_clip = (ImageClip(img_array)
                      .set_start(frame_start)
                      .set_duration(word_duration)
                      .set_position("center"))
        clips.append(frame_clip)
    
    # Giữ text hoàn chỉnh đến hết
    final_start = start + len(words) * word_duration
    if final_start < end:
        final_img = create_text_image(keyword, font, font_size, color, max_width=w - 100)
        final_clip = (ImageClip(final_img)
                     .set_start(final_start)
                     .set_duration(end - final_start)
                     .set_position("center"))
        clips.append(final_clip)
    
    return clips


def create_text_clip_char_reveal(keyword, start, end, font, font_size, color, w, h):
    """
    Hiệu ứng char reveal: text CỐ ĐỊNH vị trí, từ từ hiển thị từng ký tự
    """
    duration = end - start
    
    if len(keyword) == 0:
        return []
    
    # Tốc độ hiển thị: chia đều thời gian cho mỗi ký tự
    char_duration = max(0.03, (duration / len(keyword)) * 0.4)
    clips = []
    
    # Tạo frame cho mỗi trạng thái hiển thị (1 char, 2 chars, 3 chars...)
    for i in range(1, len(keyword) + 1):
        frame_start = start + (i - 1) * char_duration
        
        # Tạo ảnh với i ký tự đầu tiên hiển thị, các ký tự còn lại ẩn
        img_array = create_text_image_char_mask(keyword, i, font, font_size, color, max_width=w - 100)
        frame_clip = (ImageClip(img_array)
                      .set_start(frame_start)
                      .set_duration(char_duration)
                      .set_position("center"))
        clips.append(frame_clip)
    
    # Giữ text hoàn chỉnh đến hết
    final_start = start + len(keyword) * char_duration
    if final_start < end:
        final_img = create_text_image(keyword, font, font_size, color, max_width=w - 100)
        final_clip = (ImageClip(final_img)
                     .set_start(final_start)
                     .set_duration(end - final_start)
                     .set_position("center"))
        clips.append(final_clip)
    
    return clips


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

    if effect_type == 4:
        return create_text_clip_typewriter(keyword, start, end, font, font_size, color, w, h)
    
    if effect_type == 5:
        return create_text_clip_word_reveal(keyword, start, end, font, font_size, color, w, h)
    
    if effect_type == 6:
        return create_text_clip_char_reveal(keyword, start, end, font, font_size, color, w, h)

    img_array = create_text_image(keyword, font, font_size, color, max_width=w - 100)
    txt_clip = ImageClip(img_array).set_duration(duration)

    if effect_type == 1:
        txt_clip = (txt_clip
                    .set_start(start)
                    .crossfadein(0.6)
                    .crossfadeout(0.6))
    
    elif effect_type == 2:
        def zoom_effect(t):
            progress = min(t / 0.6, 1)
            return 0.3 + 0.7 * (1 - (1 - progress) ** 2)
        txt_clip = (txt_clip
                    .set_start(start)
                    .resize(zoom_effect)
                    .crossfadein(0.6)
                    .crossfadeout(0.6))
    
    elif effect_type == 3:
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
    
    else:
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
    
    if bg_color is None or font_color is None:
        color_scheme = random.choice(COLOR_SCHEMES)
        bg_color = color_scheme["bg"]
        font_color = color_scheme["text"]
    
    if font_size is None:
        font_size = calculate_font_size(resolution)
    
    if effect_type is None:
        effect_type = random.randint(1, 6)
    
    bg_clip = ColorClip(size=resolution, color=bg_color, duration=duration)
    
    clips = []
    
    for i, (kw, st, et) in enumerate(zip(keywords, start_times, end_times)):
        text_clips = create_text_clip(kw, st, et, font, font_size, font_color, w, h, effect_type)
        clips.extend(text_clips)
    
    final_clip = CompositeVideoClip([bg_clip] + clips)
    final_clip.write_videofile(output_path, fps=30, codec="libx264", audio=False, threads=4)
    
    effect_names = {1: "Fade", 2: "Zoom", 3: "Bounce", 4: "Typewriter", 5: "Word Reveal", 6: "Char Reveal"}
    print(f"✅ Video đã được tạo: {output_path}")
    print(f"🎨 Màu nền: {bg_color}, Màu chữ: {font_color}")
    print(f"✨ Hiệu ứng: {effect_names.get(effect_type, 'Unknown')}")


# # ======================
# # 🎬 VÍ DỤ SỬ DỤNG
# # ======================

# # Test effect type 6 - Char Reveal
# keywords = ["Xin chào các bạn", "Hiệu ứng từng chữ", "Mượt mà và đẹp"]
# start_times = [0, 3, 6]
# end_times = [3, 6, 9]

# create_keyword_video(
#     keywords,
#     start_times,
#     end_times,
#     duration=9.67,
#     resolution=(720, 1280),
#     bg_color=(209, 213, 219),
#     font_color="#000000",
#     font="/content/MontserratMedium-lgZ6e.otf",
#     effect_type=5,  # Effect Type 6 - Char Reveal
#     output_path="char_reveal_demo.mp4"
# )