from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont
import cv2

def apply_blur(image):
    """Áp dụng gaussian blur cho frame"""
    blurred = np.stack([
        gaussian_filter(image[:,:,i], sigma=10)
        for i in range(3)
    ], axis=2)
    # Giảm độ sáng để làm nền
    return (blurred * 0.5).astype('uint8')


def get_optimal_font_size(width, height, font_path=None):
    """
    Tự động tính toán font size phù hợp dựa trên kích thước video

    Returns:
    - font_size: kích thước font phù hợp
    """
    # Xác định orientation
    if width > height:  # Landscape 16:9
        # Font size dựa trên chiều cao (khoảng 8-10% chiều cao)
        font_size = int(height * 0.12)
    else:  # Portrait 9:16
        # Font size dựa trên chiều rộng (khoảng 12-15% chiều rộng)
        font_size = int(width * 0.15)

    return font_size


def wrap_text(text, font, max_width, draw):
    """
    Tự động xuống hàng cho text nếu quá dài

    Parameters:
    - text: nội dung text
    - font: font object
    - max_width: chiều rộng tối đa cho phép
    - draw: ImageDraw object để tính toán kích thước

    Returns:
    - lines: list các dòng text sau khi wrap
    """
    words = text.split(' ')
    lines = []
    current_line = []

    for word in words:
        # Thử thêm từ vào dòng hiện tại
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = bbox[2] - bbox[0]

        if test_width <= max_width:
            current_line.append(word)
        else:
            # Nếu dòng hiện tại không rỗng, lưu lại và bắt đầu dòng mới
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                # Trường hợp từ đơn lẻ quá dài, vẫn phải thêm vào
                lines.append(word)
                current_line = []

    # Thêm dòng cuối cùng
    if current_line:
        lines.append(' '.join(current_line))

    return lines


def create_text_clip(text, duration, size, start_time, font_path=None):
    """
    Tạo clip text bằng PIL với font size tự động điều chỉnh và auto wrap

    Parameters:
    - text: nội dung text
    - duration: thời lượng hiển thị
    - size: (width, height) của video
    - start_time: thời điểm bắt đầu
    - font_path: đường dẫn đến file .ttf (nếu None sẽ dùng font mặc định)
    """
    width, height = size

    # Tính toán font size phù hợp
    font_size = get_optimal_font_size(width, height)

    # Tính stroke width dựa trên font size
    stroke_width = max(3, int(font_size * 0.06))

    # Tính max width cho text (90% chiều rộng video để có padding)
    max_text_width = int(width * 0.9)

    def make_frame(t):
        # Tạo frame trong suốt với 3 channels (RGB)
        img = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Tải font
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                # Thử các font mặc định
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

        # Wrap text nếu cần
        lines = wrap_text(text, font, max_text_width, draw)

        # Tính toán kích thước tổng của text block (nhiều dòng)
        line_heights = []
        line_widths = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])

        # Tính line spacing (khoảng cách giữa các dòng)
        line_spacing = int(font_size * 0.2)

        # Tổng chiều cao của text block
        total_height = sum(line_heights) + line_spacing * (len(lines) - 1)

        # Vị trí bắt đầu y (căn giữa theo chiều dọc)
        start_y = (height - total_height) // 2

        # Vẽ từng dòng
        current_y = start_y
        for i, line in enumerate(lines):
            # Tính vị trí x để căn giữa dòng này
            x = (width - line_widths[i]) // 2
            y = current_y

            # Vẽ viền đen (stroke)
            for adj_x in range(-stroke_width, stroke_width+1):
                for adj_y in range(-stroke_width, stroke_width+1):
                    draw.text((x+adj_x, y+adj_y), line, font=font, fill=(0, 0, 0))

            # Vẽ text trắng
            draw.text((x, y), line, font=font, fill=(255, 255, 255))

            # Di chuyển đến dòng tiếp theo
            current_y += line_heights[i] + line_spacing

        # Convert sang numpy array RGB
        return np.array(img)

    def make_mask(t):
        # Tạo alpha mask - chỉ text có màu trắng, phần còn lại đen
        img = Image.new('L', (width, height), 0)  # L = grayscale
        draw = ImageDraw.Draw(img)

        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

        # Wrap text
        lines = wrap_text(text, font, max_text_width, draw)

        # Tính toán layout
        line_heights = []
        line_widths = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])

        line_spacing = int(font_size * 0.2)
        total_height = sum(line_heights) + line_spacing * (len(lines) - 1)
        start_y = (height - total_height) // 2

        # Vẽ từng dòng
        current_y = start_y
        for i, line in enumerate(lines):
            x = (width - line_widths[i]) // 2
            y = current_y

            # Vẽ viền
            for adj_x in range(-stroke_width, stroke_width+1):
                for adj_y in range(-stroke_width, stroke_width+1):
                    draw.text((x+adj_x, y+adj_y), line, font=font, fill=255)

            # Vẽ text
            draw.text((x, y), line, font=font, fill=255)

            current_y += line_heights[i] + line_spacing

        return np.array(img) / 255.0  # Normalize 0-1

    from moviepy.editor import VideoClip
    text_clip = VideoClip(make_frame, duration=duration)
    text_clip = text_clip.set_start(start_time)
    text_clip = text_clip.set_mask(VideoClip(make_mask, duration=duration, ismask=True))

    return text_clip


def create_keyword_videoblur(video_path, keywords, start_times, end_times, output_path="output_video.mp4", font_path=None):
    """
    Tạo video với hiệu ứng blur và hiển thị keyword

    Parameters:
    - video_path: đường dẫn đến video gốc
    - keywords: list các từ khóa cần hiển thị
    - start_times: list thời điểm bắt đầu hiển thị (giây)
    - end_times: list thời điểm kết thúc hiển thị (giây)
    - output_path: đường dẫn lưu video output
    - font_path: đường dẫn đến file font .ttf (None = dùng font mặc định)
    """

    print("Đang load video...")
    video = VideoFileClip(video_path)
    w, h = video.size

    # Hiển thị thông tin video
    print(f"📐 Kích thước video: {w}x{h}")
    if w > h:
        print("📺 Orientation: Landscape (16:9)")
        font_size = get_optimal_font_size(w, h)
        print(f"✏️  Font size: {font_size}px")
    else:
        print("📱 Orientation: Portrait (9:16)")
        font_size = get_optimal_font_size(w, h)
        print(f"✏️  Font size: {font_size}px")

    if font_path:
        print(f"🔤 Font: {font_path}")
    else:
        print("🔤 Font: System default (DejaVuSans-Bold)")

    # Tạo danh sách các khoảng thời gian cần blur
    blur_times = []

    for i in range(len(start_times)):
        start = start_times[i]
        end = end_times[i]

        blur_start = start - 0.5

        # Kiểm tra xem có keyword tiếp theo không
        if i < len(start_times) - 1:
            next_start = start_times[i + 1]
            gap = next_start - end

            # Nếu khoảng cách < 1s, giữ blur đến keyword tiếp theo
            if gap < 1:
                blur_end = next_start - 0.5
            else:
                # Nếu khoảng cách > 2s, fade out blur
                blur_end = end
        else:
            blur_end = end

        blur_times.append((blur_start, blur_end))

    # Hợp nhất các segment blur chồng chéo
    merged_blur_times = merge_segments(blur_times)

    print("Đang tạo hiệu ứng blur động...")

    # Tạo function để áp dụng blur có điều kiện theo thời gian
    def apply_conditional_blur(get_frame, t):
        frame = get_frame(t)

        # Kiểm tra xem thời điểm t có nằm trong khoảng blur không
        should_blur = False
        fade_in = False
        fade_out = False

        for blur_start, blur_end in merged_blur_times:
            if blur_start <= t <= blur_end:
                should_blur = True

                # Kiểm tra fade in (0.5s đầu)
                if t - blur_start < 0.5:
                    fade_in = True
                    fade_factor = (t - blur_start) / 0.5

                # Kiểm tra fade out (0.5s cuối) - chỉ khi không có keyword tiếp theo gần
                # Tìm keyword tiếp theo
                has_next_nearby = False
                for next_blur_start, _ in merged_blur_times:
                    if next_blur_start > blur_end and next_blur_start - blur_end < 2:
                        has_next_nearby = True
                        break

                if not has_next_nearby and blur_end - t < 0.5:
                    fade_out = True
                    fade_factor = (blur_end - t) / 0.5

                break

        if should_blur:
            # Áp dụng blur
            blurred = np.stack([
                gaussian_filter(frame[:,:,i], sigma=10)
                for i in range(3)
            ], axis=2)
            blurred = (blurred * 0.5).astype('uint8')

            # Áp dụng fade nếu cần
            if fade_in:
                frame = (frame * (1 - fade_factor) + blurred * fade_factor).astype('uint8')
            elif fade_out:
                frame = (blurred * fade_factor + frame * (1 - fade_factor)).astype('uint8')
            else:
                frame = blurred

        return frame

    print("Đang áp dụng blur...")
    # Áp dụng blur có điều kiện cho toàn bộ video
    final_video = video.fl(apply_conditional_blur)

    print("Đang thêm text...")
    # Thêm text keywords
    text_clips = []
    for i, keyword in enumerate(keywords):
        duration = end_times[i] - start_times[i]
        txt_clip = create_text_clip(keyword, duration, (w, h), start_times[i], font_path)
        text_clips.append(txt_clip)

    # Composite video với text
    final_with_text = CompositeVideoClip([final_video] + text_clips)

    print("Đang export video...")
    # Export
    final_with_text.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        fps=video.fps
    )

    # Đóng các clip
    video.close()
    final_with_text.close()

    print(f"✅ Video đã được lưu tại: {output_path}")


def merge_segments(segments):
    """Hợp nhất các segment thời gian chồng chéo"""
    if not segments:
        return []

    # Sắp xếp theo thời gian bắt đầu
    sorted_segments = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segments[0]]

    for current in sorted_segments[1:]:
        last = merged[-1]
        # Nếu overlap hoặc gần nhau (< 1s)
        if current[0] <= last[1] + 1:
            # Hợp nhất
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged


# if __name__ == "__main__":
#     # Thông tin input
#     video_path = "/content/vid1.mp4"  # Đường dẫn video của bạn

#     # Đường dẫn font (để None nếu dùng font mặc định)
#     # Ví dụ: font_path = "/path/to/your/font.ttf"
#     font_path = "/content/WixMadeforDisplay-VariableFont_wght.ttf"

#     # Danh sách keywords và thời gian
#     keywords = [
#         # "Keyworrwerwerw ewrwer",
#         # "Keyword 2",
#     ]

#     start_times = []  # Thời điểm bắt đầu (giây)
#     end_times = []   # Thời điểm kết thúc (giây)

#     # Tạo video
#     create_keyword_video(
#         video_path=video_path,
#         keywords=keywords,
#         start_times=start_times,
#         end_times=end_times,
#         output_path="output_video1.mp4",
#         font_path=font_path  # Thêm parameter font_path
#     )