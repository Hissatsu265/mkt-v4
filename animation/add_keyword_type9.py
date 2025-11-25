from moviepy.editor import VideoFileClip, CompositeVideoClip
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoClip

def get_optimal_font_size(width, height, font_path=None):
    """
    Tự động tính toán font size phù hợp dựa trên kích thước video
    """
    if width > height:  # Landscape 16:9
        font_size = int(height * 0.12)
    else:  # Portrait 9:16
        # Giảm xuống để tránh chữ quá to
        font_size = int(width * 0.12)
    
    return font_size


def wrap_text_smart(text, font, max_width, draw):
    """
    Wrap text thông minh: tự động chia nhỏ từ dài nếu cần thiết
    
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
            # Kiểm tra xem từ đơn lẻ có quá dài không
            word_bbox = draw.textbbox((0, 0), word, font=font)
            word_width = word_bbox[2] - word_bbox[0]
            
            if word_width > max_width:
                # Lưu dòng hiện tại nếu có
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = []
                
                # Chia nhỏ từ dài thành nhiều dòng
                char_width_avg = word_width / len(word)
                chars_per_line = int(max_width / char_width_avg) - 1
                
                for i in range(0, len(word), chars_per_line):
                    chunk = word[i:i + chars_per_line]
                    # Thêm dấu gạch ngang nếu không phải chunk cuối
                    if i + chars_per_line < len(word):
                        lines.append(chunk + '-')
                    else:
                        # Chunk cuối cùng: nếu còn từ tiếp theo thì để riêng, không thì thêm luôn
                        current_line = [chunk]
            else:
                # Từ không quá dài, lưu dòng hiện tại và bắt đầu dòng mới
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
    
    # Thêm dòng cuối cùng
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def create_text_clip(text, duration, size, start_time, video_clip, font_path=None):
    """
    Tạo clip text xuất hiện từng chữ trên video gốc,
    căn xuống dưới bên trái, xuất hiện nhanh (~0.4s cho 1 keyword)
    """
    width, height = size
    font_size = get_optimal_font_size(width, height)
    stroke_width = max(3, int(font_size * 0.08))
    max_text_width = int(width * 0.85)

    # Pre-load font và tính toán lines 1 lần duy nhất (cache)
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Tính toán lines trước (chỉ 1 lần)
    temp_img = Image.new('RGB', (width, height))
    temp_draw = ImageDraw.Draw(temp_img)
    lines = wrap_text_smart(text, font, max_text_width, temp_draw)
    
    # Tính line heights trước
    line_heights = [temp_draw.textbbox((0,0), line, font=font)[3] - 
                   temp_draw.textbbox((0,0), line, font=font)[1] for line in lines]
    line_spacing = int(font_size * 0.3)
    total_height = sum(line_heights) + line_spacing * (len(lines)-1)
    
    # Tính vị trí trước
    start_x = int(width * 0.08)
    start_y = height - total_height - int(height * 0.25)
    total_chars = sum(len(line) for line in lines)

    def make_frame(t):
        # Lấy frame gốc từ video tại thời điểm t + start_time
        video_time = t + start_time
        frame = video_clip.get_frame(video_time)
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # Tính số chữ hiển thị theo thời gian
        chars_to_show = min(int(total_chars * (t / 0.4)), total_chars)

        # Vẽ text
        shown_chars = 0
        current_y = start_y
        
        for i, line in enumerate(lines):
            if shown_chars >= chars_to_show:
                break
                
            # Tính text cần hiển thị trong dòng này
            chars_in_line = min(chars_to_show - shown_chars, len(line))
            visible_text = line[:chars_in_line]
            
            x = start_x
            y = current_y
            
            # Vẽ viền đen với stroke tối ưu
            for adj_x in range(-stroke_width, stroke_width + 1):
                for adj_y in range(-stroke_width, stroke_width + 1):
                    if adj_x*adj_x + adj_y*adj_y <= stroke_width*stroke_width:  # Circular stroke
                        draw.text((x + adj_x, y + adj_y), visible_text, font=font, fill=(0, 0, 0))
            
            # Vẽ chữ trắng
            draw.text((x, y), visible_text, font=font, fill=(255, 255, 255))
            
            shown_chars += chars_in_line
            current_y += line_heights[i] + line_spacing

        return np.array(img)

    # Trả về VideoClip với frame từ video gốc
    text_clip = VideoClip(make_frame, duration=duration)
    text_clip = text_clip.set_start(start_time)
    return text_clip



def create_keyword_video_noblur(video_path, keywords, start_times, end_times, output_path="output_video.mp4", font_path=None):
    """
    Tạo video với hiển thị keyword (KHÔNG CÓ BLUR)

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

    print("Đang thêm text keywords...")
    # Thêm text keywords - truyền video_clip vào
    text_clips = []
    for i, keyword in enumerate(keywords):
        duration = end_times[i] - start_times[i]
        txt_clip = create_text_clip(keyword, duration, (w, h), start_times[i], video, font_path)
        text_clips.append(txt_clip)

    # Composite video với text (KHÔNG CÓ BLUR)
    final_with_text = CompositeVideoClip([video] + text_clips)

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


# if __name__ == "__main__":
#     # Thông tin input
#     video_path = "/content/output_zoom_light_sang.mp4"  # Đường dẫn video của bạn

#     # Đường dẫn font (để None nếu dùng font mặc định)
#     font_path = "/content/WixMadeforDisplay-VariableFont_wght.ttf"

#     # Danh sách keywords và thời gian
#     keywords = [
#         "Keyworrwerwerw ewrwer",
#         "Keyword 2",
#     ]

#     start_times = [1, 2]  # Thời điểm bắt đầu (giây)
#     end_times = [2, 7]   # Thời điểm kết thúc (giây)

    # Tạo video
    # create_keyword_video_noblur(
    #     video_path=video_path,
    #     keywords=keywords,
    #     start_times=start_times,
    #     end_times=end_times,
    #     output_path="output_video.mp4",
    #     font_path=font_path
    # )