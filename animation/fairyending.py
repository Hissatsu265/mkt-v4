import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw

def auto_crop_product(image_path, output_path="cropped_product.png", margin=20, outline_thickness=5):
    """
    Tự động crop ảnh sản phẩm và thêm viền trắng mượt mà bao quanh sản phẩm.

    Args:
        image_path: Đường dẫn ảnh đầu vào
        output_path: Đường dẫn ảnh đầu ra
        margin: Khoảng cách padding xung quanh sản phẩm
        outline_thickness: Độ dày viền trắng (pixel)
    """
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img)

    # --- Tạo mask ---
    if arr.shape[2] == 4:  # Nếu có kênh alpha (ảnh xóa nền)
        alpha = arr[:, :, 3]
        mask = alpha > 0
    else:
        # Nếu không có alpha, tạo mask dựa theo độ sáng (giả sử nền trắng)
        gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        mask = gray < 250

    # --- Tìm bounding box ---
    coords = np.argwhere(mask)
    if coords.size == 0:
        print("❌ Không tìm thấy sản phẩm trong ảnh.")
        return

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    # --- Thêm margin cho cân đối ---
    y0 = max(y0 - margin, 0)
    x0 = max(x0 - margin, 0)
    y1 = min(y1 + margin, arr.shape[0])
    x1 = min(x1 + margin, arr.shape[1])

    # --- Crop vùng chứa sản phẩm ---
    cropped = arr[y0:y1, x0:x1]

    # --- TẠO VIỀN TRẮNG MƯỢT MÀ BAO QUANH SẢN PHẨM ---
    # Lấy mask của vùng crop
    if cropped.shape[2] == 4:
        crop_alpha = cropped[:, :, 3]
        crop_mask = (crop_alpha > 0).astype(np.uint8) * 255
    else:
        crop_gray = cv2.cvtColor(cropped[:, :, :3], cv2.COLOR_RGB2GRAY)
        crop_mask = (crop_gray < 250).astype(np.uint8) * 255

    # Làm mịn mask trước để giảm răng cưa
    crop_mask_smooth = cv2.GaussianBlur(crop_mask, (5, 5), 0)
    _, crop_mask_smooth = cv2.threshold(crop_mask_smooth, 127, 255, cv2.THRESH_BINARY)

    # Tìm contours của sản phẩm
    contours, _ = cv2.findContours(crop_mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Làm mịn contour bằng approxPolyDP
    smooth_contours = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)  # Giảm epsilon để giữ chi tiết
        approx = cv2.approxPolyDP(contour, epsilon, True)
        smooth_contours.append(approx)

    # Tạo ảnh lớn hơn để vẽ viền với anti-aliasing
    scale = 4  # Tăng kích thước lên 4 lần
    h, w = cropped.shape[:2]
    large_img = cv2.resize(cropped, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)

    # Scale contours lên theo tỷ lệ
    scaled_contours = [cnt * scale for cnt in smooth_contours]

    # Vẽ viền trắng trên ảnh phóng to
    cv2.drawContours(large_img, scaled_contours, -1, (255, 255, 255, 255),
                     outline_thickness * scale, lineType=cv2.LINE_AA)

    # Thu nhỏ lại về kích thước ban đầu với anti-aliasing
    cropped_with_outline = cv2.resize(large_img, (w, h), interpolation=cv2.INTER_AREA)

    # --- Căn giữa trên canvas vuông ---
    h, w = cropped_with_outline.shape[:2]
    size = max(h, w)
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = cropped_with_outline

    # --- Lưu kết quả ---
    Image.fromarray(canvas).save(output_path)
    # print(f"✅ Ảnh sản phẩm đã crop với viền trắng mượt mà: {output_path}")


# auto_crop_product(
#     "/content/Screenshot-2021-12-20-104958-Photoroom.png",
#     "product_cropped.png",
#     margin=15,
#     outline_thickness=10  # Độ dày viền
# )
class VideoProductOverlay:
    def __init__(self, video_path, image_path, output_path, duration, text_content, font_path=None, font_size=80):
        """
        Khởi tạo Video Product Overlay

        Args:
            video_path: Đường dẫn video nền
            image_path: Đường dẫn ảnh sản phẩm (PNG với nền trong suốt)
            output_path: Đường dẫn video output
            duration: Thời lượng video (giây)
            text_content: Nội dung text hiển thị
            font_path: Đường dẫn file font .ttf
            font_size: Cỡ chữ
        """
        self.video_path = video_path
        self.image_path = image_path
        self.output_path = output_path
        self.duration = duration
        self.text_content = text_content
        self.font_path = font_path
        self.font_size = font_size

        # Load video
        self.video = cv2.VideoCapture(video_path)
        self.video_fps = self.video.get(cv2.CAP_PROP_FPS)
        self.video_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Load image
        self.product_image = Image.open(image_path).convert('RGBA')

        # Detect aspect ratio
        self.aspect_ratio = self.video_width / self.video_height
        self.is_16x9 = self.aspect_ratio > 1.5
        self.is_9x16 = self.aspect_ratio < 0.7

        # Kiểm tra thời lượng video
        self.video_duration = self.video_frame_count / self.video_fps
        if self.duration > self.video_duration:
            print(f"⚠️ Cảnh báo: Duration ({self.duration}s) dài hơn video ({self.video_duration:.2f}s)")
            print(f"   Video sẽ được loop để đủ thời lượng")

        # Calculate timings
        self.move_in_duration = duration * 0.25      # Giai đoạn 1: Di chuyển vào
        self.zoom_duration = duration * 0.15         # Giai đoạn 1.5: Zoom to-nhỏ
        self.move_second_duration = duration * 0.15  # Giai đoạn 2: Di chuyển đến vị trí cuối
        self.text_display_duration = duration * 0.45 # Giai đoạn 3: Hiển thị text

        # Thời điểm bắt đầu các giai đoạn
        self.zoom_start = self.move_in_duration
        self.phase2_start = self.move_in_duration + self.zoom_duration
        self.phase3_start = self.phase2_start + self.move_second_duration

        # Text animation timing
        self.text_fade_in_duration = 0.3  # Thời gian fade in
        self.text_typing_speed = 0.05     # Giây mỗi ký tự (tốc độ đánh máy)

        # Tính toán thời gian typing
        text_length = len(self.text_content)
        self.text_typing_duration = text_length * self.text_typing_speed

        # Text fade out bắt đầu gần cuối video
        self.text_fade_out_start = self.duration - 0.5
        self.text_fade_out_duration = 0.5

        print(f"Video: {self.video_width}x{self.video_height} @ {self.video_fps} FPS")
        print(f"Video duration: {self.video_duration:.2f}s | Output duration: {self.duration}s")
        print(f"Aspect Ratio: {self.aspect_ratio:.2f}")
        print(f"Mode: {'16:9 (Landscape)' if self.is_16x9 else '9:16 (Portrait)' if self.is_9x16 else 'Other'}")
        print(f"Phase 1 (move in): 0s -> {self.zoom_start:.2f}s")
        print(f"Phase 1.5 (zoom pulse): {self.zoom_start:.2f}s -> {self.phase2_start:.2f}s")
        print(f"Phase 2 (move to final): {self.phase2_start:.2f}s -> {self.phase3_start:.2f}s")
        print(f"Phase 3 (wobble + text): {self.phase3_start:.2f}s -> {self.duration:.2f}s")
        print(f"Text typing duration: {self.text_typing_duration:.2f}s")

    def ease_in_out_cubic(self, t):
        """Hàm easing cho animation mượt mà"""
        return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

    def ease_out_back(self, t):
        """Hàm easing với hiệu ứng bounce nhẹ khi kết thúc"""
        c1 = 1.70158
        c3 = c1 + 1
        return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)

    def get_zoom_scale(self, current_time):
        """
        Tính toán scale cho hiệu ứng zoom to-nhỏ (pulse)

        Args:
            current_time: Thời gian hiện tại
        Returns:
            float: Scale factor (1.0 = kích thước gốc)
        """
        if current_time < self.zoom_start or current_time >= self.phase2_start:
            return 1.0

        # Thời gian trong giai đoạn zoom
        elapsed = current_time - self.zoom_start
        t = elapsed / self.zoom_duration

        # Tạo hiệu ứng zoom: to ra -> nhỏ lại -> về bình thường
        zoom_factor = 0.15  # Độ lớn của zoom (15% lớn hơn)
        scale = 1.0 + (math.sin(t * math.pi * 2) * zoom_factor * (1 - t))

        return scale

    def get_text_alpha_and_length(self, current_time):
        """
        Tính toán độ trong suốt và số ký tự hiển thị của text

        Args:
            current_time: Thời gian hiện tại
        Returns:
            tuple: (alpha, visible_chars)
                - alpha: 0.0 đến 1.0
                - visible_chars: số ký tự hiển thị
        """
        if current_time < self.phase3_start:
            return 0.0, 0

        elapsed = current_time - self.phase3_start
        text_length = len(self.text_content)

        # FADE IN
        if elapsed < self.text_fade_in_duration:
            alpha = elapsed / self.text_fade_in_duration
            alpha = self.ease_in_out_cubic(alpha)  # Smooth fade
            visible_chars = 0
            return alpha, visible_chars

        # TYPING EFFECT
        typing_elapsed = elapsed - self.text_fade_in_duration
        if typing_elapsed < self.text_typing_duration:
            alpha = 1.0
            visible_chars = int((typing_elapsed / self.text_typing_duration) * text_length)
            visible_chars = min(visible_chars, text_length)
            return alpha, visible_chars

        # FULLY VISIBLE
        if current_time < self.text_fade_out_start:
            return 1.0, text_length

        # FADE OUT
        fade_out_elapsed = current_time - self.text_fade_out_start
        if fade_out_elapsed < self.text_fade_out_duration:
            alpha = 1.0 - (fade_out_elapsed / self.text_fade_out_duration)
            alpha = self.ease_in_out_cubic(alpha)  # Smooth fade
            return alpha, text_length

        return 0.0, text_length

    def get_wobble_offset(self, current_time, start_time, fade_in_duration=0.3):
        """
        Tính toán độ lệch cho hiệu ứng nhấp nhô như sóng với fade-in mượt

        Args:
            current_time: Thời gian hiện tại
            start_time: Thời điểm bắt đầu wobble
            fade_in_duration: Thời gian fade-in của wobble (giây)
        """
        frequency = 1.5  # Hz - tần số sóng chậm
        amplitude = 8    # pixels - biên độ lên xuống

        elapsed = current_time - start_time
        if elapsed < 0:
            return 0, 0

        # Fade-in mượt cho wobble
        fade_multiplier = min(1.0, elapsed / fade_in_duration)
        fade_multiplier = self.ease_in_out_cubic(fade_multiplier)

        # Tính offset theo sin wave - CHỈ DI CHUYỂN THEO TRỤC Y
        angle = elapsed * frequency * 2 * math.pi
        offset_y = math.sin(angle) * amplitude * fade_multiplier

        offset_x = 0

        return int(offset_x), int(offset_y)

    def resize_product_image(self, scale=1.0):
        """
        Resize ảnh sản phẩm với scale factor

        Args:
            scale: Hệ số zoom (1.0 = kích thước gốc)
        """
        base_size = 512
        new_size = int(base_size * scale)
        return self.product_image.resize((new_size, new_size), Image.Resampling.LANCZOS)

    def get_image_position(self, current_time, img_w, img_h):
        """Tính toán vị trí ảnh dựa trên thời gian"""
        center_x = (self.video_width - img_w) // 2
        center_y = (self.video_height - img_h) // 2
        show_text = False
        wobble_x, wobble_y = 0, 0

        if self.is_16x9:
            # Animation 16:9: top → center → ZOOM → left + text
            final_x = center_x - int(self.video_width * 0.15)

            if current_time < self.zoom_start:
                # GIAI ĐOẠN 1: Di chuyển từ trên xuống giữa
                t = current_time / self.move_in_duration
                ease_t = self.ease_in_out_cubic(t)
                img_x = center_x
                img_y = int(-img_h + (center_y + img_h) * ease_t)

            elif current_time < self.phase2_start:
                # GIAI ĐOẠN 1.5: ZOOM TO-NHỎ tại center
                img_x = center_x
                img_y = center_y

            elif current_time < self.phase3_start:
                # GIAI ĐOẠN 2: Di chuyển sang trái
                t = (current_time - self.phase2_start) / self.move_second_duration
                ease_t = self.ease_out_back(t)
                img_x = int(center_x - (center_x - final_x) * ease_t)
                img_y = center_y

                # BẮT ĐẦU WOBBLE
                wobble_x, wobble_y = self.get_wobble_offset(current_time, self.phase2_start, fade_in_duration=0.5)

            else:
                # GIAI ĐOẠN 3: Vị trí cuối cùng với text + wobble
                img_x = final_x
                img_y = center_y
                show_text = True
                wobble_x, wobble_y = self.get_wobble_offset(current_time, self.phase2_start)

        elif self.is_9x16:
            # Animation 9:16: left → center → ZOOM → down + text
            final_y = center_y + int(self.video_height * 0.15)

            if current_time < self.zoom_start:
                # GIAI ĐOẠN 1: Di chuyển từ trái vào giữa
                t = current_time / self.move_in_duration
                ease_t = self.ease_in_out_cubic(t)
                img_x = int(-img_w + (center_x + img_w) * ease_t)
                img_y = center_y

            elif current_time < self.phase2_start:
                # GIAI ĐOẠN 1.5: ZOOM TO-NHỎ tại center
                img_x = center_x
                img_y = center_y

            elif current_time < self.phase3_start:
                # GIAI ĐOẠN 2: Di chuyển xuống dưới
                t = (current_time - self.phase2_start) / self.move_second_duration
                ease_t = self.ease_out_back(t)
                img_x = center_x
                img_y = int(center_y + (final_y - center_y) * ease_t)

                # BẮT ĐẦU WOBBLE
                wobble_x, wobble_y = self.get_wobble_offset(current_time, self.phase2_start, fade_in_duration=0.5)

            else:
                # GIAI ĐOẠN 3: Vị trí cuối cùng với text + wobble
                img_x = center_x
                img_y = final_y
                show_text = True
                wobble_x, wobble_y = self.get_wobble_offset(current_time, self.phase2_start)

        else:
            # Default: center
            img_x = center_x
            img_y = center_y
            if current_time > self.phase2_start:
                show_text = True
                wobble_x, wobble_y = self.get_wobble_offset(current_time, self.phase2_start)

        # Áp dụng hiệu ứng rung
        img_x += wobble_x
        img_y += wobble_y

        return img_x, img_y, show_text

    def overlay_image(self, background, overlay, x, y):
        """Chèn ảnh PNG có alpha lên background"""
        # Convert background to PIL
        bg_pil = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))

        # Paste overlay
        bg_pil.paste(overlay, (x, y), overlay)

        # Convert back to OpenCV
        return cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)

    def wrap_text(self, text, font, max_width):
        """Chia text thành nhiều dòng nếu quá dài"""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            # Tạo temporary draw để test kích thước
            temp_img = Image.new('RGB', (1, 1))
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), test_line, font=font)
            test_width = bbox[2] - bbox[0]

            if test_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def draw_text(self, frame, text, show_text, current_time):
        """Vẽ text lên frame với fade in/out và typing effect"""
        if not show_text or not text.strip():
            return frame

        # Lấy alpha và số ký tự hiển thị
        alpha, visible_chars = self.get_text_alpha_and_length(current_time)

        if alpha <= 0.0:
            return frame

        # Convert to PIL for text drawing
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Tạo layer trong suốt cho text
        text_layer = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)

        # Try to use custom font or fallback
        try:
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, self.font_size)
            else:
                font = ImageFont.truetype("arial.ttf", self.font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", self.font_size)
            except:
                font = ImageFont.load_default()

        # Calculate text area and position
        if self.is_16x9:
            text_area_x = int(self.video_width * 0.55)
            text_area_width = int(self.video_width * 0.4)
            text_y_center = self.video_height // 2
        elif self.is_9x16:
            text_area_x = int(self.video_width * 0.1)
            text_area_width = int(self.video_width * 0.8)
            text_y_center = int(self.video_height * 0.25)
        else:
            text_area_x = int(self.video_width * 0.1)
            text_area_width = int(self.video_width * 0.8)
            text_y_center = int(self.video_height * 0.8)

        # Lấy text hiển thị (typing effect)
        display_text = text[:visible_chars] if visible_chars > 0 else ""

        if not display_text:
            return frame

        # Wrap text nếu quá dài
        lines = self.wrap_text(display_text, font, text_area_width)

        # Calculate line height
        bbox = draw.textbbox((0, 0), "Test", font=font)
        line_height = bbox[3] - bbox[1] + 10

        # Calculate total text block height
        total_height = line_height * len(lines)

        # Starting Y position (centered)
        start_y = text_y_center - (total_height // 2)

        # Tính alpha cho màu
        alpha_int = int(255 * alpha)

        # Draw each line
        outline_width = 3
        for i, line in enumerate(lines):
            # Get line dimensions
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]

            # Center line horizontally
            text_x = text_area_x + (text_area_width - line_width) // 2
            text_y = start_y + i * line_height

            # Draw text outline với alpha
            for adj_x in range(-outline_width, outline_width + 1):
                for adj_y in range(-outline_width, outline_width + 1):
                    draw.text((text_x + adj_x, text_y + adj_y), line, font=font,
                             fill=(0, 0, 0, alpha_int))

            draw.text((text_x, text_y), line, font=font,
                     fill=(255, 255, 255, alpha_int))

        pil_img = pil_img.convert('RGBA')
        pil_img = Image.alpha_composite(pil_img, text_layer)
        pil_img = pil_img.convert('RGB')

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def process(self):

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.video_fps,
                             (self.video_width, self.video_height))

        total_frames = int(self.duration * self.video_fps)
        video_duration = self.video_frame_count / self.video_fps

        print(f"Processing {total_frames} frames...")

        for frame_idx in range(total_frames):
            # Calculate current time
            current_time = frame_idx / self.video_fps
            progress = (frame_idx / total_frames) * 100

            # Get video frame (loop if needed)
            video_time = (current_time / self.duration) * video_duration
            self.video.set(cv2.CAP_PROP_POS_MSEC, video_time * 1000)
            ret, frame = self.video.read()

            if not ret:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video.read()

            # Get zoom scale và resize image theo scale
            zoom_scale = self.get_zoom_scale(current_time)
            resized_product = self.resize_product_image(zoom_scale)
            img_w, img_h = resized_product.size

            # Get image position
            img_x, img_y, show_text = self.get_image_position(current_time, img_w, img_h)

            # Overlay product image
            frame = self.overlay_image(frame, resized_product, img_x, img_y)

            # Draw text với fade in/out và typing effect
            frame = self.draw_text(frame, self.text_content, show_text, current_time)

            # Write frame
            out.write(frame)

            if frame_idx % 30 == 0:
                print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")

        # Release resources
        self.video.release()
        out.release()

        print(f"\n✅ Video type 6 đã được tạo thành công: {self.output_path}")


def fairyending(VIDEO_FOLDER,FONT_FOLDER,IMAGE_PATH,OUTPUT_PATH,DURATION,FONT_SIZE):
    import os
    import random

    # === CONFIG PATHS ===
    # VIDEO_FOLDER = "/content/drive/MyDrive/1. Anymate me/16_9"   # Thư mục chứa nhiều video
    # FONT_FOLDER = "/content/drive/MyDrive/1. Anymate me/font"     # Thư mục chứa nhiều font
    # IMAGE_PATH = "/content/product_cropped (2).png"
    # OUTPUT_PATH = "output_video_smooth.mp4"
    # DURATION = 10
    # FONT_SIZE = 80

    # === RANDOM CHOICE ===
    def get_random_file(folder, extensions):
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                if any(f.lower().endswith(ext) for ext in extensions)]
        if not files:
            raise FileNotFoundError(f"No valid files found in {folder}")
        return random.choice(files)

    # Lấy video và font ngẫu nhiên
    VIDEO_PATH = get_random_file(VIDEO_FOLDER, [".mp4", ".mov", ".avi"])
    FONT_PATH = get_random_file(FONT_FOLDER, [".ttf", ".otf"])

    # === RANDOM SLOGAN LIST ===
    SLOGANS = [
        "Creating something amazing every day.",
        "Bringing ideas to life, one frame at a time.",
        "Where innovation meets inspiration.",
        "Make it shine. Make it unforgettable.",
        "Designed to inspire confidence.",
        "Crafted with passion. Built for you.",
        "Simple. Elegant. Powerful.",
        "Turning imagination into reality.",
        "Every detail matters."
    ]

    TEXT_CONTENT = random.choice(SLOGANS)

    # === PRINT CHECK ===
    # print(f"🎬 VIDEO_PATH: {VIDEO_PATH}")
    # print(f"🖋 FONT_PATH: {FONT_PATH}")
    # print(f"💬 TEXT_CONTENT: {TEXT_CONTENT}")

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Không tìm thấy video: {VIDEO_PATH}")
        return

    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Không tìm thấy ảnh: {IMAGE_PATH}")
        return
    import time

    filename = f"image_{int(time.time())}.png"
    auto_crop_product(
        IMAGE_PATH,
        filename,
        margin=15,
        outline_thickness=10  
    )
    overlay = VideoProductOverlay(
        video_path=VIDEO_PATH,
        image_path=filename,
        output_path=OUTPUT_PATH,
        duration=DURATION,
        text_content=TEXT_CONTENT,
        font_size=FONT_SIZE,
        font_path=FONT_PATH
    )

    overlay.process()
    return OUTPUT_PATH
