import os
import uuid
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.editor import ImageClip, CompositeVideoClip
import time
from config import BASE_DIR

def merge_videos(input1_path, input2_path,English=True):

    if not os.path.exists(input1_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input1_path}")
    if not os.path.exists(input2_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input2_path}")

    tmp_output = str(BASE_DIR)+"/outputs"+ f"/{uuid.uuid4().hex}.mp4"

    # Load video
    video1 = VideoFileClip(input1_path)
    video2 = VideoFileClip(input2_path)

    # Gộp video
    final = concatenate_videoclips([video1, video2])

    # Xuất ra file tạm
    final.write_videofile(tmp_output, codec="libx264", audio_codec="aac")

    # Giải phóng tài nguyên
    video1.close()
    video2.close()
    final.close()
    time.sleep(5)
    print("đã ghép vid intro xong")
    os.remove(input2_path)
# ======================add watermark==========================================
    if English:
        logo_input=str(BASE_DIR)+"/watermark/english.png"
    else:
        logo_input=str(BASE_DIR)+"/watermark/german.png"
    tmp_output1 = str(BASE_DIR)+"/outputs"+ f"/{uuid.uuid4().hex}.mp4"
    add_logo_to_video(
        video_path=tmp_output,
        logo_path=logo_input,
        output_path=tmp_output1,
        logo_size=(400, 100),  # Resize logo về 150x150
        margin=20
    )
# =======================================================
    os.rename(tmp_output1, input2_path)
    os.remove(tmp_output)


    print(f"✔ Kết quả final được lưu đè vào: {input2_path}")
 

def add_logo_to_video(video_path, logo_path, output_path, logo_size=None, margin=20):
    print("Đang đọc video...")
    video = VideoFileClip(video_path)
    
    # Đọc logo
    print("Đang đọc logo...")
    logo = ImageClip(logo_path).set_duration(video.duration)
    
    # Resize logo nếu cần
    if logo_size:
        logo = logo.resize(newsize=logo_size)
    
    # Tính toán vị trí góc dưới bên phải
    x_position = video.w - logo.w - margin
    y_position = video.h - logo.h - margin
    
    # Đặt vị trí cho logo
    logo = logo.set_position((x_position, y_position))
    
    # Ghép logo vào video
    print("Đang ghép logo vào video...")
    final_video = CompositeVideoClip([video, logo])
    
    # Xuất video
    print("Đang xuất video...")
    final_video.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        fps=video.fps
    )
    
    print(f"Hoàn thành! Video đã được lưu tại: {output_path}")
    
    video.close()
    final_video.close()
# merge_videos("/content/reindeer_northern1.mp4",
#              "/content/snowman_indoor.mp4")