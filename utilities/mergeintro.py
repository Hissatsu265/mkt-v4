import os
import uuid
from moviepy.editor import VideoFileClip, concatenate_videoclips
from config import BASE_DIR

def merge_videos(input1_path, input2_path):

    if not os.path.exists(input1_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input1_path}")
    print("sdfsdf")
    if not os.path.exists(input2_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input2_path}")
    print("sdfsdf")

    tmp_output = str(BASE_DIR)+"/outputs"+ f"/{uuid.uuid4().hex}.mp4"
    print("sdfsdf")

    # Load video
    video1 = VideoFileClip(input1_path)
    video2 = VideoFileClip(input2_path)
    print("sdfsdf")

    # Gộp video
    final = concatenate_videoclips([video1, video2])

    # Xuất ra file tạm
    final.write_videofile(tmp_output, codec="libx264", audio_codec="aac")
    print("sdfsdf")

    # Giải phóng tài nguyên
    video1.close()
    video2.close()
    final.close()

    # Lấy tên file input2 (giữ nguyên tên)
    final_name = os.path.basename(input2_path)

    # Xóa file input 2
    os.remove(input2_path)

    # Đổi tên file tạm → thành tên file input2
    os.rename(tmp_output, input2_path)

    print(f"✔ Kết quả final được lưu đè vào: {final_name}")
    print(input2_path)
    if os.path.exists(input2_path):
        print("File tồn tại")
    else:
        print("File không tồn tại")

# merge_videos("/content/reindeer_northern1.mp4",
#              "/content/snowman_indoor.mp4")