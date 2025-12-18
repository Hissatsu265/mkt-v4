# import os
# import uuid
# from moviepy.editor import VideoFileClip, concatenate_videoclips
# from moviepy.editor import ImageClip, CompositeVideoClip
# import time
# from config import BASE_DIR

# def merge_videos(input1_path, input2_path,English=True):

#     if not os.path.exists(input1_path):
#         raise FileNotFoundError(f"Không tìm thấy file: {input1_path}")
#     if not os.path.exists(input2_path):
#         raise FileNotFoundError(f"Không tìm thấy file: {input2_path}")

#     tmp_output = str(BASE_DIR)+"/outputs"+ f"/{uuid.uuid4().hex}.mp4"
#     print("Đang ghép vid intro...")
#     # Load video
#     video1 = VideoFileClip(input1_path)
#     video2 = VideoFileClip(input2_path)
#     print("Đã load xong vid intro")
#     # Gộp video
#     final = concatenate_videoclips([video1, video2])
#     print("Đang xuất vid intro...")
#     # Xuất ra file tạm
#     final.write_videofile(tmp_output, codec="libx264", audio_codec="aac")
#     print("Đã xuất xong vid intro")
#     # Giải phóng tài nguyên
#     video1.close()
#     video2.close()
#     final.close()
#     time.sleep(5)
#     print("đã ghép vid intro xong")
#     os.remove(input2_path)
# # ======================add watermark==========================================
#     if English:
#         logo_input=str(BASE_DIR)+"/watermark/english.png"
#     else:
#         logo_input=str(BASE_DIR)+"/watermark/german.png"
#     tmp_output1 = str(BASE_DIR)+"/outputs"+ f"/{uuid.uuid4().hex}1234234.mp4"
#     add_logo_to_video(
#         video_path=tmp_output,
#         logo_path=logo_input,
#         output_path=tmp_output1,
#         logo_size=(400, 100),  # Resize logo về 150x150
#         margin=20
#     )
# # =======================================================
#     os.rename(tmp_output1, input2_path)
#     os.remove(tmp_output)


#     print(f"✔ Kết quả final được lưu đè vào: {input2_path}")
 

# def add_logo_to_video(video_path, logo_path, output_path, logo_size=None, margin=20):
#     print("Đang đọc video...")
#     video = VideoFileClip(video_path)
    
#     # Đọc logo
#     print("Đang đọc logo...")
#     logo = ImageClip(logo_path).set_duration(video.duration)
    
#     # Resize logo nếu cần
#     if logo_size:
#         logo = logo.resize(newsize=logo_size)
    
#     # Tính toán vị trí góc dưới bên phải
#     x_position = video.w - logo.w - margin
#     y_position = video.h - logo.h - margin
    
#     # Đặt vị trí cho logo
#     logo = logo.set_position((x_position, y_position))
    
#     # Ghép logo vào video
#     print("Đang ghép logo vào video...")
#     final_video = CompositeVideoClip([video, logo])
    
#     # Xuất video
#     print("Đang xuất video...")
#     final_video.write_videofile(
#         output_path,
#         codec='libx264',
#         audio_codec='aac',
#         fps=video.fps
#     )
    
#     print(f"Hoàn thành! Video đã được lưu tại: {output_path}")
    
#     video.close()
#     final_video.close()
# # merge_videos("/content/reindeer_northern1.mp4",
# #              "/content/snowman_indoor.mp4")
import os
import uuid
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.editor import ImageClip, CompositeVideoClip
import time
from config import BASE_DIR

def merge_videos(input1_path, input2_path, English=True):
    if not os.path.exists(input1_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input1_path}")
    if not os.path.exists(input2_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input2_path}")

    tmp_output = str(BASE_DIR) + "/outputs" + f"/{uuid.uuid4().hex}.mp4"
    print("Đang ghép vid intro...")
    
    # Load video
    video1 = VideoFileClip(input1_path)
    video2 = VideoFileClip(input2_path)
    print("Đã load xong vid intro")
    
    # FIX 1: Chuẩn hóa FPS về 25 fps
    target_fps = 25
    if video1.fps != target_fps:
        print(f"Chuyển video1 từ {video1.fps} fps sang {target_fps} fps")
        video1 = video1.set_fps(target_fps)
    if video2.fps != target_fps:
        print(f"Chuyển video2 từ {video2.fps} fps sang {target_fps} fps")
        video2 = video2.set_fps(target_fps)
    
    # FIX 2: Xử lý trường hợp video không có audio
    # Kiểm tra xem video có audio không
    has_audio1 = video1.audio is not None
    has_audio2 = video2.audio is not None
    
    print(f"Video1 có audio: {has_audio1}, Video2 có audio: {has_audio2}")
    
    # Nếu video1 không có audio nhưng video2 có, tạo silent audio cho video1
    if not has_audio1 and has_audio2:
        print("Thêm silent audio vào video1...")
        from moviepy.audio.AudioClip import AudioClip
        silent_audio = AudioClip(lambda t: [0, 0], duration=video1.duration, fps=44100)
        video1 = video1.set_audio(silent_audio)
    
    # Nếu video2 không có audio nhưng video1 có, tạo silent audio cho video2
    if has_audio1 and not has_audio2:
        print("Thêm silent audio vào video2...")
        from moviepy.audio.AudioClip import AudioClip
        silent_audio = AudioClip(lambda t: [0, 0], duration=video2.duration, fps=44100)
        video2 = video2.set_audio(silent_audio)
    
    # Gộp video
    final = concatenate_videoclips([video1, video2], method="compose")
    print("Đang xuất vid intro...")
    
    # Xuất ra file tạm với FPS cố định
    final.write_videofile(
        tmp_output, 
        codec="libx264", 
        audio_codec="aac",
        fps=target_fps,
        preset='medium',
        threads=4
    )
    print("Đã xuất xong vid intro")
    
    # Giải phóng tài nguyên
    video1.close()
    video2.close()
    final.close()
    time.sleep(5)
    print("đã ghép vid intro xong")
    os.remove(input2_path)
    
    # ======================add watermark==========================================
    if English:
        logo_input = str(BASE_DIR) + "/watermark/english.png"
    else:
        logo_input = str(BASE_DIR) + "/watermark/german.png"
    
    tmp_output1 = str(BASE_DIR) + "/outputs" + f"/{uuid.uuid4().hex}1234234.mp4"
    add_logo_to_video(
        video_path=tmp_output,
        logo_path=logo_input,
        output_path=tmp_output1,
        logo_size=(400, 100),
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
        fps=video.fps,
        preset='medium'
    )
    
    print(f"Hoàn thành! Video đã được lưu tại: {output_path}")
    
    video.close()
    final_video.close()