from moviepy.editor import VideoFileClip, AudioFileClip

# def replace_audio_trimmed(video_path, audio_path, output_path):
#     video = VideoFileClip(str(video_path))
#     audio = AudioFileClip(str(audio_path))
#     min_duration = min(video.duration, audio.duration)
#     video = video.subclip(0, min_duration)
#     audio = audio.subclip(0, min_duration)
#     final = video.set_audio(audio)
#     final.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
#     return str(output_path)

def replace_audio_trimmed(video_path, audio_path, output_path):
    video = VideoFileClip(str(video_path))
    audio = AudioFileClip(str(audio_path))
    
    w, h = video.size

    if w > h:  # Landscape
        target_w, target_h = 1280, 720
    else:     
        target_w, target_h = 720, 1280

    if (w, h) != (target_w, target_h):
        x_center, y_center = w / 2, h / 2
        x1 = x_center - target_w / 2
        x2 = x_center + target_w / 2
        y1 = y_center - target_h / 2
        y2 = y_center + target_h / 2

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        video = video.crop(x1=x1, y1=y1, x2=x2, y2=y2)

    min_duration = min(video.duration, audio.duration)
    video = video.subclip(0, min_duration)
    audio = audio.subclip(0, min_duration)
    
    final = video.set_audio(audio)
    
    final.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

    return str(output_path)

# if __name__ == "__main__":
#     video_path = "/workspace/multitalk_verquant/merged_videoddd.mp4"
#     audio_path = "/workspace/multitalk_verquant/audio/audio_rs_demo_side_view.mp3"
#     output_path = "output_video_with_audioddd.mp4"
#     result_path = replace_audio_trimmed(video_path, audio_path, output_path)
#     print("Video with replaced audio created at:", result_path)