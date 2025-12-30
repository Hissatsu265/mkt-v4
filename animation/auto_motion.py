import os
import subprocess
import tempfile
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
from utilities.ffmpeg_wrapper import run_ffmpeg_command

def check_ffmpeg():
    from shutil import which
    if which("ffmpeg") is None:
        raise RuntimeError("Không tìm thấy ffmpeg. Cài ffmpeg và đảm bảo nó có trong PATH (ví dụ: apt install ffmpeg trên Ubuntu).")

def transcode_to_mp4(input_path, out_path):
    """Chuyển sang mp4 chuẩn h264/aac để MoviePy đọc ổn định."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        out_path
    ]
    result = run_ffmpeg_command(cmd, timeout=300, retry_count=2, log_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg transcode failed: {result.stderr}")

def ffmpeg_reverse(input_path, output_path):
    """Tạo file đảo thời gian (video + audio) bằng ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "reverse",
        "-af", "areverse",
        output_path
    ]
    result = run_ffmpeg_command(cmd, timeout=600, retry_count=2)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg reverse failed: {result.stderr}")

def parse_resolution(res_str):
    """Chuyển '1280x720' → (1280, 720)"""
    try:
        w, h = map(int, res_str.lower().split("x"))
        return w, h
    except:
        raise ValueError("Định dạng kích thước không hợp lệ. Ví dụ: '1280x720' hoặc '720x1280'")

def extend_video(input_path, output_path, target_duration, mode="pingpong", resolution=None, temp_dir=None):
    """
    Kéo dài video đến đúng target_duration giây và resize về kích thước mong muốn (nếu có).
    mode: "pingpong", "loop", "slow"
    resolution: chuỗi "1280x720" hoặc "720x1280"
    """
    check_ffmpeg()
    tmpdir = temp_dir or tempfile.mkdtemp(prefix="extend_vid_")

    try:
        # --- Load video ---
        try:
            clip = VideoFileClip(input_path)
        except Exception:
            print("Không thể load file trực tiếp bằng MoviePy, sẽ transcode sang mp4 chuẩn...")
            trans_path = os.path.join(tmpdir, "transcoded_input.mp4")
            transcode_to_mp4(input_path, trans_path)
            clip = VideoFileClip(trans_path)

        original_duration = clip.duration
        print(f"🎞 Original: {original_duration:.3f}s | Target: {target_duration:.3f}s")

        # --- Resize nếu có yêu cầu ---
        if resolution:
            w, h = parse_resolution(resolution)
            clip = clip.fx(vfx.resize, newsize=(w, h))
            print(f"📐 Đã resize video về {w}x{h}")

        # --- Nếu video dài hơn target ---
        if original_duration >= target_duration:
            final = clip.subclip(0, target_duration)
            final.write_videofile(output_path, codec="libx264", audio_codec="aac")
            print("✂️ Đã cắt video ngắn hơn target và xuất xong.")
            return

        # --- Chế độ slow motion ---
        if mode == "slow":
            speed_factor = original_duration / target_duration
            final = clip.fx(vfx.speedx, speed_factor)
            final.write_videofile(output_path, codec="libx264", audio_codec="aac")
            print("🐢 Đã làm chậm video và xuất xong.")
            return

        # --- Chế độ loop / pingpong ---
        if mode in ("pingpong", "loop"):
            if mode == "loop":
                repeat_times = int(target_duration // original_duration) + 2
                extended = concatenate_videoclips([clip] * repeat_times)
                final = extended.subclip(0, target_duration)
                final.write_videofile(output_path, codec="libx264", audio_codec="aac")
                print("🔁 Đã lặp (loop) và xuất xong.")
                return

            # --- Pingpong ---
            try:
                reversed_clip = clip.fx(vfx.time_mirror)
                combined = concatenate_videoclips([clip, reversed_clip])
                repeat_times = int(target_duration // combined.duration) + 2
                extended = concatenate_videoclips([combined] * repeat_times)
                final = extended.subclip(0, target_duration)
                final.write_videofile(output_path, codec="libx264", audio_codec="aac")
                print("🏓 Đã pingpong bằng MoviePy và xuất xong.")
                return
            except Exception:
                print("⚠️ Pingpong bằng MoviePy lỗi. Dùng ffmpeg fallback...")
                tr_in = os.path.join(tmpdir, "trans_in.mp4")
                transcode_to_mp4(input_path, tr_in)
                reversed_path = os.path.join(tmpdir, "reversed.mp4")
                ffmpeg_reverse(tr_in, reversed_path)

                clip_a = VideoFileClip(tr_in)
                clip_b = VideoFileClip(reversed_path)
                if resolution:
                    clip_a = clip_a.fx(vfx.resize, newsize=(w, h))
                    clip_b = clip_b.fx(vfx.resize, newsize=(w, h))
                combined = concatenate_videoclips([clip_a, clip_b])
                repeat_times = int(target_duration // combined.duration) + 2
                extended = concatenate_videoclips([combined] * repeat_times)
                final = extended.subclip(0, target_duration)
                final.write_videofile(output_path, codec="libx264", audio_codec="aac")
                print("✅ Đã pingpong bằng ffmpeg fallback và xuất xong.")
                return

        raise ValueError("Mode không hợp lệ: 'pingpong', 'loop', hoặc 'slow'")

    finally:
      
        pass