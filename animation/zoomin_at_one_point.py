import subprocess
import json

def apply_zoom_effect_fast(
    input_path,
    output_path="output_zoomed.mp4",
    zoom_duration=0.75,
    zoom_start_time=1.0,
    zoom_percent=0.5,
    center=(200, 500),
    end_effect=None,
    remove_mode="instant",
    crf=18
):

    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,duration,r_frame_rate',
        '-of', 'json',
        input_path
    ]
    
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(result.stdout)
        stream = video_info['streams'][0]
        
        W = int(stream['width'])
        H = int(stream['height'])
        
        # Parse frame rate
        fps_parts = stream['r_frame_rate'].split('/')
        fps = int(fps_parts[0]) / int(fps_parts[1])
        
        duration = float(stream.get('duration', 0))
        
    except Exception as e:
        print(f"[ERROR] Failed to read video info: {e}")
        return
    
    cx, cy = center
    
    # Calculate zoom scale
    zoom_scale = 1 / zoom_percent  # 0.5 = zoom 2x
    
    # Calculate start and end frames
    start_frame = int(zoom_start_time * fps)
    zoom_frames = int(zoom_duration * fps)
    end_frame = start_frame + zoom_frames
    
    # Calculate crop position (normalized 0–1)
    zoom_x = cx / W
    zoom_y = cy / H
    
    # Build FFmpeg filter
    if end_effect is None:
        # Zoom in and hold
        filter_complex = (
            f"zoompan="
            f"z='if(lt(on,{start_frame}),1,"
            f"if(lt(on,{end_frame}),1+(on-{start_frame})*({zoom_scale}-1)/{zoom_frames},{zoom_scale}))'"
            f":x='iw/2-(iw/zoom/2)+({zoom_x}-0.5)*iw/zoom'"
            f":y='ih/2-(ih/zoom/2)+({zoom_y}-0.5)*ih/zoom'"
            f":d=1"
            f":s={W}x{H}"
            f":fps={fps}"
        )
    else:
        # Zoom in then zoom out
        end_start_frame = int(end_effect * fps)
        end_zoom_frames = int(zoom_duration * fps) if remove_mode == "smooth" else 1
        end_end_frame = end_start_frame + end_zoom_frames
        
        if remove_mode == "instant":
            filter_complex = (
                f"zoompan="
                f"z='if(lt(on,{start_frame}),1,"
                f"if(lt(on,{end_frame}),1+(on-{start_frame})*({zoom_scale}-1)/{zoom_frames},"
                f"if(lt(on,{end_start_frame}),{zoom_scale},1)))'"
                f":x='if(eq(zoom,1),iw/2-(iw/zoom/2),iw/2-(iw/zoom/2)+({zoom_x}-0.5)*iw/zoom)'"
                f":y='if(eq(zoom,1),ih/2-(ih/zoom/2),ih/2-(ih/zoom/2)+({zoom_y}-0.5)*ih/zoom)'"
                f":d=1"
                f":s={W}x{H}"
                f":fps={fps}"
            )
        else:  # smooth
            filter_complex = (
                f"zoompan="
                f"z='if(lt(on,{start_frame}),1,"
                f"if(lt(on,{end_frame}),1+(on-{start_frame})*({zoom_scale}-1)/{zoom_frames},"
                f"if(lt(on,{end_start_frame}),{zoom_scale},"
                f"if(lt(on,{end_end_frame}),{zoom_scale}-(on-{end_start_frame})*({zoom_scale}-1)/{end_zoom_frames},1))))'"
                f":x='if(eq(zoom,1),iw/2-(iw/zoom/2),iw/2-(iw/zoom/2)+({zoom_x}-0.5)*iw/zoom)'"
                f":y='if(eq(zoom,1),ih/2-(ih/zoom/2),ih/2-(ih/zoom/2)+({zoom_y}-0.5)*ih/zoom)'"
                f":d=1"
                f":s={W}x{H}"
                f":fps={fps}"
            )
    
    # FFmpeg command with hardware acceleration (if available)
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-filter_complex', filter_complex,
        '-c:v', 'libx264',
        '-preset', 'fast',  # fast but keeps good quality
        '-crf', str(crf),
        '-c:a', 'copy',  # copy audio without re-encoding
        '-y',
        output_path
    ]
    
    print(f"[INFO] Processing video with FFmpeg...")
    print(f"[INFO] Zoom: {zoom_percent*100}% at ({cx}, {cy})")
    print(f"[INFO] Duration: {zoom_start_time}s - {zoom_duration}s")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] Completed! File saved at: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg failed: {e}")


def apply_zoom_effect_simple(
    input_path,
    output_path="output_zoomed.mp4",
    zoom_factor=2.0,  # 2.0 = zoom 2x
    center_x=0.5,  # 0.5 = center (0–1)
    center_y=0.5,
    zoom_start=1.0,
    zoom_duration=1.5,
    crf=18
):
    """
    Simplified version – only zooms in and holds.
    FASTEST option – best for simple use cases.
    """
    
    # Get FPS
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_path
    ]
    
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    fps_parts = result.stdout.strip().split('/')
    fps = int(fps_parts[0]) / int(fps_parts[1])
    
    start_frame = int(zoom_start * fps)
    duration_frames = int(zoom_duration * fps)
    
    filter_str = (
        f"zoompan="
        f"z='if(lte(on,{start_frame}),1,if(lte(on,{start_frame + duration_frames}),"
        f"1+(on-{start_frame})*({zoom_factor}-1)/{duration_frames},{zoom_factor}))'"
        f":x='iw/2-(iw/zoom/2)+({center_x}-0.5)*iw/zoom'"
        f":y='ih/2-(ih/zoom/2)+({center_y}-0.5)*ih/zoom'"
        f":d=1:fps={fps}"
    )
    
    cmd = [
        'ffmpeg', '-i', input_path,
        '-vf', filter_str,
        '-c:v', 'libx264',
        '-preset', 'faster',
        '-crf', str(crf),
        '-c:a', 'copy',
        '-y', output_path
    ]
    
    print("[INFO] Processing...")
    subprocess.run(cmd, check=True)
    print(f"[SUCCESS] Done! {output_path}")
