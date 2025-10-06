from moviepy.editor import VideoFileClip

def apply_zoom_effect(
    input_path,
    output_path="output_zoomed.mp4",
    zoom_duration=0.75,
    zoom_start_time=1.0,
    zoom_percent=0.5,
    center=(200, 500),
    end_effect=None,
    remove_mode="instant"  # "instant" or "smooth"
):
    """
    Apply a dynamic zoom effect to a video.
    """
    clip = VideoFileClip(input_path)

    # Original video size and aspect ratio
    W, H = clip.size
    duration_video = clip.duration
    cx, cy = center

    # ===== Validation =====
    if zoom_duration <= 0:
        print("[ERROR] zoom_duration must be greater than 0")
        return clip.write_videofile(output_path, codec="libx264")

    if zoom_start_time < 0 or zoom_start_time >= duration_video:
        print("[ERROR] zoom_start_time is invalid (out of video range)")
        return clip.write_videofile(output_path, codec="libx264")

    if end_effect is not None:
        if end_effect <= zoom_start_time:
            print("[ERROR] end_effect must be greater than zoom_start_time")
            return clip.write_videofile(output_path, codec="libx264")
        if end_effect > duration_video:
            print("[ERROR] end_effect exceeds video duration")
            return clip.write_videofile(output_path, codec="libx264")

    if not (0 < zoom_percent <= 1):
        print("[ERROR] zoom_percent must be within (0, 1] (e.g., 0.5 = 50%)")
        return clip.write_videofile(output_path, codec="libx264")

    if not (0 <= cx <= W and 0 <= cy <= H):
        print("[ERROR] center point is outside the video frame")
        return clip.write_videofile(output_path, codec="libx264")

    if remove_mode not in ["instant", "smooth"]:
        print("[ERROR] remove_mode must be either 'instant' or 'smooth'")
        return clip.write_videofile(output_path, codec="libx264")

    aspect_ratio = W / H
    min_zoom_w = int(W * zoom_percent)
    min_zoom_h = int(min_zoom_w / aspect_ratio)

    # Function to determine crop region based on time
    def dynamic_crop(t):
        if t < zoom_start_time:
            return 0, 0, W, H

        # Handle zoom-out effect
        if end_effect is not None and t >= end_effect:
            if remove_mode == "instant":
                return 0, 0, W, H
            elif remove_mode == "smooth":
                if t >= end_effect + zoom_duration:
                    return 0, 0, W, H
                else:
                    alpha = (t - end_effect) / zoom_duration
                    crop_w = int(min_zoom_w + alpha * (W - min_zoom_w))
                    crop_h = int(crop_w / aspect_ratio)
        elif t >= zoom_start_time + zoom_duration:
            crop_w, crop_h = min_zoom_w, min_zoom_h
        else:
            alpha = (t - zoom_start_time) / zoom_duration
            crop_w = int(W - alpha * (W - min_zoom_w))
            crop_h = int(crop_w / aspect_ratio)

        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(W, x1 + crop_w)
        y2 = min(H, y1 + crop_h)

        x1 = x2 - crop_w
        y1 = y2 - crop_h
        return x1, y1, x2, y2

    # Apply dynamic cropping frame-by-frame
    zoomed = clip.fl(lambda gf, t: 
        clip.crop(*dynamic_crop(t))
            .resize((W, H))
            .get_frame(t)
    )

    zoomed.set_duration(clip.duration).write_videofile(output_path, codec="libx264")


# Example usage:
# apply_zoom_effect(
#     input_path="/content/55c95f56_clip_0_cut_11.49s.mp4",
#     output_path="zoomed.mp4",
#     zoom_duration=1.5,
#     zoom_start_time=2.0,
#     zoom_percent=0.4,
#     center=(300, 400),
#     end_effect=5.0,
#     remove_mode="smooth"
# )

# apply_zoom_effect(
#     input_path="/content/55c95f56_clip_0_cut_11.49s.mp4",
#     output_path="zoomed_instant.mp4",
#     zoom_duration=1.5,
#     zoom_start_time=2.0,
#     zoom_percent=0.4,
#     center=(300, 400),
#     end_effect=5.0,
#     remove_mode="instant"
# )
