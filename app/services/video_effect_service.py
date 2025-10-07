import os
import uuid
from pathlib import Path
from typing import List
from moviepy.editor import VideoFileClip
from app.models.schemas import TransitionEffect, DollyEffect, DollyEffectType, DollyEndType
from animation.full_transition_effect import  apply_multiple_effects
from animation.zoomin_at_one_point import apply_zoom_effect_fast
from animation.zoomin import safe_create_face_zoom_video
from animation.safe_check import wait_for_file_ready
import asyncio
import numpy as np

import shutil
import math
import subprocess
def rename_video(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if os.path.exists(dst_path):
        os.remove(dst_path)

    shutil.move(src_path, dst_path)

    return dst_path
def replace_audio(video1_path, video2_path):
    if not os.path.exists(video1_path):
        raise FileNotFoundError(f"Video1 not found: {video1_path}")
    if not os.path.exists(video2_path):
        raise FileNotFoundError(f"Video2 not found: {video2_path}")

    temp_output = "temp_output.mp4"

    video1 = VideoFileClip(video1_path)
    video2 = VideoFileClip(video2_path)

    video2_with_new_audio = video2.set_audio(video1.audio)

    video2_with_new_audio.write_videofile(
        temp_output, 
        codec="libx264",
        audio_codec="aac",
        bitrate="8000k",  # Bitrate video cao (điều chỉnh theo nhu cầu)
        audio_bitrate="320k",  # Bitrate audio cao
        preset="slow",  # slower = better quality
        ffmpeg_params=["-crf", "18"],  # CRF thấp = chất lượng cao (0-51, 18 là rất tốt)
        verbose=False, 
        logger=None
    )

    video1.close()
    video2.close()
    video2_with_new_audio.close()

    os.remove(video2_path)
    os.rename(temp_output, video2_path)

    if os.path.exists(video1_path):
        os.remove(video1_path)

    print(f"✅ Replaced audio of {video2_path} with audio from {video1_path}")
    return os.path.abspath(video2_path)


class VideoEffectService:
    def __init__(self):
        pass  

    def _ensure_dolly_objects(self, dolly_effects):
        if not dolly_effects:
            return []
        
        result = []
        for dolly in dolly_effects:
            if isinstance(dolly, dict):
                dolly_obj = DollyEffect(
                    scene_index=dolly.get('scene_index'),
                    start_time=dolly.get('start_time', 0.0),
                    duration=dolly.get('duration', 0.5),
                    zoom_percent=dolly.get('zoom_percent', 0),
                    effect_type=DollyEffectType(dolly.get('effect_type')),
                    x_coordinate=dolly.get('x_coordinate'),
                    y_coordinate=dolly.get('y_coordinate'),
                    end_time=dolly.get('end_time'),
                    end_type=DollyEndType(dolly.get('end_type', 'smooth'))
                )
                result.append(dolly_obj)
            elif isinstance(dolly, DollyEffect):
                result.append(dolly)
            else:
                raise TypeError(f"Invalid dolly effect type: {type(dolly)}")
        
        return result

    async def apply_effects(self, 
                          video_path: str,
                          transition_times: List[float],
                          transition_effects: List[TransitionEffect],
                          transition_durations: List[float],
                          dolly_effects: List[DollyEffect] = None,
                          job_id: str = None) -> str:
        # video2 = 
        return self.apply_effects_sync(
            video_path, transition_times, transition_effects, 
            transition_durations, dolly_effects, job_id
        )

    def apply_effects_sync(self, 
                          video_path: str,
                          transition_times: List[float],
                          transition_effects: List[TransitionEffect],
                          transition_durations: List[float],
                          dolly_effects: List[DollyEffect] = None,
                          job_id: str = None) -> str:
        # Validate input
        if len(transition_times) != len(transition_effects) != len(transition_durations):
            raise ValueError("transition_times, transition_effects, transition_durations must have same length")
        original_videopath = video_path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")
        
        # Convert dict to DollyEffect objects if needed
        dolly_effects = self._ensure_dolly_objects(dolly_effects)
        
        # Create output path
        from config import OUTPUT_DIR
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        output_filename = f"effect_{job_id or uuid.uuid4().hex}.mp4"
        output_path = output_dir / output_filename
        
        # === APPLY EFFECTS HERE ===
        # You can access:
        # - video_path: original video path
        # - transition_times: [1.5, 3.0, 5.2, ...] - transition points
        # - transition_effects: [TransitionEffect.FADE, TransitionEffect.ZOOM_IN, ...] - effect types
        # - transition_durations: [0.5, 1.0, 0.8, ...] - duration of effects
        # - dolly_effects: list of DollyEffect objects with details
        # - job_id: ID to track progress
        # =======================Transition================================
        complex_effects=[]
        for i in range(len(transition_times)):
            if i > 0: 
                video_path = str(output_path)  
            start_time = transition_times[i] - transition_durations[i] / 2
            end_time = transition_times[i] + transition_durations[i] / 2
            effect_name = transition_effects[i]
            complex_effects.append({
                "start_time": start_time,
                "end_time": end_time,
                "effect": effect_name
            })
            # apply_multiple_effects
            # apply_effect(
            #     video_path=video_path,
            #     output_path=str(output_path),
            #     start_time=start_time,
            #     end_time=end_time,
            #     effect_name=effect_name
            # )

            # print(f"Applied effect {effect_name} from {start_time}s to {end_time}s on video {video_path}")
            # print(f"Output saved to {output_path}")
            print("================================")
        apply_multiple_effects(
            video_path=original_videopath,
            output_path=str(output_path),
            effects_list=complex_effects,
            quality="high"
        )
    
        # =======================Dolly================================
        k = 0
        print("Dolly effects processing is currently disabled in the code.")
        print(f"Received {len(dolly_effects)} dolly effects:")
        wait_for_file_ready(video_path)
        for i, dolly in enumerate(dolly_effects):
            outputpath_raw_eff = output_dir / f"raw_effect_{job_id or uuid.uuid4().hex}_step{i+1}.mp4"
            video_path = str(output_path)
            print(f"  Effect {i+1}: type={dolly.effect_type}, start={dolly.start_time}s, duration={dolly.duration}s")
            time_begin = 0
            for j in range(dolly.scene_index or 0):
                print(f"    Scene index {j+1}")
                time_begin += transition_times[j]
            print("Time begin:", time_begin)
            if dolly.effect_type == DollyEffectType.MANUAL_ZOOM:
                wait_for_file_ready(video_path)
                print(f"    Manual zoom at ({dolly.x_coordinate}, {dolly.y_coordinate}) with zoom percent {dolly.zoom_percent}%")
                print("Start time:", time_begin + dolly.start_time)
                print(dolly.end_time)
                apply_zoom_effect_fast(
                    input_path=video_path,
                    output_path=str(outputpath_raw_eff),
                    zoom_duration=dolly.duration,
                    zoom_start_time=time_begin + dolly.start_time,
                    zoom_percent=dolly.zoom_percent / 100.0,  # Convert to ratio
                    center=(dolly.x_coordinate, dolly.y_coordinate),  # Coordinates assumed in [0, 1]
                    end_effect=time_begin + dolly.end_time,  
                    remove_mode=dolly.end_type.value,
                    crf=18   # "smooth" or "instant"
                )   
            elif dolly.effect_type == DollyEffectType.AUTO_ZOOM and dolly.end_type.value == "instant":
                print(f"    Auto zoom with zoom percent {dolly.zoom_percent}%")
                safe_create_face_zoom_video(
                    input_video=video_path,
                    output_video=str(outputpath_raw_eff),
                    zoom_type="instant",
                    zoom_start_time=time_begin + dolly.start_time,
                    zoom_duration=dolly.end_time - dolly.start_time,
                    zoom_factor=2 - dolly.zoom_percent / 100.0,
                    enable_shake=False,
                    shake_intensity=1,
                    shake_start_delay=0.3
                )
            elif dolly.effect_type == DollyEffectType.AUTO_ZOOM:
                print(f"    Auto zoom with zoom percent {dolly.zoom_percent}%")
                print("afdfs", dolly.start_time + dolly.duration)
                print("afdfs", dolly.end_time - dolly.start_time)
                safe_create_face_zoom_video(
                    input_video=video_path,
                    output_video=str(outputpath_raw_eff),
                    zoom_type="gradual",
                    gradual_start_time=time_begin + dolly.start_time,
                    gradual_end_time=dolly.start_time + dolly.duration,
                    hold_duration=dolly.end_time - dolly.start_time,
                    zoom_factor=2 - dolly.zoom_percent / 100.0,
                    enable_shake=False,
                    shake_intensity=1,
                    shake_start_delay=0.3
                ) 
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"Deleted file: {output_path}")
            else:
                print("File does not exist")
            os.rename(outputpath_raw_eff, output_path)
            print("================================")

        print(f"Processing video effects for job {job_id}")
        print(f"Transition times: {transition_times}")
        print(f"Transition effects: {transition_effects}")
        print(f"Transition durations: {transition_durations}")
        print(f"Dolly effects: {len(dolly_effects or [])} effects")
        print(f"Output will be: {original_videopath}")
        print("================================")
        # print(original_videopath)
        # if dolly_effects is None or len(dolly_effects) == 0:
            # print(f"Copied video to {output_path} without dolly effects.")
            # outputtttt= str(replace_audio(original_videopath, str(output_path)))
        return str(rename_video(str(output_path), str(original_videopath)))


    def _mock_process_video_sync(self, input_path: str, output_path: str, job_id: str):
        import shutil
        import time
        time.sleep(2)  # 2 seconds for testing
        shutil.copy2(input_path, output_path)
        print(f"Mock processing completed for job {job_id}")

    def get_video_duration_sync(self, video_path: str) -> float:

        return 60.0  # 60 seconds

    async def get_video_duration(self, video_path: str) -> float:
        """
        Get video duration asynchronously using subprocess
        """
        # TODO: Implement actual duration retrieval with subprocess
        # Example:
        # process = await asyncio.create_subprocess_exec(
        #     "ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
        #     "-of", "csv=p=0", video_path,
        #     stdout=asyncio.subprocess.PIPE,
        #     stderr=asyncio.subprocess.PIPE
        # )
        # stdout, stderr = await process.communicate()
        # return float(stdout.decode('utf-8').strip())
        
        # Mock duration for testing
        return 60.0  # 60 seconds

    def validate_effects_timing(self, 
                              transition_times: List[float],
                              dolly_effects: List[DollyEffect],
                              video_duration: float):
        # Convert dict to DollyEffect objects if needed
        dolly_effects = self._ensure_dolly_objects(dolly_effects)
        
        # Check transition times
        for time_point in transition_times:
            if time_point > video_duration:
                raise ValueError(f"Transition time {time_point}s exceeds video duration {video_duration}s")
        
        # Check dolly effects
        for dolly in dolly_effects or []:
            if dolly.start_time + dolly.duration > video_duration:
                raise ValueError(f"Dolly effect (start: {dolly.start_time}s, duration: {dolly.duration}s) exceeds video duration {video_duration}s")
