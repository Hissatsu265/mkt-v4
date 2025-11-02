import os
import uuid
from pathlib import Path
from typing import List
import subprocess
import json
import random

from app.models.schemas import TransitionEffect

import aiohttp
import aiofiles
import websockets
import glob
import time
from config import SERVER_COMFYUI,WORKFLOW_INFINITETALK_PATH,BASE_DIR
from PIL import Image
server_address = SERVER_COMFYUI
from utilities.divide_audio import process_audio_file
from utilities.merge_video import concat_videos
from utilities.cut_video import cut_video,cut_audio,cut_audio_from_time
from utilities.audio_duration import get_audio_duration
from utilities.audio_processing_infinite import trim_video_start,add_silence_to_start
from utilities.check_audio_safe import wait_for_audio_ready
import asyncio
from directus.file_upload import Uploadfile_directus
from app.services.job_service import job_service
from app.services.img_service import get_random_prompt
from animation.full_transition_effect import  apply_multiple_effects
from animation.fairyending import fairyending
image_paths_product = []
image_path_sideface = []
image_paths_product_rout360=[]
video_paths_product_rout360=[]
import cv2
import numpy as np

def zoom_to_face(image_path, output_path, zoom_factor=2.0):

    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc ảnh!")
        return
    
    height, width = img.shape[:2]
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        center_x, center_y = width // 2, height // 3
        face_w, face_h = width // 3, height // 3
    else:
        if len(faces) > 1:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        center_x = x + w // 2
        center_y = y + h // 2
        face_w, face_h = w, h

    crop_width = int(width / zoom_factor)
    crop_height = int(height / zoom_factor)
    
    crop_x1 = max(0, center_x - crop_width // 2)
    crop_y1 = max(0, center_y - crop_height // 2)
    crop_x2 = min(width, crop_x1 + crop_width)
    crop_y2 = min(height, crop_y1 + crop_height)
    
    if crop_x2 - crop_x1 < crop_width:
        crop_x1 = max(0, crop_x2 - crop_width)
    if crop_y2 - crop_y1 < crop_height:
        crop_y1 = max(0, crop_y2 - crop_height)
    
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(output_path, zoomed)
    
    return zoomed


def zoom_to_face_interactive(image_path, output_path='output_zoomed.jpg'):

    original = cv2.imread(image_path)
    if original is None:
        print("Không thể đọc ảnh!")
        return
    
    zoom_levels = [1.5, 2.0, 2.5]
    
    for zoom in zoom_levels:
        temp_output = f"zoom_{zoom}x.jpg"
        print(f"\n--- Zoom {zoom}x ---")
        zoom_to_face(image_path, temp_output, zoom_factor=zoom)
    
    zoomed = zoom_to_face(image_path, output_path, zoom_factor=2.0)
    
    return zoomed

# ====================================================================
import os
from concurrent.futures import ThreadPoolExecutor

# Thread pool cho video processing (đặt ở đầu file, global)
video_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="video_concat")

async def concat_and_merge_async(job_id, results, cond_audio_path, output_path_video, cond_images):

    loop = asyncio.get_event_loop()
    def _do_concat_merge():
        """Hàm blocking - chạy trong thread pool"""
        concat_name = os.path.join(os.getcwd(), f"{job_id}_concat.mp4")
        from utilities.merge_video import concat_videos
        output_file1 = concat_videos(results, concat_name)

        from utilities.merge_video_audio import replace_audio_trimmed
        output_file = replace_audio_trimmed(output_file1, cond_audio_path, output_path_video)
        
        try:
            os.remove(output_file1)
            for file in results:
                os.remove(file)
            for path in cond_images:
                os.remove(path)
            os.remove(cond_audio_path)
        except Exception as e:
            print(f"❌ Error removing temporary files: {str(e)}")
        
        return output_file
    
    output_file = await loop.run_in_executor(video_executor, _do_concat_merge)
    return output_file
def random_transition_list(n):
    effects = [e.value for e in TransitionEffect]  # Lấy danh sách các giá trị Enum
    return [random.choice(effects) for _ in range(n)]
def adjust_end_times(start_times, end_times, audio_end):
    new_end_times = []
    for i in range(len(start_times)):
        if i < len(start_times) - 1:
            new_end = max(start_times[i + 1] - 0.2, start_times[i])
        else:
            new_end = audio_end

        if new_end - start_times[i] > 2:
            new_end -= 0.8

        new_end_times.append(new_end)
    return new_end_times
import ast

def safe_parse_color(value):
    """Chuyển string '(R, G, B)' hoặc '#HEX' về tuple RGB"""
    if isinstance(value, tuple):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("(") and value.endswith(")"):
            try:
                return tuple(ast.literal_eval(value))
            except Exception:
                pass
        elif value.startswith("#"):
            h = value.lstrip("#")
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return (0, 0, 0)  # fallback an toàn

def custom_random_sequence(n):
    if n <= 0:
        return []

    nums = [1, 2, 3, 5, 6, 7, 8,9]
    sequence = []
    last_seen = {num: -10 for num in nums}  # lưu vị trí xuất hiện gần nhất
    last = None

    for i in range(n):
        # --- Quy tắc đặc biệt cho 2 cảnh đầu ---
        if i < 2:
            # đảm bảo trong 2 cảnh đầu có ít nhất một số 1
            if i == 1 and 1 not in sequence:
                value = 1
            else:
                candidates = [x for x in nums if x != last and x != 6]  # không chọn 6 sớm
                value = random.choice(candidates)
        else:
            candidates = []
            for num in nums:
                # Không chọn trùng với cảnh trước
                if num == last:
                    continue
                # Số 6 chỉ được xuất hiện ở vị trí n-1 hoặc n-2
                if num == 6 and i < n - 1:
                    continue
                # Nếu là 1 hoặc 7, phải cách lần trước ít nhất 4 cảnh
                if num in [1, 7,9] and i - last_seen[num] < 4:
                    continue
                # Nếu là 2-6 (trừ 1 và 7), cách lần trước ít nhất 5 cảnh
                if num not in [1, 7] and i - last_seen[num] < 5:
                    continue
                # Nếu là 2 hoặc 3, phải cách lần xuất hiện gần nhất của số còn lại ít nhất 4
                if num == 2 and i - last_seen[3] < 4:
                    continue
                if num == 3 and i - last_seen[2] < 4:
                    continue

                candidates.append(num)

            # Nếu không còn candidate hợp lệ → nới lỏng điều kiện
            if not candidates:
                candidates = [x for x in nums if x != last]
                # vẫn ưu tiên không chọn 6 sớm
                if i < n - 1 and 6 in candidates:
                    candidates.remove(6)

            value = random.choice(candidates)

        sequence.append(value)
        last_seen[value] = i
        last = value

    return sequence

    # return [9,9,1,5][:n]

def custom_random_sequence111(n):
    if n <= 0:
        return []

    nums = [1, 5, 8,9]
    sequence = []
    last_seen = {num: -10 for num in nums}  # lưu vị trí xuất hiện gần nhất
    last = None

    for i in range(n):
        # --- Quy tắc đặc biệt cho 2 cảnh đầu ---
        if i < 2:
            # đảm bảo trong 2 cảnh đầu có ít nhất một số 1
            if i == 1 and 1 not in sequence:
                value = 1
            else:
                candidates = [x for x in nums if x != last and x != 6]  # không chọn 6 sớm
                value = random.choice(candidates)
        else:
            candidates = []
            for num in nums:
                # Không chọn trùng với cảnh trước
                if num == last:
                    continue
                # Số 6 chỉ được xuất hiện ở vị trí n-1 hoặc n-2
                if num == 6 and i < n - 1:
                    continue
                # Nếu là 1 hoặc 7, phải cách lần trước ít nhất 4 cảnh
                if num in [1, 7,9] and i - last_seen[num] < 4:
                    continue
                # Nếu là 2-6 (trừ 1 và 7), cách lần trước ít nhất 5 cảnh
                if num not in [1, 7] and i - last_seen[num] < 5:
                    continue
                # Nếu là 2 hoặc 3, phải cách lần xuất hiện gần nhất của số còn lại ít nhất 4
                if num == 2 and i - last_seen[3] < 4:
                    continue
                if num == 3 and i - last_seen[2] < 4:
                    continue

                candidates.append(num)

            # Nếu không còn candidate hợp lệ → nới lỏng điều kiện
            if not candidates:
                candidates = [x for x in nums if x != last]
                # vẫn ưu tiên không chọn 6 sớm
                if i < n - 1 and 6 in candidates:
                    candidates.remove(6)

            value = random.choice(candidates)

        sequence.append(value)
        last_seen[value] = i
        last = value

    return sequence

    # return [9,9,1,5][:n]  
    # return [2,3,4,2,3,4,2,3,4,2,3,4][:n]

class VideoService:
    def __init__(self):
        from config import OUTPUT_DIR
        self.output_dir = OUTPUT_DIR

    def generate_output_filename(self) -> str:
        unique_id = str(uuid.uuid4())[:8]
        timestamp = int(asyncio.get_event_loop().time())
        return unique_id, f"video_{timestamp}_{unique_id}.mp4"

    async def create_video(self, image_paths: List[str], prompts: List[str], audio_path: str, resolution: str, job_id: str) -> str:
        jobid, output_filename = self.generate_output_filename()
        output_path = self.output_dir / output_filename
        try:
            
            await job_service.update_job_status(job_id, "processing", progress=0)

            list_scene = await run_job(jobid, prompts, image_paths, audio_path, output_path,resolution)  
            print(str(output_path)) 
            # if os.path.exists(str(output_path)):
            #     print("✅ File tồn tại!")
            # else:
            #     print("❌ File không tồn tại.")
            # return str(output_path),list_scene
            path_directus= Uploadfile_directus(str(output_path))
            # if os.path.exists(str(output_path)):
            #     print("✅ File tồn tại!")
            # else:
            #     print("❌ File không tồn tại.")
            if path_directus is not None and output_path.exists() :
                print(f"Video upload successfully: {path_directus}")
                print(f"Job ID: {job_id}, Output Path: {path_directus}")
                os.remove(str(output_path))
                return str(path_directus),list_scene
            else:
                if not output_path.exists():
                    raise Exception("Cannot create video")
                else:    
                    raise Exception("Cannot upload video to Directus")
    
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise e
async def run_job(job_id, prompts, cond_images, cond_audio_path,output_path_video,resolution):
    generate_output_filename = output_path_video
    list_scene=[]
    if prompts[0]=="" or prompts[0] is None or prompts[0].lower() == "none":
        prompts[0]="A realistic video of a person confidently presenting a product they are holding. The person speaks clearly and professionally, as if explaining the product’s features in an advertisement. Their facial expression remains pleasant and natural, with slight movements to appear engaging but not exaggerated. Their hand gestures are smooth and minimal, focusing attention on the product, creating the impression of a calm and confident presenter in a product promotion video."
    if get_audio_duration(cond_audio_path) > 12:
        output_directory = "output_segments"
        os.makedirs(output_directory, exist_ok=True)
        output_paths,durations, result = process_audio_file(cond_audio_path, output_directory)
        # =========================================================================
        results=[]

        first_time=True
        last_value=None
        if len(cond_images)>1 and cond_images[1]!="":
            list_random = custom_random_sequence(len(output_paths))
        else:
            cond_images.append(cond_images[0])
            list_random = custom_random_sequence111(len(output_paths))
        # count = len(list(filter(lambda x: x != 1, list_random)))
        count = len(list(filter(lambda x: x ==2 or x==3, list_random)))
        count1=len(list(filter(lambda x: x ==7, list_random)))
        count2=len(list(filter(lambda x: x ==8, list_random)))

        index_forimgpro=0
        print("===========================")

        # ==========================================================================
        for i, output_path in enumerate(output_paths):
            await job_service.update_job_status(job_id, "processing", progress=int(i*10+10))
            if i<len(output_paths)-1:
                list_scene.append(get_audio_duration(output_path))
            
            # ==============Random image for each scene=============
            # if len(cond_images)>1:
            #     choices = [x for x in range(len(prompts)) if x != last_value] 
            #     current_value = random.choice(choices)  # chọn ngẫu nhiên
            #     last_value = current_value  # lưu 
            # else: current_value=0
            # ===============================================================================
            # print(f"Audio segment {i+1}: {output_path} (Duration: {durations[i]}s)")
            # print(cond_images)
            # print(f"Image: {cond_images[current_value]}")
            # print(f"Prompt: {prompts[current_value]}")
            clip_name=os.path.join(os.getcwd(), f"{job_id}_clip_{i}.mp4")
            audiohavesecondatstart = add_silence_to_start(output_path, job_id, duration_ms=0)
            audiohavesecondatstart=str(BASE_DIR / audiohavesecondatstart)
            print("============sgageeeee===============")
            # print("dfsdfsdfsd:   ", audiohavesecondatstart)
            # print(type(audiohavesecondatstart))

            # =================================================================
            current_value=0
            file_path = str(cond_images[current_value])

            if (list_random[i] == 9):

                from animation.addtittle import create_keyword_video

                from app.services.extract_keyword import process_keywordfromaudi

                keywords, start_times_list, end_times_list = await asyncio.to_thread(
                    process_keywordfromaudi, audiohavesecondatstart
                )
                print("==========ê======sagasgasg===========")
                if len(keywords) <= 0:
                    list_random[i] = 1
                    if i + 1 < len(list_random) and list_random[i + 1] == 1:
                        list_random[i] = 5
                    if i - 1 >= 0 and list_random[i - 1] == 1:
                        list_random[i] = 5 

                else: 
                    print("fsfsfsdf")
                    font_path_hehehehe = [
                        "/home/toan/marketing-video-ai/font/Aloevera-OVoWO.ttf",
                        "/home/toan/marketing-video-ai/font/MontserratBlack-3zOvZ.ttf",
                        "/home/toan/marketing-video-ai/font/MontserratBold-p781R.otf",
                        "/home/toan/marketing-video-ai/font/PoppinsSemibold-8l8n.otf",
                    ]
                    print("fsfsfsdf")

                    color_combos = [
                        {"name": "Navy Blue & White", "bg": "(30, 58, 138)", "text": "#ffffff"},
                        {"name": "Emerald Green & White", "bg": "(5, 150, 105)", "text": "#ffffff"},
                        {"name": "Orange & Dark Gray", "bg": "(249, 115, 22)", "text": "#1f2937"},
                        {"name": "Teal & White", "bg": "(13, 148, 136)", "text": "#ffffff"},
                        {"name": "Sky Blue & White", "bg": "(2, 132, 199)", "text": "#ffffff"},
                        {"name": "Magenta & White", "bg": "(192, 38, 211)", "text": "#ffffff"},
                        {"name": "Gray & Black", "bg": "(156, 163, 175)", "text": "#000000"},
                        {"name": "White & Black", "bg": "(255, 255, 255)", "text": "#000000"},
                    ]
                    
                    time_video=get_audio_duration(audiohavesecondatstart)

                    new_end_times = adjust_end_times(start_times_list, end_times_list, time_video)

                    selected = random.choice(color_combos)

                    font_path= random.choice(font_path_hehehehe)

                    resolution_tuple =(1280,720) if resolution == "16:9" else (720,1280)
                    bg_color = safe_parse_color(selected['bg'])
                    # font_color = safe_parse_color(selected['text'])
                    await asyncio.to_thread(
                        create_keyword_video,
                        keywords,                      # 1
                        start_times_list,              # 2
                        new_end_times,                 # 3
                        time_video,                    # 4 = duration
                        resolution_tuple,              # 5 = resolution
                        font_path,                     # 6 = font (theo định nghĩa hàm)
                        bg_color,                # 7 = bg_color
                        selected['text'],              # 8 = font_color
                        None,                          # 9 = font_size (None để tự tính)
                        random.randint(1, 2),          # 10 = effect_type
                        clip_name                      # 11 = output_path
                    )

            if (list_random[i] == 1):
                output=await generate_video_cmd(
                    prompt=prompts[current_value],
                    cond_image=str(file_path),# 
                    cond_audio_path=audiohavesecondatstart, 
                    output_path=clip_name,
                    job_id=job_id,
                    resolution=resolution
                )
            elif (list_random[i] == 5):
                clip_name111=os.path.join(os.getcwd(), f"{job_id}_zoomin_{i}.png")
                zoom_to_face(str(file_path), clip_name111, zoom_factor=2.0)
                output=await generate_video_cmd(
                    prompt="A realistic video of a person confidently giving a lecture. Their face remains neutral and professional, without expressions or head movement. Their hands move up and down slowly and naturally to emphasize their words without swinging their arms from side to side, creating the impression of a teacher explaining a lesson.",
                    cond_image=clip_name111,# 
                    cond_audio_path=audiohavesecondatstart, 
                    output_path=clip_name,
                    job_id=job_id,
                    resolution=resolution
                )
                os.remove(clip_name111)
            elif (list_random[i] == 9):
                print("heh")

            else:
                output=await generate_video_fast(
                    prompt=prompts[current_value],
                    cond_image=str(cond_images[1]),
                    cond_audio_path=audiohavesecondatstart, 
                    output_path=clip_name,
                    job_id=job_id,
                    resolution=resolution,
                    type=list_random[i],
                    first_time=first_time,
                    howmuch=count,
                    index=index_forimgpro,
                    howmuch1=count1,
                    howmuch2=count2,
                    avt_image=str(cond_images[0])
                )
                first_time=False
                if list_random[i]!=8 and list_random[i]!=7 and list_random[i]!=6 :
                    index_forimgpro+=1
            # =========================add tittle===================================
            # from animation.addtittle import add_text_to_video
            # from app.services.extract_keyword import process_keywordfromaudi

            # keywords, start_times_list, end_times_list = await asyncio.to_thread(
            #     process_keywordfromaudi, audiohavesecondatstart
            # )
            # # keywords, start_times_list, end_times_list = process_keywordfromaudi(audiohavesecondatstart)

            # clip_name1=os.path.join(os.getcwd(), f"{job_id}_clip11_{i}.mp4")

            # font_path_hehehehe = [
            #     "/home/toan/marketing-video-ai/font/Aloevera-OVoWO.ttf",
            #     "/home/toan/marketing-video-ai/font/MontserratBlack-3zOvZ.ttf",
            #     "/home/toan/marketing-video-ai/font/MontserratBold-p781R.otf",
            #     "/home/toan/marketing-video-ai/font/PoppinsSemibold-8l8n.otf",
            # ]

            # font_path= random.choice(font_path_hehehehe)
            # if len(start_times_list) > 0:
            #     y_pos = 0.4 if resolution == "16:9" else 0.68
            #     await asyncio.to_thread(
            #         add_text_to_video,
            #         clip_name,
            #         keywords,
            #         start_times_list,
            #         end_times_list,
            #         font_path,
            #         clip_name1,
            #         True,  
            #         2,      
            #         y_pos  
            #     )
            # ================================================================
            print(list_random,"]]]]]]]]]]]]]]]]]]]]]]]]]ư")
            trim_video_start(clip_name, duration=0.5)
            output_file=cut_video(clip_name, get_audio_duration(output_path)-0.5) 
            results.append(output_file)
            try:
                os.remove(output_path)
                # os.remove(clip_name1)
                os.remove(clip_name)
                os.remove(audiohavesecondatstart)
            except Exception as e:
                print(f"❌ Error removing temporary file {output_path}: {str(e)}")
        # print("44444444444444")
        # concat_name=os.path.join(os.getcwd(), f"{job_id}_concat_{i}.mp4")
        # output_file1 = concat_videos(results, concat_name)
        # from utilities.merge_video_audio import replace_audio_trimmed
        # output_file = replace_audio_trimmed(output_file1,cond_audio_path,output_path_video)
        # # print("5555555555555555")
        # try:
        #     os.remove(output_file1)
        #     for file in results:
        #         os.remove(file)
        #     for path in cond_images:
        #         os.remove(path)
        #     os.remove(cond_audio_path)
        # except Exception as e:
        #     print(f"❌ Error removing temporary files: {str(e)}")
        # return list_scene
        global image_paths_product
        for file_anh in image_paths_product:
            await delete_file_async(str(file_anh))
        image_paths_product=[]
        output_file = await concat_and_merge_async(
            job_id=job_id,
            results=results,
            cond_audio_path=cond_audio_path,
            output_path_video=output_path_video,
            cond_images=cond_images
        )
        # ===========================transition==================================
        video_name1111111 = os.path.join(os.getcwd(), f"{job_id}clip_hehehehehe.mp4")
        
        for i in range(1, len(list_scene)):
            list_scene[i] = list_scene[i-1] + list_scene[i]
        for i in range(len(list_scene)):
            list_scene[i] = list_scene[i] - 0.5*(i+1) 

        complex_effects=[]
        transition_effects = random_transition_list(len(list_scene))
        for i in range(len(list_scene)):
            start_time = list_scene[i] - 1 / 2
            end_time = list_scene[i] + 1 / 2
            effect_name = transition_effects[i]
            complex_effects.append({
                "start_time": start_time,
                "end_time": end_time,
                "effect": effect_name
            })
            print("================================")
        # apply_multiple_effects(
        #     video_path=output_path_video,
        #     output_path=str(video_name1111111),
        #     effects_list=complex_effects,
        #     quality="high"
        # )
        # os.remove(output_path_video)
        # os.rename(video_name1111111, output_path_video)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,  # Dùng default ThreadPoolExecutor
            apply_multiple_effects,
            output_path_video,
            str(video_name1111111),
            complex_effects,
            "high"
        )
        await delete_file_async(output_path_video)
        await asyncio.to_thread(os.rename, video_name1111111, output_path_video)

        #========================================================
        
        return list_scene
    # =========================================================================================
    # =========================================================================================
    # =========================================================================================    

    else:
        audiohavesecondatstart = add_silence_to_start(cond_audio_path, job_id, duration_ms=500)
        generate_output_filename=os.path.join(os.getcwd(), f"{job_id}_noaudio.mp4")
        if wait_for_audio_ready(audiohavesecondatstart, min_size_mb=0.02, max_wait_time=60, min_duration=2.0):
            print("Detailed check passed!")

        # =================================================================
        file_path = str(cond_images[0])
        file_root, file_ext = os.path.splitext(file_path)
        await job_service.update_job_status(job_id, "processing", progress=50)
        output=await generate_video_cmd(
            prompt=prompts[0], 
            cond_image=file_path, 
            cond_audio_path=audiohavesecondatstart, 
            output_path=generate_output_filename,
            job_id=job_id,
            resolution=resolution
        )  
        await job_service.update_job_status(job_id, "processing", progress=97)
        from utilities.merge_video_audio import replace_audio_trimmed
        tempt=trim_video_start(generate_output_filename, duration=0.5)
        output_file = replace_audio_trimmed(generate_output_filename,cond_audio_path,output_path_video)
        try:
            os.remove(str(generate_output_filename))
            os.remove(str(audiohavesecondatstart))
            os.remove(str(cond_audio_path))
            os.remove(str(file_path))
        except Exception as e:
            print(f"❌ Error removing temporary files: {str(e)}")
        
        return list_scene
# ============================================================================================

import signal

# async def start_comfyui():
#     process = await asyncio.create_subprocess_exec(
#         "python3", "main.py",
#         cwd=str(BASE_DIR / "ComfyUI"),  # chỗ chứa main.py
#         stdout=asyncio.subprocess.PIPE,
#         stderr=asyncio.subprocess.PIPE
#     )
#     print("🚀 ComfyUI started (PID:", process.pid, ")")
#     return process
import socket
async def wait_for_port_async(host: str, port: int, timeout: int = 60) -> bool:
 
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"✅ ComfyUI port {port} đã sẵn sàng!")
                return True
        except (OSError, ConnectionRefusedError):
            await asyncio.sleep(2)
    print(f"❌ Hết thời gian {timeout}s mà ComfyUI vẫn chưa mở port {port}.")
    return False


async def start_comfyui():
    HOST = "127.0.0.1"
    PORT = 8188
    process = await asyncio.create_subprocess_exec(
        "python3", "main.py",
        cwd=str(BASE_DIR / "ComfyUI"),  # chỗ chứa main.py
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    print(f"🚀 ComfyUI started (PID: {process.pid}) — đang chờ server mở port {PORT}...")

    ready = await wait_for_port_async(HOST, PORT, timeout=120)

    if not ready:
        print("⚠️ ComfyUI không khởi động được đúng cách (port không mở).")
    else:
        print("🎉 ComfyUI sẵn sàng kết nối!")

    return process
async def stop_comfyui(process):
    if process and process.returncode is None:
        print("🛑 Stopping ComfyUI...")
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            print("⚠️ Force killing ComfyUI...")
            process.kill()
            await process.wait()
async def load_workflow(path="workflow.json"):
    async with aiofiles.open(path, "r", encoding='utf-8') as f:
        content = await f.read()
        return json.loads(content)

async def queue_prompt(workflow):
    client_id = str(uuid.uuid4())
    
    payload = {
        "prompt": workflow, 
        "client_id": client_id
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://{server_address}/prompt",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                result = await response.json()
                result["client_id"] = client_id
                return result
            else:
                raise Exception(f"Failed to queue prompt: {response.status}")

async def wait_for_completion(prompt_id, client_id):
    
    websocket_url = f"ws://{server_address}/ws?clientId={client_id}"
    
    try:
        async with websockets.connect(websocket_url) as websocket:
            
            total_nodes = 0
            completed_nodes = 0
            
            while True:
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    
                    if isinstance(msg, str):
                        data = json.loads(msg)                        
                        # print(f"📨 Rêcive message: {data.get('type', 'unknown')}")
                        
                        if data["type"] == "execution_start":
                            print(f"🚀 Start workflow: {data.get('data', {}).get('prompt_id')}")
                        
                        elif data["type"] == "executing":
                            node_id = data["data"]["node"]
                            # current_prompt_id = data["data"]["prompt_id"]
                            current_prompt_id = data.get("data", {}).get("prompt_id")

                            if current_prompt_id == prompt_id:
                                if node_id is None:
                                    print("🎉 Workflow Complete!")
                                    return True
                                else:
                                    completed_nodes += 1
                                    print(f"⚙️  Processing node: {node_id} ({completed_nodes} )")
                        
                        elif data["type"] == "progress":
                            progress_data = data.get("data", {})
                            value = progress_data.get("value", 0)
                            max_value = progress_data.get("max", 100)
                            node = progress_data.get("node")
                            percentage = (value / max_value * 100) if max_value > 0 else 0
                            print(f"📊 Node {node}: {value}/{max_value} ({percentage:.1f}%)")
                        
                        elif data["type"] == "execution_error":
                            print(f"❌ Error: {data}")
                            return False
                            
                        elif data["type"] == "execution_cached":
                            cached_nodes = data.get("data", {}).get("nodes", [])
                            print(f"💾 {len(cached_nodes)} nodes cached")
                
                except asyncio.TimeoutError:
                    print("⏰ WebSocket timeout, waiting...")
                    continue
                except Exception as e:
                    print(f"❌ Error WebSocket: {e}")
                    break
                    
    except Exception as e:
        print(f"❌ Cannot connect to WebSocket: {e}")
        # print("🔄 Fallback: Kiểm tra file output định kỳ...")
        
        return await wait_for_completion_fallback(prompt_id)

async def wait_for_completion_fallback(prompt_id):
    start_time = time.time()
    
    while True:
        await asyncio.sleep(2)  
        
        video_path = await find_latest_video("my_custom_video")
        if video_path and os.path.exists(video_path):
            file_time = os.path.getmtime(video_path)
            if file_time > start_time:
                print("✅ Findout video!")
                return True
        
        if time.time() - start_time > 300:
            print("⏰ Timeout waiting for video")
            return False

async def find_latest_video(prefix, output_dir=str(BASE_DIR / "ComfyUI/output")):    
    def _find_files():
        patterns = [
            f"{prefix}*audio*.mp4", 
            f"{prefix}_*-audio.mp4"
        ]
        
        all_files = []
        for pattern in patterns:
            files = glob.glob(os.path.join(output_dir, pattern))
            all_files.extend(files)
        
        if not all_files:
            print(f"🔍 Cannot find video match prefix '{prefix}' in {output_dir}")
            all_mp4 = glob.glob(os.path.join(output_dir, "*.mp4"))
            if all_mp4:
                print(f"📁 file .mp4 we have now:")
                for f in sorted(all_mp4, key=os.path.getmtime, reverse=True)[:5]:
                    print(f"   {f} (modified: {time.ctime(os.path.getmtime(f))})")
            return None
        
        latest_file = max(all_files, key=os.path.getmtime)
        print(f"📁 Finding newest file: {latest_file}")
        return latest_file
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _find_files)

# ========== Hàm chính được cập nhật ==========
async def generate_video_cmd(prompt, cond_image, cond_audio_path, output_path, job_id,resolution,negative_prompt=""):
    comfy_process = await start_comfyui()
    # await asyncio.sleep(20)  # đợi server ComfyUI khởi động (có thể tăng nếu load model chậm)

    try:
        print("🔄 Loading workflow...")
        workflow = await load_workflow(str(BASE_DIR) + "/" + WORKFLOW_INFINITETALK_PATH)  
# ===========================================================================
        # workflow["203"]["inputs"]["image"] = cond_image
        # workflow["125"]["inputs"]["audio"] = cond_audio_path
        
        # if prompt.strip() == "" or prompt is None or prompt == "none":
        #     # workflow["135"]["inputs"]["positive_prompt"] = "Mouth moves in sync with speech. A person is sitting in a side-facing position, with their face turned toward the left side of the frame and the eyes look naturally forward in that left-facing direction without shifting. Speaking naturally, as if having a conversation. He always kept his posture and gaze straight without turning his head."    
        #     workflow["135"]["inputs"]["positive_prompt"] = "Mouth moves in sync with speech. A person is sitting in a side-facing position, with their face turned toward the left side of the frame and the eyes look naturally forward in that left-facing direction without shifting. Speaking naturally, as if having a conversation. He mostly keeps his posture and gaze straight without turning his head, but occasionally makes small, natural gestures with the head or hands to emphasize the speech, adding subtle liveliness to the video."    
        # else:
        #     workflow["135"]["inputs"]["positive_prompt"] = prompt
            
        # workflow["135"]["inputs"]["negative_prompt"] = "change perspective, bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        # wf_h=448
        # wf_w=448
        # if resolution == "1080x1920":
        #     wf_w = 1080
        #     wf_h = 1920
        # elif resolution=="1920x1080":
        #     wf_w = 1920
        #     wf_h = 1080
        # elif resolution=="720x1280":
        #     wf_w = 720
        #     wf_h = 1280
        #     # workflow["208"]["inputs"]["frame_window_size"] = 41
        # elif resolution=="480x854": 
        #     wf_w = 480
        #     wf_h = 854
        # elif resolution=="854x480": 
        #     wf_w = 854
        #     wf_h = 480
        # elif resolution=="1280x720":    
        #     wf_w = 1280
        #     wf_h = 720 
        
        #     # workflow["208"]["inputs"]["frame_window_size"] = 41
        # img = Image.open(cond_image)
        # width_real, height_real = img.size
        # workflow["211"]["inputs"]["value"] = width_real
        # workflow["212"]["inputs"]["value"] = height_real

        # workflow["211"]["inputs"]["value"] = 608
        # workflow["212"]["inputs"]["value"] = 608
        # img.close()
        # prefix = job_id
        # workflow["131"]["inputs"]["filename_prefix"] = prefix

# ===========================================================================
        workflow["284"]["inputs"]["image"] = cond_image
        workflow["125"]["inputs"]["audio"] = cond_audio_path
        
        workflow["270"]["inputs"]["value"] = int(get_audio_duration(cond_audio_path)*30)
        # =============================================================
        
        
        if prompt.strip() == "" or prompt is None or prompt.lower() == "none" :
            workflow["241"]["inputs"]["positive_prompt"] = "A realistic video of a person confidently giving a lecture. Their face remains neutral and professional, without strong expressions or noticeable head movement. Their hands move up and down slowly and naturally to emphasize key points, without swinging side to side, creating the impression of a teacher clearly explaining a lesson."    
        else:
            print("dùng prompt của mình")
            workflow["241"]["inputs"]["positive_prompt"] = prompt
            
        workflow["241"]["inputs"]["negative_prompt"] = "bright tones, overexposed, blurred details, move, head movement, subtitles, style, works, paintings, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        wf_h=448
        wf_w=448
    
        if resolution == "1:1":
            wf_w = 720
            wf_h = 720
        elif resolution=="16:9":    
            wf_w = 1088
            wf_h = 608
        elif resolution=="9:16":
            wf_w = 544
            wf_h = 960
        elif resolution=="720":
            wf_w = 448
            wf_h = 800
        elif resolution=="720_16:9":
            wf_h = 448
            wf_w = 800
        if negative_prompt!="":
            print("tạo video type 7)))))))))")
            workflow["241"]["inputs"]["negative_prompt"] = negative_prompt
            if resolution == "1:1":
                wf_w = 640
                wf_h = 640
            elif resolution=="16:9":    
                wf_w = 960
                wf_h = 544 
            elif resolution=="9:16":
                wf_w = 544
                wf_h = 960
            elif resolution=="720":
                wf_w = 448
                wf_h = 800
            elif resolution=="720_16:9":
                wf_h = 448
                wf_w = 800
        workflow["245"]["inputs"]["value"] = wf_w
        workflow["246"]["inputs"]["value"] = wf_h
        # workflow["245"]["inputs"]["value"] = 448
        # workflow["246"]["inputs"]["value"] = 800
        # img.close()

        prefix = job_id
        workflow["131"]["inputs"]["filename_prefix"] = prefix

# ===========================================================================
        print("📤 Sending workflow to ComfyUI...")

        resp = await queue_prompt(workflow)
        prompt_id = resp["prompt_id"]
        client_id = resp["client_id"]
        print(f"✅ Workflow sent! Prompt ID: {prompt_id}")

        success = await wait_for_completion(prompt_id, client_id)

        if not success:
            print("❌ Workflow failed")
            return None

        print("🔍 Searching for the generated video...")

        video_path = await find_latest_video(prefix)
        
        if video_path:
            await delete_file_async(str(video_path.replace("-audio.mp4",".mp4")))
            await delete_file_async(str(video_path.replace("-audio.mp4",".png")))
            file_size = os.path.getsize(video_path)
            print(f"📏 File size: {file_size / (1024*1024):.2f} MB")
            # wf_w = 720
            # wf_h = 1280
            if resolution=="16:9":    
                wf_w = 1280
                wf_h = 720 
            elif resolution=="9:16":
                wf_w = 720
                wf_h = 1280
            await scale_video(
                input_path=video_path,
                output_path=output_path,
                target_w=wf_w,
                target_h=wf_h
            )
            await delete_file_async(str(video_path))

            return output_path
        else:
            print("❌ Cannot findout video")
            return None
    finally:
        await stop_comfyui(comfy_process)

async def move_file_async(src_path, dst_path):
    def move_file():
        os.rename(src_path, dst_path)
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, move_file)
async def delete_file_async(file_path: str):
    def delete_file():
        if os.path.exists(file_path):
            os.remove(file_path)
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, delete_file)
# ===============================================================
import cv2
import os

async def scale_video(input_path, output_path, target_w, target_h):

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not exist: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    scale_w = target_w / orig_w
    scale_h = target_h / orig_h
    scale = max(scale_w, scale_h)
    
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    info = {
        'original_size': (orig_w, orig_h),
        'target_size': (target_w, target_h),
        'scale_factor': scale,
        'final_size': (new_w, new_h),
        'fps': fps,
        'total_frames': total_frames
    }
    
    print(f"Original video: {orig_w}x{orig_h}")
    print(f"Target size: {target_w}x{target_h}")
    print(f"Scale factor: {scale:.4f}")
    print(f"Final size: {new_w}x{new_h}")

    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        scaled_frame = cv2.resize(frame, (new_w, new_h))
        
        out.write(scaled_frame)
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    
    print(f"Video saved: {output_path}")
    
    return info
# ==================================================================
from asyncio import Semaphore

video_semaphore = Semaphore(2)

async def generate_video_fast(prompt, cond_image, cond_audio_path, output_path, job_id, resolution, type,first_time=True,howmuch=1,index=0,howmuch1=0,howmuch2=0,avt_image=""):
    width=720
    height=1280
    str__kl="720x1280"
    video_bg="/home/toan/marketing-video-ai/backgrund_vid/9_16"
    if resolution=="16:9" or resolution=="720_16:9":
        width=1280
        height=720
        str__kl="1280x720"
        video_bg="/home/toan/marketing-video-ai/backgrund_vid/16_9"
    elif resolution=="9:16":
        width=720
        height=1280
    elif resolution=="1:1": 
        width=720
        height=720
    global image_paths_product 
    global image_paths_product_rout360
    global video_paths_product_rout360
    global image_path_sideface

    if len(image_paths_product)==0 or len(image_paths_product) <howmuch-1 or first_time:
        print("==========================================")
        process = None
        server_address= "127.0.0.1:8188"
        process = await start_comfyui1()
        # await asyncio.sleep(8)
        try:
            for i in range(howmuch):
                job_id1 = str(uuid.uuid4())
                image_path = await generate_image_with_comfyui(
                            width=width,
                            height=height,
                            job_id=job_id1,
                            input_image=cond_image
                        )
                # print(image_path)
                image_paths_product.append(image_path[0])
            video_paths_product_rout360=[]
            image_paths_product_rout360=[]
            # =================================================
            for i in range(howmuch1):
                job_id1 = str(uuid.uuid4())
                prompt_pairs = [
                    {
                        "image": "Create a realistic photo of the input product placed on a small pedestal or stand in a bright white studio corner. \
The background is pure white with soft natural light coming from one side, creating gentle shadows and realistic reflections on the surface. \
The product should look clean, sharp, and naturally lit — as if photographed in a professional studio with a minimal setup. \
Keep the product’s shape and texture unchanged.",
                        "video": "Create a smooth cinematic 360-degree rotation video of the input product. The product stays centered on a clean studio background with soft lighting and realistic reflections. The camera slowly orbits around the product in a complete circle, showing all sides with natural motion and focus depth. Keep the product perfectly detailed and consistent with the input image. Use gentle motion blur and subtle shadows to make it feel realistic. The style should look like a professional product commercial shot with high-end studio lighting."
                    },
                    {
                        "image": "Create a realistic image of the input product placed on a small stand or pedestal in a dark studio environment. \
The background is deep black or dark grey, illuminated by a focused light source that highlights the product’s contours and reflections. \
Subtle shadows and a faint rim light enhance depth and texture, giving the scene a premium professional look. \
Keep the product realistic and unchanged, with no distortion.",
                        "video": "Create a smooth cinematic 360-degree rotation video of the input product. The product stays centered on a clean studio background with soft lighting and realistic reflections. The camera slowly orbits around the product in a complete circle, showing all sides with natural motion and focus depth. Keep the product perfectly detailed and consistent with the input image. Use gentle motion blur and subtle shadows to make it feel realistic. The style should look like a professional product commercial shot with high-end studio lighting."
                    },
                    {
                        "image": "A cinematic dark photo of the product surrounded by ice cubes and frost. The product emits a soft blue glow reflecting on nearby ice. Steam rises gently in the background. The lighting is cold and moody, emphasizing clarity and chill freshness.",
                        "video": "A cinematic ultra-realistic slow-motion video of the product standing on an icy surface, surrounded by frost and mist. Ice cubes fall from above and scatter around the product, bouncing and sliding on the cold surface. The product emits a soft blue glow reflecting on the ice. Subtle steam rises in the background, with cold atmospheric lighting and depth of field, emphasizing clarity and chill freshness. 4K, realistic lighting, shallow depth of field, high frame rate, macro lens effect, cinematic color grading."
                    },
                    {
                        "image": "Generate a premium advertisement image using the given product. Place the product on a smooth matte surface with soft smoke drifting beneath and subtle fog layers wrapping around the base. Add elegant key lighting from above and a warm rim light from behind to emphasize shape and texture. The background is minimalist with dark tones fading to light, cinematic contrast, ultra-detailed shadows, and soft reflections for a sophisticated and modern mood.",
                        "video": "a slow-motion advertising video where the product stands on a dark matte surface. Soft smoke waves drift beneath and around it, illuminated by subtle moving light beams from the sides. The camera gently pans and tilts to highlight the product’s shape and logo. The environment feels cinematic, mysterious, and luxurious with a warm-to-cool gradient light"
                    }

                ]

                selected_pair = random.choice(prompt_pairs)
                image_prompt = selected_pair["image"]
                video_prompt = selected_pair["video"]
                image_path = await generate_image_with_comfyui(
                            width=width,
                            height=height,
                            job_id=job_id1,
                            input_image=cond_image,
                            prompt=image_prompt+ str(i)+" ."
                        )
                # print(image_path)
                image_paths_product_rout360.append(image_path[0])
                video_paths_product_rout360.append(video_prompt)
# ========================================================================================
            image_path_sideface=[]
            for i in range(howmuch2):

                job_id1 = str(uuid.uuid4())
                image_path = await generate_image_with_comfyui(
                            width=width,
                            height=height,
                            job_id=job_id1,
                            input_image=avt_image,
                            type_sideface="sideface",
                        )
                
                # print(image_path)
                # image_path_sideface.append(image_path[0])
                image_path_sideface.insert(0, image_path[0])

        except Exception as e:
            print(f"❌ Error creating image with ComfyUI: {e}")
            raise
        finally:
            print(howmuch2)
            await stop_comfyui1(process)


    if type == 2:
        from animation.zoom_in_effect import zoom_and_light_effect
        print("===========================================")
        print(image_paths_product)
        print("===========================================")
        async with video_semaphore:  
            await asyncio.to_thread(
                zoom_and_light_effect,
                image_path=image_paths_product[index],
                audio_path=cond_audio_path,
                output_path=output_path,
                zoom_factor=1.3,
                zoom_portion=0.9,
                light_portion=0.6
            )
    elif type == 3:
        from animation.zoom_out_effect import zoom_and_light_effect1
        
        async with video_semaphore:  
            await asyncio.to_thread(
                zoom_and_light_effect1,
                image_path=image_paths_product[index],
                audio_path=cond_audio_path,
                output_path=output_path,
                zoom_factor=1.3,
                zoom_portion=0.9,
                light_portion=0.5
            )
    elif type ==4:
        from animation.ken_burn import ken_burns_effect
        
        async with video_semaphore:  
            await asyncio.to_thread(
                ken_burns_effect,
                image_path=image_paths_product[index],
                audio_path=cond_audio_path, 
                output_path=output_path
            )
    elif type ==6:
        time_videotype6=get_audio_duration(cond_audio_path)
        async with video_semaphore:  
            await asyncio.to_thread(
                fairyending,
                VIDEO_FOLDER=video_bg,
                FONT_FOLDER="/home/toan/marketing-video-ai/font",
                IMAGE_PATH=cond_image,
                OUTPUT_PATH=output_path,
                DURATION=time_videotype6,
                FONT_SIZE=80
            )
    elif type ==7:
        time_videotype6=get_audio_duration(cond_audio_path)
        from animation.auto_motion import extend_video
        jobid = uuid.uuid4().hex
        filename = f"{jobid}.mp4"
        output=await generate_video_cmd(
                    prompt=video_paths_product_rout360[0],
                    cond_image=image_paths_product_rout360[0],
                    cond_audio_path="/home/toan/marketing-video-ai/directus/english_girl_3s.wav", 
                    output_path=filename,
                    job_id=jobid,
                    resolution=resolution,
                    negative_prompt="human, people, text, watermark, logo, extra object, overexposure, low-quality, distortion, blur, messy background, cartoon, unrealistic texture"
                )
        os.remove(image_paths_product_rout360[0])
        video_paths_product_rout360.pop(0)
        image_paths_product_rout360.pop(0)
        
        # ========================================
        async with video_semaphore:  
            await asyncio.to_thread(
                extend_video,
                input_path=filename,
                output_path=output_path,
                target_duration=time_videotype6,
                mode="pingpong",
                resolution=str__kl 
            )
        os.remove(filename)
    elif type ==8:
        print("sấdasdasdasdasdasdasdasdasdasd")
        time_videotype6=get_audio_duration(cond_audio_path)
        jobid = uuid.uuid4().hex
        number = random.randint(1, 1000)
        output=await generate_video_cmd(
                    prompt="A realistic video of a person confidently giving a lecture in front of a indoor background. The person’s face is turned to one side, maintaining that direction throughout the video without ever facing the camera directly. Their expression remains neutral and professional, with no head movement. Their hands moves slowly, naturally, and with subtle variation to emphasize their words, creating the impression of a teacher explaining a lesson.",
                    cond_image=image_path_sideface[0],
                    cond_audio_path=cond_audio_path, 
                    output_path=output_path,
                    job_id=jobid,
                    resolution=resolution,
                    negative_prompt="change perspective, bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards "+str(number)
                )
        os.remove(image_path_sideface[0])
        
        image_path_sideface.pop(0)      
    # await delete_file_async(str(image_paths_product[index]))
    return output_path
 
# ==========================================================================================
async def generate_image_with_comfyui( width,height, job_id ,input_image=None,prompt=None,type_sideface=None):
    # process = None
    # server_address= "127.0.0.1:8188"
    # process = await start_comfyui1()
    
    # await asyncio.sleep(8)
    try:
        print("🔄 Loading workflow...")
        if type_sideface=="sideface":
            workflow_path="/home/toan/marketing-video-ai/workflow/Qwen+Image+Edit+Plus(2509)+Basic+Version (1)_api.json"
        else:
            workflow_path="/home/toan/anymateme-visualengine/workflow/Qwen IMAGE Edit 2509 Three Image Edit_api (2).json"
        # print(f"Workflow path: {workflow_path}")
        workflow = await load_workflow1(workflow_path)
        # print(input_image)
        workflow["78"]["inputs"]["image"] = input_image if input_image else "none"
        workflow["3"]["inputs"]["steps"]=8
        
        if "111" in workflow:
            if prompt is not None:
                workflow["111"]["inputs"]["prompt"] = prompt
            else:
                nsdaaff=get_random_prompt()
                workflow["111"]["inputs"]["prompt"] = nsdaaff
            if type_sideface=="sideface":
                print("change side??????????????????????hehe")
                workflow["111"]["inputs"]["prompt"] = "change the pose of the person to the reference image. keep the same background"
            # print(nsdaaff)
            # print("||||||||||||||||||||||||||||||||||||||||||||||")
        # ======================================================================
        if "108" in workflow and type_sideface=="sideface":
            image_paths_ref16_9 = [
                "/home/toan/marketing-video-ai/sideface_image_ref/5.png",
                "/home/toan/marketing-video-ai/sideface_image_ref/6.png"
            ]
            image_paths_ref9_16=[
                "/home/toan/marketing-video-ai/sideface_image_ref/916_1.png",
                "/home/toan/marketing-video-ai/sideface_image_ref/916_2.png"
            ]
            if width > height:
                random_image = random.choice(image_paths_ref16_9)
            else:
                random_image = random.choice(image_paths_ref9_16)
            workflow["108"]["inputs"]["image"] = random_image
        # =================================================================
        if "112" in workflow:
            workflow["112"]["inputs"]["width"] = width
            workflow["112"]["inputs"]["height"] = height
        if "110" in workflow :
            workflow["110"]["inputs"]["prompt"] = "human, text, watermark, logo, extra objects, hands, people, human, low quality, blurry, distorted, messy background, overexposed, unrealistic shadows, poor lighting"

            if type_sideface=="sideface":
                number = random.randint(1, 1000)

                workflow["110"]["inputs"]["prompt"] = "bright tones, overexposed, blurred details, move, head movement, subtitles, style, works, paintings, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs "+str(number)

        prefix = f"{job_id}/{job_id}"
        if "60" in workflow:
            workflow["60"]["inputs"]["filename_prefix"] = prefix
        
        print("📤 Sending workflow to ComfyUI...")
        resp = await queue_prompt1(workflow, server_address)

        prompt_id = resp["prompt_id"]
        client_id = resp["client_id"]
        print(f"✅ Workflow sent! Prompt ID: {prompt_id}")
        
        success = await wait_for_completion1(prompt_id, client_id, server_address)
        if not success:
            print("❌ Workflow Failed")
            return None

        print("🔍 Searching for created image...")
        image_path = await find_image_by_id(job_id)
        return image_path
        
    except Exception as e:
        print(f"❌ Error creating image with ComfyUI: {e}")
        raise
    finally:
        # await stop_comfyui1(process)
        print("done 1 image")
# ===================================================================================

async def load_workflow1(path):
    async with aiofiles.open(path, "r", encoding='utf-8') as f:
        content = await f.read()
        return json.loads(content)

async def queue_prompt1(workflow, server_address):
    client_id = str(uuid.uuid4())
    
    payload = {
        "prompt": workflow, 
        "client_id": client_id
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://{server_address}/prompt",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                result = await response.json()
                result["client_id"] = client_id
                return result
            else:
                raise Exception(f"Failed to queue prompt: {response.status}")

async def wait_for_completion1(prompt_id, client_id, server_address):
    print(f"Connecting WebSocket to monitor progress...")
    
    websocket_url = f"ws://{server_address}/ws?clientId={client_id}"
    
    try:
        async with websockets.connect(websocket_url) as websocket:
            print("✅  Đã kết nối WebSocket")
            
            completed_nodes = 0
            
            while True:
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    
                    if isinstance(msg, str):
                        data = json.loads(msg)                        
                        print(f"📨 Receive message: {data.get('type', 'unknown')}")
                        
                        if data["type"] == "execution_start":
                            print(f"🚀 Start workflow execution with prompt_id: {data.get('data', {}).get('prompt_id')}")
                        
                        elif data["type"] == "executing":
                            node_id = data["data"]["node"]
                            current_prompt_id = data.get("data", {}).get("prompt_id")

                            if current_prompt_id == prompt_id:
                                if node_id is None:
                                    print("🎉 Workflow Complete!")
                                    return True
                                else:
                                    completed_nodes += 1
                                    print(f"⚙️  Processing node: {node_id} ({completed_nodes} Completed nodes)")
                        
                        elif data["type"] == "progress":
                            progress_data = data.get("data", {})
                            value = progress_data.get("value", 0)
                            max_value = progress_data.get("max", 100)
                            node = progress_data.get("node")
                            percentage = (value / max_value * 100) if max_value > 0 else 0
                            print(f"📊 Node {node}: {value}/{max_value} ({percentage:.1f}%)")
                        
                        elif data["type"] == "execution_error":
                            print(f"❌ Execution error: {data}")
                            return False
                            
                        elif data["type"] == "execution_cached":
                            cached_nodes = data.get("data", {}).get("nodes", [])
                            print(f"💾 {len(cached_nodes)} cached nodes")
                
                except asyncio.TimeoutError:
                    print("⏰ WebSocket timeout, waiting...")
                    continue
                except Exception as e:
                    print(f"❌ Error WebSocket: {e}")
                    break
                    
    except Exception as e:
        print(f"❌ Cannot connect to WebSocket: {e}")
        return False

async def find_image_by_id(image_id, output_dir=str("ComfyUI/output")):
    def _find_files():
        # PARENT_DIR = os.path.dirname(BASE_DIR)
        output_dir="/home/toan/anymateme-visualengine/ComfyUI/output"
        # output_dir = os.path.join(PARENT_DIR, "ComfyUI/output")
        target_dir = os.path.join(output_dir, str(image_id))
        if not os.path.exists(target_dir):
            print(f"❌ Directory not found: {target_dir}")
            return None
        
        pattern = os.path.join(target_dir, f"{image_id}*.png")
        files = glob.glob(pattern)
        
        if not files:
            print(f"🔍 No file found with id '{image_id}' in {target_dir}")
            return None
        # print("====================================")
        # print(files)
        t=files[0]
        files.append(t)
        # latest_file = max(files, key=os.path.getmtime)
        # print(f"📁 File found: {latest_file}")
        # return latest_file
        files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
        # print(files_sorted)
        # print("====================================")
    
        return files_sorted[:1]

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _find_files)
import asyncio
import aiohttp
import time
from pathlib import Path


# ====== HÀM KHỞI CHẠY / TẮT COMFYUI ======
async def start_comfyui1():
    HOST = "127.0.0.1"
    PORT = 8188
    COMFYUI_DIR = "/home/toan/anymateme-visualengine/ComfyUI"
    process = await asyncio.create_subprocess_exec(
        "python3", "main.py",
        cwd=str(COMFYUI_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    print(f"🚀 ComfyUI started (PID: {process.pid})")
    ready = await wait_for_port_async(HOST, PORT, timeout=120)

    if not ready:
        print("⚠️ ComfyUI không khởi động được đúng cách (port không mở).")
    else:
        print("🎉 ComfyUI sẵn sàng kết nối!")

    return process

async def stop_comfyui1(process):
    if process and process.returncode is None:
        print("🛑 Stopping ComfyUI...")
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
            print("✅ ComfyUI stopped gracefully.")
        except asyncio.TimeoutError:
            print("⚠️ Force killing ComfyUI...")
            process.kill()
            await process.wait()
