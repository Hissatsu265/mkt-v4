import os
import uuid
from pathlib import Path
from typing import List, Dict
import subprocess
import json
import random
import psutil
from app.models.schemas import TransitionEffect
import shutil

import aiohttp
import aiofiles
import websockets
import glob
import time
from config import SERVER_COMFYUI,WORKFLOW_INFINITETALK_PATH,BASE_DIR
from PIL import Image
server_address = SERVER_COMFYUI
from utilities.divide_audio import process_audio_file
from utilities.generate_prompt import generate_prompts
from utilities.merge_video import concat_videos
from utilities.cut_video import cut_video,cut_audio,cut_audio_from_time
from utilities.audio_duration import get_audio_duration
from utilities.audio_processing_infinite import trim_video_start,add_silence_to_start
from utilities.check_audio_safe import wait_for_audio_ready
from utilities.extract_fulltext import process_audio_to_fulltext

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
event=""
full_text_ofauido=""

# ComfyUI Process Pool - one instance per worker
comfyui_processes: Dict[int, asyncio.subprocess.Process] = {}  # worker_id -> process
comfyui_ports: Dict[int, int] = {}  # worker_id -> port
comfyui_locks: Dict[int, asyncio.Lock] = {}  # worker_id -> lock for that instance

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
        concat_name = os.path.join(os.getcwd(), f"{job_id}_concat.mp4")
        from utilities.merge_video import concat_videos
        output_file1 = concat_videos(results, concat_name)

        from utilities.merge_video_audio import replace_audio_trimmed
        output_file = replace_audio_trimmed(output_file1, cond_audio_path, output_path_video)
        
        try:
            os.remove(output_file1)
            # print("fdfsdfsdfsdfsfd")
            for file in results:
                os.remove(file)
            # print("sfdfsdfsdf")
            for path in cond_images:
                if path != "" and os.path.exists(path):
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
# def adjust_end_times(start_times, end_times, audio_end):
#     new_end_times = []
#     for i in range(len(start_times)):
#         if i < len(start_times) - 1:
#             new_end = max(start_times[i + 1] - 0.2, start_times[i])
#         else:
#             new_end = audio_end

#         if new_end - start_times[i] > 2:
#             new_end -= 0.8

#         new_end_times.append(new_end)
#     return new_end_times
def adjust_end_times(start_times, end_times, audio_end):
    new_end_times = []
    for i in range(len(start_times)):
        if i < len(start_times) - 1:
            new_end = max(start_times[i + 1] - 0.2, end_times[i])
        else:
            new_end = audio_end

        if new_end - start_times[i] > 1.5:
            new_end = start_times[i]+1.5
      
        new_end_times.append(new_end)
    last = new_end_times[-1]
    if last>audio_end:
      new_end_times[-1]=audio_end
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
# =================================================
def generate_valid_sequence(length=10):
    nums = [1, 2, 3, 5, 7, 8, 9, 6]
    sequence = []
    last_seen = {num: -10 for num in nums}

    for i in range(length):
        candidates = []

        for num in nums:
            # 🔹 1. 4 cảnh đầu chỉ dùng [1,9,8,7]
            if i < 4 and num not in [1, 9, 8, 7]:
                continue
            # 🔹 2. Số 6 chỉ được xuất hiện ở cuối
            if num == 6 and i != length - 1:
                continue
            # 🔹 3. Không trùng với cảnh trước
            if i > 0 and num == sequence[-1]:
                continue
            # 🔹 4. 2 và 3 cách nhau ít nhất 6 cảnh
            if num == 2 and i - last_seen[3] < 6:
                continue
            if num == 3 and i - last_seen[2] < 6:
                continue
            # 🔹 5. Mỗi số chỉ được lặp lại sau ít nhất 6 cảnh
            if i - last_seen[num] < 6:
                continue

            candidates.append(num)

        # 🔸 Nếu là cảnh thứ 2 mà chưa có 1 → bắt buộc chọn 1
        if i == 1 and 1 not in sequence:
            value = 1
        else:
            if not candidates:
                return None
            value = random.choice(candidates)

        sequence.append(value)
        last_seen[value] = i

    # 🔹 6. Đảm bảo số 1 có trong 2 cảnh đầu
    if 1 not in sequence[:2]:
        return None

    return sequence


def generate_presets(count=10, length=10):
    presets = []
    attempts = 0
    while len(presets) < count and attempts < 1000:
        seq = generate_valid_sequence(length)
        attempts += 1
        if seq and seq not in presets:
            presets.append(seq)
    return presets
# =====================================
import wave
import struct

def create_silent_wav(filename, duration_seconds, sample_rate=44100):
    num_samples = int(duration_seconds * sample_rate)

    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)          # mono
        wav_file.setsampwidth(2)          # 16 bit
        wav_file.setframerate(sample_rate)

        for _ in range(num_samples):
            wav_file.writeframes(struct.pack("<h", 0))

    print("Created:", filename)

# =====================================
def custom_random_sequence(n):
    # preset_sequences = generate_presets(count=5, length=max(10, n))
    # chosen_sequence = random.choice(preset_sequences)
    # return chosen_sequence[:n]
    # ===============tà đạo===========================
    presets = [
        # [1, 9, 8, 9, 7, 5, 2],
        # [1, 8, 9, 5, 7, 1, 6],
        # [1, 9, 5, 9, 8, 1, 3]
        [1,  8,  7, 5, 2],
        [1, 8,  5, 7, 1, 6],
        [1,  5,  8, 1, 3]
    ]
    arr = random.choice(presets)
    result = []
    while len(result) < n:
        result.extend(arr)
    return result[:n]
    # return [9,9,9,1,1,1][:n]
# ====================================================
def check_time_gap(start_times, end_times):
    # trường hợp chỉ có 1 start time
    if len(start_times) == 1:
        return 1

    # trường hợp có 2 start time
    if len(start_times) == 2:
        gap = start_times[1] - end_times[0]
        if gap < 0.8:
            return 1
        else:
            return 2

    # các trường hợp khác trả về 2
    return 2
def custom_random_sequence222(n):
    if n <= 0:
        return []

    nums = [1, 8, 5]
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
                # Nếu là 1 hoặc 7, phải cách lần trước ít nhất 4 cảnh
                if num in [1, 7,9] and i - last_seen[num] < 4:
                    continue
                # Nếu là 2-6 (trừ 1 và 7), cách lần trước ít nhất 5 cảnh
                if num not in [1, 7] and i - last_seen[num] < 5:
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
def custom_random_sequence111(n):
    if n <= 0:
        return []

    nums = [1, 8]
    sequence = []

    # đánh dấu lần xuất hiện
    last_seen = {num: -10 for num in nums}
    last = None

    for i in range(n):

        # --- Cảnh đầu tiên luôn là số 1 ---
        if i == 0:
            value = 1

        # --- Cảnh thứ 2: vẫn đảm bảo có 1 trong 2 cảnh đầu ---
        elif i == 1:
            if 1 not in sequence:
                value = 1
            else:
                candidates = [x for x in nums if x != last]
                value = random.choice(candidates)

        else:
            candidates = []
            for num in nums:
                # không trùng với cảnh trước
                if num == last:
                    continue
                # nếu là 1 hoặc 7 hoặc 9 cần cách ít nhất 4 cảnh
                if num in [1, 7, 9] and i - last_seen[num] < 4:
                    continue
                # nếu là số khác thì cách 5 cảnh
                if num not in [1, 7] and i - last_seen[num] < 5:
                    continue

                candidates.append(num)

            # nếu không có lựa chọn hợp lệ thì nới lỏng
            if not candidates:
                candidates = [x for x in nums if x != last]

            value = random.choice(candidates)

        sequence.append(value)
        last_seen[value] = i
        last = value

    return sequence
    # return [9,9,9,9,9,9,9,1,5][:n]  
    # return [2,3,4,2,3,4,2,3,4,2,3,4][:n]

def get_video_path(resolution: str, character, background):
    BASE_DIR11=str(BASE_DIR)+"/intro_vid"
    char_name = character.value if hasattr(character, "value") else character
    bg_name = background.value if hasattr(background, "value") else background

    target_dir = os.path.join(BASE_DIR11, resolution, char_name, bg_name)

    if not os.path.isdir(target_dir):
        return None

    videos = [
        f for f in os.listdir(target_dir)
        if f.lower().endswith((".mp4", ".mov", ".mkv"))
    ]
    if not videos:
        return None
    return os.path.join(target_dir, random.choice(videos))
class VideoService:
    def __init__(self):
        from config import OUTPUT_DIR
        self.output_dir = OUTPUT_DIR

    def generate_output_filename(self) -> str:
        unique_id = str(uuid.uuid4())[:8]
        timestamp = int(asyncio.get_event_loop().time())
        return unique_id, f"video_{timestamp}_{unique_id}.mp4"

    async def create_video(self, image_paths: List[str], prompts: List[str], audio_path: str, resolution: str, job_id: str,background:str,character:str, worker_id: int = 0, gpu_id: int = 0) -> str:
        # print("Starting video creation...")
        # print(character)
        # print(background)
        # print(resolution)
        
        # time.sleep(2)
        # await job_service.update_job_status(job_id, "processing", progress=0)
        # return "https://cms.anymateme.pro/assets/425ad513-d54c-4faa-8098-1b36feb06729",[]
        print("BASEDIR: ",BASE_DIR)
        global full_text_ofauido

        full_text_ofauido=""
        jobid, output_filename = self.generate_output_filename()

        jobid=job_id
        output_path = self.output_dir / output_filename
        try:
            
            await job_service.update_job_status(job_id, "processing", progress=0)
            list_scene = await run_job(jobid, prompts, image_paths, audio_path, output_path,resolution,character,background, worker_id, gpu_id)    
            print("Uploading video to Directus...")

            path_directus= Uploadfile_directus(str(output_path))
            print("dfsf")
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
async def run_job(job_id, prompts, cond_images, cond_audio_path,output_path_video,resolution,character,background, worker_id: int = 0, gpu_id: int = 0):
    generate_output_filename = output_path_video
    list_scene=[]
    
    if prompts[0]=="" or prompts[0] is None or prompts[0].lower() == "none":
        prompts[0]="A realistic video of a person confidently presenting a product they are holding. The person speaks clearly and professionally, as if explaining the product’s features in an advertisement. Their facial expression remains pleasant and natural, with slight movements to appear engaging but not exaggerated. Their hand gestures are smooth and minimal, focusing attention on the product, creating the impression of a calm and confident presenter in a product promotion video."
    if get_audio_duration(cond_audio_path) > 10:
        output_directory = "output_segments"
        os.makedirs(output_directory, exist_ok=True)
        output_paths,durations, result = process_audio_file(cond_audio_path, output_directory)
        # =========================================================================
        results=[]
        global event
        first_time=True
        last_value=None
        if len(cond_images)>1 and cond_images[1]!="":
            list_random = custom_random_sequence(len(output_paths))
        else:
            cond_images.append(cond_images[0])
            if event=="Christmas":
                list_random = custom_random_sequence111(len(output_paths))
                prompts[0]="A festive cartoon-style video of a character in a holiday environment. The background has subtle ambient motion, soft light shifts, and gentle environmental details to make the scene lively and realistic. The character is standing straight, calm, and natural, without any exaggerated movements or expressions"
            else:
                list_random = custom_random_sequence222(len(output_paths))
                prompts[0]="A realistic video of a person confidently giving a lecture. Their face remains neutral and professional, without expressions or head movement. Their hands moves up and down slowly and naturally to emphasize his words without swinging his arms from side to side, creating the impression of a teacher explaining a lesson."
        print("=====================")
        print("Random sequence for transition effects:", list_random)
        time.sleep(20)
        list_random=[8,3,8,1,1,1,1,1,1,1,1][:len(output_paths)]
        print("=====================")

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
                full_text=""
                from animation.addtittle import create_keyword_video

                from app.services.extract_keyword import process_keywordfromaudi

                keywords, start_times_list1, end_times_list,full_text = await asyncio.to_thread(
                    process_keywordfromaudi, audiohavesecondatstart
                )
                # print(keywords," hehehe")
                # print(start_times_list1," hehehe")
                # print(end_times_list," hehaeehe")
                new_end_times = []
                start_times_list = []
                time_audio=get_audio_duration(audiohavesecondatstart)
                if len(keywords) > 0 and len(start_times_list1) > 0 and len(new_end_times) > 0:
                    start_times_list = [max(0, t - 0.4) for t in start_times_list1]
                    new_end_times = adjust_end_times(start_times_list, end_times_list, time_audio)
                else:
                    keywords =[]

# ========================================================================
                ratio="16:9"
                if  resolution=="9:16":
                    ratio="9:16"
                # ====================================================
                global full_text_ofauido
                full_text_ofauido = await asyncio.to_thread(
                    process_audio_to_fulltext,cond_audio_path
                )
                print("Full text of audio:", full_text_ofauido)
                image_prompt11, video_prompt111= await asyncio.to_thread(
                    generate_prompts, full_text_ofauido,str(full_text)
                )
                print("=========PROMPTs for scene ",i,"=============")
                print("Image Prompt:", image_prompt11)
                print("Video Prompt:", video_prompt111)
                # ====================================================
                from utilities.text_to_image import generate_image
                result= await asyncio.to_thread(
                    generate_image, str(image_prompt11),ratio
                )
                result_text2image_path=""
                if result.get("success", False):
                    result_text2image_path = result['file_path']
                else:
                    list_random[i] = 1

                # print(result_text2image_path," àdfsdfsdfheheheeh")
# ============================================================================
                
                clip_name_test=os.path.join(os.getcwd(), f"{job_id}_clip_{i}_test.mp4")

                if time_audio<=5.5 and list_random[i] == 9:
                    from animation.zoom_in_type9 import zoom_and_light_effect_type9
                    async with video_semaphore:  
                        await asyncio.to_thread(
                            zoom_and_light_effect_type9,
                            image_path=result_text2image_path,
                            duration=time_audio,
                            output_path=clip_name_test,
                            zoom_factor=1.3,
                            zoom_portion=0.8,
                            light_portion=0.9
                        )
                    
                    from animation.add_keyword_type9 import create_keyword_video_noblur
                    await asyncio.to_thread(
                        create_keyword_video_noblur,
                        video_path=clip_name_test,
                        keywords=keywords,
                        start_times=start_times_list,
                        end_times=new_end_times,
                        output_path=clip_name,
                        font_path=str(BASE_DIR)+"/font/MontserratMedium-lgZ6e.otf"
                    )
                    os.remove(result_text2image_path)
                    os.remove(clip_name_test)
                elif list_random[i] == 9:
                    # ==============================create silent file==========
                    silent_file=os.path.join(os.getcwd(), f"{job_id}_silent.wav")
                    create_silent_wav(silent_file, time_audio)
                    # =======================================================
                    clip_name_test=os.path.join(os.getcwd(), f"{job_id}_clip_{i}_test.mp4")
                    output=await generate_video_cmd(
                        prompt="The background features soft seasonal lighting, gentle ambient motion, and subtle details that enhance a warm Christmas atmosphere. A stylized cartoon character moves naturally with smooth, mild gestures. Facial features remain calm and steady without exaggerated expressions. Body movements stay relaxed and fluid, creating a believable and festive animated look.",
                        cond_image=result_text2image_path,
                        cond_audio_path=silent_file,
                        output_path=clip_name_test,
                        job_id=job_id,
                        resolution=resolution,
                        worker_id=worker_id,
                        gpu_id=gpu_id
                    )
                    os.remove(result_text2image_path)
                    os.remove(silent_file)
                    tttttttt=check_time_gap(start_times_list, new_end_times)
                    if tttttttt==1:
                        from animation.add_keyword_type9_blur import create_keyword_videoblur
                        await asyncio.to_thread(
                            create_keyword_videoblur,
                            video_path=clip_name_test,
                            keywords=keywords,
                            start_times=start_times_list,
                            end_times=new_end_times,
                            output_path=clip_name,
                            font_path=str(BASE_DIR)+"/font/MontserratMedium-lgZ6e.otf"
                        )
                    else:
                        from animation.add_keyword_type9 import create_keyword_video_noblur
                        await asyncio.to_thread(
                            create_keyword_video_noblur,
                            video_path=clip_name_test,
                            keywords=keywords,
                            start_times=start_times_list,
                            end_times=new_end_times,
                            output_path=clip_name,
                            font_path=str(BASE_DIR)+"/font/MontserratMedium-lgZ6e.otf"
                        )
                    os.remove(clip_name_test)


# ========================================================================                

                # if len(keywords) <= 0:
                #     list_random[i] = 1
                #     if i + 1 < len(list_random) and list_random[i + 1] == 1:
                #         list_random[i] = 5
                #     if i - 1 >= 0 and list_random[i - 1] == 1:
                #         list_random[i] = 5 

                # else: 
                #     font_path_hehehehe = [
                #         "/home/toan/marketing-video-ai/font/MontserratMedium-lgZ6e.otf",
                #         "/home/toan/marketing-video-ai/font/MontserratSemibold-8M8PB.otf",
                #         "/home/toan/marketing-video-ai/font/RobotoBoldItalic-4e0x.ttf",
                #     ]

                #     color_combos = [
                #         {"name": "White & Black", "bg": "(255, 255, 255)", "text": "#000000"},
                #         {"name": "Light Gray & Black", "bg": "(229, 231, 235)", "text": "#000000"},
                #         {"name": "Medium Gray & Black", "bg": "(209, 213, 219)", "text": "#000000"},
                #     ]
                    
                #     time_video=get_audio_duration(audiohavesecondatstart)

                #     new_end_times = adjust_end_times(start_times_list, end_times_list, time_video)

                #     selected = random.choice(color_combos)

                #     font_path= random.choice(font_path_hehehehe)

                #     resolution_tuple =(1280,720) if resolution == "16:9" else (720,1280)
                #     bg_color = safe_parse_color(selected['bg'])
                #     # font_color = safe_parse_color(selected['text'])
                #     await asyncio.to_thread(
                #         create_keyword_video,
                #         keywords,                      # 1
                #         start_times_list,              # 2
                #         new_end_times,                 # 3
                #         time_video,                    # 4 = duration
                #         resolution_tuple,              # 5 = resolution
                #         font_path,                     # 6 = font (theo định nghĩa hàm)
                #         bg_color,                      # 7 = bg_color
                #         selected['text'],              # 8 = font_color
                #         None,                          # 9 = font_size (None để tự tính)
                #         random.choice([5, 6]),         # 10 = effect_type
                #         clip_name                      # 11 = output_path
                #     )
# =========================================================================
            if (list_random[i] == 1):
                output=await generate_video_cmd(
                    prompt=prompts[current_value],
                    cond_image=str(file_path),#
                    cond_audio_path=audiohavesecondatstart,
                    output_path=clip_name,
                    job_id=job_id,
                    resolution=resolution,
                    worker_id=worker_id,
                    gpu_id=gpu_id
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
                    resolution=resolution,
                    worker_id=worker_id,
                    gpu_id=gpu_id
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
                    avt_image=str(cond_images[0]),

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
            # print(list_random,"]]]]]]]]]]]]]]]]]]]]]]]]]ư")
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
        # video_name1111111 = os.path.join(os.getcwd(), f"{job_id}clip_hehehehehe.mp4")
        
        # for i in range(1, len(list_scene)):
        #     list_scene[i] = list_scene[i-1] + list_scene[i]
        # for i in range(len(list_scene)):
        #     list_scene[i] = list_scene[i] - 0.5*(i+1) 

        # complex_effects=[]
        # transition_effects = random_transition_list(len(list_scene))
        # for i in range(len(list_scene)):
        #     start_time = list_scene[i] - 1 / 2
        #     end_time = list_scene[i] + 1 / 2
        #     effect_name = transition_effects[i]
        #     complex_effects.append({
        #         "start_time": start_time,
        #         "end_time": end_time,
        #         "effect": effect_name
        #     })
        #     print("================================")
      
        # loop = asyncio.get_event_loop()
        # await loop.run_in_executor(
        #     None,  
        #     apply_multiple_effects,
        #     output_path_video,
        #     str(video_name1111111),
        #     complex_effects,
        #     "high"
        # )
        # await delete_file_async(output_path_video)
        # await asyncio.to_thread(os.rename, video_name1111111, output_path_video)

        #========================================================
        
        return list_scene
    # =========================================================================================
    # =========================================================================================
    # =========================================================================================    

    else:
        watermark_english=True
        fffffffff=prompts[0].strip()
        if fffffffff.lower()=="german":
            watermark_english=False 
        
        # ===========================================================
        path_intovid_base_resolution="916"
        if resolution=="16:9":
            path_intovid_base_resolution="169"
        video_intro = get_video_path(
            resolution=path_intovid_base_resolution,
            character=character,
            background=background
        )
        # if video:
        #     print("Found video:", video)
        # else:
        #     print("No video available")
        # ===========================================================
        if event=="Christmas" and prompts[0].strip()=="":
            prompts[0]="A festive cartoon-style video of a character in a holiday environment. The background has subtle ambient motion, soft light shifts, and gentle environmental details to make the scene lively and realistic. The character is standing straight, calm, and natural, without any exaggerated movements or expressions"
            # print("use base promtp")
        elif event=="Christmas" and len(prompts[0].strip()) < 10:
            print("using our prompt <><<>><<<<>>>>")
            prompts[0]="A festive cartoon-style video of a character in a holiday environment. The background has subtle ambient motion, soft light shifts, and gentle environmental details to make the scene lively and realistic. The character is standing straight, calm, and natural, without any exaggerated movements or expressions"
        elif prompts[0].strip()=="":

            prompts[0]="A realistic video of a person confidently giving a lecture. Their face remains neutral and professional, without expressions or head movement. Their hands moves up and down slowly and naturally to emphasize his words without swinging his arms from side to side, creating the impression of a teacher explaining a lesson."

        if background=="workshop" and event=="Christmas":
            prompts[0]="A festive cartoon style video of a character in a holiday workshop environment. The background shows soft ambient motion, warm light shifts, and gentle environmental details to keep the scene lively and believable. The character stands straight, calm, and natural, without exaggerated movements or expressions. In the background, a small wooden toy car moves across the scene one time then stop, passing smoothly behind the character without drawing too much attention."
        # if background=="firework" and event=="Christmas":
        #     prompts[0]="A high-quality 3D cartoon holiday video based on the reference image. Santa Claus stands centered, straight, calm, and natural, mostly still. Light snow falls gently. Fireworks appear one at a time, including golden chrysanthemum fireworks, soft willow fireworks, subtle crackle effects, and occasional fan-shaped launches from distant rooftops and far behind the character, rising at varied heights, bursting elegantly, and fully fading out before the next appears. No ring-shaped fireworks. Cinematic lighting, steady camera, warm festive night mood."
        # elif background=="northern" and event=="Christmas":
        #     prompts[0]="A charming cartoon style holiday video where the environment feels gently alive. Tree decorations shimmer with mild blinking patterns, soft light ripples across the snow, and tiny particles float slowly in the air. The character holds a steady stance, appearing calm and natural."
        #     print("use prompt of workshop")
        # print( prompts[0])
        # print(background)
        # print(character)
        # import time
        # time.sleep(15)
        if background=="firework" and event=="Christmas":
            os.remove(str(cond_images[0]))
      
            # ==================replace character image for firework background=========
            if character=="santa" and resolution=="16:9":
                src_image=str(BASE_DIR)+"/character_image_firework/169/santa/1767077592-acf7c974.jpg"
            elif character=="santa" and resolution=="9:16":
                src_image=str(BASE_DIR)+"/character_image_firework/916/santa/first_frame1 (1).jpeg"

            if character=="elf" and resolution=="16:9":
                src_image=str(BASE_DIR)+"/character_image_firework/169/elf/10e98c15-cad6-4025-99ea-f89f62e2ee9a_0.jpg"
            elif character=="elf" and resolution=="9:16":
                src_image=str(BASE_DIR)+"/character_image_firework/916/elf/1767059929-012a2700.jpg"
            
            if character=="reindeer" and resolution=="16:9":
                src_image =str(BASE_DIR)+"/character_image_firework/169/reindeer/1767059803-0635a074.jpg"
            elif character=="reindeer" and resolution=="9:16":
                src_image=str(BASE_DIR)+"/character_image_firework/916/reindeer/1767060685-d9bb08fd (1).jpg"

            if character=="snowman" and resolution=="16:9":
                src_image=str(BASE_DIR)+"/character_image_firework/169/snowman/1767081598-efe84df9.jpg"
            elif character=="snowman" and resolution=="9:16":
                src_image=str(BASE_DIR)+"/character_image_firework/916/snowman/Gemini_Generated_Image_1b943r1b943r1b94.png"
            
            os.makedirs(os.path.dirname(str(cond_images[0])), exist_ok=True)
            shutil.copy2(src_image, str(cond_images[0]))
        print("============sgageeeee===============")
        # =================================================================
      
        audiohavesecondatstart = add_silence_to_start(cond_audio_path, job_id, duration_ms=500)
        generate_output_filename=os.path.join(os.getcwd(), f"{job_id}_noaudio.mp4")
        if wait_for_audio_ready(audiohavesecondatstart, min_size_mb=0.02, max_wait_time=60, min_duration=2.0):
            print("Detailed check passed!")
        print("============sgageeeee===============")
        # =================================================================
        file_path = str(cond_images[0])
        file_root, file_ext = os.path.splitext(file_path)
        await job_service.update_job_status(job_id, "processing", progress=50)
        print("Generating video with prompt:", prompts[0])
        output=await generate_video_cmd(
            prompt=prompts[0],
            cond_image=file_path,
            cond_audio_path=audiohavesecondatstart,
            output_path=generate_output_filename,
            job_id=job_id,
            resolution=resolution,
            worker_id=worker_id,
            gpu_id=gpu_id
        )  
        await job_service.update_job_status(job_id, "processing", progress=97)
        # =========================replace audio===================================
        from utilities.merge_video_audio import replace_audio_trimmed
        tempt=trim_video_start(generate_output_filename, duration=0.5)
        output_file = replace_audio_trimmed(generate_output_filename,cond_audio_path,output_path_video)
        print("===>done replace audio")
        # ===========================transition==================================
        from utilities.mergeintro import merge_videos
        if video_intro:
            await asyncio.to_thread(
                merge_videos,
                str(video_intro),
                str(output_path_video),
                watermark_english
            )
        else:
            print("No video intro available")
        print("===>done merge intro")
        if os.path.exists(output_path_video):
            print("File tồn tại")
        else:
            print("File không tồn tại")
        try:
            os.remove(str(generate_output_filename))
            os.remove(str(audiohavesecondatstart))
            os.remove(str(cond_audio_path))
            os.remove(str(file_path))
        except Exception as e:
            print(f"❌ Error removing temporary files: {str(e)}")
        print("===>done remove temp")
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
async def wait_for_port_async(host: str, port: int, timeout: int = 200) -> bool:
 
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

    ready = await wait_for_port_async(HOST, PORT, timeout=400)

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

# ========== Multi-Worker ComfyUI Process Pool ==========
async def start_comfyui_for_worker(worker_id: int, gpu_id: int = 0) -> tuple:
    """Start a dedicated ComfyUI instance for a specific worker on a specific GPU"""
    from config import COMFYUI_BASE_PORT

    port = COMFYUI_BASE_PORT + worker_id
    host = "127.0.0.1"

    print(f"[Worker {worker_id}] 🚀 Starting ComfyUI on GPU {gpu_id}, port {port}...")

    # Create environment with GPU assignment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Isolate to specific GPU

    process = await asyncio.create_subprocess_exec(
        "python3", "main.py", "--port", str(port),
        cwd=str(BASE_DIR / "ComfyUI"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env  # Pass environment with GPU assignment
    )

    print(f"[Worker {worker_id}] ComfyUI started (PID: {process.pid}), waiting for port {port}...")

    ready = await wait_for_port_async(host, port, timeout=400)

    if not ready:
        print(f"[Worker {worker_id}] ⚠️ ComfyUI failed to start on port {port}")
        raise Exception(f"ComfyUI failed to start on port {port}")

    print(f"[Worker {worker_id}] ✅ ComfyUI ready on port {port}")

    return process, port

async def stop_comfyui_for_worker(worker_id: int):
    """Stop ComfyUI instance for a specific worker"""
    if worker_id not in comfyui_processes:
        return

    process = comfyui_processes[worker_id]
    port = comfyui_ports.get(worker_id, "unknown")

    if process and process.returncode is None:
        print(f"[Worker {worker_id}] 🛑 Stopping ComfyUI (port {port})...")
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
            print(f"[Worker {worker_id}] ✅ ComfyUI stopped")
        except asyncio.TimeoutError:
            print(f"[Worker {worker_id}] ⚠️ Force killing ComfyUI...")
            process.kill()
            await process.wait()

    # Cleanup
    comfyui_processes.pop(worker_id, None)
    comfyui_ports.pop(worker_id, None)

async def get_or_start_comfyui(worker_id: int, gpu_id: int = 0) -> tuple:
    """Get existing ComfyUI for worker or start a new one"""
    # Initialize lock if needed
    if worker_id not in comfyui_locks:
        comfyui_locks[worker_id] = asyncio.Lock()

    async with comfyui_locks[worker_id]:
        # Check if already running
        if worker_id in comfyui_processes:
            process = comfyui_processes[worker_id]
            if process.returncode is None:  # Still running
                port = comfyui_ports[worker_id]
                print(f"[Worker {worker_id}] Reusing existing ComfyUI on GPU {gpu_id}, port {port}")
                return process, port

        # Start new instance with GPU assignment
        process, port = await start_comfyui_for_worker(worker_id, gpu_id)
        comfyui_processes[worker_id] = process
        comfyui_ports[worker_id] = port

        return process, port

async def load_workflow(path="workflow"):
    async with aiofiles.open(path, "r", encoding='utf-8') as f:
        content = await f.read()
        return json.loads(content)

async def queue_prompt(workflow, server_address_override=None):
    client_id = str(uuid.uuid4())
    server = server_address_override or server_address  # Use override if provided

    payload = {
        "prompt": workflow,
        "client_id": client_id
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://{server}/prompt",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                result = await response.json()
                result["client_id"] = client_id
                return result
            else:
                raise Exception(f"Failed to queue prompt: {response.status}")

async def wait_for_completion(prompt_id, client_id, server_address_override=None):
    server = server_address_override or server_address
    websocket_url = f"ws://{server}/ws?clientId={client_id}"
    
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
                            error_data = data.get("data", {})
                            node_id = error_data.get("node_id", "unknown")
                            node_type = error_data.get("node_type", "unknown")
                            exception_message = error_data.get("exception_message", "No message")
                            exception_type = error_data.get("exception_type", "Unknown")
                            traceback_text = error_data.get("traceback", "No traceback")

                            print(f"❌ WORKFLOW ERROR DETAILS:")
                            print(f"   Node ID: {node_id}")
                            print(f"   Node Type: {node_type}")
                            print(f"   Exception: {exception_type}: {exception_message}")
                            print(f"   Traceback: {traceback_text[:500]}")  # First 500 chars
                            print(f"   Full error data: {error_data}")
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
async def generate_video_cmd(prompt, cond_image, cond_audio_path, output_path, job_id,resolution, worker_id: int = 0, gpu_id: int = 0, negative_prompt=""):
    # Get or start ComfyUI for this worker with GPU assignment
    comfy_process, port = await get_or_start_comfyui(worker_id, gpu_id)
    server_address_local = f"127.0.0.1:{port}"

    try:
        print("🔄 Loading workflow...")
        workflow = await load_workflow(str(BASE_DIR) + "/" + WORKFLOW_INFINITETALK_PATH)  
       
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
            wf_w = 1040
            wf_h = 592
        elif resolution=="9:16":
            wf_w = 592
            wf_h = 1040
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
                wf_w = 1120
                wf_h = 640 
            elif resolution=="9:16":
                wf_w = 592
                wf_h = 1040
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

        resp = await queue_prompt(workflow, server_address_local)
        prompt_id = resp["prompt_id"]
        client_id = resp["client_id"]
        print(f"✅ Workflow sent! Prompt ID: {prompt_id}")

        success = await wait_for_completion(prompt_id, client_id, server_address_local)

        if not success:
            print(f"[Worker {worker_id}] ❌ Workflow failed - keeping ComfyUI running for debugging")
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
                wf_w = 1920
                wf_h = 1080
            elif resolution=="9:16":
                wf_w = 1080
                wf_h = 1920
            await scale_video(
                input_path=video_path,
                output_path=output_path,
                target_w=wf_w,
                target_h=wf_h
            )
            await delete_file_async(str(video_path))

            # Only check RAM pressure after successful completion
            if check_ram_status():
                print(f"[Worker {worker_id}] 🚨 RAM pressure detected after successful job, restarting ComfyUI...")
                await stop_comfyui_for_worker(worker_id)
                # Will auto-restart on next job

            return output_path
        else:
            print(f"[Worker {worker_id}] ❌ Cannot find output video")
            return None
    finally:
        print(f"[Worker {worker_id}] ✅ Finished generate_video_cmd")

        # await stop_comfyui(comfy_process)
        # comfy_process = await start_comfyui()

# --- HÀM KIỂM TRA RAM ---
def check_ram_status():
    RAM_LIMIT_GB = 70 
    AVAILABLE_RAM_THRESHOLD_PERCENT = 15
    mem = psutil.virtual_memory()
    used_ram_gb = mem.used / (1024**3) 
    available_ram_percent = mem.available / mem.total * 100 
    print(f"📊 RAM Usage: {used_ram_gb:.2f} GB (Used) | {mem.percent}% (Used Percent)")
    print(f"   RAM Available: {available_ram_percent:.2f}%")
    
    if used_ram_gb > RAM_LIMIT_GB:
        print(f"⚠️ CẢNH BÁO: RAM Used ({used_ram_gb:.2f} GB) vượt quá giới hạn {RAM_LIMIT_GB} GB.")
        return True 
    
    if available_ram_percent < AVAILABLE_RAM_THRESHOLD_PERCENT:
        print(f"⚠️ CẢNH BÁO: RAM Available ({available_ram_percent:.2f}%) quá thấp (dưới {AVAILABLE_RAM_THRESHOLD_PERCENT}%).")
        return True 
    print("✅ RAM status is normal.")  
    return False 
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
    video_bg=str(BASE_DIR)+"/backgrund_vid/9_16"
    if resolution=="16:9" or resolution=="720_16:9":
        width=1280
        height=720
        str__kl="1280x720"
        video_bg=str(BASE_DIR)+"/backgrund_vid/16_9"
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
#                     {
#                         "image": "Create a realistic photo of the input product placed on a small pedestal or stand in a bright white studio corner. \
# The background is pure white with soft natural light coming from one side, creating gentle shadows and realistic reflections on the surface. \
# The product should look clean, sharp, and naturally lit — as if photographed in a professional studio with a minimal setup. \
# Keep the product’s shape and texture unchanged.",
#                         "video": "Create a smooth cinematic 180-degree rotation video of the input product. The product stays centered on a clean studio background with soft lighting and realistic reflections. The camera slowly orbits around the product in a complete circle, showing all sides with natural motion and focus depth. Keep the product perfectly detailed and consistent with the input image. Use gentle motion blur and subtle shadows to make it feel realistic. The style should look like a professional product commercial shot with high-end studio lighting."
#                     },
                    {
                        "image": "Create a realistic image of the input product placed on a small stand or pedestal in a dark studio environment. \
The background is deep black or dark grey, illuminated by a focused light source that highlights the product’s contours and reflections. \
Subtle shadows and a faint rim light enhance depth and texture, giving the scene a premium professional look. \
Keep the product realistic and unchanged, with no distortion.",
                        "video": "Create a smooth cinematic 180-degree rotation video of the input product. The product stays centered on a clean studio background with soft lighting and realistic reflections. The camera slowly orbits around the product in a complete circle, showing all sides with natural motion and focus depth. Keep the product perfectly detailed and consistent with the input image. Use gentle motion blur and subtle shadows to make it feel realistic. The style should look like a professional product commercial shot with high-end studio lighting."
                    },
                    {
                        "image": "Generate a premium advertisement image using the given product. Place the product on a smooth matte surface with soft smoke drifting beneath and subtle fog layers wrapping around the base. Add elegant key lighting from above and a warm rim light from behind to emphasize shape and texture. The background is minimalist with dark tones fading to light, cinematic contrast, ultra-detailed shadows, and soft reflections for a sophisticated and modern mood.",
                        "video": "a slow-motion advertising video where the product stands on a dark matte surface. Soft smoke waves drift beneath and around it, illuminated by subtle moving light beams from the sides. The camera gently pans and tilts to highlight the product’s shape and logo. The environment feels cinematic, mysterious, and luxurious with a warm-to-cool gradient light"
                    },
                    {
                        "image": "Generate a luxurious product advertisement image where the product stands on a textured stone pedestal. Dense golden smoke curls around the base and slowly fades into the background. Use warm amber and orange lighting to create depth and highlight the product’s silhouette. The background transitions from dark bronze to soft gold mist with a cinematic vignette. The overall look is bold, premium, and dramatic.",
                        "video": "a cinematic macro-style product video featuring the item on a rough stone pedestal. Thick amber smoke rolls gently across the base, illuminated by warm, flickering lights. The camera moves slowly from low angle to front view, emphasizing power and luxury. The color tone is warm and cinematic with deep shadows and glowing highlights."
                    },
                    {
                        "image": "Generate a cinematic product showcase placed on a slightly reflective glass pedestal. Soft white smoke drifts and swirls at the base, illuminated by cool blue underlights. The background fades from deep navy to soft cyan with faint volumetric light rays. The mood is futuristic, premium, and clean with crisp reflections and fine atmospheric haze. Key lighting comes from the top with subtle side glows for a high-end commercial look.",
                        "video": "a slow, cinematic product reveal video where the item rests on a glass pedestal surrounded by drifting white smoke. Blue light pulses softly from below, interacting with the fog. The camera makes a smooth orbit move while focus shifts across the surface details. The tone is high-tech, elegant, and mysterious with a refined cool color palette."
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
            global event
            print(event)
            for i in range(howmuch2):
                if event=="Christmas":
                    job_id1 = str(uuid.uuid4())
                    prompt_schoice=[
                        "Keep the exact same cartoon character style, proportions, colors, and facial features from the input image. Do NOT change the character design. Only change the background and environment. The character is sitting in a cozy Christmas living room with a decorated tree, warm lights, and gentle snow falling outside the window.",
                        "Keep the exact same cartoon character style, proportions, colors, and facial features from the input image. Do NOT change the character design. Only change the background and environment. The character is holding a glowing Christmas lantern. Warm lighting and soft textures, designed for holiday-themed video creation.",
                        "Keep the exact same cartoon character style, proportions, colors, and facial features from the input image. Do NOT change the character design. Only change the background and environment. The character stands in front of a snowy cabin decorated with Christmas lights.",
                        "Keep the exact same cartoon character style, proportions, colors, and facial features from the input image. Do NOT change the character design. Only change the background and environment. The character is standing inside a modern airport terminal decorated with Christmas ornaments and warm lights.",
                        "Keep the exact same cartoon character style, proportions, colors, and facial features from the input image. Do NOT change the character design. Only change the background and environment. The character is sitting on a large red chair next to a big decorated Christmas tree shining with fairy lights.",
                    ]
                    image_path = await generate_image_with_comfyui(
                                width=width,
                                height=height,
                                job_id=job_id1,
                                input_image=avt_image,
                                prompt=str(random.choice(prompt_schoice)),
                                type_sideface="event_christmas",
                            )
                else:
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
                FONT_FOLDER=str(BASE_DIR)+"/font",
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
                    cond_audio_path=str(BASE_DIR)+"/directus/english_girl_3s.wav",
                    output_path=filename,
                    job_id=jobid,
                    resolution=resolution,
                    worker_id=worker_id,
                    gpu_id=gpu_id,
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
        time_videotype6=get_audio_duration(cond_audio_path)
        jobid = uuid.uuid4().hex
        number = random.randint(1, 1000)
        if event=="Christmas":
            prompt_evesdf = "A festive cartoon-style video of a character in a holiday environment. The background has subtle ambient motion, soft light shifts, and gentle environmental details to make the scene lively and realistic. The character is standing straight, calm, and natural, without any exaggerated movements or expressions"
        else:
            prompt_evesdf = "A realistic video of a person confidently giving a lecture in front of a indoor background. The person’s face is turned to one side, maintaining that direction throughout the video without ever facing the camera directly. Their expression remains neutral and professional, with no head movement. Their hands moves slowly, naturally, and with subtle variation to emphasize their words, creating the impression of a teacher explaining a lesson.",
        output=await generate_video_cmd(
                    prompt=prompt_evesdf,
                    cond_image=image_path_sideface[0],
                    cond_audio_path=cond_audio_path,
                    output_path=output_path,
                    job_id=jobid,
                    resolution=resolution,
                    worker_id=worker_id,
                    gpu_id=gpu_id,
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
            workflow_path=str(BASE_DIR)+"/workflow/QWen_change_pose.json"
        else:
            # workflow_path="/home/toan/anymateme-visualengine/workflow/Qwen IMAGE Edit 2509 Three Image Edit_api (2).json"
            workflow_path=str(BASE_DIR)+"/workflow/QWen_gen_1_image.json"
        workflow = await load_workflow1(workflow_path)
        workflow["78"]["inputs"]["image"] = input_image if input_image else "none"
        workflow["3"]["inputs"]["steps"]=8
        
        if "111" in workflow:
            if prompt is not None:
                workflow["111"]["inputs"]["prompt"] = prompt
            else:
                nsdaaff=get_random_prompt()
                workflow["111"]["inputs"]["prompt"] = nsdaaff
            if type_sideface=="sideface":
                workflow["111"]["inputs"]["prompt"] = "change the pose of the person to the reference image. keep the same background and take off the headphones."
        # ======================================================================
        if "108" in workflow and type_sideface=="sideface":
            image_paths_ref16_9 = [
                str(BASE_DIR)+"/sideface_image_ref/5.png",
                str(BASE_DIR)+"/sideface_image_ref/6.png"
            ]
            image_paths_ref9_16=[
                str(BASE_DIR)+"/sideface_image_ref/916_1.png",
                str(BASE_DIR)+"/sideface_image_ref/916_2.png"
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
            elif type_sideface is not None:
                print("dùng negative prompt khác nè")
                workflow["110"]["inputs"]["prompt"] = "bright tones, overexposed, blurred details, move, head movement, subtitles, style, works, paintings, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs"
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
        output_dir=str(BASE_DIR)+"/ComfyUI/output"
        target_dir = os.path.join(output_dir, str(image_id))
        if not os.path.exists(target_dir):
            print(f"❌ Directory not found: {target_dir}")
            return None
        
        pattern = os.path.join(target_dir, f"{image_id}*.png")
        files = glob.glob(pattern)
        
        if not files:
            print(f"🔍 No file found with id '{image_id}' in {target_dir}")
            return None
        t=files[0]
        files.append(t) 
        files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
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
    COMFYUI_DIR = str(BASE_DIR)+"/anymateme-visualengine/ComfyUI"
    process = await asyncio.create_subprocess_exec(
        "python3", "main.py",
        cwd=str(COMFYUI_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    print(f"🚀 ComfyUI started (PID: {process.pid})")
    ready = await wait_for_port_async(HOST, PORT, timeout=400)

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


# ========== Shutdown Handler ==========
async def shutdown_all_comfyui():
    """Shutdown all ComfyUI instances on app exit"""
    for worker_id in list(comfyui_processes.keys()):
        await stop_comfyui_for_worker(worker_id)
    print("✅ All ComfyUI instances stopped")

