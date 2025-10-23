# import os
# import uuid
# from pathlib import Path
# from typing import List
# import subprocess
# import json
# import random

# import aiohttp
# import aiofiles
# import websockets
# import glob
# import time
# from config import SERVER_COMFYUI,WORKFLOW_INFINITETALK_PATH,BASE_DIR
# from PIL import Image
# server_address = SERVER_COMFYUI
# from utilities.divide_audio import process_audio_file
# from utilities.merge_video import concat_videos
# from utilities.cut_video import cut_video,cut_audio,cut_audio_from_time
# from utilities.audio_duration import get_audio_duration
# from utilities.audio_processing_infinite import trim_video_start,add_silence_to_start
# from utilities.check_audio_safe import wait_for_audio_ready
# # from paddvideo import add_green_background,replace_green_screen,crop_green_background,resize_and_pad
# # from app.services.create_video_infinitetalk import load_workflow,wait_for_completion,queue_prompt,find_latest_video
# import asyncio
# from directus.file_upload import Uploadfile_directus
# class VideoService:
#     def __init__(self):
#         from config import OUTPUT_DIR
#         self.output_dir = OUTPUT_DIR

#     def generate_output_filename(self) -> str:
#         unique_id = str(uuid.uuid4())[:8]
#         timestamp = int(asyncio.get_event_loop().time())
#         return unique_id, f"video_{timestamp}_{unique_id}.mp4"

#     async def create_video(self, image_paths: List[str], prompts: List[str], audio_path: str, resolution: str, job_id: str) -> str:
#         jobid, output_filename = self.generate_output_filename()
#         output_path = self.output_dir / output_filename
#         try:
            
#             from app.services.job_service import job_service
#             await job_service.update_job_status(job_id, "processing", progress=99)

#             list_scene = await run_job(jobid, prompts, image_paths, audio_path, output_path,resolution)  
#             print(str(output_path)) 
#             # if os.path.exists(str(output_path)):
#             #     print("✅ File tồn tại!")
#             # else:
#             #     print("❌ File không tồn tại.")
#             # return str(output_path),list_scene
#             path_directus= Uploadfile_directus(str(output_path))
#             # if os.path.exists(str(output_path)):
#             #     print("✅ File tồn tại!")
#             # else:
#             #     print("❌ File không tồn tại.")
#             if path_directus is not None and output_path.exists() :
#                 print(f"Video upload successfully: {path_directus}")
#                 print(f"Job ID: {job_id}, Output Path: {path_directus}")
#                 # os.remove(str(output_path))
#                 return str(path_directus),list_scene
#             else:
#                 raise Exception("Cannot upload video to Directus or Video creation failed - output file not found")
    
#         except Exception as e:
#             if output_path.exists():
#                 output_path.unlink()
#             raise e
# async def run_job(job_id, prompts, cond_images, cond_audio_path,output_path_video,resolution):
#     print("resolution: ",resolution)
#     generate_output_filename = output_path_video
#     list_scene=[]
#     if get_audio_duration(cond_audio_path) > 20:
#         output_directory = "output_segments"
#         os.makedirs(output_directory, exist_ok=True)
#         output_paths,durations, result = process_audio_file(cond_audio_path, output_directory)
#         results=[]
#         last_value=None
#         for i, output_path in enumerate(output_paths):
#             if i<len(output_paths)-1:
#                 list_scene.append(get_audio_duration(output_path))
#             # ==============Random image for each scene=============
#             if len(cond_images)>1:
#                 choices = [x for x in range(len(prompts)) if x != last_value] 
#                 current_value = random.choice(choices)  # chọn ngẫu nhiên
#                 last_value = current_value  # lưu 
#             else: current_value=0
#             # ===============================================================================
#             # print(f"Audio segment {i+1}: {output_path} (Duration: {durations[i]}s)")
#             # print(cond_images)
#             # print(f"Image: {cond_images[current_value]}")
#             # print(f"Prompt: {prompts[current_value]}")
#             clip_name=os.path.join(os.getcwd(), f"{job_id}_clip_{i}.mp4")
#             audiohavesecondatstart = add_silence_to_start(output_path, job_id, duration_ms=0)
#             audiohavesecondatstart=str(BASE_DIR / audiohavesecondatstart)
#             # print("dfsdfsdfsd:   ", audiohavesecondatstart)
#             # print(type(audiohavesecondatstart))
       
#             # =================================================================
#             file_path = str(cond_images[current_value])
        
#             output=await generate_video_cmd(
#                 prompt=prompts[current_value],
#                 cond_image=str(file_path),# 
#                 cond_audio_path=audiohavesecondatstart, 
#                 output_path=clip_name,
#                 job_id=job_id,
#                 resolution=resolution
#             )
#             trim_video_start(clip_name, duration=0.5)
#             output_file=cut_video(clip_name, get_audio_duration(output_path)-0.5) 
#             results.append(output_file)
#             try:
#                 # os.remove(pad_file)
#                 # os.remove(crop_file)
#                 os.remove(output_path)
#                 os.remove(clip_name)
#                 os.remove(audiohavesecondatstart)
#             except Exception as e:
#                 print(f"❌ Error removing temporary file {output_path}: {str(e)}")

#         concat_name=os.path.join(os.getcwd(), f"{job_id}_concat_{i}.mp4")
#         output_file1 = concat_videos(results, concat_name)
#         from utilities.merge_video_audio import replace_audio_trimmed
#         output_file = replace_audio_trimmed(output_file1,cond_audio_path,output_path_video)

#         try:
#             os.remove(output_file1)
#             for file in results:
#                 os.remove(file)
#             for path in cond_images:
#                 os.remove(path)
#             os.remove(cond_audio_path)
#         except Exception as e:
#             print(f"❌ Error removing temporary files: {str(e)}")
#         return list_scene
#     else:
#         audiohavesecondatstart = add_silence_to_start(cond_audio_path, job_id, duration_ms=500)
#         generate_output_filename=os.path.join(os.getcwd(), f"{job_id}_noaudio.mp4")
#         if wait_for_audio_ready(audiohavesecondatstart, min_size_mb=0.02, max_wait_time=60, min_duration=2.0):
#             print("Detailed check passed!")

#         # =================================================================
#         file_path = str(cond_images[0])
#         file_root, file_ext = os.path.splitext(file_path)

#         output=await generate_video_cmd(
#             prompt=prompts[0], 
#             cond_image=file_path, 
#             cond_audio_path=audiohavesecondatstart, 
#             output_path=generate_output_filename,
#             job_id=job_id,
#             resolution=resolution
#         )  
#         from utilities.merge_video_audio import replace_audio_trimmed
#         tempt=trim_video_start(generate_output_filename, duration=0.5)
#         output_file = replace_audio_trimmed(generate_output_filename,cond_audio_path,output_path_video)
#         try:
#             print("sdgsfsfgfsgfg",generate_output_filename)
#             os.remove(str(generate_output_filename))
#             os.remove(str(audiohavesecondatstart))
#             os.remove(str(cond_audio_path))
#             os.remove(str(file_path))
#         except Exception as e:
#             print(f"❌ Error removing temporary files: {str(e)}")

#         return list_scene
# # ============================================================================================

# import asyncio
# import signal

# async def start_comfyui():
#     process = await asyncio.create_subprocess_exec(
#         "python3", "main.py",
#         cwd=str(BASE_DIR / "ComfyUI"),  # chỗ chứa main.py
#         stdout=asyncio.subprocess.PIPE,
#         stderr=asyncio.subprocess.PIPE
#     )
#     print("🚀 ComfyUI started (PID:", process.pid, ")")
#     return process
# async def stop_comfyui(process):
#     if process and process.returncode is None:
#         print("🛑 Stopping ComfyUI...")
#         process.terminate()
#         try:
#             await asyncio.wait_for(process.wait(), timeout=10)
#         except asyncio.TimeoutError:
#             print("⚠️ Force killing ComfyUI...")
#             process.kill()
#             await process.wait()
# async def load_workflow(path="workflow.json"):
#     async with aiofiles.open(path, "r", encoding='utf-8') as f:
#         content = await f.read()
#         return json.loads(content)

# async def queue_prompt(workflow):
#     client_id = str(uuid.uuid4())
    
#     payload = {
#         "prompt": workflow, 
#         "client_id": client_id
#     }
    
#     async with aiohttp.ClientSession() as session:
#         async with session.post(
#             f"http://{server_address}/prompt",
#             json=payload,
#             headers={"Content-Type": "application/json"}
#         ) as response:
#             if response.status == 200:
#                 result = await response.json()
#                 result["client_id"] = client_id
#                 return result
#             else:
#                 raise Exception(f"Failed to queue prompt: {response.status}")

# async def wait_for_completion(prompt_id, client_id):
    
#     websocket_url = f"ws://{server_address}/ws?clientId={client_id}"
    
#     try:
#         async with websockets.connect(websocket_url) as websocket:
            
#             total_nodes = 0
#             completed_nodes = 0
            
#             while True:
#                 try:
#                     msg = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    
#                     if isinstance(msg, str):
#                         data = json.loads(msg)                        
#                         print(f"📨 Rêcive message: {data.get('type', 'unknown')}")
                        
#                         if data["type"] == "execution_start":
#                             print(f"🚀 Start workflow: {data.get('data', {}).get('prompt_id')}")
                        
#                         elif data["type"] == "executing":
#                             node_id = data["data"]["node"]
#                             # current_prompt_id = data["data"]["prompt_id"]
#                             current_prompt_id = data.get("data", {}).get("prompt_id")

#                             if current_prompt_id == prompt_id:
#                                 if node_id is None:
#                                     print("🎉 Workflow Complete!")
#                                     return True
#                                 else:
#                                     completed_nodes += 1
#                                     print(f"⚙️  Processing node: {node_id} ({completed_nodes} )")
                        
#                         elif data["type"] == "progress":
#                             progress_data = data.get("data", {})
#                             value = progress_data.get("value", 0)
#                             max_value = progress_data.get("max", 100)
#                             node = progress_data.get("node")
#                             percentage = (value / max_value * 100) if max_value > 0 else 0
#                             print(f"📊 Node {node}: {value}/{max_value} ({percentage:.1f}%)")
                        
#                         elif data["type"] == "execution_error":
#                             print(f"❌ Error: {data}")
#                             return False
                            
#                         elif data["type"] == "execution_cached":
#                             cached_nodes = data.get("data", {}).get("nodes", [])
#                             print(f"💾 {len(cached_nodes)} nodes cached")
                
#                 except asyncio.TimeoutError:
#                     print("⏰ WebSocket timeout, waiting...")
#                     continue
#                 except Exception as e:
#                     print(f"❌ Error WebSocket: {e}")
#                     break
                    
#     except Exception as e:
#         print(f"❌ Cannot connect to WebSocket: {e}")
#         # print("🔄 Fallback: Kiểm tra file output định kỳ...")
        
#         return await wait_for_completion_fallback(prompt_id)

# async def wait_for_completion_fallback(prompt_id):
#     start_time = time.time()
    
#     while True:
#         await asyncio.sleep(2)  
        
#         video_path = await find_latest_video("my_custom_video")
#         if video_path and os.path.exists(video_path):
#             file_time = os.path.getmtime(video_path)
#             if file_time > start_time:
#                 print("✅ Findout video!")
#                 return True
        
#         if time.time() - start_time > 300:
#             print("⏰ Timeout waiting for video")
#             return False

# async def find_latest_video(prefix, output_dir=str(BASE_DIR / "ComfyUI/output")):    
#     def _find_files():
#         patterns = [
#             f"{prefix}*audio*.mp4", 
#             f"{prefix}_*-audio.mp4"
#         ]
        
#         all_files = []
#         for pattern in patterns:
#             files = glob.glob(os.path.join(output_dir, pattern))
#             all_files.extend(files)
        
#         if not all_files:
#             print(f"🔍 Cannot find video match prefix '{prefix}' in {output_dir}")
#             all_mp4 = glob.glob(os.path.join(output_dir, "*.mp4"))
#             if all_mp4:
#                 print(f"📁 file .mp4 we have now:")
#                 for f in sorted(all_mp4, key=os.path.getmtime, reverse=True)[:5]:
#                     print(f"   {f} (modified: {time.ctime(os.path.getmtime(f))})")
#             return None
        
#         latest_file = max(all_files, key=os.path.getmtime)
#         print(f"📁 Finding newest file: {latest_file}")
#         return latest_file
    
#     loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(None, _find_files)

# # ========== Hàm chính được cập nhật ==========
# async def generate_video_cmd(prompt, cond_image, cond_audio_path, output_path, job_id,resolution):
#     comfy_process = await start_comfyui()
#     await asyncio.sleep(15)  # đợi server ComfyUI khởi động (có thể tăng nếu load model chậm)

#     try:
#         print("🔄 Loading workflow...")
#         workflow = await load_workflow(str(BASE_DIR) + "/" + WORKFLOW_INFINITETALK_PATH)  
# # ===========================================================================
#         # workflow["203"]["inputs"]["image"] = cond_image
#         # workflow["125"]["inputs"]["audio"] = cond_audio_path
        
#         # if prompt.strip() == "" or prompt is None or prompt == "none":
#         #     # workflow["135"]["inputs"]["positive_prompt"] = "Mouth moves in sync with speech. A person is sitting in a side-facing position, with their face turned toward the left side of the frame and the eyes look naturally forward in that left-facing direction without shifting. Speaking naturally, as if having a conversation. He always kept his posture and gaze straight without turning his head."    
#         #     workflow["135"]["inputs"]["positive_prompt"] = "Mouth moves in sync with speech. A person is sitting in a side-facing position, with their face turned toward the left side of the frame and the eyes look naturally forward in that left-facing direction without shifting. Speaking naturally, as if having a conversation. He mostly keeps his posture and gaze straight without turning his head, but occasionally makes small, natural gestures with the head or hands to emphasize the speech, adding subtle liveliness to the video."    
#         # else:
#         #     workflow["135"]["inputs"]["positive_prompt"] = prompt
            
#         # workflow["135"]["inputs"]["negative_prompt"] = "change perspective, bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
#         # wf_h=448
#         # wf_w=448
#         # if resolution == "1080x1920":
#         #     wf_w = 1080
#         #     wf_h = 1920
#         # elif resolution=="1920x1080":
#         #     wf_w = 1920
#         #     wf_h = 1080
#         # elif resolution=="720x1280":
#         #     wf_w = 720
#         #     wf_h = 1280
#         #     # workflow["208"]["inputs"]["frame_window_size"] = 41
#         # elif resolution=="480x854": 
#         #     wf_w = 480
#         #     wf_h = 854
#         # elif resolution=="854x480": 
#         #     wf_w = 854
#         #     wf_h = 480
#         # elif resolution=="1280x720":    
#         #     wf_w = 1280
#         #     wf_h = 720 
        
#         #     # workflow["208"]["inputs"]["frame_window_size"] = 41
#         # img = Image.open(cond_image)
#         # width_real, height_real = img.size
#         # workflow["211"]["inputs"]["value"] = width_real
#         # workflow["212"]["inputs"]["value"] = height_real

#         # workflow["211"]["inputs"]["value"] = 608
#         # workflow["212"]["inputs"]["value"] = 608
#         # img.close()
#         # prefix = job_id
#         # workflow["131"]["inputs"]["filename_prefix"] = prefix

# # ===========================================================================
#         workflow["284"]["inputs"]["image"] = cond_image
#         workflow["125"]["inputs"]["audio"] = cond_audio_path
        
#         workflow["270"]["inputs"]["value"] = int(get_audio_duration(cond_audio_path)*30)
#         # =============================================================
        
        
#         if prompt.strip() == "" or prompt is None or prompt.lower() == "none" :
#             workflow["241"]["inputs"]["positive_prompt"] = "A realistic video of a person confidently giving a lecture. Their face remains neutral and professional, without strong expressions or noticeable head movement. Their hands move up and down slowly and naturally to emphasize key points, without swinging side to side, creating the impression of a teacher clearly explaining a lesson."    
#         else:
#             print("dùng prompt của mình")
#             workflow["241"]["inputs"]["positive_prompt"] = prompt
            
#         workflow["241"]["inputs"]["negative_prompt"] = "bright tones, overexposed, blurred details, move, head movement, subtitles, style, works, paintings, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
#         wf_h=448
#         wf_w=448
#         if resolution == "1:1":
#             wf_w = 720
#             wf_h = 720
#         elif resolution=="16:9":    
#             wf_w = 1040
#             wf_h = 592 
#         elif resolution=="9:16":
#             wf_w = 592
#             wf_h = 1040
#         elif resolution=="720":
#             wf_w = 448
#             wf_h = 800
#         elif resolution=="720_16:9":
#             wf_h = 448
#             wf_w = 800

#         workflow["245"]["inputs"]["value"] = wf_w
#         workflow["246"]["inputs"]["value"] = wf_h
#         # workflow["245"]["inputs"]["value"] = 448
#         # workflow["246"]["inputs"]["value"] = 800
#         # img.close()

#         prefix = job_id
#         workflow["131"]["inputs"]["filename_prefix"] = prefix

# # ===========================================================================
#         print("📤 Sending workflow to ComfyUI...")

#         resp = await queue_prompt(workflow)
#         prompt_id = resp["prompt_id"]
#         client_id = resp["client_id"]
#         print(f"✅ Workflow sent! Prompt ID: {prompt_id}")

#         success = await wait_for_completion(prompt_id, client_id)

#         if not success:
#             print("❌ Workflow failed")
#             return None

#         print("🔍 Searching for the generated video...")

#         video_path = await find_latest_video(prefix)
        
#         if video_path:
#             await delete_file_async(str(video_path.replace("-audio.mp4",".mp4")))
#             await delete_file_async(str(video_path.replace("-audio.mp4",".png")))
#             file_size = os.path.getsize(video_path)
#             print(f"📏 File size: {file_size / (1024*1024):.2f} MB")
#             # wf_w = 720
#             # wf_h = 1280
#             if resolution=="16:9":    
#                 wf_w = 1280
#                 wf_h = 720 
#             elif resolution=="9:16":
#                 wf_w = 720
#                 wf_h = 1280
#             await scale_video(
#                 input_path=video_path,
#                 output_path=output_path,
#                 target_w=wf_w,
#                 target_h=wf_h
#             )
#             await delete_file_async(str(video_path))

#             return output_path
#         else:
#             print("❌ Cannot findout video")
#             return None
#     finally:
#         await stop_comfyui(comfy_process)


# async def move_file_async(src_path, dst_path):
#     def move_file():
#         os.rename(src_path, dst_path)
    
#     loop = asyncio.get_event_loop()
#     await loop.run_in_executor(None, move_file)
# async def delete_file_async(file_path: str):
#     def delete_file():
#         if os.path.exists(file_path):
#             os.remove(file_path)
    
#     loop = asyncio.get_event_loop()
#     await loop.run_in_executor(None, delete_file)
# # ===============================================================
# import cv2
# import os

# async def scale_video(input_path, output_path, target_w, target_h):

#     if not os.path.exists(input_path):
#         raise FileNotFoundError(f"File not exist: {input_path}")
    
#     cap = cv2.VideoCapture(input_path)
#     if not cap.isOpened():
#         raise ValueError(f"Cannot open video: {input_path}")
    
#     orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     scale_w = target_w / orig_w
#     scale_h = target_h / orig_h
#     scale = max(scale_w, scale_h)
    
#     new_w = int(orig_w * scale)
#     new_h = int(orig_h * scale)
    
#     info = {
#         'original_size': (orig_w, orig_h),
#         'target_size': (target_w, target_h),
#         'scale_factor': scale,
#         'final_size': (new_w, new_h),
#         'fps': fps,
#         'total_frames': total_frames
#     }
    
#     print(f"Original video: {orig_w}x{orig_h}")
#     print(f"Target size: {target_w}x{target_h}")
#     print(f"Scale factor: {scale:.4f}")
#     print(f"Final size: {new_w}x{new_h}")

    
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))
    
#     frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         scaled_frame = cv2.resize(frame, (new_w, new_h))
        
#         out.write(scaled_frame)
        
#         frame_count += 1
        
#         if frame_count % 30 == 0:
#             progress = (frame_count / total_frames) * 100
#             print(f"Progress: {progress:.1f}%")
    
#     cap.release()
#     out.release()
#     # cv2.destroyAllWindows()
    
#     print(f"Video saved: {output_path}")
    
#     return info
import os
import uuid
from pathlib import Path
from typing import List
import subprocess
import json
import random

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
image_paths_product = []
# ====================================================================
import os
from concurrent.futures import ThreadPoolExecutor

# Thread pool cho video processing (đặt ở đầu file, global)
video_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="video_concat")

async def concat_and_merge_async(job_id, results, cond_audio_path, output_path_video, cond_images):
    """
    Chạy concat và merge video trong background thread.
    Không block event loop nhưng vẫn đợi kết quả.
    """
    loop = asyncio.get_event_loop()
    
    def _do_concat_merge():
        """Hàm blocking - chạy trong thread pool"""
        concat_name = os.path.join(os.getcwd(), f"{job_id}_concat.mp4")
        
        # Concat videos
        from utilities.merge_video import concat_videos
        output_file1 = concat_videos(results, concat_name)
        
        # Merge audio
        from utilities.merge_video_audio import replace_audio_trimmed
        output_file = replace_audio_trimmed(output_file1, cond_audio_path, output_path_video)
        
        # Cleanup
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
    
    # Chạy trong thread pool, không block event loop
    output_file = await loop.run_in_executor(video_executor, _do_concat_merge)
    return output_file

def custom_random_sequence(n):
    if n <= 0:
        return []
    
    nums = [1, 2, 3, 4]
    sequence = []
    last = None
    not_one_count = 0  # đếm số lần không ra 1 liên tiếp
    
    for i in range(n):
        # --- Quy tắc đặc biệt cho lượt 1 và 2 ---
        if i == 0:
            value = random.choice(nums)
        elif i == 1 and 1 not in sequence:
            value = 1
        else:
            # tạo danh sách các lựa chọn hợp lệ
            candidates = [x for x in nums if x != last]
            
            # Nếu đã quá 2 lần không ra 1 → bắt buộc ra 1
            if not_one_count >= 2:
                value = 1
            else:
                value = random.choice(candidates)
        
        # đảm bảo không ra 1 quá thưa khi n < 3
        if i == n - 1 and n < 3 and 1 not in sequence and value != 1:
            value = 1
        
        # cập nhật đếm
        not_one_count = 0 if value == 1 else not_one_count + 1
        
        sequence.append(value)
        last = value
    
    # return sequence
    # return [2,2,2,2,2][:n]  
    return [2,3,4,2,3,4,2,3,4,2,3,4][:n]

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
    print("resolution: ",resolution)
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
        # first_time=True
        last_value=None
        list_random = custom_random_sequence(len(output_paths))
        count = len(list(filter(lambda x: x != 1, list_random)))
        index_forimgpro=0
        # ==========================================================================
        for i, output_path in enumerate(output_paths):
            await job_service.update_job_status(job_id, "processing", progress=int((i+1)/len(output_paths)*100))
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
            # print("dfsdfsdfsd:   ", audiohavesecondatstart)
            # print(type(audiohavesecondatstart))

            # =================================================================
            current_value=0
            file_path = str(cond_images[current_value])
            
            if (list_random[i] == 1):
                output=await generate_video_cmd(
                    prompt=prompts[current_value],
                    cond_image=str(file_path),# 
                    cond_audio_path=audiohavesecondatstart, 
                    output_path=clip_name,
                    job_id=job_id,
                    resolution=resolution
                )
            else:

                output=await generate_video_fast(
                    prompt=prompts[current_value],
                    cond_image=str(cond_images[1]),
                    cond_audio_path=audiohavesecondatstart, 
                    output_path=clip_name,
                    job_id=job_id,
                    resolution=resolution,
                    type=list_random[i],
                    # first_time=first_time,
                    howmuch=count,
                    index=index_forimgpro
                )
                # first_time=False
                index_forimgpro+=1
            # print("111111111111111111111111111111")
            trim_video_start(clip_name, duration=0.5)
            # print("22222222222222222222222")
            output_file=cut_video(clip_name, get_audio_duration(output_path)-0.5) 
            # print("333333333333333333333333")
            results.append(output_file)
            try:
                # os.remove(pad_file)
                # os.remove(crop_file)
                os.remove(output_path)
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
        print("ádfsdfsfsdfsdf")
        output_file = await concat_and_merge_async(
            job_id=job_id,
            results=results,
            cond_audio_path=cond_audio_path,
            output_path_video=output_path_video,
            cond_images=cond_images
        )
        
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
            print("sdgsfsfgfsgfg",generate_output_filename)
            os.remove(str(generate_output_filename))
            os.remove(str(audiohavesecondatstart))
            os.remove(str(cond_audio_path))
            os.remove(str(file_path))
        except Exception as e:
            print(f"❌ Error removing temporary files: {str(e)}")

        return list_scene
# ============================================================================================

import signal

async def start_comfyui():
    process = await asyncio.create_subprocess_exec(
        "python3", "main.py",
        cwd=str(BASE_DIR / "ComfyUI"),  # chỗ chứa main.py
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    print("🚀 ComfyUI started (PID:", process.pid, ")")
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
                        print(f"📨 Rêcive message: {data.get('type', 'unknown')}")
                        
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
async def generate_video_cmd(prompt, cond_image, cond_audio_path, output_path, job_id,resolution):
    comfy_process = await start_comfyui()
    await asyncio.sleep(15)  # đợi server ComfyUI khởi động (có thể tăng nếu load model chậm)

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

async def generate_video_fast(prompt, cond_image, cond_audio_path, output_path, job_id, resolution, type,first_time=True,howmuch=1,index=0):
    width=720
    height=1280
    if resolution=="16:9":
        width=1280
        height=720
    elif resolution=="9:16":
        width=720
        height=1280
    elif resolution=="1:1": 
        width=720
        height=720
    global image_paths_product 

    if len(image_paths_product)==0 or len(image_paths_product) <howmuch-1:
        print("==========================================")
        process = None
        server_address= "127.0.0.1:8188"
        process = await start_comfyui1()
        await asyncio.sleep(8)
        try:
            for i in range(howmuch):
                image_path = await generate_image_with_comfyui(
                            width=width,
                            height=height,
                            job_id=job_id,
                            input_image=cond_image
                        )
                print(image_path)
                image_paths_product.append(image_path[0])
        except Exception as e:
            print(f"❌ Error creating image with ComfyUI: {e}")
            raise
        finally:
            await stop_comfyui1(process)


    if type == 2:
        from animation.zoom_in_effect import zoom_and_light_effect
        
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
    await delete_file_async(str(image_paths_product[index]))
    return output_path
 
# ==========================================================================================
async def generate_image_with_comfyui( width,height, job_id ,input_image=None):
    # process = None
    # server_address= "127.0.0.1:8188"
    # process = await start_comfyui1()
    
    # await asyncio.sleep(8)
    try:
        print("🔄 Loading workflow...")
        workflow_path="/home/toan/anymateme-visualengine/workflow/Qwen IMAGE Edit 2509 Three Image Edit_api (2).json"
        print(f"Workflow path: {workflow_path}")
        workflow = await load_workflow1(workflow_path)
        print(input_image)
        workflow["78"]["inputs"]["image"] = input_image if input_image else "none"
        
        if "111" in workflow:
            nsdaaff=get_random_prompt()
            workflow["111"]["inputs"]["prompt"] = nsdaaff
            print(nsdaaff)
            print("||||||||||||||||||||||||||||||||||||||||||||||")
        
        if "112" in workflow:
            workflow["112"]["inputs"]["width"] = width
            workflow["112"]["inputs"]["height"] = height
        if "110" in workflow :
            workflow["110"]["inputs"]["prompt"] = "human, text, watermark, logo, extra objects, hands, people, human, low quality, blurry, distorted, messy background, overexposed, unrealistic shadows, poor lighting"
            
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
        print("====================================")
        print(files)
        t=files[0]
        files.append(t)
        # latest_file = max(files, key=os.path.getmtime)
        # print(f"📁 File found: {latest_file}")
        # return latest_file
        files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
        print(files_sorted)
        print("====================================")
    
        return files_sorted[:1]

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _find_files)
import asyncio
import aiohttp
import time
from pathlib import Path


# ====== HÀM KHỞI CHẠY / TẮT COMFYUI ======
async def start_comfyui1():
    COMFYUI_DIR = "/home/toan/anymateme-visualengine/ComfyUI"

    print(COMFYUI_DIR)
    process = await asyncio.create_subprocess_exec(
        "python3", "main.py",
        cwd=str(COMFYUI_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    print(f"🚀 ComfyUI started (PID: {process.pid})")
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

