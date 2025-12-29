import torch
from PIL import Image
from app.api.backend.models.llm_models import MistralModel_v03
from app.api.backend.models.image_models import FluxImageModel
from app.api.backend.services.lora_service import LoRAService
from app.api.backend.utils.helpers import extract_answer, extract_answer1, warning_message, get_aspect_ratio_dimensions, image_to_base64
from app.api.backend.schemas.requests import ImageGenerationRequest, ImageGenerationResponse
from typing import Optional
from app.api.backend.services.llm_client import PromptEnhancerClient
from app.config import SERVER_COMFYUI, WORKFLOW_IMAGE_PATH, BASE_DIR,WORKFLOW_2IMAGE_TO_IMAGE,WORKFLOW_IMAGE_POSE,WORKFLOW_IMAGE_PATH_1Image,WORKFLOW_INPAINT

import os
import subprocess
import uuid
import json
import glob
import time
import aiohttp
import aiofiles
import websockets
import asyncio

import math
def process(a, b, lower=650, upper=2000):
    if a/b>3.9 and b/a<0.3:
        lower=500
        upper=2050
    a1=a
    b1=b
    if lower <= a <= upper and lower <= b <= upper: 
      print("No need to resize")
    else:
      gcd = math.gcd(a, b)
      a_r = a // gcd
      b_r = b // gcd

      k = 1
      while True:
          a1 = a_r * k
          b1 = b_r * k
          if lower <= a1 <= upper and lower <= b1 <= upper:
              break
          k += 1
          if k > 100000: 
              raise ValueError("No suitable coefficient found within reasonable range.")

    def next_div16(n):
        return n if n % 16 == 0 else n + (16 - n % 16)

    a2 = next_div16(a1)
    b2 = next_div16(b1)

    return {
        'a1': a1,
        'b1': b1,
        'a2': a2,
        'b2': b2,
    }


def center_crop_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    width, height = image.size
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    return image.crop((left, top, right, bottom))
# ========================================================================================================

class ImageGenerationService:
    _instance: Optional['ImageGenerationService'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    async def _wait_for_comfyui(self, timeout=120):
        """Wait until ComfyUI server is ready."""
        url = f"http://{self.host}:{self.port}"
        start = asyncio.get_event_loop().time()

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            print("✅ ComfyUI is ready!")
                            return True
                except Exception:
                    pass

                await asyncio.sleep(2)
                if asyncio.get_event_loop().time() - start > timeout:
                    raise TimeoutError("ComfyUI did not start in time")

    def __init__(self):
        if not hasattr(self, '_setup_done'):
            # self.llm_model = None
            self.client_prompt = None
            self.image_model = None
            self.lora_service = LoRAService()
            self._setup_done = True
            self._initialize_style_configs()
            self.workspace = "/home/toan/anymateme-visualengine/ComfyUI"
            self.host = "127.0.0.1"
            self.port = "8188"
            self._initialized = False
            self.comfy_process = None
    
    def _initialize_style_configs(self):
        """Initialize style configurations"""
        self.style_rules = {
            "Realistic": "Ultra-realistic, 8K UHD, highly detailed, photo-realistic lighting and textures, cinematic composition.",
            "Cartoon": "Cartoon style, bold black outlines, flat and vibrant colors, exaggerated features, anime and comic influence.",
            "DigitalArt": "Digital painting, fantasy or sci-fi themes, soft shading, dynamic lighting, high-resolution concept art feel.",
            "Sketch": "Hand-drawn pencil sketch, rough linework, monochrome or grayscale, raw and minimal artistic style.",
            "Cyberpunk": "Futuristic cyberpunk style, neon lights, dark cityscape, high-tech elements, dystopian atmosphere.",
            "Fantasy": "Fantasy art style, mythical creatures, magical landscapes, vibrant colors, epic and adventurous themes.",
            "LoRA": "",
            "Artistic": "Creative artistic style, expressive brushstrokes, abstract elements, bold colors, emotional and imaginative atmosphere.",
            "Minimal": "Minimalist design, clean composition, soft neutral colors, simple shapes, lots of whitespace, modern aesthetic.",
            "Vintage": "Vintage style, warm tones, retro textures, film grain, nostalgic look, old-fashioned composition.",
            "Anime": "Anime art style, expressive characters, bold outlines, vibrant colors, dynamic action scenes, Japanese animation influence."
        }
        
        self.style_negative_prompts = {
            "Fantasy":(
                "Deformed anatomy, disfigured face, extra or missing limbs or fingers, bad proportions."
                "Realistic, photo-realistic, 8k, ultra detailed, skin texture, photographic lighting, shadows, reflections, "
                "noise, wrinkles, blemishes, over-detailed anatomy, complex backgrounds, gloomy mood, gritty"
            ),
            "Realistic": (
                "Deformed anatomy, disfigured face, extra or missing limbs or fingers, bad proportions."
                "Cartoon, anime, 3D render, sketch style, low quality, blurry, out of frame, text artifacts."
            ),
            "Cartoon": (
                "(realistic, photo-realistic, 8k, ultra detailed, skin texture), photographic lighting, shadows, reflections, "
                "noise, wrinkles, blemishes, over-detailed anatomy, complex backgrounds, gloomy mood, gritty"
            ),
            "DigitalArt": (
                "(photo-realistic, real photo, sketch, pixelated, cartoon), blurry, low contrast, washed out colors, "
                "flat composition, poorly rendered details, inconsistent lighting, muddy textures"
            ),
            "Sketch": (
                "(photo-realistic, digital painting, colored, cgi, cartoon, anime), vibrant colors, smooth textures, "
                "glossy effects, full shading, clean lines, 3d elements, depth of field"
            ),
            "Cyberpunk": (
                "(medieval, nature landscape, rural, bright sunlight), overexposed, plain clothing, traditional architecture, "
                "fantasy themes, cute style, dull neon, low contrast, non-futuristic elements"
            ),
            "LoRA": (
                "low quality, blurry, distorted, bad anatomy, poorly rendered, inconsistent style"
            ),
        }
    
    @property
    def is_initialized(self) -> bool:
        return (self._initialized and 
                # self.llm_model is not None and
                self.client_prompt is not None and 
                self.image_model is not None)
    
    async def initialize_models(self):
        if self._initialized:
            print("Models already initialized, skipping...")
            return
            
        try:
    #    /home/toan/anymateme-visualengine/ComfyUI/main.py
            # main_py = os.path.join(self.workspace, "main.py")
            # print(f"🚀 Starting ComfyUI from: {main_py}")
            # self.comfy_process = subprocess.Popen(
            #     ["python3", main_py],
            #     cwd=self.workspace,
            #     stdout=subprocess.PIPE,
            #     stderr=subprocess.STDOUT,
            # )

            # # Wait until ComfyUI is ready
            # await self._wait_for_comfyui()
            
# =================================================================
            self.client_prompt = PromptEnhancerClient()
            # print("Initializing LLM model...")
            # self.llm_model = MistralModel_v03("mistral7Bv03")
            # await asyncio.to_thread(self.llm_model.load_model)
            print("Initializing image model...")
            self.image_model = FluxImageModel()
            # await asyncio.to_thread(self.image_model.load_model)
            


            self._initialized = True
            print("Models initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            self._initialized = False
            raise
    
    def _ensure_initialized(self):
        """Ensure models are initialized before use"""
        if not self.is_initialized:
            raise RuntimeError("Models not initialized. Call initialize_models() first.")
    
    def check_person_prompt(self, prompt: str) -> tuple[bool, str]:
        self._ensure_initialized()
        try:
            # response = extract_answer(self.llm_model.check(prompt).strip().lower())
            # type_of_blocked_prompt = extract_answer(self.llm_model.check_type(prompt).strip().lower())
            # response = self.client_prompt.check_prompt_safety(prompt).strip().lower()
            # type_of_blocked_prompt = self.client_prompt.classify_prompt_type(prompt).strip().lower()

            return False, "none"
        except Exception as e:
            print(f"Error checking prompt: {e}")
            return False, "none"
        
    def create_warning_image(self) -> Image.Image:
        return Image.new("RGB", (512, 512), (255, 255, 255))
        
    async def generate_imag11e(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        try:
            # Ensure models are initialized
            self._ensure_initialized()
            
            print("Starting image generation...")
            actual_style = "LoRA" if request.use_custom_lora else request.style
            # print("===========================================")
            # print(request)
            # print(request.style)
            rulebase = self.style_rules.get(actual_style, "") + " "
            
            result, type_of_blocked_prompt = self.check_person_prompt(request.user_prompt)

            print(f"Prompt check result: {result}, Type: {type_of_blocked_prompt}")
            
            if result or type_of_blocked_prompt in ["public-figure", "inappropriate"]:
                warning_img = self.create_warning_image()
                return ImageGenerationResponse(
                    success=False,
                    message=warning_message(type_of_blocked_prompt),
                    image_base64=image_to_base64(warning_img),
                    warning=warning_message(type_of_blocked_prompt)
                )
            
            lora_loaded = False
            lora_status = "LoRA not used"
            
            if request.use_custom_lora:
                lora_loaded, lora_status = self.lora_service.load_custom_lora(
                    self.image_model.pipe, request.lora_repo, request.use_custom_lora, request.safetensors
                )
                if not lora_loaded:
                    print(f"Warning: {lora_status}")
            elif request.style == "Realistic":
                lora_loaded, lora_status = self.lora_service.load_realistic_lora(
                    self.image_model.pipe, request.style
                )
            try:
                enhanced_prompt = self.client_prompt.enhance_prompt(request.user_prompt, actual_style)
                if enhanced_prompt is None:
                    enhanced_prompt = request.user_prompt
            except Exception as e:
                print(f"⚠️ Prompt enhancement failed: {e}")
                enhanced_prompt = request.user_prompt

            no_text_prompt = "Avoid any form of writing, signage, labels, or visible characters in the scene. "
            final_prompt = rulebase + enhanced_prompt 
            
            width, height = get_aspect_ratio_dimensions(request.aspect, request.width, request.height)
            result = process(width, height)
            negative_prompt_base = (
                "Deformed anatomy, disfigured face, extra or missing limbs or fingers, bad proportions."
                "low quality, blurry, out of frame, text artifacts."
                "distorted text, unreadable text, gibberish writing, foreign characters, blurry text, incorrect spelling"
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(42)
            generation_kwargs = {
                "prompt": no_text_prompt + final_prompt,
                "negative_prompt": negative_prompt_base,
                "height": result['b2'],
                "width": result['a2'],
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_steps,
                "generator": generator
            }

            if lora_loaded and request.use_custom_lora:
                generation_kwargs["joint_attention_kwargs"] = {"scale": request.lora_scale}

            output = self.image_model.pipe(**generation_kwargs)
            image = output.images[0]
            image= center_crop_image(image, result['a1'], result['b1'])

            used_memory = torch.cuda.memory_allocated(device=device) / 1024**2
            peak_memory = torch.cuda.max_memory_allocated(device=device) / 1024**2

            if request.use_custom_lora:
                lora_info = f"Custom LoRA: {lora_status} (scale: {request.lora_scale})"
            elif request.style == "Realistic":
                lora_info = f"Auto LoRA: XLabs-AI/flux-RealismLora (scale: {request.lora_scale})"
            else:
                lora_info = "LoRA: Disabled"

            memory_info = f"Used Memory: {used_memory:.2f}MB\nPeak: {peak_memory:.2f}MB\n{lora_info}\nSettings: {width}x{height}, Steps: {request.num_steps}, Guidance: {request.guidance_scale}"

            return ImageGenerationResponse(
                success=True,
                message="Image generated successfully",
                image_base64=image_to_base64(image),
                final_prompt=final_prompt,
                system_info=memory_info
            )
            
        except RuntimeError as e:
            # Handle initialization errors specifically
            print(f"Service not initialized: {e}")
            warning_img = self.create_warning_image()
            return ImageGenerationResponse(
                success=False,
                message="Service not ready. Please try again.",
                image_base64=image_to_base64(warning_img),
                warning="Service initialization required."
            )
        except Exception as e:
            print(f"Error while generating image: {e}")
            warning_img = self.create_warning_image()
            return ImageGenerationResponse(
                success=False,
                message="Image generation failed",
                image_base64=image_to_base64(warning_img),
                warning="An unexpected error occurred."
            )
    
    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        try:
            # Ensure models are initialized
            self._ensure_initialized()
            # print("Starting image slide generation...")
            # actual_style = "Realistic"
            # rulebase = self.style_rules.get(actual_style, "") + " "
            
            # result, type_of_blocked_prompt = self.check_person_prompt(request.user_prompt)
            
            # request.style = "None"

            # # lora_loaded = False
            # # lora_status = "LoRA not used"
            # # lora_loaded, lora_status = self.lora_service.load_realistic_lora(
            # #     self.image_model.pipe, request.style
            # # )
            
            # enhanced_prompt = extract_answer(self.client_prompt.enhance_slide_prompt(request.user_prompt).strip().lower())
            # # enhanced_prompt = extract_answer(self.llm_model.generate_imageslide(request.user_prompt).strip().lower())
            # prompt11111 = extract_answer1(enhanced_prompt)
            
            # final_prompt = request.theme + " " + prompt11111
            # print("Starting image generation...")
            # actual_style = "LoRA" if request.use_custom_lora else request.style
            # print("============tttttttttt===============================")
            # print(request)
            # print(actual_style)
            # rulebase = self.style_rules.get(actual_style, "") + " "
            
            # result, type_of_blocked_prompt = self.check_person_prompt(request.user_prompt)

            # print(f"Prompt check result: {result}, Type: {type_of_blocked_prompt}")
            
            # if result or type_of_blocked_prompt in ["public-figure", "inappropriate"]:
            #     warning_img = self.create_warning_image()
            #     return ImageGenerationResponse(
            #         success=False,
            #         message=warning_message(type_of_blocked_prompt),
            #         image_base64=image_to_base64(warning_img),
            #         warning=warning_message(type_of_blocked_prompt)
            #     )
            
            # lora_loaded = False
            # lora_status = "LoRA not used"
            
            # if request.use_custom_lora:
            #     lora_loaded, lora_status = self.lora_service.load_custom_lora(
            #         self.image_model.pipe, request.lora_repo, request.use_custom_lora, request.safetensors
            #     )
            #     if not lora_loaded:
            #         print(f"Warning: {lora_status}")
            # elif request.style == "Realistic":
            #     lora_loaded, lora_status = self.lora_service.load_realistic_lora(
            #         self.image_model.pipe, request.style
            #     )
            # try:
            #     enhanced_prompt = self.client_prompt.enhance_prompt(request.user_prompt, actual_style)
            #     if enhanced_prompt is None:
            #         enhanced_prompt = request.user_prompt
            # except Exception as e:
            #     print(f"⚠️ Prompt enhancement failed: {e}")
            #     enhanced_prompt = request.user_prompt

            # no_text_prompt = "Avoid any form of writing, signage, labels, or visible characters in the scene. "
            # final_prompt = rulebase + enhanced_prompt
            # ===================================================================
            width, height = get_aspect_ratio_dimensions(request.aspect, request.width, request.height)
            # print(f"=========================")
            # print(f"Converted aspect {request.aspect} to dimensions: {width}x{height}")
            # print("=========================",request.dict())
            negative_prompt_base = (
                "text, letters, words, captions, subtitles, handwriting, signs, labels, logos, watermarks, "
                "distorted text, unreadable text, gibberish writing, foreign characters, blurry text, incorrect spelling"
            )
            # ================================================================
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # generator = torch.Generator(device=device).manual_seed(42)
            # generation_kwargs = {
            #     "prompt": final_prompt,
            #     "negative_prompt": negative_prompt_base,
            #     "height": height,
            #     "width": width,
            #     "guidance_scale": request.guidance_scale,
            #     "num_inference_steps": request.num_steps,
            #     "generator": generator
            # }

            # if lora_loaded and request.use_custom_lora:
            #     generation_kwargs["joint_attention_kwargs"] = {"scale": request.lora_scale}
                
            # output = self.image_model.pipe(**generation_kwargs)
            # image = output.images[0]
            job_id = str(uuid.uuid4())[:8]
            PARENT_DIR = os.path.dirname(BASE_DIR)
            FULL_PATH = os.path.join(PARENT_DIR, WORKFLOW_IMAGE_PATH.lstrip("/"))
            print(request.input_image,"fddddddddddddddddđ")

            print(request)
            print("=====================================================")
            # import time
            # time.sleep(15)
            print("Generating image with ComfyUI...")
            print(width)
            print(height)
            print("ttttttttttttttttttttttttttt")
            input_image2_path = request.input_image2 if (request.input_image2 and os.path.exists(request.input_image2)) else None
            image_path = await generate_image_with_comfyui(
                prompt=request.user_prompt,
                negative_prompt=request.negative_prompt,
                width=width,
                height=height,
                guidance_scale=request.guidance_scale,
                num_steps=request.num_steps,
                job_id=job_id,
                server_address=SERVER_COMFYUI,
                workflow_path=str(FULL_PATH),
                type_generation=request.type_generation,
                input_image=request.input_image,
                input_image2=input_image2_path
            )
            # # print(f"Generated image path: {image_path}")
            # if image_path is None:
            #     raise Exception("ComfyUI failed to generate image")
            # # print("=====================================================")
            # from PIL import Image
            # image = Image.open(image_path)
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # image_base64 = image_to_base64(image) 
            # os.remove(image_path) 
            # # =========================================================================
            # used_memory = torch.cuda.memory_allocated(device=device) / 1024**2
            # peak_memory = torch.cuda.max_memory_allocated(device=device) / 1024**2

            # memory_info = f"Used Memory: {used_memory:.2f}MB\nPeak: {peak_memory:.2f}MB\nSettings: {width}x{height}"

            # return ImageGenerationResponse(
            #     success=True,
            #     message="Image generated successfully",
            #     image_base64=image_base64,
            #     final_prompt=final_prompt,
            #     system_info=memory_info
            # )
            t=image_path[0]
            image_path.append(t)
            # print(image_path,"====================")
            # if not image_path or len(image_path) < 2:
            #     raise Exception("ComfyUI failed to generate both images")

            from PIL import Image

            device = "cuda" if torch.cuda.is_available() else "cpu"

            image1 = Image.open(image_path[0])
            image1_base64 = image_to_base64(image1)
            # os.remove(image_path[0])
            image2 = Image.open(image_path[0])
            image2_base64 = image_to_base64(image2)
            os.remove(image_path[0])

            # Memory info
            used_memory = torch.cuda.memory_allocated(device=device) / 1024**2
            peak_memory = torch.cuda.max_memory_allocated(device=device) / 1024**2
            memory_info = f"Used Memory: {used_memory:.2f}MB\nPeak: {peak_memory:.2f}MB\nSettings: {width}x{height}"
            # if request.user_prompt is not None and request.user_prompt != "" and request.user_prompt() != "none":
            #     print("User prompt provided, using it directly.")
            # else: request.user_prompt = "A hyper-realistic high-quality photo where the person in the original image is naturally interacting with the product placed in the scene. The person is realistically holding, touching, or standing next to the object with a natural posture and hand placement. The product looks proportional, seamlessly integrated into the scene, with correct perspective, lighting, reflections, and shadows matching the environment. The person’s appearance, expression, clothing, and background remain unchanged. The overall photo looks natural, authentic, and photorealistic, suitable for professional commercial advertising."
            # print(request.user_prompt,"====================")
            return ImageGenerationResponse(
                success=True,
                message="Images generated successfully",
                image1_base64=image1_base64,
                image2_base64=image2_base64,
                final_prompt=request.user_prompt,
                system_info=memory_info
            )
            
        except RuntimeError as e:
            # Handle initialization errors specifically
            print(f"Service not initialized: {e}")
            warning_img = self.create_warning_image()
            return ImageGenerationResponse(
                success=False,
                message="Service not ready. Please try again.",
                image1_base64=image_to_base64(warning_img),
                warning="Service initialization required."
            )
        except Exception as e:
            print(f"Error while generating image: {e}")
            warning_img = self.create_warning_image()
            return ImageGenerationResponse(
                success=False,
                message="Image generation failed",
                image_base64=image_to_base64(warning_img),
                warning="An unexpected error occurred."
            )
# =============================================================================================


async def load_workflow(path):
    async with aiofiles.open(path, "r", encoding='utf-8') as f:
        content = await f.read()
        return json.loads(content)

async def queue_prompt(workflow, server_address):
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

async def wait_for_completion(prompt_id, client_id, server_address):
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
                        # print(f"📨 Receive message: {data.get('type', 'unknown')}")
                        
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

async def find_image_by_id(image_id, output_dir=str(BASE_DIR / "ComfyUI/output")):
    def _find_files():
        PARENT_DIR = os.path.dirname(str(BASE_DIR))
        output_dir = os.path.join(PARENT_DIR, "ComfyUI/output")
        # output_dir="/home/toan/anymateme-visualengine/ComfyUI/output"
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

# BASE_DIR = Path(__file__).resolve().parent

# ====== HÀM KHỞI CHẠY / TẮT COMFYUI ======
# async def start_comfyui():
#     print(BASE_DIR,"========BASEDIR====================ssss")
#     # "/home/toan/anymateme-visualengine/app "
#     parent_dir = os.path.dirname(str(BASE_DIR))
#     COMFYUI_DIR = os.path.join(parent_dir, "ComfyUI")

#     # COMFYUI_DIR = "/home/toan/anymateme-visualengine/ComfyUI"

#     print(COMFYUI_DIR)
#     process = await asyncio.create_subprocess_exec(
#         "python3", "main.py",
#         cwd=str(COMFYUI_DIR),
#         stdout=asyncio.subprocess.PIPE,
#         stderr=asyncio.subprocess.PIPE
#     )
#     print(f"🚀 ComfyUI started (PID: {process.pid})")
#     return process
import socket
async def wait_for_port_async(host, port, timeout=60):
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
    print(BASE_DIR, "========BASEDIR====================")
    parent_dir = os.path.dirname(str(BASE_DIR))
    COMFYUI_DIR = os.path.join(parent_dir, "ComfyUI")

    # Nếu bạn biết chắc ComfyUI chạy ở port nào (thường là 8188)
    HOST = "127.0.0.1"
    PORT = 8188

    print(f"📂 Đang khởi động ComfyUI tại: {COMFYUI_DIR}")
    process = await asyncio.create_subprocess_exec(
        "python3", "main.py",
        cwd=str(COMFYUI_DIR),
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
            print("✅ ComfyUI stopped gracefully.")
        except asyncio.TimeoutError:
            print("⚠️ Force killing ComfyUI...")
            process.kill()
            await process.wait()

# ====== HÀM KIỂM TRA SERVER COMFYUI SẴN SÀNG ======
async def wait_for_comfyui_ready(server_address, timeout=60):
    print("⏳ Waiting for ComfyUI to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{server_address}/system_stats") as resp:
                    if resp.status == 200:
                        print("✅ ComfyUI is ready!")
                        return True
        except Exception:
            await asyncio.sleep(2)
    print("❌ ComfyUI did not start within timeout.")
    return False

async def generate_image_with_comfyui(prompt, negative_prompt, width, height, guidance_scale, num_steps, job_id, server_address, workflow_path,input_image=None,input_image2=None, type_generation="normal" ):

    process = None
    process = await start_comfyui()

    t=0
    if "[change pose]" in prompt:
        t=1
    elif "[change clothes]" in prompt:
        t=2

    if type_generation.lower() == "image_edit" and input_image2 is not None and t!=1:
        try:
            print("🔄 Loading workflow...")
            parent_dir = os.path.dirname(str(BASE_DIR))
            workflow_path = os.path.join(parent_dir,WORKFLOW_2IMAGE_TO_IMAGE)

            print(f"Workflow path: {workflow_path}")
            workflow = await load_workflow(workflow_path)
            print(input_image)
            workflow["78"]["inputs"]["image"] = input_image if input_image else "none"
            workflow["139"]["inputs"]["image"] = input_image2 if input_image2 else "none"
            
            if "111" in workflow:
                if t==2:
                    workflow["111"]["inputs"]["prompt"] = "Replace the person’s outfit in image 1 with the clothes from image 2. Keep the person’s identity, body shape, pose, face, lighting, and background unchanged. The new clothes should look realistic, well-fitted, and consistent with the scene."                 
                elif prompt is not None and prompt != "" and prompt.lower() != "none" and prompt.lower() != "default":
                    workflow["111"]["inputs"]["prompt"] = prompt
            if "112" in workflow:
                workflow["112"]["inputs"]["width"] = width
                workflow["112"]["inputs"]["height"] = height
            if "110" in workflow :
                workflow["110"]["inputs"]["prompt"] = "bright tones, overexposed, blurred details, head movement, subtitles, style, works, paintings, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs"
                
            prefix = f"{job_id}/{job_id}"
            if "60" in workflow:
                workflow["60"]["inputs"]["filename_prefix"] = prefix
            
            print("📤 Sending workflow to ComfyUI...")
            resp = await queue_prompt(workflow, server_address)

            prompt_id = resp["prompt_id"]
            client_id = resp["client_id"]
            print(f"✅ Workflow sent! Prompt ID: {prompt_id}")
            
            success = await wait_for_completion(prompt_id, client_id, server_address)
            
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
            await stop_comfyui(process)
    elif type_generation.lower() == "inpaint" and input_image2 is not None:
        try:
            print("🔄 Loading workflow...")
            parent_dir = os.path.dirname(str(BASE_DIR))
            workflow_path = os.path.join(parent_dir,WORKFLOW_INPAINT)

            print(f"Workflow path: {workflow_path}")
            workflow = await load_workflow(workflow_path)
            print(input_image)
            workflow["71"]["inputs"]["image"] = input_image if input_image else "none"
            workflow["137"]["inputs"]["image"] = input_image2 if input_image2 else "none"
            
            if "131" in workflow:
                # if t==2:
                #     workflow["131"]["inputs"]["text"] = "Replace the person’s outfit in image 1 with the clothes from image 2. Keep the person’s identity, body shape, pose, face, lighting, and background unchanged. The new clothes should look realistic, well-fitted, and consistent with the scene."                 
                # elif prompt is not None and prompt != "" and prompt.lower() != "none" and prompt.lower() != "default":
                workflow["131"]["inputs"]["text"] = prompt
            if "138" in workflow:
                workflow["138"]["inputs"]["width"] = width
                workflow["138"]["inputs"]["height"] = height
            # if "110" in workflow :
            #     workflow["110"]["inputs"]["prompt"] = "bright tones, overexposed, blurred details, head movement, subtitles, style, works, paintings, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs"
                
            prefix = f"{job_id}/{job_id}"
            if "60" in workflow:
                workflow["60"]["inputs"]["filename_prefix"] = prefix
            
            print("📤 Sending workflow to ComfyUI...")
            resp = await queue_prompt(workflow, server_address)

            prompt_id = resp["prompt_id"]
            client_id = resp["client_id"]
            print(f"✅ Workflow sent! Prompt ID: {prompt_id}")
            
            success = await wait_for_completion(prompt_id, client_id, server_address)
            
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
            await stop_comfyui(process)

    elif type_generation.lower() == "change_pose" and input_image2 is not None:
        try:
            print("🔄 Loading workflow...")
            parent_dir = os.path.dirname(str(BASE_DIR))
            workflow_path = os.path.join(parent_dir,WORKFLOW_IMAGE_POSE)

            print(f"Workflow path: {workflow_path}")
            workflow = await load_workflow(workflow_path)
            print(input_image)
            workflow["78"]["inputs"]["image"] = input_image if input_image else "none"
            workflow["108"]["inputs"]["image"] = input_image2 if input_image2 else "none"
            
            if "111" in workflow:
                if prompt is not None and prompt != "" and prompt.lower() != "none" and prompt.lower() != "default":
                    workflow["111"]["inputs"]["prompt"] = prompt
                    t=2
                else:
                    workflow["111"]["inputs"]["prompt"] = "Change the person’s pose in image 1 to match the pose in image 3 while keeping their identity, face, body shape, clothes, lighting, and background unchanged. Make the result natural and realistic."
             
            if "112" in workflow:
                workflow["112"]["inputs"]["width"] = width
                workflow["112"]["inputs"]["height"] = height
            if "110" in workflow :
            #     workflow["110"]["inputs"]["prompt"] = "cartoon, 3d render, CGI, unrealistic, bad anatomy, distorted body, extra limbs, blurry face, unnatural lighting, wrong shadows, low quality, deformed hands, plastic skin, artifacts, oversaturated, low detail, watermark, text, logo"
            # elif prompt.lower() != "default":
                # print("<<dùng negative prompt mới>>======")
                workflow["110"]["inputs"]["prompt"] = "bright tones, overexposed, blurred details, head movement, subtitles, style, works, paintings, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs"
                
            prefix = f"{job_id}/{job_id}"
            if "60" in workflow:
                workflow["60"]["inputs"]["filename_prefix"] = prefix
            
            print("📤 Sending workflow to ComfyUI...")
            resp = await queue_prompt(workflow, server_address)

            prompt_id = resp["prompt_id"]
            client_id = resp["client_id"]
            print(f"✅ Workflow sent! Prompt ID: {prompt_id}")
            
            success = await wait_for_completion(prompt_id, client_id, server_address)
            
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
            await stop_comfyui(process)
    else:
        try:
            print("🔄 Loading workflow...")
            parent_dir = os.path.dirname(str(BASE_DIR))
            workflow_path = os.path.join(parent_dir, WORKFLOW_IMAGE_PATH_1Image)
            workflow = await load_workflow(workflow_path)
            workflow["78"]["inputs"]["image"] = input_image if input_image else "none"
            
            if "111" in workflow:
                if prompt is not None and prompt != "" and prompt.lower() != "none" and prompt.lower() != "default":
                    workflow["111"]["inputs"]["prompt"] = prompt
                    t=2
                elif prompt.lower() == "default":
                    import random
                    t=1
                    prompts = [
                        "A hyper-realistic high-quality photo where the person from the original image is actively holding the product with a clearly visible grip, both hands naturally adjusted to interact with the item. The body posture is slightly repositioned to face the camera directly, creating a confident commercial look. The background shows a bright, modern retail store with similar products on shelves. Lighting is strong, balanced, and professional, illuminating the person and product realistically.",
                        "A hyper-realistic high-quality outdoor photo where the person from the original image firmly holds the product with natural hand placement and a comfortable body posture facing the camera. The product is clearly visible and proportional. The background is a bright, green garden under soft sunlight. Lighting highlights the face and product beautifully, producing a vibrant, realistic advertising scene.",
                        "A hyper-realistic indoor photo where the person from the original image is clearly holding or presenting the product with one or both hands in front of the body, facing the camera directly. The posture is slightly repositioned for a natural, confident pose. The scene is a bright modern home or living room with warm, soft lighting that enhances the product and the person’s features. Everything looks realistic, balanced, and professional.",
                        "A hyper-realistic outdoor photo where the person from the original image firmly holds the product with visible hand contact and a naturally adjusted pose facing the camera. The person appears confident and relaxed, the product clearly visible and integrated into the scene. The background is a sunny park with trees and grass under bright daylight. Lighting is vivid and natural, enhancing the commercial realism of the shot."
                    ]
                    workflow["111"]["inputs"]["prompt"] = random.choice(prompts)
                else: 
                    workflow["111"]["inputs"]["prompt"] = "A hyper-realistic high-quality photo where the person in the original image is naturally interacting with the product placed in the scene. The person is realistically holding, touching, or standing next to the object with a natural posture and hand placement. The product looks proportional, seamlessly integrated into the scene, with correct perspective, lighting, reflections, and shadows matching the environment. The person’s appearance, expression, clothing, and background remain unchanged. The overall photo looks natural, authentic, and photorealistic, suitable for professional commercial advertising."
            
            if "112" in workflow:
                workflow["112"]["inputs"]["width"] = width
                workflow["112"]["inputs"]["height"] = height
            if "110" in workflow and negative_prompt is not None and negative_prompt != "":
                workflow["110"]["inputs"]["prompt"] = "bright tones, overexposed, blurred details, move, head movement, subtitles, style, works, paintings, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs"
            else:
                workflow["110"]["inputs"]["prompt"] = "bright tones, overexposed, blurred details, move, head movement, subtitles, style, works, paintings, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs"
            if prompt.lower() != "default":
                workflow["110"]["inputs"]["prompt"] = "bright tones, overexposed, blurred details, move, head movement, subtitles, style, works, paintings, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs"
                
            prefix = f"{job_id}/{job_id}"
            if "60" in workflow:
                workflow["60"]["inputs"]["filename_prefix"] = prefix
            
            print("📤 Sending workflow to ComfyUI...")
            resp = await queue_prompt(workflow, server_address)

            prompt_id = resp["prompt_id"]
            client_id = resp["client_id"]
            print(f"✅ Workflow sent! Prompt ID: {prompt_id}")
            
            success = await wait_for_completion(prompt_id, client_id, server_address)
            
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
            await stop_comfyui(process)