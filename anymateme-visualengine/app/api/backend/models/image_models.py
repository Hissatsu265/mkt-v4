import torch
from diffusers import AutoencoderTiny,DiffusionPipeline,FluxPipeline

class FluxImageModel:
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        
        # self.pipe = FluxPipeline.from_pretrained(
        #     "black-forest-labs/FLUX.1-dev", 
        #     torch_dtype=torch.bfloat16
        # ).to(self.device)
        # self.pipe.enable_model_cpu_offload()
        if self.device == "cuda":
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Detected GPU memory: {total_vram_gb:.2f} GB")
            
            if total_vram_gb > 32:
                print("Loading full precision model (requires >32GB VRAM)...")
                self.pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-dev", 
                    torch_dtype=torch.bfloat16
                ).to(self.device)
                self.pipe.enable_model_cpu_offload()
            else:
                print("Loading quantized INT8 model (for <=32GB VRAM)...")
                self.pipe = FluxPipeline.from_pretrained(
                    "diffusers/FLUX.1-dev-torchao-int8",
                    torch_dtype=torch.bfloat16,
                    use_safetensors=False,
                ).to(self.device)
        else:
            raise RuntimeError("CUDA device not found. GPU is required for this model.")
    def generate_image(self, prompt: str, negative_prompt: str, width: int, height: int, 
                      guidance_scale: float, num_steps: int, lora_scale: float = None):
        if self.pipe is None:
            raise RuntimeError("Model isn't load. Call load_model() before.")
            
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        generation_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_steps,
            "generator": generator
        }
        
        if lora_scale is not None:
            generation_kwargs["joint_attention_kwargs"] = {"scale": lora_scale}
            
        output = self.pipe(**generation_kwargs)
        return output.images[0]
    
    def get_memory_stats(self):
        if not torch.cuda.is_available():
            return "CPU mode - no GPU memory stats"
            
        used_memory = torch.cuda.memory_allocated(device=self.device) / 1024**2
        peak_memory = torch.cuda.max_memory_allocated(device=self.device) / 1024**2
        
        return f"Used Memory: {used_memory:.2f}MB\nPeak: {peak_memory:.2f}MB"