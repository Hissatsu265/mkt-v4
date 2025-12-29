import os
class LoRAService:
    def __init__(self):
        self.current_loaded_lora = None
        self.current_lora_repo = None
        self.current_lora_repo_weight = ""
    
    def load_custom_lora(self, pipe, lora_repo: str, use_lora: bool = True,safetensors:str=""):
        """Tải LoRA tùy chỉnh"""
        # Nếu LoRA bị tắt, gỡ bỏ LoRA hiện tại
        if not use_lora:
            if self.current_loaded_lora is not None:
                print(f"LoRA disabled, unloading current LoRA: {self.current_lora_repo}")
                try:
                    pipe.unload_lora_weights()
                    self.current_loaded_lora = None
                    self.current_lora_repo = None
                    print("LoRA unloaded successfully")
                except Exception as e:
                    print(f"Error unloading LoRA: {e}")
            return False, "LoRA disabled"

        # Nếu không có repo, bỏ qua
        if not lora_repo or lora_repo.strip() == "":
            if self.current_loaded_lora is not None:
                print("No LoRA repo provided, unloading current LoRA")
                try:
                    pipe.unload_lora_weights()
                    self.current_loaded_lora = None
                    self.current_lora_repo = None
                    print("LoRA unloaded successfully")
                except Exception as e:
                    print(f"Error unloading LoRA: {e}")
            return False, "No LoRA repository provided"

        # Nếu cùng repo đã được tải, bỏ qua
        if self.current_lora_repo == lora_repo.strip() and self.current_lora_repo_weight== safetensors:
            print(f"LoRA for {lora_repo} already loaded, skipping...")
            return True, f"LoRA already loaded: {lora_repo}"

        try:
            if self.current_loaded_lora is not None:
                print(f"Unloading current LoRA: {self.current_lora_repo}")
                pipe.unload_lora_weights()
                self.current_loaded_lora = None
                self.current_lora_repo = None
            print(f"Loading LoRA from: {lora_repo}")
            print(f"Using safetensors: {safetensors}")
            if safetensors!="":
                pipe.load_lora_weights(
                    lora_repo.strip(),
                    weight_name=safetensors,
                    low_cpu_mem_usage=True
                )
            else:
                pipe.load_lora_weights(
                    lora_repo.strip(),
                    low_cpu_mem_usage=True
                )
            self.current_lora_repo_weight = safetensors

            self.current_loaded_lora = True
            self.current_lora_repo = lora_repo.strip()
            print(f"Successfully loaded LoRA from {lora_repo}")
            return True, f"Successfully loaded LoRA: {lora_repo}"

        except Exception as e:
            print(f"Error loading LoRA from {lora_repo}: {e}")
            self.current_loaded_lora = None
            self.current_lora_repo = None
            return False, f"Error loading LoRA: {str(e)}"

    def load_realistic_lora(self, pipe, style: str):
        """Tải LoRA realism tự động cho style Realistic"""
        if style == "Realistic":
            return self.load_custom_lora(pipe, "XLabs-AI/flux-RealismLora", True)
        return False, "Not realistic style"
        
    def get_lora_status(self):
        """Lấy trạng thái LoRA hiện tại"""
        if self.current_loaded_lora:
            return f"Loaded: {self.current_lora_repo}"
        return "No LoRA loaded"