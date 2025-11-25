import torch
import re
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from app.api.backend.utils.helpers import clean_prompt

def build_system_prompt(prompt: str, style: str) -> str:
    style_guide = {
        "Realistic": "Imagine you're capturing a real-life moment with a camera. Focus on lifelike lighting, textures, and natural details.",
        "Cartoon": "Imagine you're illustrating a cartoon scene. Emphasize exaggerated expressions, simple shapes, and playful details.",
        "Digital Art": "Imagine you're creating a vivid digital painting. Think fantasy settings, dramatic poses, and surreal environments.",
        "Sketch": "Imagine you're sketching with pencil on paper. Focus on line work, posture, and structural outlines of the scene.",
        "Cyberpunk": "Imagine a futuristic, neon-lit city. Emphasize high-tech elements, glowing signs, dark streets, and a dystopian feel.",
        "Fantasy": "Imagine a magical world filled with mythical creatures and enchanting landscapes. Focus on vibrant colors, epic themes, and adventurous settings.",
        "LoRA": "",
        "Artistic": "Imagine a creative artwork inspired by various art styles. Focus on expressive brushstrokes, abstract elements, bold colors, and an emotional atmosphere.",
        "Minimal": "Imagine a clean and simple scene with lots of white space. Focus on basic shapes, soft colors, and a modern, uncluttered look.",
        "Vintage": "Imagine an old-fashioned photograph or poster. Emphasize warm tones, film grain, retro details, and a nostalgic atmosphere.",
        "Anime": "Imagine a Japanese anime style scene. Focus on expressive characters, bold outlines, vibrant colors, and dynamic backgrounds."
    }

    style_note = style_guide.get(style, "")

    return f"""You are a professional prompt enhancer for text-to-image generation models like FLUX.

Your task is to take a short and simple image prompt and expand it into a rich, vivid, and coherent description that helps a generative model visualize the scene clearly.

Requirements:
- Focus only on visual elements: the main subject, their appearance, posture, actions, objects around, background, and setting.
- Do NOT include rendering styles, lighting techniques, camera types, aspect ratio, or any form of visible text, letters, signs, or written words in the scene.

Style Guidance: {style_note}

The expanded prompt must be between **30 to 50 words**.
Use descriptive and concrete language that paints a visual scene without mentioning any text or writing elements.

Short Prompt: "{prompt}"
Expanded Prompt:
"""

class BaseLLM:
    def __init__(self, model_name: str, max_new_tokens=100, temperature=0.8, top_p=0.9):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def clean_output(self, input_text: str, outputs, raw_input_tokens: int) -> tuple[str, int]:
        rulebase = ""
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_tokens = outputs.shape[-1] - raw_input_tokens

        match = re.search(r"Expanded Prompt:\s*(.*)", raw_output, re.DOTALL)
        if match:
            expanded_prompt = match.group(1).strip()
            return clean_prompt(expanded_prompt + rulebase), output_tokens
        else:
            return clean_prompt(raw_output.strip() + rulebase), output_tokens

    def load_model(self):
        raise NotImplementedError

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class MistralModel_v03(BaseLLM):
    def load_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

    def generate(self, prompt: str, style: str = "Realistic") -> str:
        input_text = build_system_prompt(prompt, style)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        input_token_count = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )

        output_text, _ = self.clean_output(input_text, outputs, input_token_count)
        return output_text
    
    def check_type(self, prompt: str) -> str:
        classification_prompt = f"""You are a safety classifier.
For the given prompt, classify whether it explicitly describes or requests the generation of:
- a human or person (realistic or fictional) as the main subject,
- a public figure, celebrity, or historical figure (named or clearly implied),
- or an avatar/profile image of a human.
Return one of the following labels ONLY:
- "inappropriate": The prompt contains or implies inappropriate, explicit, or unsafe content.
- "none": The prompt does not fall into any of the categories above.
- "main-human": The prompt describes or requests a fictional or realistic human figure as the main subject of the image.
- "public-figure": The prompt mentions or implies the likeness of a real person such as a celebrity, politician, or historical figure.
- "avatar": The prompt asks for an avatar or profile picture of a human.

Just return one of these labels: "none", "main-human", "public-figure", "avatar", or "inappropriate".
Do not include any explanation. Do not return anything else.
Prompt: {prompt}
Answer:
"""
        inputs = self.tokenizer(classification_prompt, return_tensors="pt").to(self.model.device)
        input_token_count = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.0,
                do_sample=False,
                top_p=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        output_text, _ = self.clean_output(classification_prompt, outputs, input_token_count)
        return output_text.strip().lower()
    def generate_imageslide(self, prompt: str) -> str:
        print(f"Generating image slide with prompt: {prompt}")
        classification_prompt = f"""You are a visual prompt enhancer for a text-to-image generation model.

Your task is to expand a short image prompt into a detailed, vivid, and realistic visual description that clearly illustrates a concept, profession, or activity for use in a presentation slide.

Requirements:
- The image must be realistic and contextually accurate, designed to support and clarify educational or informational slide content.
- Focus on visual clarity and relevance: describe the main subject, their appearance, action, and surrounding environment.
- Avoid artistic styles, cinematic effects, lighting descriptions, or emotionally charged language.
- Use concise, specific, and neutral language that directly reflects the intended subject matter.

Output Format:
- Write a complete sentence or paragraph between **10 to 30 words**

prompt: {prompt}
Answer:
"""
        inputs = self.tokenizer(classification_prompt, return_tensors="pt").to(self.model.device)
        input_token_count = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.0,
                do_sample=False,
                top_p=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        output_text, _ = self.clean_output(classification_prompt, outputs, input_token_count)
        return output_text.strip().lower()
    def check(self, prompt: str) -> str:
        classification_prompt = f""" You are a safety classifier.
Answer with only "yes" or "no".

Does the following prompt explicitly describe or request the generation of:
  - a human or person (realistic or fictional),
  - a public figure, celebrity, or historical figure,
  - or an avatar/profile image of a human?

Return "yes" ONLY if the prompt:
  - Requests an avatar or profile picture of a person
  - Describes a fictional or realistic human figure as the main subject
  - Mentions or implies the likeness of a real person (e.g., "Elon Musk", "Hitler", "a US president")

Return "no" if the prompt refers to general scenes (e.g., people in a crowd, group photos, or background presence).

Prompt: {prompt}
Answer:
"""

        inputs = self.tokenizer(classification_prompt, return_tensors="pt").to(self.model.device)
        input_token_count = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=3,
                temperature=0.0,
                do_sample=False,
                top_p=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        output_text, _ = self.clean_output(classification_prompt, outputs, input_token_count)
        return output_text.strip().lower()