
import gradio as gr
import requests
import base64
from PIL import Image
import io

BACKEND_URL = "http://localhost:6005/api/v1/ai/generate-image"  # Endpoint mặc định
SLIDE_BACKEND_URL = "http://localhost:6005/api/v1/ai/slide-image"  # Endpoint cho slide

def generate_image_from_api(
    prompt_input, method, model_choice, style, aspect,
    use_custom_lora, lora_repo, lora_scale, safetensors_name,
    guidance_scale, steps, width, height, image_type, theme_input
):
    endpoint_url = SLIDE_BACKEND_URL if image_type == "slide" else BACKEND_URL
    

    payload = {
        "user_prompt": prompt_input,
        "method": method,
        "model_choice": model_choice,
        "style": style if image_type != "slide" else "",  # Không gửi style cho slide
        "aspect": aspect,
        "use_custom_lora": use_custom_lora if image_type != "slide" else False,  # Không dùng custom LoRA cho slide
        "lora_repo": lora_repo if (use_custom_lora and image_type != "slide") else "",
        "lora_scale": lora_scale if (use_custom_lora and image_type != "slide") else 0.8,
        "safetensors": safetensors_name if (use_custom_lora and image_type != "slide" and safetensors_name.strip()) else "",
        "guidance_scale": guidance_scale,
        "num_steps": steps,
        "width": width if aspect == "Custom" else 0,
        "height": height if aspect == "Custom" else 0
    }

    # Thêm theme vào payload nếu là slide
    if image_type == "slide":
        payload["theme"] = theme_input

    try:
        response = requests.post(endpoint_url, json=payload)
        response.raise_for_status()
        data = response.json()

        if not data["success"]:
            return None, "⚠️ " + data.get("message", "Unknown error"), data.get("system_info", "")

        image_bytes = base64.b64decode(data["image_base64"])
        image = Image.open(io.BytesIO(image_bytes))  # 👈 chuyển bytes sang PIL Image

        return image, data.get("final_prompt", ""), data.get("system_info", "")

    except Exception as e:
        return None, "❌ Error: " + str(e), ""

# UI
with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ FLUX Custom LoRA Generator with Style Enhancement")
    gr.Markdown("### Load your own LoRA weights or use built-in styles!")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Enter your idea")
            
            image_type = gr.Radio(
                ["normal", "slide"], 
                value="normal", 
                label="Image Type",
                info="Choose 'slide' for presentation images"
            )
            
            theme_input = gr.Textbox(
                label="Theme",
                placeholder="Enter theme for slide (e.g., business, education, technology)",
                visible=False,
                info="Specify the theme/style for your slide presentation"
            )
            
            method = gr.Radio(["mistral7Bv03"], value="mistral7Bv03", label="Prompt Handling Method")
            model_choice = gr.Radio(["FLUX"], value="FLUX", label="Diffusion Model")

            use_custom_lora = gr.Checkbox(
                value=False,
                label="Use Custom LoRA",
                info="Check to use your own LoRA repository"
            )
            lora_repo_input = gr.Textbox(
                label="LoRA Repository",
                placeholder="e.g., username/lora-name or huggingface-repo-path",
                visible=False,
                info="Enter the HuggingFace repository path for your LoRA weights"
            )
            safetensors_input = gr.Textbox(
                label="Safetensors File Name",
                placeholder="e.g., pytorch_lora_weights.safetensors (optional)",
                visible=False,
                info="Specify the safetensors file name if the repository has multiple files"
            )
            lora_scale_slider = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                step=0.1,
                value=0.8,
                label="LoRA Scale",
                visible=False,
                info="Controls the strength of LoRA effect"
            )
            style_dropdown = gr.Dropdown(
                ["Realistic", "Cartoon", "Digital Art", "Sketch", "Cyberpunk"],
                value="Realistic",
                label="Select Style"
            )
            aspect_dropdown = gr.Dropdown(
                ["1:1", "3:2", "2:3", "16:9", "Custom"],
                value="1:1",
                label="Select Aspect Ratio"
            )

            with gr.Row():
                guidance_scale_slider = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    step=0.5,
                    value=9.5,
                    label="Guidance Scale"
                )
                steps_slider = gr.Slider(
                    minimum=5,
                    maximum=50,
                    step=1,
                    value=15,
                    label="Inference Steps"
                )

            with gr.Row():
                width_slider = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=1024,
                    label="Width",
                    visible=False
                )
                height_slider = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=1024,
                    label="Height",
                    visible=False
                )

            def toggle_lora_controls(use_lora):
                return (
                    gr.update(visible=use_lora),  # lora_repo_input
                    gr.update(visible=use_lora),  # safetensors_input
                    gr.update(visible=use_lora),  # lora_scale_slider
                    gr.update(visible=not use_lora)  # style_dropdown - Ẩn style nếu dùng custom LoRA
                )

            def toggle_custom_dimensions(aspect):
                is_custom = aspect == "Custom"
                return (
                    gr.update(visible=is_custom),
                    gr.update(visible=is_custom)
                )

            def toggle_image_type_controls(img_type):
                is_slide = img_type == "slide"
                return (
                    gr.update(visible=not is_slide),  # use_custom_lora
                    gr.update(visible=not is_slide),  # style_dropdown
                    gr.update(visible=False if is_slide else None),  # lora_repo_input (ẩn nếu là slide)
                    gr.update(visible=False if is_slide else None),  # safetensors_input (ẩn nếu là slide)
                    gr.update(visible=False if is_slide else None),  # lora_scale_slider (ẩn nếu là slide)
                    gr.update(visible=is_slide)  # theme_input (hiện nếu là slide)
                )

            # Event handlers
            use_custom_lora.change(
                toggle_lora_controls,
                inputs=[use_custom_lora],
                outputs=[lora_repo_input, safetensors_input, lora_scale_slider, style_dropdown]
            )

            aspect_dropdown.change(
                toggle_custom_dimensions,
                inputs=[aspect_dropdown],
                outputs=[width_slider, height_slider]
            )

            image_type.change(
                toggle_image_type_controls,
                inputs=[image_type],
                outputs=[use_custom_lora, style_dropdown, lora_repo_input, safetensors_input, lora_scale_slider, theme_input]
            )

            generate_button = gr.Button("Generate Image")

        with gr.Column():
            output_image = gr.Image(label="Generated Image")
            final_prompt_out = gr.Textbox(label="Used Prompt")
            stats_out = gr.Textbox(label="System Info")

    generate_button.click(
        generate_image_from_api,
        inputs=[
            prompt_input, method, model_choice, style_dropdown, aspect_dropdown,
            use_custom_lora, lora_repo_input, lora_scale_slider, safetensors_input,
            guidance_scale_slider, steps_slider, width_slider, height_slider, image_type, theme_input
        ],
        outputs=[output_image, final_prompt_out, stats_out]
    )

demo.launch(share=True)