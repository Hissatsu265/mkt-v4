import gradio as gr
import requests
import json

# Configuration - change these to match your API endpoints
API_BASE_URL = "http://localhost:8001/api/v1"

# Basic API call function
def call_api(endpoint, data):
    """Make a request to the API"""
    try:
        response = requests.post(endpoint, data=data, timeout=60)
        if response.status_code == 200:
            return response.json()
        return {"error": f"API returned status code {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# Service functions
def translation_service(text, target_language):
    """Translation service"""
    if not text.strip():
        return "Please enter text to translate", ""
    
    try:
        result = call_api(f"{API_BASE_URL}/translate", {
            "source_text": text, 
            "target_language": target_language
        })
        
        if "error" in result:
            return f"Error: {result['error']}", ""
        
        if "data" in result and "translated_text" in result["data"]:
            # Format metadata
            meta = result.get("meta", {})
            metadata = f"Processing time: {meta.get('response_time', 0):.2f} seconds\n"
            metadata += f"Source length: {meta.get('source_char_count', 0)} characters\n"
            metadata += f"Result length: {meta.get('target_char_count', 0)} characters"
            
            return result["data"]["translated_text"], metadata
        else:
            return "Invalid response from API", ""
    except Exception as e:
        return f"Error: {str(e)}", ""

def plain_language_service(text):
    """Plain language conversion service"""
    if not text.strip():
        return "Please enter text to convert", ""
    
    try:
        result = call_api(f"{API_BASE_URL}/plain-translation", {
            "text": text,
        })
        
        if "error" in result:
            return f"Error: {result['error']}", ""
        
        if "data" in result and "translation" in result["data"]:
            # Format metadata
            meta = result.get("meta", {})
            metadata = f"Processing time: {meta.get('response_time', 0):.2f} seconds\n"
            metadata += f"Source words: {meta.get('source_word_count', 0)}\n"
            metadata += f"Result words: {meta.get('translated_word_count', 0)}"
            
            return result["data"]["translation"], metadata
        else:
            return "Invalid response from API", ""
    except Exception as e:
        return f"Error: {str(e)}", ""

def slide_content_service(content, template_type):
    """Single slide content generation service"""
    if not content.strip():
        return "Please enter content to generate a slide from", ""
    
    try:
        result = call_api(f"{API_BASE_URL}/generate-slides", {
            "content": content,
            "template_types": template_type
        })
        
        if "error" in result:
            return f"Error: {result['error']}", ""
        
        if "slides" in result and len(result["slides"]) > 0:
            slide = result["slides"][0]
            
            # Format slide as plain text
            formatted = f"--- {slide.get('title', 'Slide')} ---\n\n"
            
            # Format content based on template type
            content_obj = slide.get("content", {})
            template_type = slide.get("template_type", "")
            
            if template_type == "bullet_points" and "bullets" in content_obj:
                for bullet in content_obj["bullets"]:
                    formatted += f"• {bullet}\n"
            
            elif template_type == "numbered_list" and "numbered_items" in content_obj:
                for i, item in enumerate(content_obj["numbered_items"], 1):
                    formatted += f"{i}. {item}\n"
            
            elif template_type == "two_column" and "left_column" in content_obj and "right_column" in content_obj:
                formatted += "Left Column:\n"
                for item in content_obj["left_column"]:
                    formatted += f"• {item}\n"
                
                formatted += "\nRight Column:\n"
                for item in content_obj["right_column"]:
                    formatted += f"• {item}\n"
            
            elif template_type == "paragraph_blocks" and "paragraphs" in content_obj:
                for para in content_obj["paragraphs"]:
                    formatted += f"{para}\n\n"
            
            elif template_type == "hierarchical_outline":
                if "level1" in content_obj:
                    for item in content_obj["level1"]:
                        formatted += f"• {item}\n"
                        
                    if "level2" in content_obj:
                        for item in content_obj["level2"]:
                            formatted += f"  - {item}\n"
                            
                        if "level3" in content_obj:
                            for item in content_obj["level3"]:
                                formatted += f"    * {item}\n"
            
            else:
                # Fallback for unknown template types
                formatted += json.dumps(content_obj, indent=2)
            
            metadata = f"Generated slide using template: {template_type}"
            
            return formatted, metadata
        else:
            return "Invalid response from API or no slides generated", ""
    except Exception as e:
        return f"Error: {str(e)}", ""

def speech_generation_service(content, speech_style, language):
    """Speech generation service"""
    if not content.strip():
        return "Please enter content to generate speech from", ""
    
    try:
        result = call_api(f"{API_BASE_URL}/single-slide-speech", {
            "slide_content": content,
            "speech_style": speech_style,
            "language": language
        })
        
        if "error" in result:
            return f"Error: {result['error']}", ""
        
        if "speech" in result:
            speech_text = result["speech"]
            
            # Format metadata
            word_count = result.get('word_count', 0)
            seconds = result.get('approximate_seconds', 0)
            response_time = result.get('response_time', '0s')
            
            metadata = f"Word count: {word_count} words\n"
            metadata += f"Duration: ~{seconds} seconds\n"
            metadata += f"Language: {result.get('language', language)}\n"
            metadata += f"Style: {result.get('speech_style', speech_style)}\n"
            metadata += f"Processing time: {response_time}"
            
            return speech_text, metadata
        else:
            return "Invalid response from API", ""
    except Exception as e:
        return f"Error: {str(e)}", ""

# Custom theme for enhanced UI - Try to use a compatible version
try:
    custom_theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    )
except Exception:
    # Fallback to default theme if the theme system is different
    custom_theme = None

# Translation interface
def translation_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 🌐 Translation Service")
        gr.Markdown("Translate text between multiple languages with high accuracy and natural results.")
        
        with gr.Row():
            with gr.Column(scale=6):
                input_text = gr.Textbox(
                    label="Text to Translate",
                    placeholder="Enter the text you want to translate...",
                    lines=5
                )
                
                with gr.Row():
                    languages = [
                        "English", "German", "Spanish", "French", "Chinese", "Japanese",
                        "Russian", "Arabic", "Portuguese", "Italian"
                    ]
                    target_lang = gr.Dropdown(
                        label="Target Language",
                        choices=languages,
                        value="English"
                    )
                    
                    btn = gr.Button("Translate", variant="primary")
            
            with gr.Column(scale=6):
                output_text = gr.Textbox(
                    label="Translation Result",
                    placeholder="Translation will appear here...",
                    lines=5,
                    show_copy_button=True
                )
                info_text = gr.Textbox(
                    label="Information",
                    lines=3
                )
        
        def process_translation(text, target_lang):
            if not text.strip():
                return "Please enter text to translate", ""
                
            result, metadata = translation_service(text, target_lang)
            return result, metadata
        
        btn.click(
            fn=process_translation,
            inputs=[input_text, target_lang],
            outputs=[output_text, info_text]
        )
    
    return demo

# Plain language interface
def plain_language_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 📝 Plain Language Conversion")
        gr.Markdown("Convert complex German text to simple, easy-to-understand language.")
        
        with gr.Row():
            with gr.Column(scale=6):
                input_text = gr.Textbox(
                    label="German Text",
                    placeholder="Geben Sie hier Ihren deutschen Text ein...",
                    lines=5
                )
                
                btn = gr.Button("Convert to Plain Language", variant="primary")
            
            with gr.Column(scale=6):
                output_text = gr.Textbox(
                    label="Plain Language Result",
                    placeholder="Simplified text will appear here...",
                    lines=5,
                    show_copy_button=True
                )
                info_text = gr.Textbox(
                    label="Information",
                    lines=3
                )
        
        def process_plain_language(text):
            if not text.strip():
                return "Please enter text to convert", ""
                
            result, metadata = plain_language_service(text)
            return result, metadata
        
        btn.click(
            fn=process_plain_language,
            inputs=[input_text],
            outputs=[output_text, info_text]
        )
    
    return demo

# Slide content creation interface
def slide_content_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 📊 Single Slide Content Creation")
        gr.Markdown("Generate professional slide content using various template formats.")
        
        with gr.Row():
            with gr.Column(scale=6):
                content = gr.Textbox(
                    label="Content",
                    placeholder="Enter your content ideas, concepts, or information...",
                    lines=8
                )
                
                with gr.Row():
                    # Updated to match backend template types
                    template_types = [
                        "bullet_points", "numbered_list", "two_column", 
                        "paragraph_blocks", "hierarchical_outline"
                    ]
                    
                    template_type = gr.Dropdown(
                        label="Slide Template",
                        choices=template_types,
                        value="bullet_points"
                    )
                    
                    btn = gr.Button("Generate Slide Content", variant="primary")
            
            with gr.Column(scale=6):
                output_text = gr.Textbox(
                    label="Generated Slide Content",
                    placeholder="Slide content will appear here...",
                    lines=10,
                    show_copy_button=True
                )
                info_text = gr.Textbox(
                    label="Information",
                    lines=3
                )
        
        # Help accordion with examples
        with gr.Accordion("Template Types Guide", open=False):
            gr.Markdown("""
            ### Template Types
            
            - **Bullet Points**: Simple list of key points, perfect for summarizing concepts
            - **Numbered List**: Sequential items for processes, timelines, or prioritized information
            - **Two Column**: Left side for main topics, right side for details or explanations
            - **Paragraph Blocks**: Text blocks with headings for more detailed explanations
            - **Hierarchical Outline**: Multi-level organization for complex, nested information
            
            ### Example Input: Digital Marketing
            
            For a topic like "Digital Marketing Strategies," the different templates would organize the information:
            
            - **Bullet Points**: A simple list of different strategies
            - **Numbered List**: A prioritized approach to implementing strategies
            - **Two Column**: Strategy names on left, benefits on right
            - **Paragraph Blocks**: Detailed explanation of each major strategy
            - **Hierarchical Outline**: Main strategies with sub-tactics and specific implementation details
            """)
        
        def process_slide_content(content, template_type):
            if not content.strip():
                return "Please enter content to generate a slide from", ""
            
            try:
                response = requests.post(
                    f"{API_BASE_URL}/generate-slides", 
                    data={
                        "content": content,
                        "template_types": template_type
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Format the output based on the template type
                    if result.get("slides") and len(result["slides"]) > 0:
                        slide = result["slides"][0]
                        formatted_output = f"--- {slide.get('title', 'Slide')} ---\n\n"
                        
                        content_obj = slide.get("content", {})
                        template_type = slide.get("template_type", "")
                        
                        if template_type == "bullet_points" and "bullets" in content_obj:
                            for bullet in content_obj["bullets"]:
                                formatted_output += f"• {bullet}\n"
                        
                        elif template_type == "numbered_list" and "numbered_items" in content_obj:
                            for i, item in enumerate(content_obj["numbered_items"], 1):
                                formatted_output += f"{i}. {item}\n"
                        
                        elif template_type == "two_column" and "left_column" in content_obj and "right_column" in content_obj:
                            formatted_output += "Left Column:\n"
                            for item in content_obj["left_column"]:
                                formatted_output += f"• {item}\n"
                            
                            formatted_output += "\nRight Column:\n"
                            for item in content_obj["right_column"]:
                                formatted_output += f"• {item}\n"
                        
                        elif template_type == "paragraph_blocks" and "paragraphs" in content_obj:
                            for para in content_obj["paragraphs"]:
                                formatted_output += f"{para}\n\n"
                        
                        elif template_type == "hierarchical_outline":
                            if "level1" in content_obj:
                                for item in content_obj["level1"]:
                                    formatted_output += f"• {item}\n"
                                    
                                if "level2" in content_obj:
                                    for item in content_obj["level2"]:
                                        formatted_output += f"  - {item}\n"
                                        
                                    if "level3" in content_obj:
                                        for item in content_obj["level3"]:
                                            formatted_output += f"    * {item}\n"
                        
                        # Prepare metadata
                        metadata = f"Job ID: {result.get('job_id', 'N/A')}\n"
                        metadata += f"Template: {template_type}\n"
                        metadata += f"Response Time: {result.get('response_time', 'N/A')} seconds"
                        
                        return formatted_output, metadata
                    
                    else:
                        return "No slides generated", ""
                
                else:
                    return f"Error: {response.json().get('error', 'Unknown error')}", ""
            
            except Exception as e:
                return f"Error: {str(e)}", ""
        
        btn.click(
            fn=process_slide_content,
            inputs=[content, template_type],
            outputs=[output_text, info_text]
        )
    
    return demo

# Speech generation interface
def speech_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 🔊 Speech Generation")
        gr.Markdown("Generate concise, engaging speech content for presentations or landing pages.")
        
        with gr.Row():
            with gr.Column(scale=6):
                content = gr.Textbox(
                    label="Content",
                    placeholder="Enter content to transform into speech...",
                    lines=5
                )
                
                with gr.Row():
                    speech_styles = [
                        "formal", "conversational", "persuasive", 
                        "educational", "inspirational", "plain_language"
                    ]
                    
                    speech_style = gr.Dropdown(
                        label="Speech Style",
                        choices=speech_styles,
                        value="conversational"
                    )
                
                with gr.Row():
                    languages = ["en", "de"]
                    language_names = {
                        "en": "English",
                        "de": "German"
                    }
                    
                    language = gr.Radio(
                        label="Language",
                        choices=languages,
                        value="en"
                    )
                
                btn = gr.Button("Generate Speech", variant="primary")
            
            with gr.Column(scale=6):
                output_text = gr.Textbox(
                    label="Generated Speech",
                    placeholder="Speech will appear here...",
                    lines=5,
                    show_copy_button=True
                )
                
                # Stats component for speech metrics
                gr.Markdown("### Speech Metrics")
                info_text = gr.Textbox(
                    label="",
                    lines=5
                )
        
        # Help accordion with examples
        with gr.Accordion("Speech Styles Guide", open=False):
            gr.Markdown("""
            ### Speech Styles
            
            - **Formal**: Professional tone with precise language and logical structure
            - **Conversational**: Friendly, approachable tone with everyday language
            - **Persuasive**: Compelling tone with action-oriented language
            - **Educational**: Informative tone with clear explanations
            - **Inspirational**: Uplifting tone with dynamic, future-focused language
            - **Plain Language**: Simple vocabulary with short, direct sentences
            
            ### Language Notes
            
            - **English**: All speech styles supported
            - **German**: Supports formal (Sie) or informal (Du) addressing
            
            ### Output Format
            
            The speech is generated to be approximately 50 words long (about 15-20 seconds when spoken),
            making it ideal for slide introductions or landing page headings.
            """)
        
        def process_speech(content, style, lang):
            if not content.strip():
                return "Please enter content to generate speech from", ""
            
            result, metadata = speech_generation_service(content, style, lang)
            return result, metadata
        
        btn.click(
            fn=process_speech,
            inputs=[content, speech_style, language],
            outputs=[output_text, info_text]
        )
    
    return demo

# Main app that combines interfaces
# Check if theme is supported
if custom_theme:
    demo = gr.Blocks(theme=custom_theme, title="Anymateme EduHub")
else:
    demo = gr.Blocks(title="Anymateme EduHub")

with demo:
    gr.Markdown("# Anymateme EduHub")
    gr.Markdown("Educational resources and tools for students and educators")
    
    with gr.Tabs():
        with gr.TabItem("Translation"):
            translation_interface()
        
        with gr.TabItem("Plain Language"):
            plain_language_interface()
        
        with gr.TabItem("Slide Content"):
            slide_content_interface()
            
        with gr.TabItem("Speech Generation"):
            speech_interface()
    
    # Footer with additional information
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### About")
            gr.Markdown("Anymateme EduHub provides educational resources and tools for students and educators.")
        
        with gr.Column(scale=1):
            gr.Markdown("### Support")
            gr.Markdown("For questions or support, contact info@anymateme.com")
        
        with gr.Column(scale=1):
            gr.Markdown("### © 2025 Anymateme EduHub")
            gr.Markdown("Version 1.0.0-beta")

# Launch the demo
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)