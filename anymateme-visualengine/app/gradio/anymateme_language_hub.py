import gradio as gr
import requests

# Configuration - change these to match your API endpoints
API_BASE_URL = "http://localhost:6004/api/v1"

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
        return "Please enter text to translate"
    
    try:
        result = call_api(f"{API_BASE_URL}/translate", {
            "source_text": text, 
            "target_language": target_language
        })
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if "data" in result and "translated_text" in result["data"]:
            return result["data"]["translated_text"]
        else:
            return "Invalid response from API"
    except Exception as e:
        return f"Error: {str(e)}"

def plain_language_service(text):
    """Plain language conversion service"""
    if not text.strip():
        return "Please enter text to convert"
    
    try:
        result = call_api(f"{API_BASE_URL}/plain-translation", {
            "text": text,
        })
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if "data" in result and "translation" in result["data"]:
            return result["data"]["translation"]
        else:
            return "Invalid response from API"
    except Exception as e:
        return f"Error: {str(e)}"

# Custom CSS for enhanced UI with dark theme
custom_css = """
body {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
}

.gradio-container {
    background-color: #0f172a !important;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

.header-logo {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
}

.header-logo img {
    max-height: 60px;
    margin-right: 1rem;
}

.header-content h1 {
    margin: 0;
    font-size: 2.5rem;
    background: linear-gradient(90deg, #3b82f6, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.title-link {
    text-decoration: none;
}

.tab-header {
    margin-top: 0 !important;
    margin-bottom: 1rem !important;
    font-size: 1.5rem !important;
    color: #4f46e5 !important;
}

.tab-description {
    margin-bottom: 1.5rem !important;
    color: #94a3b8 !important;
}

.function-section {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #1e293b;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    margin-bottom: 1rem;
}

.footer {
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid #334155;
    color: #94a3b8;
    font-size: 0.875rem;
}

.button-primary {
    background-color: #e97434 !important;
    color: white !important;
    border: none !important;
}

.button-primary:hover {
    background-color: #ea580c !important;
}

/* Input/output styling */
.input-area textarea, .output-area textarea {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 0.375rem !important;
    transition: border-color 0.15s ease-in-out !important;
}

.input-area textarea:focus, .output-area textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
}

.input-area label, .output-area label {
    color: #e2e8f0 !important;
}

/* Dropdown styling */
.gradio-dropdown {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
}

/* Tab styling */
.tabs > div:first-child {
    border-bottom: 1px solid #334155 !important;
}

.tabs > div:first-child button {
    color: #94a3b8 !important;
}

.tabs > div:first-child button[data-selected="true"] {
    color: #e2e8f0 !important;
    border-bottom-color: #6366f1 !important;
}

/* Examples styling */
.examples {
    margin-top: 1rem;
    padding: 0.75rem;
    background-color: #1e293b;
    border-radius: 0.375rem;
    font-size: 0.875rem;
}

.examples-title {
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #94a3b8;
}

.example-item {
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    background-color: #334155;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
    display: inline-block;
    cursor: pointer;
    transition: background-color 0.15s ease-in-out;
    color: #e2e8f0;
}

.example-item:hover {
    background-color: #475569;
}

/* Footer link */
.footer-link {
    color: #60a5fa;
    text-decoration: none;
}

.footer-link:hover {
    text-decoration: underline;
}

/* Fix for copy button */
.copy-button {
    background-color: #334155 !important;
    color: #e2e8f0 !important;
    border: 1px solid #475569 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

.copy-button:hover {
    background-color: #475569 !important;
}

/* Make sure all icon buttons are visible */
button svg {
    color: #e2e8f0 !important;
}

/* Override Gradio's default copy button styles */
.output-area .wrap .copy_button {
    opacity: 1 !important;
    visibility: visible !important;
    color: #e2e8f0 !important;
    background-color: #334155 !important;
}

.output-area .wrap .copy_button:hover {
    background-color: #475569 !important;
}
"""

# Translation interface
def translation_interface():
    with gr.Blocks() as demo:
        with gr.Column(elem_classes="function-section"):
            gr.Markdown("## 🌐 Translation Service", elem_classes="tab-header")
            gr.Markdown("Translate text between multiple languages with high accuracy and natural results.", 
                        elem_classes="tab-description")
            
            with gr.Row():
                with gr.Column(scale=6, elem_classes="input-area"):
                    input_text = gr.Textbox(
                        label="Text to Translate",
                        placeholder="Enter the text you want to translate...",
                        lines=8
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
                        
                        btn = gr.Button("Translate", variant="primary", elem_classes="button-primary")
                
                with gr.Column(scale=6, elem_classes="output-area"):
                    output_text = gr.Textbox(
                        label="Translation Result",
                        placeholder="Translation will appear here...",
                        lines=8,
                        show_copy_button=True,
                        elem_classes="output-text-with-copy"
                    )
            
            # Translation examples
            with gr.Accordion("Examples (click to try)", open=False, elem_classes="examples"):
                gr.Markdown("### Quick examples to try:", elem_classes="examples-title")
                with gr.Row():
                    example1 = gr.Button("Hello, how are you today?", elem_classes="example-item")
                    example2 = gr.Button("Ich freue mich, Sie kennenzulernen.", elem_classes="example-item")
                    example3 = gr.Button("Hola, ¿cómo estás?", elem_classes="example-item")
            
            example1.click(fn=lambda: "Hello, how are you today?", inputs=None, outputs=input_text)
            example2.click(fn=lambda: "Ich freue mich, Sie kennenzulernen.", inputs=None, outputs=input_text)
            example3.click(fn=lambda: "Hola, ¿cómo estás?", inputs=None, outputs=input_text)
        
        def process_translation(text, target_lang):
            if not text.strip():
                return "Please enter text to translate"
                
            result = translation_service(text, target_lang)
            return result
        
        btn.click(
            fn=process_translation,
            inputs=[input_text, target_lang],
            outputs=output_text
        )
    
    return demo

# Plain language interface
def plain_language_interface():
    with gr.Blocks() as demo:
        with gr.Column(elem_classes="function-section"):
            gr.Markdown("## 📝 Plain Language Conversion", elem_classes="tab-header")
            gr.Markdown("Convert complex German text to simple, easy-to-understand language.", 
                        elem_classes="tab-description")
            
            with gr.Row():
                with gr.Column(scale=6, elem_classes="input-area"):
                    input_text = gr.Textbox(
                        label="German Text",
                        placeholder="Geben Sie hier Ihren deutschen Text ein...",
                        lines=8
                    )
                    
                    btn = gr.Button("Convert to Plain Language", variant="primary", elem_classes="button-primary")
                
                with gr.Column(scale=6, elem_classes="output-area"):
                    output_text = gr.Textbox(
                        label="Plain Language Result",
                        placeholder="Simplified text will appear here...",
                        lines=8,
                        show_copy_button=True,
                        elem_classes="output-text-with-copy"
                    )
            
            # Plain language examples
            with gr.Accordion("Examples (click to try)", open=False, elem_classes="examples"):
                gr.Markdown("### Quick examples to try:", elem_classes="examples-title")
                with gr.Row():
                    example1 = gr.Button("Die Antragstellung für ein Reisevisum erfordert die Vorlage aller erforderlichen Unterlagen gemäß Paragraph 5 der Einreisebestimmungen.", 
                                        elem_classes="example-item")
                    example2 = gr.Button("Die moderne Technologie hat heutzutage die Art und Weise, wie wir Informationen verarbeiten und kommunizieren, grundlegend transformiert.", 
                                        elem_classes="example-item")
            
            example1.click(
                fn=lambda: "Die Antragstellung für ein Reisevisum erfordert die Vorlage aller erforderlichen Unterlagen gemäß Paragraph 5 der Einreisebestimmungen.", 
                inputs=None, 
                outputs=input_text
            )
            example2.click(
                fn=lambda: "Die moderne Technologie hat heutzutage die Art und Weise, wie wir Informationen verarbeiten und kommunizieren, grundlegend transformiert.", 
                inputs=None, 
                outputs=input_text
            )
        
        def process_plain_language(text):
            if not text.strip():
                return "Please enter text to convert"
                
            result = plain_language_service(text)
            return result
        
        btn.click(
            fn=process_plain_language,
            inputs=[input_text],
            outputs=output_text
        )
    
    return demo

# Main app that combines interfaces
demo = gr.Blocks(css=custom_css, title="Anymateme EduHub")

with demo:
    # Header with logo and hyperlink to company website
    with gr.Row(elem_classes="header-logo"):
        gr.HTML("""
        <div class="header-logo">
            <a href="https://anymateme.com/" target="_blank" class="title-link">
                <img src="file/logo.png" alt="Anymateme Logo">
            </a>
            <div class="header-content">
                <h1>EduHub</h1>
            </div>
        </div>
        """)
    
    gr.Markdown("### Educational resources and tools for students and educators")
    
    with gr.Tabs():
        with gr.TabItem("Translation"):
            translation_interface()
        
        with gr.TabItem("Plain Language"):
            plain_language_interface()
    
    # Footer with additional information and hyperlink
    with gr.Row(elem_classes="footer"):
        with gr.Column(scale=1):
            gr.Markdown("### About")
            gr.Markdown("Anymateme EduHub provides educational resources and tools for students and educators.")
        
        with gr.Column(scale=1):
            gr.Markdown("### Support")
            gr.Markdown('For questions or support, contact <a href="mailto:info@anymateme.com" class="footer-link">info@anymateme.com</a>')
        
        with gr.Column(scale=1):
            gr.Markdown("### © 2025 Anymateme GmbH")
            gr.Markdown('Visit our <a href="https://anymateme.com/" target="_blank" class="footer-link">website</a> for more information')

# Launch the demo
if __name__ == "__main__":
    # Set the server name and port
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)