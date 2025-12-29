import gradio as gr
import requests
import json
import time
from datetime import datetime
import tempfile
import os

# Define API endpoints
API_BASE_URL = "http://localhost:8000/api/v1"  # Change this directly in the code if needed
TRANSLATION_ENDPOINT = f"{API_BASE_URL}/translate"
PLAIN_LANGUAGE_ENDPOINT = f"{API_BASE_URL}/plain-translation"

# Define language options
TARGET_LANGUAGES_PRIMARY = {
    "1": "English",
    "2": "German",
    "3": "Spanish",
    "4": "French",
    "5": "Chinese",
    "6": "Japanese",
    "7": "Thai",
    "8": "Russian",
    "9": "Arabic",
    "10": "Portuguese",
    "11": "Czech",
    "12": "Danish",
    "13": "Dutch",
    "14": "Finnish",
    "15": "Hungarian",
    "16": "Italian",
    "17": "Korean",
    "18": "Norwegian",
    "19": "Polish",
    "20": "Swedish",
    "21": "Turkish",
    "22": "Ukrainian"
}

# Convert to list for dropdown
LANGUAGE_OPTIONS = list(TARGET_LANGUAGES_PRIMARY.values())

# Custom CSS for dark theme modern UI
custom_css = """
:root {
    --background-color: #111827;
    --surface-color: #1F2937;
    --primary-color: #3B82F6;
    --secondary-color: #4F46E5;
    --text-color: #F9FAFB;
    --text-muted: #9CA3AF;
    --border-color: #374151;
    --success-color: #10B981;
    --error-color: #EF4444;
}

body {
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
}

.app-container {
    max-width: 1200px;
    margin: 0 auto;
}

.app-header {
    padding: 24px 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 24px;
}

.app-title {
    color: white;
    font-size: 32px !important;
    font-weight: 700 !important;
    margin-bottom: 4px !important;
}

.app-subtitle {
    color: var(--text-muted);
    font-size: 16px !important;
    margin-top: 0 !important;
}

.status-container {
    background-color: var(--surface-color);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 24px;
}

.status-label {
    font-weight: 600;
    color: #6366F1;
    background-color: #312E81;
    padding: 4px 12px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 8px;
}

.status-online {
    display: flex;
    align-items: center;
    background-color: rgba(16, 185, 129, 0.1);
    color: var(--success-color);
    border-radius: 6px;
    padding: 8px 12px;
    font-weight: 600;
}

.status-offline {
    display: flex;
    align-items: center;
    background-color: rgba(239, 68, 68, 0.1);
    color: var(--error-color);
    border-radius: 6px;
    padding: 8px 12px;
    font-weight: 600;
}

.stat-box {
    background-color: var(--surface-color);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
}

.stat-label {
    font-weight: 600;
    color: #6366F1;
    background-color: #312E81;
    padding: 4px 12px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 12px;
}

.stat-item {
    color: var(--text-color);
    margin-bottom: 8px;
    font-size: 16px;
}

.stat-value {
    color: var(--text-color);
    font-size: 22px;
    font-weight: 700;
}

.stat-unit {
    color: var(--text-muted);
    font-size: 16px;
    font-weight: 500;
}

.input-box, .output-box {
    background-color: var(--surface-color);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
}

.input-label, .output-label {
    font-weight: 600;
    color: #6366F1;
    background-color: #312E81;
    padding: 4px 12px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 12px;
}

.action-btn {
    width: 100% !important;
    margin-top: 16px !important;
    background-color: var(--primary-color) !important;
    color: white !important;
    font-weight: 600 !important;
    height: 48px !important;
    border-radius: 6px !important;
}

.download-btn {
    background-color: var(--secondary-color) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}

footer {
    text-align: center;
    padding: 24px 0;
    color: var(--text-muted);
    font-size: 14px;
    border-top: 1px solid var(--border-color);
    margin-top: 40px;
}

/* Fix tab styling for dark theme */
.tab-nav {
    border-bottom-color: var(--border-color) !important;
}

.tab-nav button {
    color: var(--text-muted) !important;
}

.tab-nav-active {
    color: var(--primary-color) !important;
    border-color: var(--primary-color) !important;
}

/* Fix textbox styling for dark theme */
textarea {
    background-color: #374151 !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
    resize: none !important;
}

/* Fix dropdown styling */
.gr-dropdown {
    background-color: #374151 !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
}

/* Style code blocks */
.code-editor {
    background-color: #374151 !important;
    color: var(--text-color) !important;
    border-radius: 6px !important;
}

/* Download links styling */
.download-links a button {
    background-color: var(--secondary-color) !important;
    color: white !important;
    border: none !important;
    padding: 8px 16px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    font-weight: 600 !important;
    margin-right: 10px !important;
    margin-top: 12px !important;
}
"""

# Helper functions
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            return True, "API Online"
        else:
            return False, f"API Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "API Offline"
    except Exception as e:
        return False, f"Error: {str(e)}"

def format_stats_simple(stats_data):
    """Format stats data into HTML for better presentation"""
    if not stats_data or stats_data == "No data available":
        return ""
    
    # Extract values from stats string
    lines = stats_data.strip().split("\n")
    stats_dict = {}
    
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            stats_dict[key.strip()] = value.strip()
    
    # Create HTML output
    html = """<div class="stat-box">
    <div class="stat-label">Stats</div>"""
    
    if "Processing Time" in stats_dict:
        time_value = stats_dict["Processing Time"].replace("s", "")
        html += f"""
        <div class="stat-item">
            Processing Time: <span class="stat-value">{time_value}</span><span class="stat-unit">s</span>
        </div>
        """
    
    if "Source Words" in stats_dict:
        html += f"""
        <div class="stat-item">
            Source Words: <span class="stat-value">{stats_dict["Source Words"]}</span>
        </div>
        """
    
    if "Result Words" in stats_dict:
        html += f"""
        <div class="stat-item">
            Result Words: <span class="stat-value">{stats_dict["Result Words"]}</span>
        </div>
        """
    
    if "Source Characters" in stats_dict:
        html += f"""
        <div class="stat-item">
            Source Characters: <span class="stat-value">{stats_dict["Source Characters"]}</span>
        </div>
        """
    
    if "Result Characters" in stats_dict:
        html += f"""
        <div class="stat-item">
            Result Characters: <span class="stat-value">{stats_dict["Result Characters"]}</span>
        </div>
        """
    
    html += "</div>"
    
    return html

def translate_text(source_text, target_language):
    """Call translation API endpoint"""
    if not source_text.strip():
        return "Please enter text to translate", None, None
    
    try:
        response = requests.post(
            TRANSLATION_ENDPOINT,
            data={"source_text": source_text, "target_language": target_language}
        )
        
        if response.status_code == 200:
            result = response.json()
            translated_text = result.get("data", {}).get("translated_text", "")
            
            # Extract metrics
            processing_time = result.get('meta', {}).get('processing_time_seconds', 0)
            source_chars = result.get('meta', {}).get('source_char_count', 0)
            target_chars = result.get('meta', {}).get('target_char_count', 0)
            
            # Create formatted metrics
            stats = f"""Processing Time: {processing_time:.2f}s
Source Characters: {source_chars}
Result Characters: {target_chars}"""
            
            # JSON response for display
            json_str = json.dumps(result, indent=2)
            
            return translated_text, stats, json_str
        else:
            return f"Error: API returned status code {response.status_code}", None, None
    except Exception as e:
        return f"Connection error: {str(e)}", None, None

def convert_to_plain_language(text):
    """Call plain language API endpoint"""
    if not text.strip():
        return "Please enter text to convert", None, None
    
    try:
        response = requests.post(
            PLAIN_LANGUAGE_ENDPOINT,
            data={"text": text}
        )
        
        if response.status_code == 200:
            result = response.json()
            plain_text = result.get("translation", "")
            
            # Extract metrics
            processing_time = result.get('metadata', {}).get('processing_time', 0)
            source_words = result.get('metadata', {}).get('source_word_count', 0)
            result_words = result.get('metadata', {}).get('translated_word_count', 0)
            
            # Create formatted metrics
            stats = f"""Processing Time: {processing_time:.2f}s
Source Words: {source_words}
Result Words: {result_words}"""
            
            # JSON response for display
            json_str = json.dumps(result, indent=2)
            
            return plain_text, stats, json_str
        else:
            return f"Error: API returned status code {response.status_code}", None, None
    except Exception as e:
        return f"Connection error: {str(e)}", None, None

def process_file(file, process_type, target_language=None):
    """Process uploaded file"""
    if file is None:
        return "Please upload a file", None, None
    
    try:
        # Read the file content properly
        with open(file.name, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if process_type == "translation":
            if not target_language:
                return "Please select a target language", None, None
            return translate_text(content, target_language)
        else:  # plain language
            return convert_to_plain_language(content)
    except Exception as e:
        return f"Error processing file: {str(e)}", None, None

# Create download links function
def create_download_links(result_text, json_text, prefix="download"):
    """Create HTML with download links instead of using file component"""
    if not result_text or result_text == "No data available" or result_text.startswith("Error:") or result_text.startswith("Please "):
        return "No content available to download"
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    html = """<div class="download-links" style="margin-top: 10px;">"""
    
    # Encoded result text for download
    text_filename = f"{prefix}_{timestamp}.txt"
    
    # Add button for text download
    html += f"""
    <a href="data:text/plain;charset=utf-8,{result_text}" download="{text_filename}" style="text-decoration: none; margin-right: 10px;">
        <button>
            Download Result Text
        </button>
    </a>
    """
    
    # Add button for JSON download if available
    if json_text and json_text != "Error response" and json_text != "No data available":
        json_filename = f"api_response_{timestamp}.json"
        html += f"""
        <a href="data:application/json;charset=utf-8,{json_text}" download="{json_filename}" style="text-decoration: none;">
            <button>
                Download API Response
            </button>
        </a>
        """
    
    html += "</div>"
    return html

# Create wrapper functions for each download type
def create_translation_links(result_text, json_text):
    return create_download_links(result_text, json_text, "translation")

def create_plain_links(result_text, json_text):
    return create_download_links(result_text, json_text, "plain_language")

def create_file_links(result_text, json_text):
    return create_download_links(result_text, json_text, "processed_file")

# Create Gradio interface
with gr.Blocks(css=custom_css, title="Content Processor", theme=gr.themes.Base()) as demo:
    # Header
    with gr.Row(elem_classes="app-header"):
        gr.Markdown('<h1 class="app-title">Content Processor</h1>')
        gr.Markdown('<p class="app-subtitle">Transform content between languages and formats using advanced language models</p>')
    
    # API Status
    api_online, status_message = check_api_health()
    status_class = "status-online" if api_online else "status-offline"
    status_icon = "✅" if api_online else "❌"
    
    with gr.Row(elem_classes="status-container"):
        with gr.Column():
            gr.HTML(f'<div class="status-label">API Status</div>')
            gr.HTML(f'<div class="{status_class}">{status_icon} {status_message}</div>')
    
    # Main content
    with gr.Tabs():
        # Translation Tab
        with gr.TabItem("Multilingual Translation"):
            with gr.Row():
                with gr.Column():
                    with gr.Column(elem_classes="input-box"):
                        gr.HTML('<div class="input-label">Source Text (Any Language)</div>')
                        source_text = gr.Textbox(
                            label="",
                            placeholder="Enter text to translate...",
                            lines=10,
                            show_label=False
                        )
                    
                    with gr.Row():
                        target_language = gr.Dropdown(
                            choices=LANGUAGE_OPTIONS,
                            label="Target Language",
                            value=LANGUAGE_OPTIONS[0]
                        )
                    
                    translate_btn = gr.Button("Translate", variant="primary", elem_classes="action-btn")
                
                with gr.Column():
                    with gr.Column(elem_classes="output-box"):
                        gr.HTML('<div class="output-label">Translation Result</div>')
                        translation_result = gr.Textbox(
                            label="",
                            lines=10,
                            interactive=False,
                            show_label=False
                        )
                    
                    # Stats in HTML format
                    translation_stats_raw = gr.Textbox(visible=False)
                    translation_stats_html = gr.HTML()
                    
                    # API Response viewer
                    with gr.Accordion("View API Response", open=False):
                        translation_json_display = gr.Code(
                            language="json",
                            label="API Response",
                            show_label=True
                        )
                    
                    # Hidden field for storing API response
                    translation_json = gr.Textbox(visible=False)
                    
                    # Download links
                    translation_download_links = gr.HTML()
                    translation_download = gr.Button("Download Results", variant="secondary", elem_classes="download-btn")
            
            # Connect components
            translate_output = translate_btn.click(
                fn=translate_text,
                inputs=[source_text, target_language],
                outputs=[translation_result, translation_stats_raw, translation_json]
            )
            
            # Update stats and JSON display
            translate_output.then(
                fn=format_stats_simple,
                inputs=translation_stats_raw,
                outputs=translation_stats_html
            )
            
            translate_output.then(
                fn=lambda x: x,
                inputs=translation_json,
                outputs=translation_json_display
            )
            
            # Add download functionality
            translation_download.click(
                fn=create_translation_links,
                inputs=[translation_result, translation_json],
                outputs=translation_download_links
            )
        
        # Plain Language Tab
        with gr.TabItem("Plain Language Conversion"):
            with gr.Row():
                with gr.Column():
                    with gr.Column(elem_classes="input-box"):
                        gr.HTML('<div class="input-label">German Text</div>')
                        plain_source = gr.Textbox(
                            label="",
                            placeholder="Enter German text to convert to Leichte Sprache...",
                            lines=10,
                            show_label=False
                        )
                    
                    plain_btn = gr.Button("Convert to Plain Language", variant="primary", elem_classes="action-btn")
                
                with gr.Column():
                    with gr.Column(elem_classes="output-box"):
                        gr.HTML('<div class="output-label">Plain Language Result</div>')
                        plain_result = gr.Textbox(
                            label="",
                            lines=10,
                            interactive=False,
                            show_label=False
                        )
                    
                    # Stats in HTML format
                    plain_stats_raw = gr.Textbox(visible=False)
                    plain_stats_html = gr.HTML()
                    
                    # API Response viewer
                    with gr.Accordion("View API Response", open=False):
                        plain_json_display = gr.Code(
                            language="json",
                            label="API Response",
                            show_label=True
                        )
                    
                    # Hidden field for storing API response
                    plain_json = gr.Textbox(visible=False)
                    
                    # Download links
                    plain_download_links = gr.HTML()
                    plain_download = gr.Button("Download Results", variant="secondary", elem_classes="download-btn")
            
            # Connect components
            plain_output = plain_btn.click(
                fn=convert_to_plain_language,
                inputs=plain_source,
                outputs=[plain_result, plain_stats_raw, plain_json]
            )
            
            # Update stats and JSON display
            plain_output.then(
                fn=format_stats_simple,
                inputs=plain_stats_raw,
                outputs=plain_stats_html
            )
            
            plain_output.then(
                fn=lambda x: x,
                inputs=plain_json,
                outputs=plain_json_display
            )
            
            # Add download functionality
            plain_download.click(
                fn=create_plain_links,
                inputs=[plain_result, plain_json],
                outputs=plain_download_links
            )
        
        # File Processing Tab
        with gr.TabItem("File Processing"):
            with gr.Row():
                with gr.Column():
                    with gr.Column(elem_classes="input-box"):
                        gr.HTML('<div class="input-label">Upload File</div>')
                        file_upload = gr.File(label="", show_label=False)
                    
                    process_options = gr.Radio(
                        ["Translate to another language", "Convert to plain language (German only)"],
                        label="Processing Option",
                        value="Translate to another language"
                    )
                    
                    file_target_lang = gr.Dropdown(
                        choices=LANGUAGE_OPTIONS,
                        label="Target Language (for translation)",
                        value=LANGUAGE_OPTIONS[0],
                        visible=True
                    )
                    
                    def update_language_visibility(option):
                        return gr.update(visible=(option == "Translate to another language"))
                    
                    process_options.change(
                        fn=update_language_visibility,
                        inputs=process_options,
                        outputs=file_target_lang
                    )
                    
                    process_file_btn = gr.Button("Process File", variant="primary", elem_classes="action-btn")
                
                with gr.Column():
                    with gr.Column(elem_classes="output-box"):
                        gr.HTML('<div class="output-label">Processing Result</div>')
                        file_result = gr.Textbox(
                            label="",
                            lines=10,
                            interactive=False,
                            show_label=False
                        )
                    
                    # Stats in HTML format
                    file_stats_raw = gr.Textbox(visible=False)
                    file_stats_html = gr.HTML()
                    
                    # API Response viewer
                    with gr.Accordion("View API Response", open=False):
                        file_json_display = gr.Code(
                            language="json",
                            label="API Response",
                            show_label=True
                        )
                    
                    # Hidden field for storing API response
                    file_json = gr.Textbox(visible=False)
                    
                    # Download links
                    file_download_links = gr.HTML()
                    file_download = gr.Button("Download Results", variant="secondary", elem_classes="download-btn")
            
            # Connect components
            def process_file_handler(file, option, language):
                process_type = "translation" if option == "Translate to another language" else "plain"
                return process_file(file, process_type, language if process_type == "translation" else None)
            
            file_output = process_file_btn.click(
                fn=process_file_handler,
                inputs=[file_upload, process_options, file_target_lang],
                outputs=[file_result, file_stats_raw, file_json]
            )
            
            # Update stats and JSON display
            file_output.then(
                fn=format_stats_simple,
                inputs=file_stats_raw,
                outputs=file_stats_html
            )
            
            file_output.then(
                fn=lambda x: x,
                inputs=file_json,
                outputs=file_json_display
            )
            
            # Add download functionality
            file_download.click(
                fn=create_file_links,
                inputs=[file_result, file_json],
                outputs=file_download_links
            )

    # Footer
    gr.HTML('<footer>Content Processor | Advanced Language Tools</footer>')
    
    # API Documentation Section
    with gr.Accordion("API Documentation", open=False):
        gr.Markdown(f"""
        # API Documentation
        
        This application uses a REST API backend for processing text. You can interact with the API directly using curl or any HTTP client.
        
        ## Translation API
        
        **Endpoint:** `{API_BASE_URL}` 
        
        **Method:** POST
        
        **Example curl command:**
        ```bash
        curl -X POST \\
          {API_BASE_URL}/translate \\
          -H "Content-Type: application/x-www-form-urlencoded" \\
          -d "source_text=Hello world&target_language=German"
        ```
        
        **Response format:**
        ```json
        {{
          "data": {{
            "translated_text": "Hallo Welt"
          }},
          "meta": {{
            "processing_time_seconds": 0.35,
            "source_char_count": 11,
            "target_char_count": 10
          }}
        }}
        ```
        
        ## Plain Language API
        
        **Endpoint:** `{API_BASE_URL}`
        
        **Method:** POST
        
        **Example curl command:**
        ```bash
        curl -X POST \\
          {API_BASE_URL}/plain-translation \\
          -H "Content-Type: application/x-www-form-urlencoded" \\
          -d "text=Dieser Text ist sehr komplex und sollte in einfache Sprache übersetzt werden."
        ```
        
        **Response format:**
        ```json
        {{
          "translation": "Dieser Text ist einfach.",
          "metadata": {{
            "processing_time": 0.42,
            "source_word_count": 12,
            "translated_word_count": 4
          }}
        }}
        ```
        
        ## Health Check API
        
        **Endpoint:** `{API_BASE_URL}/health`
        
        **Method:** GET
        
        **Example curl command:**
        ```bash
        curl -X GET {API_BASE_URL}/health
        ```
        
        **Response format:**
        ```json
        {{
          "status": "ok",
          "message": "API is running"
        }}
        ```
        """)

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)