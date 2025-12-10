#!/bin/bash

# Start ComfyUI main.py in the background
cd /workspace/ComfyUI
python main.py &

# Wait a moment for ComfyUI to initialize
sleep 5

# Start your marketing video AI application
cd /workspace/marketing-video-ai
source venv/bin/activate
python run.py
