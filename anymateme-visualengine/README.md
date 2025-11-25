Step-by-step instructions for setting up and running the multi-style image generation API with ComfyUI.

## 📋 System Requirements

### Recommended Specifications (Tested Configuration)
- **Python:** 3.10.12
- **PyTorch:** 2.8.0+cu128
- **CUDA:** 12.8
- **NumPy:** 1.26.4
- **GPU:** CUDA-compatible GPU (recommended)
- **Storage:** At least 60GB of free disk space

---

## 🚀 Installation Guide

### Step 1: Install Dependencies

Install required libraries from the requirements files in sequential order:

```bash
# Install dependencies from requirements files in order
cd /ComfyUI/custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
cd /comfyui_controlnet_aux
pip install -r requirements.txt

pip install -r requirements.txt
pip install -r requirements1.txt

# Install ONNX and ONNX Runtime
pip install onnx onnxruntime

# Reinstall numpy with specific version
pip uninstall numpy -y
pip install numpy==1.26.4
pip install pyngrok
# Install PyTorch with CUDA 12.8 support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
```

### Step 2: Download Models

Run the script to download necessary models:

```bash
bash download_model.sh
```

⏳ **Note:** The model download process may take several minutes depending on your internet speed.

### Step 3: Install Ngrok (For Public Link)

```bash
# Install pyngrok
pip install pyngrok

# Sign up for a free account at: https://ngrok.com/
# After registration, get your authtoken and run:
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
```

---

## ⚙️ Configuration

### Change Endpoint Name

To modify the endpoint name, edit the file:
```
app/api/routes/image_gen/image_gen_route.py
```

Go to **line 374** and change the endpoint name as desired.

### Configure Ngrok (In run.py)

The `run.py` file is configured with ngrok to create a public URL. You can:
- Enable/disable ngrok tunnel
- Customize uvicorn parameters

---

## 🎯 Running the Application

### Start the Server

From the project root directory, run:

```bash
python run.py
```

After running, you will see:
- ✅ **Local URL:** `http://localhost:8001`
