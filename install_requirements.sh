#!/bin/bash

# ==============================================
# Script: install_custom_nodes.sh
# Mục đích: Cài đặt các thư viện cho custom nodes của ComfyUI
# ==============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Đường dẫn gốc tới custom_nodes
CUSTOM_NODES_DIR="$SCRIPT_DIR/ComfyUI/custom_nodes"

echo "📂 Đường dẫn custom nodes: $CUSTOM_NODES_DIR"
cd "$CUSTOM_NODES_DIR" || { echo "❌ Không tìm thấy thư mục custom_nodes!"; exit 1; }

# Danh sách các custom node cần cài
NODES=(
    "ComfyUI-WanVideoWrapper"
    "InfiniteTalk"
    "audio-separation-nodes-comfyui"
    "comfyui-kjnodes"
    "comfyui-videohelpersuite"
    "ComfyUI-MelBandRoFormer"
)

# Lặp qua từng node
for NODE in "${NODES[@]}"; do
    NODE_PATH="$CUSTOM_NODES_DIR/$NODE"
    REQ_FILE="$NODE_PATH/requirements.txt"

    if [ -d "$NODE_PATH" ]; then
        echo "-------------------------------------------------"
        echo "📦 Đang xử lý: $NODE"
        cd "$NODE_PATH" || continue

        if [ -f "$REQ_FILE" ]; then
            echo "📘 Đang cài thư viện từ $REQ_FILE..."
            pip install -r requirements.txt --no-cache-dir
        else
            echo "⚠️  Không có file requirements.txt trong $NODE"
        fi

        cd "$CUSTOM_NODES_DIR" || exit
    else
        echo "⚠️  Bỏ qua: $NODE (thư mục không tồn tại)"
    fi
done

echo "✅ Hoàn tất cài đặt tất cả custom nodes!"