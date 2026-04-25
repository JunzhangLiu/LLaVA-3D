#!/bin/bash
# Download LLaVA-3D-7B from Hugging Face to checkpoints/llava-3d-7b (no CLI required).
# Usage: ./download_model.sh [target_dir]
# Default: checkpoints/llava-3d-7b

set -e
REPO_ID="ChaimZhu/LLaVA-3D-7B"
LOCAL_DIR="${1:-checkpoints/llava-3d-7b}"

mkdir -p "$LOCAL_DIR"
echo "Downloading $REPO_ID to $LOCAL_DIR ..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO_ID', local_dir='$LOCAL_DIR')
"
echo "Done. Model saved under $LOCAL_DIR"
