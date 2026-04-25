#!/bin/bash
# Download LLaVA-3D-Demo-Data from Hugging Face and put it under ./demo.
# Usage: run from LLaVA-3D root: ./download_demo.sh [target_dir]
# Default: ./demo
# See: https://huggingface.co/datasets/ChaimZhu/LLaVA-3D-Demo-Data

set -e
REPO_ID="ChaimZhu/LLaVA-3D-Demo-Data"
LOCAL_DIR="${1:-demo}"

mkdir -p "$LOCAL_DIR"
echo "Downloading dataset $REPO_ID to $LOCAL_DIR ..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '$REPO_ID',
    repo_type='dataset',
    local_dir='$LOCAL_DIR',
)
"
echo "Done. Demo data saved under $LOCAL_DIR"
