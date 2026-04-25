#!/bin/bash
# Install flash-attn via pre-built wheel (no CUDA toolkit / nvcc required).
# Matches: Python 3.10, PyTorch 2.10, CUDA 12.8 (cu128)

set -e
WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3%2Bcu128torch2.10-cp310-cp310-linux_x86_64.whl"

echo "Installing flash-attn from pre-built wheel (cu128, torch2.10, cp310)..."
pip install "$WHEEL_URL"
echo "Done. Verify with: python -c 'import flash_attn; print(flash_attn.__version__)'"
