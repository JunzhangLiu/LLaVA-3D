# CUDA_VISIBLE_DEVICES=0 python llava/eval/run_llava_3d.py \
#     --model-path ChaimZhu/LLaVA-3D-7B \
#     --video-path ./demo/scannet/scene0356_00 \
#     --query "Is there a red bucket in the scene?"


# One GPU: needs that GPU mostly free (check nvidia-smi; other jobs caused your OOM).
# CUDA_VISIBLE_DEVICES=3 python llava/eval/model_scanqa.py \
#         --model-path /home/chenyt/LLaVA-3D/checkpoints/llava-3d-7b \
#         --question-file playground/data/annotations/ScanQA_v1.0_val.json \
#         --answers-file ./llava-3d-7b-scanqa_answer_val.json \
#         --video-folder /mnt/disk4/chenyt/LLaVA-3D/playground/data/scanet_v2/OpenDataLab___ScanNet_v2/raw/scans \
#         --prune_layer_ratio 5:0.05 8:0.1 14:0.2

# Default: single visible GPU + full model on that GPU (avoids meta-tensor and cross-GPU matmul errors).
# CUDA_VISIBLE_DEVICES=0 python -u llava/eval/model_scanqa.py \
#         --model-path /home/chenyt/LLaVA-3D/checkpoints/llava-3d-7b \
#         --question-file playground/data/annotations/ScanQA_v1.0_val.json \
#         --answers-file ./llava-3d-7b-scanqa_answer_val.json \
#         --video-folder /mnt/disk4/chenyt/LLaVA-3D/playground/data/scanet_v2/OpenDataLab___ScanNet_v2/raw/scans \
#         --prune_layer_ratio 5:0.05 8:0.1 14:0.2 \
#         --device-map single

# Incomplete ScanNet / ScanQA: only scene0001_* … scene0060_* and skip scenes with no .../video/ yet:
# CUDA_VISIBLE_DEVICES=0 python -u llava/eval/model_scanqa.py \
#         --model-path /home/chenyt/LLaVA-3D/checkpoints/llava-3d-7b \
#         --question-file playground/data/annotations/ScanQA_v1.0_val.json \
#         --answers-file ./llava-3d-7b-scanqa_answer_val_scene60.json \
#         --video-folder /mnt/disk4/chenyt/LLaVA-3D/playground/data/scanet_v2/OpenDataLab___ScanNet_v2/raw/scans \
#         --prune_layer_ratio 5:0.05 8:0.1 14:0.2 \
#         --device-map single \
#         --max-scene-index 60 \
#         --skip-missing-video

# Run Large Scene
CUDA_VISIBLE_DEVICES=2 python -u llava/eval/model_scanqa.py \
        --model-path /home/chenyt/LLaVA-3D/checkpoints/llava-3d-7b \
        --question-file playground/data/annotations/ScanQA_v1.0_val.json \
        --answers-file ./llava-3d-7b-scanqa_answer_val_scene60.json \
        --video-folder /mnt/disk4/chenyt/LLaVA-3D/playground/data/scanet_v2/OpenDataLab___ScanNet_v2/raw/scans \
        --prune_layer_ratio 5:0.05 8:0.1 14:0.2 \
        --device-map single \
        --max-scene-index 60 \
        --skip-missing-video \
        --extra-scenes 2 \
        --extra-scene-translation 50 \
        --debug-scene-overlap \
        --debug-overlap-stride 16 \
        --debug-overlap-margin 0

# Multi-GPU sharding (`--device-map auto`) can hit device mismatches in the 3D tower unless all activations are moved per-layer; prefer a single large GPU or 8-bit loading instead.
# CUDA_VISIBLE_DEVICES=0,1 python -u llava/eval/model_scanqa.py ... --device-map auto