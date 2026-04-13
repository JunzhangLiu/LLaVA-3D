# CUDA_VISIBLE_DEVICES=0 python llava/eval/run_llava_3d.py \
#     --model-path ChaimZhu/LLaVA-3D-7B \
#     --video-path ./demo/scannet/scene0356_00 \
#     --query "Is there a red bucket in the scene?"


CUDA_VISIBLE_DEVICES=1 python llava/eval/model_scanqa.py \
        --model-path ChaimZhu/LLaVA-3D-7B \
        --question-file playground/data/annotations/ScanQA_v1.0_val.json \
        --answers-file ./llava-3d-7b-scanqa_answer_val.json \
        --video-folder playground/data/scanet_v2/OpenDataLab___ScanNet_v2/raw/scans\
        # --prune_layer_ratio 5:0.05 8:0.1 14:0.2 