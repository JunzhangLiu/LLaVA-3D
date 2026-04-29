import argparse
import torch
import os
import json
import random
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_videos, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        num_clusters=args.num_clusters,
    )

    with open(args.question_file, 'r') as file:
        all_questions = json.load(file)

    # Optionally restrict to the first N unique scenes (for partial-dataset runs)
    if args.max_scene_index is not None:
        unique_scenes = sorted(set(q["video"] for q in all_questions))[:args.max_scene_index + 1]
        scene_set = set(unique_scenes)
        all_questions = [q for q in all_questions if q["video"] in scene_set]

    questions = get_chunk(all_questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        video_file = line["video"]
        video_path = os.path.join(args.video_folder, video_file)
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        try:
            videos_dict = process_videos(
                video_path,
                processor['video'],
                mode='random',
                device=model.device,
                text=cur_prompt
            )
        except Exception as e:
            if args.skip_missing_video:
                continue
            raise

        images_tensor    = videos_dict['images'].to(model.device, dtype=torch.bfloat16)
        depths_tensor    = videos_dict['depths'].to(model.device, dtype=torch.bfloat16)
        poses_tensor     = videos_dict['poses'].to(model.device, dtype=torch.bfloat16)
        intrinsics_tensor = videos_dict['intrinsics'].to(model.device, dtype=torch.bfloat16)

        # Concatenate extra scenes along the views dimension (large-scene stress test)
        if args.extra_scenes > 0:
            other_qs = [q for q in all_questions if q["video"] != video_file]
            sampled = random.sample(other_qs, min(args.extra_scenes, len(other_qs)))
            for i, extra_q in enumerate(sampled):
                extra_path = os.path.join(args.video_folder, extra_q["video"])
                try:
                    extra_dict = process_videos(extra_path, processor['video'], mode='random',
                                                device=model.device, text=cur_prompt)
                except Exception:
                    continue
                # Shift camera translations so the extra scene occupies a separate region
                extra_poses = extra_dict['poses'].to(model.device, dtype=torch.bfloat16)
                extra_poses[:, :, 0, 3] += (i + 1) * args.extra_scene_translation
                images_tensor     = torch.cat([images_tensor,     extra_dict['images'].to(model.device, dtype=torch.bfloat16)], dim=1)
                depths_tensor     = torch.cat([depths_tensor,     extra_dict['depths'].to(model.device, dtype=torch.bfloat16)], dim=1)
                poses_tensor      = torch.cat([poses_tensor,      extra_poses], dim=1)
                intrinsics_tensor = torch.cat([intrinsics_tensor, extra_dict['intrinsics'].to(model.device, dtype=torch.bfloat16)], dim=1)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                depths=depths_tensor,
                poses=poses_tensor,
                intrinsics=intrinsics_tensor,
                image_sizes=None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=512,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/llava3d-v1.5-7b-task-v3-tuning")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="playground/data/LLaVA-3D-Pretrain")
    parser.add_argument("--question-file", type=str, default="playground/data/annotations/llava3d_sqa3d_val_question.json")
    parser.add_argument("--answers-file", type=str, default="./llava3d_sqa3d_val_answer_pred.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    # Method 3: spatial k-means clustering
    parser.add_argument("--num-clusters", type=int, default=None,
                        help="Use k-means pooling with K clusters instead of voxel pooling")
    # Large-scene stress test
    parser.add_argument("--extra-scenes", type=int, default=0,
                        help="Number of additional scenes to concatenate per query")
    parser.add_argument("--extra-scene-translation", type=float, default=50.0,
                        help="X-axis offset (metres) applied to each extra scene's camera poses")
    # Dataset helpers
    parser.add_argument("--max-scene-index", type=int, default=None,
                        help="Only evaluate on the first N unique scenes (0-indexed)")
    parser.add_argument("--skip-missing-video", action="store_true",
                        help="Skip scenes whose video folder cannot be loaded")
    args = parser.parse_args()

    eval_model(args)
