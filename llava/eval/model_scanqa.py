import argparse
import re
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import random

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.eval.scanqa_text_utils import answer_match, clean_answer
from llava.model.multimodal_encoder.unproject import unproject
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


_SCENENET_ID_RE = re.compile(r"^scene(\d{4})_")


def scannet_scene_index(scene_id: str):
    m = _SCENENET_ID_RE.match(scene_id or "")
    return int(m.group(1)) if m else None


def resolve_video_root(video_folder: str, scene_id: str):
    """Return scene root if RGB-D video exists under raw ScanNet or legacy layout."""
    vf = os.path.expanduser(video_folder)
    direct = os.path.join(vf, scene_id)
    legacy = os.path.join(vf, "scannet", scene_id)
    for root in (direct, legacy):
        vdir = os.path.join(root, "video")
        if os.path.isdir(vdir) and os.listdir(vdir):
            return root
    return None


def filter_questions(questions, args):
    """Optionally restrict to a max ScanNet scene index and/or scenes with extracted video."""
    out = questions
    if args.max_scene_index is not None:
        kept = []
        for q in out:
            idx = scannet_scene_index(q.get("scene_id", ""))
            if idx is None or idx > args.max_scene_index:
                continue
            kept.append(q)
        out = kept
    if args.skip_missing_video:
        vf = args.video_folder
        kept = []
        for q in out:
            if resolve_video_root(vf, q.get("scene_id", "")) is not None:
                kept.append(q)
        out = kept
    return out


def _scene_xyz_aabb(depths_vhw: torch.Tensor, poses_v44: torch.Tensor, intr_v44: torch.Tensor, stride: int):
    """Compute a coarse AABB (min/max xyz) in world coords for one scene.

    Inputs are per-scene tensors:
      - depths_vhw: (V, H, W)
      - poses_v44: (V, 4, 4)
      - intr_v44: (V, 4, 4) or (4, 4)
    """
    if intr_v44.dim() == 2:
        intr_v44 = intr_v44.unsqueeze(0).repeat(depths_vhw.shape[0], 1, 1)
    # Add batch dim
    depths = depths_vhw[None, ...]
    poses = poses_v44[None, ...]
    intr = intr_v44[None, ...]
    xyz = unproject(intr, poses, depths)  # (1, V, H, W, 3)
    if stride and stride > 1:
        xyz = xyz[:, :, ::stride, ::stride, :]
        depths = depths[:, :, ::stride, ::stride]
    valid = depths > 0
    if valid.any():
        # xyz is (B, V, H, W, 3) and valid is (B, V, H, W).
        # Boolean indexing with (B,V,H,W) mask yields (N, 3) points.
        pts = xyz[valid]
        mn = pts.min(dim=0).values
        mx = pts.max(dim=0).values
    else:
        # Degenerate: no valid depth
        mn = torch.tensor([float("inf")] * 3, device=xyz.device, dtype=xyz.dtype)
        mx = torch.tensor([float("-inf")] * 3, device=xyz.device, dtype=xyz.dtype)
    return mn, mx


def _aabb_overlap(mn1, mx1, mn2, mx2, margin: float):
    """Return True if 3D AABBs overlap (with optional margin)."""
    if margin and margin > 0:
        mn1 = mn1 - margin
        mx1 = mx1 + margin
        mn2 = mn2 - margin
        mx2 = mx2 + margin
    overlap = (mn1 <= mx2).all() and (mn2 <= mx1).all()
    if not overlap:
        return False
    # Require overlap on all 3 axes (positive intersection)
    inter_min = torch.maximum(mn1, mn2)
    inter_max = torch.minimum(mx1, mx2)
    return (inter_max > inter_min).all().item()


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # device_map="auto" can leave custom submodules (e.g. video_tower) on meta; .to() then fails.
    device_map = "auto" if args.device_map == "auto" else {"": 0}
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device_map=device_map,
        prune_ratio=args.prune_layer_ratio,
    )

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    with open(args.question_file, 'r') as file:
        questions = json.load(file)
    n_before = len(questions)
    questions = filter_questions(questions, args)
    if args.max_scene_index is not None or args.skip_missing_video:
        print(f"ScanQA questions after filter: {len(questions)} (was {n_before})", flush=True)
    if len(questions) == 0:
        raise SystemExit(
            "No questions left after filtering (--max-scene-index / --skip-missing-video). "
            "Check paths and filters."
        )

    # Build a pool of candidate scenes (post-filter) for optional multi-scene mixing.
    # We sample extra scenes from this pool and concatenate their sampled frames.
    rng = random.Random(args.extra_scene_seed)
    scene_pool = []
    if args.extra_scenes > 0:
        seen = set()
        vf = os.path.expanduser(args.video_folder)
        for q in questions:
            sid = q.get("scene_id", "")
            if sid in seen:
                continue
            root = resolve_video_root(vf, sid)
            if root is None:
                continue
            seen.add(sid)
            scene_pool.append((sid, root))
        if len(scene_pool) <= 1:
            raise SystemExit(
                f"--extra-scenes {args.extra_scenes} requested, but scene pool has only {len(scene_pool)} scene(s) "
                "after filtering. Increase --max-scene-index or disable --skip-missing-video."
            )

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    acc =refined_acc =current=0
    pbar = tqdm(questions)
    for line in pbar:
        idx = line["question_id"]
        scene_id = line["scene_id"]
        vf = os.path.expanduser(args.video_folder)
        # Raw ScanNet layout: <video-folder>/<scene_id>/video/0.jpg
        # Legacy / pretrain layout: <video-folder>/scannet/<scene_id>/...
        video_path = resolve_video_root(vf, scene_id)
        if video_path is None:
            raise FileNotFoundError(
                f"No non-empty video/ under {os.path.join(vf, scene_id)} or "
                f"{os.path.join(vf, 'scannet', scene_id)} for scene_id={scene_id}. "
                "Extract .sens or use --skip-missing-video when filtering the question list."
            )
        qs = line["question"]
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

        # Optional: mix in extra random scenes by concatenating their sampled frames.
        # IMPORTANT: Poses/intrinsics are used for backprojecting depth to 3D world coords.
        # If we concatenate multiple unrelated scenes, we must prevent their coordinate frames
        # from colliding in a single global space. We do this by translating extra scenes far away.
        scene_roots = [(scene_id, video_path)]
        if args.extra_scenes > 0:
            candidates = [(sid, root) for (sid, root) in scene_pool if sid != scene_id]
            if len(candidates) <= args.extra_scenes:
                extra = candidates
            else:
                extra = rng.sample(candidates, k=args.extra_scenes)
            scene_roots.extend(extra)

        video_infos = []
        for j, (sid, root) in enumerate(scene_roots):
            vi = processor["video"].preprocess(
                root,
                return_tensors="pt",
                mode="random",
                device=model.device,
                text=cur_prompt,
            )
            # Separate each additional scene by a large translation in "world" coordinates.
            # This keeps each scene internally consistent while avoiding mixed-scene overlap
            # during voxelization / pooling.
            if j > 0 and args.extra_scene_translation and args.extra_scene_translation > 0:
                T = torch.eye(4, dtype=vi["poses"].dtype, device=vi["poses"].device)
                T[0, 3] = float(j) * float(args.extra_scene_translation)
                # vi["poses"] is [V, 4, 4] (one world-from-camera pose per view).
                # Left-multiply by T for each view to translate the whole scene.
                vi["poses"] = (T.unsqueeze(0) @ vi["poses"])
            video_infos.append(vi)

        if args.debug_scene_overlap and len(video_infos) > 1:
            # Compute coarse world-coordinate AABBs per scene and raise if any overlap.
            boxes = []
            for j, vi in enumerate(video_infos):
                mn, mx = _scene_xyz_aabb(
                    vi["depth_images"],
                    vi["poses"],
                    vi["intrinsic"],
                    stride=max(int(args.debug_overlap_stride), 1),
                )
                boxes.append((j, mn, mx))
            for a in range(len(boxes)):
                for b in range(a + 1, len(boxes)):
                    ja, mna, mxa = boxes[a]
                    jb, mnb, mxb = boxes[b]
                    if _aabb_overlap(mna, mxa, mnb, mxb, margin=float(args.debug_overlap_margin)):
                        raise RuntimeError(
                            "Detected overlapping 3D AABBs after multi-scene concatenation. "
                            f"Pair indices=({ja},{jb}) scene_ids=({scene_roots[ja][0]},{scene_roots[jb][0]}). "
                            "Increase --extra-scene-translation or set --debug-overlap-margin."
                        )

        images_cat = torch.cat([vi["images"] for vi in video_infos], dim=0).unsqueeze(0)
        depths_cat = torch.cat([vi["depth_images"] for vi in video_infos], dim=0).unsqueeze(0)
        poses_cat = torch.cat([vi["poses"] for vi in video_infos], dim=0).unsqueeze(0)
        intr_cat = torch.cat([vi["intrinsic"] for vi in video_infos], dim=0).unsqueeze(0)

        images_tensor = images_cat.to(model.device, dtype=torch.bfloat16)
        depths_tensor = depths_cat.to(model.device, dtype=torch.bfloat16)
        poses_tensor = poses_cat.to(model.device, dtype=torch.bfloat16)
        intrinsics_tensor = intr_cat.to(model.device, dtype=torch.bfloat16)

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
        pred_answer = clean_answer(outputs)
        ref_captions = [clean_answer(gt_answer) for gt_answer in line['answers']]
        tmp_acc, tmp_refined_acc = answer_match(pred_answer, ref_captions)
        current+=1
        acc += tmp_acc
        refined_acc += tmp_refined_acc


        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
        pbar.set_description(f"acc {acc*100/current:.2f} refined {refined_acc*100/current:.2f}")
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
    parser.add_argument("--prune_layer_ratio", nargs="+", default=[])
    parser.add_argument(
        "--device-map",
        type=str,
        default="single",
        choices=["auto", "single"],
        help="Use 'single' (default) to load the full model on cuda:0 and avoid meta-tensor errors on video_tower; use 'auto' for HF multi-GPU sharding.",
    )
    parser.add_argument(
        "--max-scene-index",
        type=int,
        default=None,
        help="If set, only run questions whose scene_id matches sceneNNNN_* with NNNN <= this "
        "value (e.g. 60 keeps scene0001_00 … scene0060_01, drops scene0061_00).",
    )
    parser.add_argument(
        "--skip-missing-video",
        action="store_true",
        help="After other filters, drop questions whose scene has no non-empty .../video/ folder "
        "(partial ScanNet extract / incomplete download).",
    )
    parser.add_argument(
        "--extra-scenes",
        type=int,
        default=0,
        help="If >0, for each question concatenate frames from this many additional random scenes "
        "sampled from the filtered scene pool (post --max-scene-index / --skip-missing-video).",
    )
    parser.add_argument(
        "--extra-scene-seed",
        type=int,
        default=0,
        help="RNG seed for choosing extra random scenes when --extra-scenes > 0.",
    )
    parser.add_argument(
        "--extra-scene-translation",
        type=float,
        default=50.0,
        help="Translate each extra scene by N meters along +X in the pose world frame before concatenation. "
        "This prevents unrelated scenes from sharing the same global voxel grid when poses are used for 3D pooling. "
        "Set 0 to disable (not recommended if --extra-scenes > 0).",
    )
    parser.add_argument(
        "--debug-scene-overlap",
        action="store_true",
        help="If set and --extra-scenes > 0, compute coarse 3D bounding boxes per scene from depth+poses and "
        "raise an error if any boxes overlap (indicates mixed-scene coordinate collision).",
    )
    parser.add_argument(
        "--debug-overlap-stride",
        type=int,
        default=16,
        help="Stride for subsampling pixels when computing overlap AABBs (larger is faster, less accurate).",
    )
    parser.add_argument(
        "--debug-overlap-margin",
        type=float,
        default=0.0,
        help="Expand each AABB by this margin (meters) before checking overlap.",
    )
    args = parser.parse_args()

    result = {}
    for pair in args.prune_layer_ratio:
        k, v = pair.split(":")
        result[int(k)] = float(v)
    args.prune_layer_ratio = result
    eval_model(args)
