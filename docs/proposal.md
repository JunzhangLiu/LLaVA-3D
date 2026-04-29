# Scaling LLaVA-3D for Large-Scale 3D Scene Understanding via Efficient Token Compression

COMP5422 Deep 2D & 3D Visual Scene Understanding — HKUST (Prof. Dan Xu)

**Team:** WU Yuyao · CHAN Nga Teng · LIU Junzhang · WONG Wei Ming

This repository is a fork of [LLaVA-3D](https://github.com/JunzhangLiu/LLaVA-3D) extended with token compression strategies and large-scale scene evaluation.

---

## Problem Statement

[LLaVA-3D](https://arxiv.org/abs/2409.18125) adapts the LLaVA multimodal architecture to 3D scenes by augmenting 2D visual tokens with 3D positional embeddings from depth maps, enabling a LLaMA backbone to reason about spatial structure and answer queries about 3D environments.

While effective on small indoor scenes, the architecture has fundamental scalability limitations:

1. **Token explosion** — large or multi-room scenes produce many 3D patches. Token count grows with the number of views and scene elements.
2. **Quadratic attention cost** — transformer-based LLMs have O(n²) complexity with sequence length, making long token sequences slow and memory-intensive.
3. **Context window limits** — the LLaMA backbone has a fixed context length. Scenes that exceed it require aggressive token reduction that may discard important spatial information.
4. **Untested scalability** — all original paper experiments use single small indoor rooms; performance on large or multi-room scenes is unknown.

**Goal:** Investigate and improve LLaVA-3D's scalability to large 3D scenes through token compression, while measuring the accuracy vs. efficiency tradeoff across multiple strategies.

---

## Methods to Optimise

The original LLaVA-3D architecture is kept frozen (no retraining of the vision encoder or LLM backbone). Compression is inserted as a lightweight add-on at different points in the pipeline.

### Overview

| # | Method | Type | Owner | Status |
|---|---|---|---|---|
| 1 | Attention-Score Pruning | Training-free | JunzhangLiu | Done (partial eval) |
| 2 | ForestPrune | Training-free | — | Exploring |
| 3 | Spatial Clustering / Superpoint Aggregation | Training-free | — | Not started |
| 4 | Hierarchical Scene Representations | Architectural | — | Not started |
| 5 | Q-Former / Learned Token Compression | Trained | WU Yuyao | In progress |

---

### Method 1 — Attention-Score Based Pruning

**Simple idea:** When the LLM processes visual tokens, not all of them get equal attention — some patches (e.g. a blank wall) barely affect the output. Method 1 watches which tokens receive low attention in early layers and simply drops them before they reach later layers. Fewer tokens in → faster inference, less memory.

**How it works:**
1. During the LLM forward pass, at certain early layers (5, 8, 14), collect the attention scores for all visual tokens.
2. Drop the bottom X% of tokens by score at each of those layers (5%, 10%, 20% respectively).
3. The remaining tokens continue through the rest of the network as normal.

**Why no training is needed:** The attention scores come directly from the model's own weights — we're just using the model's own opinion of what's important.

- Implemented in `LLaVA-3D/llava/model/language_model/llava_llama.py`
- Activated via `--prune_layer_ratio 5:0.05 8:0.1 14:0.2` flag on eval scripts

**Results on ScanQA val (first 60 scenes, single scene):**

| Method | Raw EM@1 | Refined EM@1 | Speed (scenes/s) |
|---|---|---|---|
| Baseline (original LLaVA-3D) | 28.37 | 44.65 | 1.52 |
| Attn prune (5:5%, 8:10%, 14:20%) | 28.74 | 44.89 | 2.18 |

~43% speedup with no accuracy loss.

> **Known bug:** `--prune_layer_ratio` is currently silently ignored in the large-scene eval because the flag is passed to HuggingFace `from_pretrained()` but never forwarded to the model. Fix: explicitly set `model.model.prune_ratio` after loading in `builder.py`.

---

### Method 2 — ForestPrune

**Simple idea:** Instead of looking at attention scores like Method 1, ForestPrune builds a tree-like structure over the tokens and decides which branches (groups of tokens) are redundant enough to cut. Tokens that cover similar regions or carry similar information get pruned together.

**Why it's interesting:** It's also training-free (plug in, no retraining needed), but the pruning decision is more structured than just looking at a single attention score — it considers the relationship between tokens, not just each token individually.

**In practice:** You'd wrap the visual token sequence with the ForestPrune module before it enters the LLM. It outputs a shorter sequence with the unimportant tokens removed.

- Reference: https://github.com/luminousllsa/ForestPrune
- Found via: [Awesome-Token-Compress](https://github.com/daixiangzi/Awesome-Token-Compress) survey (shared in WeChat April 11)
- No retraining required.
- Status: to be integrated and benchmarked.

---

### Method 3 — Spatial Clustering / Superpoint Aggregation

**Simple idea:** Imagine the 3D scene as a point cloud. Many points are part of the same flat wall, floor, or table — they look almost identical and carry redundant information. Instead of giving the LLM a separate token for every single patch, group nearby patches together and send one averaged token per group.

**How it works:**
1. Take all the 3D patch positions (x, y, z coordinates in the scene).
2. Cluster them by proximity — e.g. k-means or voxel grid (divide space into a regular grid, one cell = one cluster).
3. Average the visual features of all patches in the same cluster into a single token.
4. Feed the LLM these cluster tokens instead of the original patches.

**Why it helps:** A large scene might have 3000+ patches, but most of the scene is flat surfaces. Clustering might reduce this to 200–300 meaningful regions without losing the objects the question is about.

**No training required** for the clustering step — it's pure geometry. Optionally, a small learnable aggregator can be added to do smarter averaging.

- Status: not started.

---

### Method 4 — Hierarchical Scene Representations

**Simple idea:** Think of how Google Maps works — you zoom out to see the whole city (coarse), then zoom into your street (fine). Apply the same idea to the scene tokens: have two levels of detail, and let the model decide which level to focus on based on the question.

**How it works:**
1. Build a "coarse" set of tokens — one token per large region of the scene (e.g. left side of room, right side, ceiling, floor). These are like the zoomed-out view.
2. Keep a "fine" set of tokens — the original per-patch tokens for areas of interest.
3. The LLM first attends to coarse tokens to understand the overall layout, then selectively looks at fine tokens for the regions relevant to the question.

**Why it helps:** For a question like "what is on the shelf?", the model can ignore the floor and ceiling entirely at the coarse level, then zoom into just the shelf area using fine tokens. This avoids processing every patch at full resolution.

**Difference from Method 3:** Method 3 always collapses everything into one fixed level. Method 4 keeps two levels and attends to both — more flexible, but also more complex to implement.

- May be combined with Method 3 (coarse level = clusters, fine level = original patches).
- Requires changes to the pooling / projection stage of LLaVA-3D.
- Status: not started.

---

### Method 5 — Q-Former / Learned Token Compression

**Simple idea:** Train a small "summariser" module that reads all the visual tokens and produces a fixed-size summary — say, always 64 tokens regardless of how big the scene is. The LLM only ever sees these 64 summary tokens, never the raw thousands of patches.

**How it works:**
1. Start with N visual tokens from the scene (could be 3000+ for a large scene).
2. Feed them through a Q-Former: a small transformer with M learnable "query" vectors. Each query attends over all N tokens and pulls out the information it needs.
3. Output: exactly M tokens (e.g. 64), one per query vector.
4. These M tokens go into the LLM instead of the original N.

**Why it's powerful:** Unlike pruning (which just drops tokens), the Q-Former can mix information from multiple tokens into one output token. It learns what to keep and what to compress based on the training data — so it gets smarter over time.

**The tradeoff:** It needs training. You need to fine-tune the Q-Former on 3D vision-language data so it learns to preserve the information the LLM actually needs for answering questions.

**Reference:** Inspired by BLIP-2's Q-Former architecture, widely used in vision-language models.

- Output size M is fixed → inference cost is always predictable, regardless of scene size.
- Requires training on 3D vision-language data (e.g. ScanQA / SQA3D).
- Status: in progress (WU Yuyao).

---

## Large Scene Stress Test

This is an **evaluation setup**, not a compression method. It synthesises large scenes to stress-test the model and measure how well compression methods hold up beyond single-room inputs.

**Implementation** (`LLaVA-3D/llava/eval/model_scanqa.py`):
- For each question, keep the original `(scene_id, question)` as the primary scene.
- Randomly sample 2 additional scenes from the available pool.
- Apply a translation offset of `j × 50m` along the X-axis to each extra scene's poses, keeping each scene internally consistent while separating their point clouds in world space (prevents voxel pooling from merging them).
- Concatenate all frames, depth maps, poses, and intrinsics into one long sequence.
- Feed to the model as a single "large scene".

**Results (original model, no compression, first 60 scenes, ScanQA):**

| Setup | Raw EM@1 | Refined EM@1 |
|---|---|---|
| Single scene | 28.47 | 46.17 |
| 3 scenes concatenated | 26.70 | 43.07 |

The ~1.8 / 3.1 point drop is pure baseline degradation from more tokens. The key next experiment is running compression methods on this setup to see how much of the drop they recover.

---

## Datasets

### ScanQA

3D question answering grounded in ScanNet RGB-D scenes. Questions ask about objects, attributes, and spatial relationships within a single room.

- Val set: ~4,675 questions
- Question file (on server): `playground/data/annotations/ScanQA_v1.0_val.json`
- Answer output: `llava-3d-7b-scanqa_answer_val.json` (pushed to `main` branch)
- Eval script: `LLaVA-3D/llava/eval/model_scanqa.py`
- Metric: EM@1 (raw string match + refined with text normalisation)
- Paper baselines: Voxel-0.2 → 27.0 | FPS-1024 → 26.3

### SQA3D

Situated 3D Question Answering. Questions are grounded in an embodied viewpoint within the scene (position + orientation given), making spatial reasoning more explicit.

- Test set: 3,519 questions
- Raw files (in repo): `LLaVA-3D/SQA3D/LLM/v1_balanced_questions_test_scannetv2.json` and `v1_balanced_sqa_annotations_test_scannetv2.json`
- Eval script: `LLaVA-3D/llava/eval/model_sqa3d.py`
- Needs: a reformatted `llava3d_sqa3d_val_question.json` merging both files and filtered to available scenes
- Metric: EM@1
- Paper baselines: Voxel-0.2 → 55.6 | FPS-1024 → 55.2

---

## Evaluation Metrics

| Category | Metric | How Measured |
|---|---|---|
| Accuracy | ScanQA EM@1 (raw + refined) | Built-in eval scripts |
| Accuracy | SQA3D EM@1 | Built-in eval scripts |
| Efficiency | Token count per scene | Log `len(visual_tokens)` per forward pass |
| Efficiency | Inference time (s/scene) | `time.time()` around `model.generate()` |
| Efficiency | GPU memory (GB) | `torch.cuda.max_memory_allocated()` |

---

## Results Table

| Method | Token count | Time (s/scene) | GPU mem (GB) | ScanQA EM@1 | SQA3D EM@1 |
|---|---|---|---|---|---|
| Baseline — Voxel 0.2 (paper) | ~3096 dynamic | ~0.2s | — | 27.0 | 55.6 |
| Baseline — FPS 1024 (paper) | 1024 fixed | ~0.2s | — | 26.3 | 55.2 |
| Baseline — our repro (single scene) | TBD | TBD | TBD | 28.47 | TBD |
| Baseline — large scene (3 scenes) | TBD | TBD | TBD | 26.70 | TBD |
| Method 1 — Attn pruning (single scene) | TBD | ~0.46s | TBD | 28.74 | TBD |
| Method 1 — Attn pruning (large scene) | TBD | TBD | TBD | TBD | TBD |
| Method 2 — ForestPrune | TBD | TBD | TBD | TBD | TBD |
| Method 3 — Spatial clustering | TBD | TBD | TBD | TBD | TBD |
| Method 5 — Q-Former | TBD | TBD | TBD | TBD | TBD |

---

## To-Do

- [ ] Fix pruning bug in `builder.py` so `--prune_layer_ratio` actually takes effect
- [ ] Run Method 1 (attn pruning) + large scene combined — **the core missing result**
- [ ] Add token count + GPU memory logging to eval scripts
- [ ] Create `llava3d_sqa3d_val_question.json` from `SQA3D/LLM/` data, filtered to available scenes
- [ ] Run SQA3D eval for all methods
- [ ] Run ablation on pruning ratios (vary layer/ratio combos, plot accuracy vs. speed)
- [ ] Integrate and benchmark ForestPrune
- [ ] Implement Q-Former compression module
- [ ] Explore spatial clustering approach

---

## Setup

```bash
git clone --recurse-submodules git@github.com:Alvin0523/comp5422_3d.git
cd comp5422_3d/LLaVA-3D
pixi install
```

Model weights load from HuggingFace at runtime (`ChaimZhu/LLaVA-3D-7B`). ScanNet scene data (RGB-D frames + poses) must be downloaded separately — see `download_data.py` and `download_demo.sh`.

### Run ScanQA eval (baseline)

```bash
python llava/eval/model_scanqa.py \
  --model-path ChaimZhu/LLaVA-3D-7B \
  --video-folder /path/to/scannet \
  --question-file playground/data/annotations/ScanQA_v1.0_val.json \
  --answers-file ./scanqa_answers.json \
  --max-scene-index 59 \
  --skip-missing-video
```

### Run with attention pruning

```bash
python llava/eval/model_scanqa.py \
  ... \
  --prune_layer_ratio 5:0.05 8:0.1 14:0.2
```

### Run large scene eval

```bash
python llava/eval/model_scanqa.py \
  ... \
  --extra-scenes 2 \
  --extra-scene-translation 50
```

### Run large scene + pruning combined

```bash
python llava/eval/model_scanqa.py \
  ... \
  --extra-scenes 2 \
  --extra-scene-translation 50 \
  --prune_layer_ratio 5:0.05 8:0.1 14:0.2
```

---

## Project Timeline

| Week | Task | Owner |
|---|---|---|
| 1–2 | Reproduce baseline; set up data pipeline; record baseline token counts and inference time | All |
| 3–4 | Implement Method 1 (attn pruning) + Method 5 (Q-Former) | Split |
| 3–4 | Large scene stress test implementation | Split |
| 5 | Combine strategies; run ablation studies; compare efficiency vs. accuracy tradeoffs | All |
| 6 | Final presentation and report | All |

---

## Repository Structure

```
comp5422_3d/
├── recipe.md                              # This file
└── LLaVA-3D/                             # Fork of ChaimZhu/LLaVA-3D
    ├── llava/
    │   ├── eval/
    │   │   ├── model_scanqa.py            # ScanQA eval (+ large scene + pruning flags)
    │   │   ├── model_sqa3d.py             # SQA3D eval
    │   │   └── scanqa_evaluator.py        # EM@1 scoring
    │   └── model/
    │       ├── builder.py                 # Model loading (pruning bug fix needed here)
    │       └── language_model/
    │           └── llava_llama.py         # Attention pruning implementation
    ├── SQA3D/                             # SQA3D benchmark submodule
    │   └── LLM/
    │       ├── v1_balanced_questions_test_scannetv2.json
    │       └── v1_balanced_sqa_annotations_test_scannetv2.json
    ├── inference.sh                       # Example eval commands
    └── download_data.py                   # ScanNet download helper
```
