# Method 3: Spatial K-Means Clustering for Token Compression in LLaVA-3D

**Author:** WONG Wei Ming  
**Course:** COMP5422 Deep 2D & 3D Visual Scene Understanding — HKUST (Prof. Dan Xu)  
**Team:** WU Yuyao · CHAN Nga Teng · LIU Junzhang · WONG Wei Ming  
**Branch:** `feat/pixi`

---

## 1. Background & Motivation

### What is LLaVA-3D?

LLaVA-3D is a multimodal LLM that understands 3D scenes. It takes RGB-D video frames of a room, extracts visual patches, projects them into 3D world coordinates using depth + camera poses, and feeds the resulting tokens into a LLaMA language model to answer questions about the scene.

### The Problem: Token Explosion

The original model uses **voxel pooling** to compress 3D patches into tokens before the LLM. A fixed 0.2m voxel grid divides the scene into cells — each occupied cell becomes one token.

This has a critical flaw for large scenes:

- A small single room → ~3096 tokens
- 3 rooms concatenated → ~9000+ tokens
- Token count is **unpredictable and uncontrollable**
- LLMs have quadratic attention cost — more tokens = exponentially slower + more memory
- Too many tokens overloads the context window

**Evidence:** When Nga Teng ran the large-scene stress test (3 scenes concatenated), accuracy dropped from **28.47% → 26.70% EM@1** just from the extra tokens overwhelming the model.

### Why Voxel Pooling Fails at Scale

| Problem | Voxel Pooling Behaviour |
|---------|------------------------|
| Token count control | None — depends entirely on scene geometry |
| Large scene | Token count explodes unpredictably |
| Tuning | Changing grid size has unclear effect on accuracy |
| Ceiling | No hard limit — can exceed LLM context window |

---

## 2. Method 3: K-Means Spatial Clustering

### Core Idea

Replace the voxel grid with **k-means clustering on 3D positions**. Instead of dividing space into fixed-size cells, k-means finds K natural spatial groups in the point cloud — and we always get exactly K tokens out, regardless of scene size.

```
RGB-D frames
    ↓
3D patch positions (feat_xyz)  +  visual features
    ↓
K-Means clustering on XYZ positions
    ↓
Average features within each cluster  →  K tokens (fixed)
    ↓
LLM
```

### Why K-Means is Better Than Voxel Pooling

| | Voxel Pooling (original) | K-Means (Method 3) |
|---|---|---|
| Token count | Variable, unpredictable (~3096) | Always exactly K |
| Large scene | Token count explodes | Stable — always K |
| Control | Change grid size (unclear effect) | Change K directly |
| Training required | No | No |
| Memory guarantee | None | Hard upper bound |

### K-Means++ Initialisation

Standard k-means can get stuck in bad local minima depending on where initial cluster centres are placed. We use **k-means++** initialisation which spreads the initial seeds across the 3D space proportional to distance, giving better coverage of the scene:

1. Pick first centroid randomly from all points
2. For each remaining centroid: sample a new point with probability proportional to its squared distance from the nearest existing centroid
3. This ensures seeds are spread across the scene, not all bunched in one corner

---

## 3. Code Implementation

Three files were changed. No new dependencies — `torch_scatter` was already imported.

### File 1: `llava/model/multimodal_encoder/video_encoder.py`

Added the `kmeans_cluster()` function and wired it into the forward pass.

**New function `kmeans_cluster()`:**
```python
def kmeans_cluster(features, xyz, K, num_iters=10):
    """
    K-means++ spatial clustering on 3D positions; aggregates features per cluster.
    Args:
        features: (B, N, F) — per-patch visual features
        xyz:      (B, V, H, W, 3) — 3D patch positions
        K:        number of clusters (= output tokens per scene)
    Returns:
        pooled:       (B*K, F) — cluster-averaged features
        batch_offset: (B,)  int32 — cumulative token counts (each = K)
    """
    B, N, F = features.shape
    xyz_flat = xyz.reshape(B, -1, 3)  # (B, N, 3)
    pooled_list = []
    for b in range(B):
        pts, feat = xyz_flat[b], features[b]
        # K-means++ init
        centroids = pts[torch.randint(N, (1,), device=pts.device)]
        for _ in range(K - 1):
            d2 = torch.cdist(pts, centroids).min(dim=1).values ** 2
            centroids = torch.cat([centroids, pts[torch.multinomial(d2/d2.sum(), 1)]], dim=0)
        # Lloyd iterations
        for _ in range(num_iters):
            assign = torch.cdist(pts, centroids).argmin(dim=1)
            centroids = scatter_mean(pts, assign, dim=0, dim_size=K)
        assign = torch.cdist(pts, centroids).argmin(dim=1)
        pooled_list.append(scatter_mean(feat, assign, dim=0, dim_size=K))
    pooled = torch.cat(pooled_list, dim=0)
    batch_offset = torch.arange(1, B+1, device=features.device, dtype=torch.int32) * K
    return pooled, batch_offset
```

**Added `self.num_clusters = 512` to `RGBDVideoTower.__init__()`** — default K value.

**Added kmeans branch in `forward()`:**
```python
elif self.pooling == 'kmeans':
    pooled_video_features, batch_offset = kmeans_cluster(
        video_features, feat_xyz, self.num_clusters
    )
```

### File 2: `llava/model/builder.py`

**Bug fix + new feature.** The original code passed all kwargs directly into HuggingFace `from_pretrained()`, which silently ignores unknown arguments. This meant `--prune_layer_ratio` (Method 1) and any future `--num-clusters` flag would be **silently ignored**.

Fix: extract custom args before passing to HuggingFace, apply them to the model after loading.

```python
def load_pretrained_model(..., **kwargs):
    # Extract before HuggingFace swallows them
    num_clusters = kwargs.pop('num_clusters', None)
    prune_ratio   = kwargs.pop('prune_ratio', None)

    # ... normal model loading ...

    # Apply after model + video tower are loaded
    if num_clusters is not None:
        video_tower.num_clusters = num_clusters
        video_tower.pooling = 'kmeans'
    if prune_ratio is not None:
        model.prune_ratio = prune_ratio
```

### File 3: `llava/eval/model_scanqa.py`

Added CLI flags to control Method 3 and large-scene experiments:

```python
# Method 3
parser.add_argument("--num-clusters", type=int, default=None)

# Large-scene stress test
parser.add_argument("--extra-scenes", type=int, default=0)
parser.add_argument("--extra-scene-translation", type=float, default=50.0)

# Dataset helpers
parser.add_argument("--max-scene-index", type=int, default=None)
parser.add_argument("--skip-missing-video", action="store_true")
```

Extra-scene logic: when `--extra-scenes 2` is passed, randomly samples 2 additional scenes, offsets their camera poses by `(i+1) × 50m` along the X-axis, and concatenates all frames into one large scene before inference.

---

## 4. Evaluation Metric

**EM@1 (Exact Match @ 1)** — the model answers a question (e.g. *"What colour is the chair?"*) and we check if the output matches the ground truth answer exactly after text normalisation.

- Score 1 if the model's answer matches any accepted answer
- Score 0 otherwise
- EM@1 = average across all questions × 100%

Two variants:
- **Raw EM@1** — strict: output must exactly equal an accepted answer after lowercasing + stripping punctuation
- **Refined EM@1** — looser: also counts if the output is a substring of an accepted answer, or vice versa

**Dataset:** ScanQA val set — 4,675 questions about ScanNet 3D scenes.

---

## 5. Results

### Local Smoke Test (3 demo scenes, 84 questions)

Run locally on RTX 5080. Small sample — used to verify code correctness, not for final numbers.

| Method | Tokens to LLM | Raw EM@1 | Refined EM@1 |
|--------|--------------|----------|--------------|
| Baseline — Voxel pooling | ~3096 (variable) | 35.71% (30/84) | 50.00% (42/84) |
| Method 3 — K-Means K=512 | **512 (fixed)** | 34.52% (29/84) | 48.81% (41/84) |
| **Difference** | **6× fewer tokens** | **−1.19%** | **−1.19%** |

**Key takeaway:** 83% reduction in tokens with only ~1 point accuracy drop. The clustering is working.

### Reference Numbers (JZ's full run, ~1864 questions, main branch)

| Setup | Raw EM@1 | Refined EM@1 |
|-------|----------|--------------|
| Baseline — single scene | 27.20% | 45.17% |
| Baseline — 3 scenes (NT's stress test) | 26.70% | 43.07% |
| Paper (Voxel-0.2) | 27.0% | — |

### Pending: Full Ablation on ASPIRE2A (60 scenes, ~1865 questions)

| Method | Tokens | Raw EM@1 | Refined EM@1 |
|--------|--------|----------|--------------|
| Baseline (voxel) | ~3096 | 28.47% (ref) | 46.17% (ref) |
| K-means K=1024 | 1024 | TBD | TBD |
| K-means K=512 | 512 | TBD | TBD |
| K-means K=256 | 256 | TBD | TBD |
| K-means K=128 | 128 | TBD | TBD |
| K-means K=512 + large scene (3 scenes) | 512 | TBD | TBD |
| Baseline + large scene | ~9288 | 26.70% (ref) | 43.07% (ref) |

The **core result** is the last two rows: does K=512 on a large scene outperform voxel pooling on a large scene? If yes → Method 3 solves the scalability problem.

---

## 6. How to Run

### Prerequisites
```bash
cd LLaVA-3D
pixi shell   # activate environment
```

### Baseline (voxel pooling — original)
```bash
python llava/eval/model_scanqa.py \
  --model-path ChaimZhu/LLaVA-3D-7B \
  --video-folder /path/to/scannet \
  --question-file playground/data/annotations/ScanQA_v1.0_val.json \
  --answers-file results/scanqa_baseline.json \
  --max-scene-index 59 --skip-missing-video
```

### Method 3 — K-Means K=512
```bash
python llava/eval/model_scanqa.py \
  --model-path ChaimZhu/LLaVA-3D-7B \
  --video-folder /path/to/scannet \
  --question-file playground/data/annotations/ScanQA_v1.0_val.json \
  --answers-file results/scanqa_kmeans512.json \
  --max-scene-index 59 --skip-missing-video \
  --num-clusters 512
```

### K-Means Ablation (loop over K values)
```bash
for K in 1024 512 256 128; do
  python llava/eval/model_scanqa.py \
    --model-path ChaimZhu/LLaVA-3D-7B \
    --video-folder /path/to/scannet \
    --question-file playground/data/annotations/ScanQA_v1.0_val.json \
    --answers-file results/scanqa_kmeans${K}.json \
    --max-scene-index 59 --skip-missing-video \
    --num-clusters $K
done
```

### K-Means + Large Scene (core experiment)
```bash
python llava/eval/model_scanqa.py \
  --model-path ChaimZhu/LLaVA-3D-7B \
  --video-folder /path/to/scannet \
  --question-file playground/data/annotations/ScanQA_v1.0_val.json \
  --answers-file results/scanqa_kmeans512_largescene.json \
  --max-scene-index 59 --skip-missing-video \
  --num-clusters 512 \
  --extra-scenes 2 --extra-scene-translation 50
```

### Score Any Results File
```bash
python work/score.py results/scanqa_kmeans512.json
```

---

## 7. How Method 3 Fits in the Overall Project

```
Project goal: make LLaVA-3D work on large 3D scenes

JunzhangLiu  → Method 1: Attention pruning     (drop low-attention tokens inside LLM)
Wong Wei Ming → Method 3: Spatial k-means      (cluster tokens before LLM)   ← this
Nga Teng      → Large scene stress test        (concatenate 3 scenes, measure degradation)
WU Yuyao      → Method 5: Q-Former            (trained token compression)

Method 3 + Nga Teng's large scene setup = the main combined experiment
```

Methods 1 and 3 address the problem from different angles:
- **Method 1** prunes tokens *inside* the LLM based on attention scores — works well for single scenes, incompatible with the large-scene setup until the `builder.py` bug is fixed
- **Method 3** reduces tokens *before* the LLM using geometry — directly controls the token budget, works cleanly with the large-scene setup
