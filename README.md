# COMP5422 Project — Scaling LLaVA-3D for Large 3D Scenes

**Course:** COMP5422 Deep 2D & 3D Visual Scene Understanding, HKUST (Prof. Dan Xu)  
**Team:** WU Yuyao · CHAN Nga Teng · LIU Junzhang · WONG Wei Ming

This is a fork of [LLaVA-3D](https://github.com/ZCMax/LLaVA-3D) extended with token compression strategies to improve scalability on large 3D scenes. The original model is kept frozen — all compression is inserted as a lightweight plug-in with no retraining.

## Methods

| # | Method | Status |
|---|---|---|
| 1 | Attention-Score Pruning | Done |
| 2 | ForestPrune | Exploring |
| 3 | Spatial K-Means Clustering | Done (ablation pending) |
| 4 | Hierarchical Scene Representations | Not started |
| 5 | Q-Former / Learned Token Compression | In progress |

## Setup

```bash
git clone --recurse-submodules git@github.com:Alvin0523/comp5422_3d.git
cd comp5422_3d/LLaVA-3D
pixi install
```

## Run Commands

```bash
# Baseline (original voxel pooling)
python llava/eval/model_scanqa.py \
  --model-path ChaimZhu/LLaVA-3D-7B \
  --video-folder /path/to/scannet \
  --question-file playground/data/annotations/ScanQA_v1.0_val.json \
  --answers-file results/scanqa_baseline.json \
  --max-scene-index 59 --skip-missing-video

# Method 1 — Attention pruning
python llava/eval/model_scanqa.py ... \
  --prune_layer_ratio 5:0.05 8:0.1 14:0.2

# Method 3 — K-Means K=512
python llava/eval/model_scanqa.py ... \
  --num-clusters 512

# Large-scene stress test (3 scenes concatenated)
python llava/eval/model_scanqa.py ... \
  --extra-scenes 2 --extra-scene-translation 50

# Score any results file
python llava/eval/score_scanqa.py results/scanqa_baseline.json
```

## Results (ScanQA val, first 60 scenes)

| Method | Tokens | Raw EM@1 | Refined EM@1 |
|---|---|---|---|
| Baseline — single scene | ~3096 | 28.47% | 46.17% |
| Baseline — 3 scenes concat | ~9288 | 26.70% | 43.07% |
| Method 1 — Attn pruning | ~3096 | 28.74% | 44.89% |
| Method 3 — K-Means K=512 | 512 | 34.52% | 48.81% |
| Method 3 — K=512 full run | TBD | TBD | TBD |

## Docs

- `docs/proposal.md` — full design doc (all methods, datasets, evaluation metrics, timeline)
- `docs/method3_report.md` — Method 3 detailed report (algorithm, code walkthrough, ablation plan)
- `demo/README.md` — how to run smoke tests on the 3 local demo scenes

---

<br>
<p align="center">
<h1 align="center"><strong>LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities</strong></h1>
  <p align="center">
	<br>
    <a href='https://zcmax.github.io//' target='_blank'>Chenming Zhu</a>&emsp;
	<a href='https://tai-wang.github.io/' target='_blank'>Tai Wang*</a>&emsp;
    <a href='https://zhangwenwei.cn/' target='_blank'>Wenwei Zhang</a>&emsp;
    <a href='https://oceanpang.github.io/' target='_blank'>Jiangmiao Pang</a>&emsp;
	<a href='https://xh-liu.github.io//' target='_blank'>Xihui Liu*</a>&emsp;
    <br>
    The University of Hong Kong&emsp;Shanghai AI Laboratory
    <br>
  </p>
</p>


<div id="top" align="center">

<a href="https://arxiv.org/abs/2409.18125" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-LLaVA--3D-red?logo=arxiv" height="20" />
</a>
<a href="(https://zcmax.github.io/projects/LLaVA-3D/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-LLaVA--3D-blue.svg" height="20" />
</a>
<a href="https://huggingface.co/datasets/ChaimZhu/LLaVA-3D-Data" target="_blank">
    <img alt="HF Dataset: LLaVA-3D-Data" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Dataset-LLaVA--3D--Data-ffc107?color=ffc107&logoColor=white" height="20" />
</a>
<a href="https://huggingface.co/ChaimZhu/LLaVA-3D-7B" target="_blank">
    <img alt="HF Dataset: LLaVA-3D-7B" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-LLaVA--3D--7B-ffc107?color=ffc107&logoColor=white" height="20" />
</a>


</div>


## 🏠 Introducing LLaVA-3D
<!-- ![Teaser](assets/teaser.jpg) -->

<div style="text-align: center;">
    <img src="assets/llava-3d-teaser-combine-v2.png" alt="Dialogue_Teaser" width=100% >
</div>
LLaVA-3D could perform both 2D and 3D vision-language tasks. The left block (b) shows that compared with previous 3D LMMs, our LLaVA-3D achieves state-of-the-art performance across a wide range of 3D benchmarks while maintaining a comparable performance on various 2D benchmarks compared with LLaVA-1.5. The middle block (c) demonstrates that LLaVA-3D is built on the 2D LMM: LLaVA, and leverages 3D patches to endow it with 3D spatial awareness, enabling it to perform various 3D vision-and-language tasks in the physical world. The right blocks (d) and (e) highlights the significantly faster convergence and inference speeds of LLaVA-3D compared to existing 3D LMMs.

## 🔥 News
- [2025-07-11] :hearts: Our paper is accepted by ICCV 2025! See u in Hawaii! We release the full `LLaVA-3D-Instruct-86OK` data on [HuggingFace](https://huggingface.co/datasets/ChaimZhu/LLaVA-3D-Data)!
- [2024-11-29] We update the custom data instruction tuning tutorial, now you can train the model on your own dataset!
- [2024-10-19] We release the inference codes with checkpoints as well as the image and 3D scene demos. You can chat with LLaVA-3D with your own machines.
- [2024-09-28] We release the [paper](https://arxiv.org/abs/2409.18125) of LLaVA-3D. &#x1F389;

<!-- contents with emoji -->
## 📋 Contents
- [🔍 Model Architecture](#-model-architecture)
- [🔨 Install](#-install)
- [📦 Model Zoo](#-model-zoo)
- [🤖 Demo](#-demo)
- [📝 TODO List](#-todo-list)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)

## 🔍 Model Architecture
<p align="center">
  <img src="assets/llava-3d-method-v13.png" align="center" width="100%">
</p>
LLaVA-3D Architecture. Based on LLaVA, we directly add the corresponding 3D position embeddings to 2D patch visual tokens of multi-view images to construct the 3D Patches, then the 3D Patches will undergo 3D pooling and be sent into the projection layer of LLaVA to map into the LLM space and align with the LLM using 3D-visual-language data.


## 🔨 Install
We test our codes under the following environment:
* Python 3.10
* Pytorch 2.1.0
* CUDA Version 11.8

To start: 
1. Clone this repository.

```bash
git clone https://github.com/ZCMax/LLaVA-3D.git
cd LLaVA-3D
```



2. Install Pixi (if not already installed):

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```


3. Install all dependencies using Pixi:

```sh
pixi install
```

After installation, you can:
- Run commands directly with `pixi run <task>` (no need to activate the environment first)
- Or enter the Pixi environment shell (like `venv`/`conda`):

```sh
pixi shell
```
This gives you an interactive shell with all dependencies available.

**Training dependencies:**
To install and use extra packages for training (e.g., deepspeed, wandb, flash-attn), activate the training environment:

```sh
pixi install -e train
pixi shell -e train
```
or run a command with training dependencies:
```sh
pixi run -e train <task>
```

3. Download the [Camera Parameters File](https://drive.google.com/file/d/1a-1MCFLkfoXNgn9XdlmS9Gnzplrzw7vf/view?usp=drive_link) and put the json file under the `./playground/data/annotations`.

**Important:** You must also download or generate `embodiedscan_infos_full.json` and place it in `./playground/data/annotations/`. See `docs/fintune_custom_data.md` for the required format if you want to generate your own.


## 📦 Model Zoo

The trained model checkpoints are available [here](https://huggingface.co/ChaimZhu/LLaVA-3D-7B). Currently we only provide the 7B model, and we will continue to update the model zoo.

## ⚡ Pixi Environment Manager

This project uses [Pixi](https://pixi.sh/) for environment and task management. If you don't have Pixi installed, run:

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

See the [Pixi documentation](https://pixi.sh/docs/) for more details.
## 🤖 Demo

We currently support single image as inputs for 2D tasks and posed RGB-D images as inputs for 3D tasks. You can run the demo by using the script `llava/eval/run_llava_3d.py`. For 2D tasks, use the `image-file` parameter, and for 3D tasks, use the `video-path` parameter to provide the corresponding data. Here, we provide some demos as examples:

### 2D Tasks

```Shell
pixi run demo-2d
```

### 3D Tasks

We provide the demo scene [here](https://huggingface.co/datasets/ChaimZhu/LLaVA-3D-Demo-Data). **To use the demo, you must clone the dataset using git-lfs:**

```sh
sudo apt-get install git-lfs  # or see https://git-lfs.com for your OS
git lfs install
git clone https://huggingface.co/datasets/ChaimZhu/LLaVA-3D-Demo-Data
mv LLaVA-3D-Demo-Data/* ./demo/
```
This ensures all large files are downloaded correctly. If you skip git-lfs, you may get only pointers, not the actual data.

1. 3D Question Answering

```Shell
pixi run demo-3d-qa
```

2. 3D Dense Captioning

```Shell
pixi run demo-3d-caption
```

3. 3D Localization

```Shell
pixi run demo-3d-loc
```


## 📝 TODO List

- \[x\] Release the training and inference code.
- \[x\] Release the checkpoint, demo data and script.
- \[x\] Release the training datasets.
- \[ \] Release the full code.

## 🔗 Citation

If you find our work and this codebase helpful, please consider starring this repo 🌟 and cite:

```bibtex
@article{zhu2024llava,
  title={LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D-awareness},
  author={Zhu, Chenming and Wang, Tai and Zhang, Wenwei and Pang, Jiangmiao and Liu, Xihui},
  journal={arXiv preprint arXiv:2409.18125},
  year={2024}
}
```

## 📄 License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 👏 Acknowledgements

This repo benefits from [3D-LLM](https://github.com/UMass-Foundation-Model/3D-LLM), [LLaVA](https://github.com/haotian-liu/LLaVA), and [ODIN](https://github.com/ayushjain1144/odin).
