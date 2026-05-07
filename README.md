<h1 align="center">
  Continuous-Time Distribution Matching for Few-Step Diffusion Distillation
</h1>

<div align="center">

<a href="https://arxiv.org/abs/2509.16117" style="display: inline-block;">
    <img src="https://img.shields.io/badge/arXiv%20paper-2509.16117-b31b1b.svg" alt="arXiv" style="height: 20px; vertical-align: middle;">
</a>&nbsp;
<a href="https://research.nvidia.com/labs/dir/DiffusionNFT" style="display: inline-block;">
    <img src="https://img.shields.io/badge/Project_page-Website-green" alt="project page" style="height: 20px; vertical-align: middle;">
</a>&nbsp;
<a href="https://huggingface.co/worstcoder/SD3.5M-DiffusionNFT-MultiReward" style="display: inline-block;">
    <img src="https://img.shields.io/badge/Model-HuggingFace-blue" alt="model" style="height: 20px; vertical-align: middle;">
</a>&nbsp;

</div>

<p align="center">
  <a href="#algorithm-overview">Algorithm Overview</a> •
  <a href="#4-nfe-generation-results">Results</a> •
  <a href="#inference">Inference</a> •
  <a href="#training">Training</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="assets/teaser.png" width="95%" alt="Teaser: High-quality images generated with only 4 NFE">
</p>

## Algorithm Overview

<p align="center">
  <img src="assets/pipe.png" width="90%" alt="Pipeline overview of Continuous-Time Distribution Matching">
</p>

**Overview of Continuous-Time Distribution Matching (CDM).** **Top:** Our approach employs a dynamic continuous time schedule during backward simulation, sampling intermediate anchors uniformly from (0, 1]. **Bottom Left:** CFG augmentation (CA) and distribution matching (DM) operate on this dynamic schedule to align text-image conditions and data distributions at on-trajectory anchors. **Bottom Right:** To address inter-anchor inconsistency, the proposed CDM objective explicitly extrapolates off-trajectory latents using the predicted velocity.

## 4-NFE Generation Results

### SD3.5-Medium

<p align="center">
  <img src="assets/sd3.png" width="90%" alt="SD3.5-Medium 4-NFE generation samples">
</p>

### LongCat

<p align="center">
  <img src="assets/longcat.png" width="90%" alt="LongCat 4-NFE generation samples">
</p>

---

## Inference

```bash
# Clone this repository
git clone https://github.com/byliutao/cdm.git
cd cdm

# [Optional] Use HuggingFace mirror if huggingface.co is not accessible
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="hf_xxx"

# Create and activate the inference environment
conda create -n cdm_infer python=3.10
conda activate cdm_infer
pip install -r config/requirements_infer.txt

# Run inference
python scripts/infer/sd3_m.py   # SD3.5-Medium
python scripts/infer/longcat.py # LongCat
```

## Training

```bash
# Clone this repository
git clone https://github.com/NVlabs/DiffusionNFT.git
cd DiffusionNFT

# Create and activate the training environment
conda create -n cdm_train python=3.10
conda activate cdm_train
pip install -r config/requirements_train.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation  # May take 1-2 hours

# Launch training with FSDP2
accelerate launch --config_file config/accelerate_fsdp2.yaml \
    --num_processes 1 -m scripts.train \
    --config config/config.py:sd3      # SD3.5-Medium

accelerate launch --config_file config/accelerate_fsdp2.yaml \
    --num_processes 1 -m scripts.train \
    --config config/config.py:longcat  # LongCat
```

## Evaluation

Evaluation is split into two phases: **image generation** and **metric computation**.

### Step 1 — Export a checkpoint to a pipeline

```bash
conda activate cdm_train

python -m scripts.save \
    --experiment_dir "logs/experiments/sd3/test" \
    --output_dir "logs/pipelines/test" \
    --checkpoint_steps "10"
```

### Step 2 — Generate images

```bash
accelerate launch --num_processes 1 -m scripts.eval \
    --phase generate \
    --model_path "logs/pipelines/test/checkpoint-2000" \
    --eval_metrics imagereward clipscore pickscore hpsv2 hpsv3 aesthetic ocr dpgbench fid \
    --output_dir "logs/evaluations/test" \
    --base_model sd3 \
    --save_images
```

### Step 3 — Compute metrics

```bash
# Create a separate environment for evaluation dependencies
conda create -n cdm_eval python=3.10
conda activate cdm_eval
pip install -r config/requirements_eval.txt
pip install image-reward --no-deps
pip install fairseq --no-deps

# NOTE: If running on multiple GPUs, download checkpoints on 1 GPU first.
# For FID evaluation, place COCO 2014 val images under: dataset/coco2014val_10k/images

accelerate launch --num_processes 1 -m scripts.eval \
    --phase evaluate \
    --eval_metrics imagereward clipscore pickscore hpsv2 hpsv3 aesthetic ocr dpgbench fid \
    --output_dir "logs/evaluations/test"
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

If our work assists your research, please consider giving us a star ⭐ or citing us:

```bibtex
```
