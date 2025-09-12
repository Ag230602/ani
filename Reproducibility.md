# 🧪 Reproducibility Report

Course: CS 5588 – Data Science Capstone
Assignment: Reproducibility Lab
Team / Student: [Adrija Ghosh]
GitHub Repo: https://github.com/salesforce/ULIP

Date: [Fill in 06/23/2002]

1. Upstream Repository

Original Project Repo: https://github.com/salesforce/ULIP

Commit Hash Used: (captured automatically in env_metadata.json)

Citation: ULIP: Learning Unified Representations of Language, Images, and Point Clouds for 3D Understanding (NeurIPS 2022).

2. Environment

Python version: 3.12.11 (from Colab environment)

Operating System / Platform: Linux (Colab VM, glibc 2.35)

Hardware: CPU + GPU (NVIDIA Tesla T4/A100, depending on Colab session)

Frameworks & Libraries:

Torch = 2.0.1

Torchvision = 0.15.2

Transformers = 4.36.2

Diffusers/Accelerate not needed for ULIP

PyTorch Lightning = 2.1.3

Others: omegaconf, imageio, einops, moviepy, kornia, decord

Dependency file: requirements.txt (from repo)

3. Data

Datasets Used:

ModelNet40 Point Clouds – Benchmark for 3D object recognition

Text Annotations – Category names or natural language prompts

Source/URL: (to be replaced with actual download links provided in ULIP repo/docs)

Pretrained Checkpoint: ULIP pretrained weights (pretrained/ulip_ckpt.pth)

Preprocessing: unzip archives, ensure directory layout matches --pc_root and --text_root

4. Baseline Run

Command:

python main_zero_shot.py \
    --dataset modelnet40 \
    --pc_root data/ModelNet40_PC \
    --text_root data/texts \
    --ckpt pretrained/ulip_ckpt.pth


Output Location: runs/baseline/ (log: logs/baseline_log.txt)

Results / Metrics: To be filled after execution (e.g., zero-shot accuracy on ModelNet40).

5. Controlled Variation

Parameter Changed: --batch_size 128

Command:

python main_zero_shot.py \
    --dataset modelnet40 \
    --pc_root data/ModelNet40_PC \
    --text_root data/texts \
    --ckpt pretrained/ulip_ckpt.pth \
    --batch_size 128


Output Location: runs/variation/ (log: logs/variation_log.txt)

Comparison Table (example):

Run	Batch Size	Accuracy	Notes
Baseline	default	(TBD)	matches paper claim
Variation	128	(TBD)	check scaling effect
6. Reproducibility Notes

Steps taken:

Logged environment & commit hash

Fixed random seed = 42

Structured outputs under runs/ and logs/

Challenges: dataset download links missing in repo, GPU availability in Colab limits batch size

Deviations: reduced batch size for Colab testing

7. Reflection

ULIP’s unified representation learning is directly relevant to multimodal AI projects (e.g., combining 3D + text for retrieval/classification).

Easy: cloning repo, environment setup with pinned requirements.

Hard: dataset acquisition + ensuring pretrained checkpoint compatibility.

Lesson: Always provide environment logs (env_metadata.json), pin requirements, and document data sources for reproducibility.