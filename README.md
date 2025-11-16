# Text2Img Generative Model  
### COCO 2017 • Stable Diffusion • Diffusers • PyTorch

This repository contains the implementation for **Milestone 1 and Milestone 2** of the Text-to-Image Generative Modelling project.  
The project uses the **COCO 2017 dataset** along with a **Stable Diffusion (v1.5) pipeline** for conditional text-to-image generation.

---

## Project Structure

text2img_Generative_Model/
│
├── code/
│ ├── data_prep_coco.py # Build (image, caption) CSV from COCO
│ ├── gen_baseline.py # Milestone 1 & 2 baseline generator
│ ├── eval_fid_is.py # FID + Inception Score evaluation
│ └── milestone2_baseline.py # Extended baseline experiments
│
├── data/
│ └── manifests/
│ └── coco_train1k.csv # 1,000 curated COCO samples
│
├── sample_outputs/
│ ├── milestone1_output/ # Milestone 1 sample images
│ └── milestone2_output/ # Milestone 2 experiments
│
├── .gitignore
└── README.md

---

# Milestone 1 — Dataset Setup & First Generation

### ✔ Goal  
Prepare COCO subset, build a text-embedding pipeline, and produce the first 5–10 generated images.

---

## Dataset Preparation — COCO 2017

We use:

- **train2017 images** (118,000)
- **captions_train2017.json**
- Filtered **1k samples** with ≥ 5 words captions

CSV generation:

```bash
python code/data_prep_coco.py \
  --images_root /path/to/coco/train2017 \
  --annotations /path/to/coco/annotations/captions_train2017.json \
  --out_csv data/manifests/coco_train1k.csv \
  --k 1000 \
  --min_len 5


The dataset file:
data/manifests/coco_train1k.csv

Text Encoder Pipeline
We use the CLIP text encoder inside Stable Diffusion (HuggingFace Diffusers):
Tokenization → CLIP embedding
Embedding shape: [batch, 77, 768]
Conditioning injected into UNet during diffusion steps

Baseline Stable Diffusion Generation
Example Milestone-1 run:
python code/gen_baseline.py \
  --val_csv data/manifests/coco_train1k.csv \
  --outdir sample_outputs/milestone1_output \
  --steps 30 \
  --guidance 7.5 \
  --num 10 \
  --seed 42

Outputs stored in:
sample_outputs/milestone1_output/

Milestone 2 — Conditional Generation Experiments
✔ Goal
Run guided diffusion experiments (vary steps/guidance) and produce 256+ samples.

Baseline Conditional Generation
Core experiment:
python code/gen_baseline.py \
  --val_csv data/manifests/coco_train1k.csv \
  --outdir sample_outputs/milestone2_output/baseline \
  --steps 30 \
  --guidance 7.5 \
  --num 256 \
  --seed 42

Guidance Scale Experiments
We vary classifier-free guidance:
Example:
python code/milestone2_baseline.py \
  --val_csv data/manifests/coco_train1k.csv \
  --outdir sample_outputs/milestone2_output/gs12 \
  --steps 30 \
  --guidance 12 \
  --num 128 \
  --seed 42

Diffusion Step Experiments
Evaluate generation quality vs speed:
Example:
python code/milestone2_baseline.py \
  --val_csv data/manifests/coco_train1k.csv \
  --outdir sample_outputs/milestone2_output/steps60 \
  --steps 60 \
  --guidance 7.5 \
  --num 128 \
  --seed 42

Key Observations (Milestone 2)
✔ Higher guidance improves prompt alignment
But may reduce diversity or introduce artifacts.
✔ More diffusion steps improve sharpness
But cost more time and give diminishing returns beyond ~40–50 steps.
✔ Failure cases
Wrong object count
Distorted hands/limbs
Incorrect spatial relations
Hallucinated background details
