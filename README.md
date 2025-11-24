# Text2Img Generative Model  
### COCO 2017 • Stable Diffusion • Diffusers • PyTorch • CLIP • FID/IS/CLIP Evaluation

This repository contains the implementation for **Milestone 1, Milestone 2, and Milestone 3** of the Text-to-Image Generative Modelling project.  
The project uses the **COCO 2017 dataset**, **CLIP text encoder**, and a **Stable Diffusion (v1.5) pipeline** for conditional text-to-image generation and quantitative evaluation.

---

# Project Structure

```
text2img_Generative_Model/
│
├── code/
│   ├── data_prep_coco.py
│   ├── gen_baseline.py
│   ├── eval_fid_is.py
│   └── milestone2_baseline.py
│
├── data/
│   └── manifests/
│       └── coco_train1k.csv
│
├── milestone2/
│   ├── milestone2_baseline.py
│   └── sample_outputs/
│
├── milestone3/
│   ├── generate_fid_images.py
│   ├── calculate_fid.py
│   ├── calculate_inception_score.py
│   ├── calculate_clip_similarity.py
│   ├── cfg_step_experiments.py
│   ├── scheduler_comparison.py
│   ├── resize_reference_images.py
│   │
│   ├── generated_images/
│   ├── reference_images/
│   ├── reference_images_resized/
│   │
│   ├── results/
│   │   ├── fid_results.txt
│   │   ├── inception_score_results.txt
│   │   ├── clip_similarity_results.txt
│   │   ├── cfg_step_results.csv
│   │   └── scheduler_results.csv
│   │
│   └── plots/
│       ├── cfg_vs_time.png
│       ├── steps_vs_time.png
│       ├── scheduler_comparison.png
│       └── metrics_summary.png
│
└── README.md
```

---

# Milestone 1 — Dataset Setup & First Generation

- Prepared COCO subset  
- Implemented CLIP embedding pipeline  
- Generated initial 5–10 SD images  

---

# Milestone 2 — Conditional Generation Experiments

- Integrated CLIP tokenizer + encoder  
- Integrated Stable Diffusion v1.5  
- Generated baseline images  
- Conducted guidance-scale and step experiments  

Outputs stored in:
```
sample_outputs/milestone2/
```

---

# Milestone 3 — Quantitative Evaluation

Milestone 3 evaluates the text-to-image system using statistical metrics:

### ✔ Generated 100 SD images  
### ✔ Prepared COCO reference dataset  
### ✔ Computed:
- FID  
- Inception Score  
- CLIP Similarity Score  

### ✔ Parameter Studies:
- CFG × steps (30 experiments)  
- Scheduler comparison (DDIM, DPM++, Euler)

### ✔ Visualization Plots:
- cfg_vs_time.png  
- steps_vs_time.png  
- scheduler_comparison.png  
- metrics_summary.png  

Stored in:
```
milestone3/plots/
```

---

# Running the Project (HPC)

```
salloc -p gpu --gres=gpu:1 --mem=16G -t 02:00:00
srun --jobid=<id> --pty bash
module load anaconda3/2024.06
module load cuda/12.1.1
conda activate text2img
```

---

# Dependencies

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors pillow
pip install clean-fid torch-fidelity matplotlib seaborn tqdm
```

---

# License  
Academic use only — Northeastern University.
