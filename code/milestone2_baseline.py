import os
import torch
import logging
from datetime import datetime
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionUpscalePipeline,
)


def setup_logger(name: str) -> logging.Logger:
    # repo_root = .../text2img
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(sh)

    logger.info(f"Logging to {log_file}")
    return logger


logger = setup_logger("milestone2_baseline")

# -------------------------------------------------
# Device / dtype
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
logger.info(f"Using device: {device}, dtype={dtype}")

# -------------------------------------------------
# 1. Load Stable Diffusion pipeline
# -------------------------------------------------
logger.info("Loading Stable Diffusion v1.5 pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
).to(device)

# Use better sampler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Speed / memory tweaks
pipe.enable_attention_slicing()
try:
    pipe.enable_xformers_memory_efficient_attention()
    logger.info("Enabled xFormers memory efficient attention.")
except Exception as e:
    logger.info(f"Could not enable xFormers attention: {e}")

# Disable NSFW checker so it never blanks images
pipe.safety_checker = lambda images, **kwargs: (images, [False])

logger.info("Attempting to load x4 upscaler (optional)...")
try:
    upscaler = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=dtype,
    ).to(device)
    upscaler.enable_attention_slicing()
    HAS_UPSCALER = True
    logger.info("Upscaler loaded successfully.")
except Exception as e:
    logger.info(f"Could not load upscaler ({e}). Continuing without it.")
    upscaler = None
    HAS_UPSCALER = False

NEGATIVE_PROMPT = (
    "low quality, blurry, distorted face, deformed, extra limbs, bad anatomy, "
    "crooked eyes, ugly, watermark, text, logo, grainy, oversaturated"
)

# -------------------------------------------------
# 2. Generation function (NO manual embeddings)
# -------------------------------------------------
def generate_image(
    prompt: str,
    guidance_scale: float = 8.0,
    num_steps: int = 40,
    height: int = 512,
    width: int = 512,
    use_upscaler: bool = False,  # keep False to avoid black images
):
    logger.info(f"Generating for prompt='{prompt}' guidance={guidance_scale}")

    # Base SD generation (this is stable and should NOT be black)
    with torch.autocast(
        device_type=device,
        dtype=torch.float16,
        enabled=(device == "cuda"),
    ):
        out = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        )

    img = out.images[0]

    # Optional upscaler – disabled by default since it caused issues before
    if use_upscaler and HAS_UPSCALER:
        logger.info("Running x4 upscaler...")
        with torch.autocast(
            device_type=device,
            dtype=torch.float16,
            enabled=(device == "cuda"),
        ):
            up_out = upscaler(
                prompt=prompt,
                image=img,
                num_inference_steps=50,
                guidance_scale=0.0,
            )
        img = up_out.images[0]

    return img

# -------------------------------------------------
# 3. Run Milestone 2 generations (100 images)
# -------------------------------------------------
import csv

repo_root = Path(__file__).resolve().parent.parent
output_dir = repo_root / "sample_outputs" / "milestone2_output"
os.makedirs(output_dir, exist_ok=True)

# Load 100 captions from your dataset CSV
CSV_PATH = "../data/manifests/coco_train1k.csv"   # or coco_train100.csv if you created one

prompts = []
with open(CSV_PATH, "r") as f:
    r = csv.DictReader(f)
    for row in r:
        prompts.append(row["caption"])
        if len(prompts) >= 100:
            break

logger.info(f"\nLoaded {len(prompts)} prompts for Milestone 2 generation.\n")

logger.info("Starting Milestone 2 improved baseline generation...\n")

for i, prompt in enumerate(prompts):
    img = generate_image(prompt)
    save_path = output_dir / f"sample_{i:05d}.png"
    img.save(save_path)
    logger.info(f"Saved: {save_path}")

logger.info("\n=== Milestone 2 Complete ===")
logger.info(f"Images saved in: {output_dir}")
print("\n=== Milestone 2 Complete ===")
print("Images saved in:", output_dir)
