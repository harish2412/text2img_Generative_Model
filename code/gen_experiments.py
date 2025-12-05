import os
import csv
import argparse
import logging
from typing import Optional
from datetime import datetime
from pathlib import Path

import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionUpscalePipeline,
    DDIMScheduler,
)


def setup_logger(name: str) -> logging.Logger:
    """
    Simple file + console logger.
    """
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


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted face, deformed, extra limbs, bad anatomy, "
    "crooked eyes, ugly, watermark, text, logo, grainy, oversaturated"
)


def build_pipeline(args, logger):
    """
    Build SD pipeline with requested scheduler, optional LoRA, optional upscaler.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"Using device: {device}, dtype={dtype}")

    logger.info(f"Loading Stable Diffusion model: {args.model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    ).to(device)

    # Scheduler choice
    if args.scheduler == "dpm":
        logger.info("Using DPMSolverMultistepScheduler")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        logger.info("Using DDIMScheduler")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Memory / speed tweaks
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("Enabled xFormers memory efficient attention.")
    except Exception as e:
        logger.info(f"Could not enable xFormers attention: {e}")

    # Disable NSFW checker so it never blanks images
    pipe.safety_checker = lambda images, **kwargs: (images, [False])

    # Optional LoRA (expects diffusers-style LoRA adapter directory)
    if args.lora_path:
        logger.info(f"Loading LoRA weights from: {args.lora_path}")
        try:
            pipe.load_lora_weights(args.lora_path)
            logger.info("LoRA weights loaded successfully.")
        except Exception as e:
            logger.info(f"Could not load LoRA weights ({e}). Continuing without LoRA.")

    # Optional upscaler
    upscaler = None
    has_upscaler = False
    if args.use_upscaler:
        logger.info("Attempting to load x4 upscaler...")
        try:
            upscaler = StableDiffusionUpscalePipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler",
                torch_dtype=dtype,
            ).to(device)
            upscaler.enable_attention_slicing()
            has_upscaler = True
            logger.info("Upscaler loaded successfully.")
        except Exception as e:
            logger.info(f"Could not load upscaler ({e}). Continuing without it.")
            upscaler = None
            has_upscaler = False

    return pipe, upscaler, has_upscaler, device, dtype


def generate_image(
    pipe,
    upscaler,
    has_upscaler: bool,
    device: str,
    prompt: str,
    guidance_scale: float,
    num_steps: int,
    height: int,
    width: int,
    negative_prompt: Optional[str],
    use_upscaler: bool,
    generator: Optional[torch.Generator],
    logger: logging.Logger,
):
    """
    Generate one image from prompt using SD (and optionally upscaler).
    """
    logger.info(
        f"Prompt='{prompt[:80]}...' | "
        f"guidance={guidance_scale}, steps={num_steps}, size={width}x{height}, "
        f"neg={'ON' if negative_prompt else 'OFF'}, upscaler={use_upscaler}"
    )

    with torch.autocast(
        device_type=device,
        dtype=torch.float16,
        enabled=(device == "cuda"),
    ):
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )

    img = out.images[0]

    # Optional 2-stage upscaling
    if use_upscaler and has_upscaler and upscaler is not None:
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


def main():
    logger = setup_logger("gen_experiments")

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent

    ap = argparse.ArgumentParser(
        description="General generation script for experiments (guidance, steps, scheduler, LoRA, etc.)."
    )
    ap.add_argument(
        "--csv_path",
        type=str,
        default="../data/manifests/coco_train1k.csv",
        help="CSV with at least a 'caption' column.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(repo_root / "sample_outputs" / "experiments"),
        help="Output directory for generated images.",
    )
    ap.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion model id.",
    )
    ap.add_argument(
        "--num_images",
        type=int,
        default=200,
        help="Number of images to generate.",
    )
    ap.add_argument(
        "--guidance",
        type=float,
        default=8.0,
        help="Classifier-free guidance scale.",
    )
    ap.add_argument(
        "--num_steps",
        type=int,
        default=40,
        help="Number of diffusion steps.",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height.",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width.",
    )
    ap.add_argument(
        "--scheduler",
        choices=["dpm", "ddim"],
        default="dpm",
        help="Sampler / noise schedule.",
    )
    ap.add_argument(
        "--no_negative_prompt",
        action="store_true",
        help="Disable the default negative prompt.",
    )
    ap.add_argument(
        "--use_upscaler",
        action="store_true",
        help="Use x4 upscaler as a second stage.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    ap.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Optional path to LoRA weights directory (for finetuned Model C).",
    )

    args = ap.parse_args()
    logger.info("=== gen_experiments configuration ===")
    logger.info(str(args))

    # Build pipeline
    pipe, upscaler, has_upscaler, device, dtype = build_pipeline(args, logger)

    # Negative prompt toggle
    negative_prompt = None if args.no_negative_prompt else DEFAULT_NEGATIVE_PROMPT
    logger.info(
        f"Negative prompt: {'DISABLED' if negative_prompt is None else 'ENABLED'}"
    )

    # -------- CSV path resolution (fixed) --------
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        # Resolve relative to the 'code/' directory where this script lives
        csv_path = (script_path.parent / args.csv_path).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    # --------------------------------------------

    # Load prompts
    prompts = []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        if "caption" not in r.fieldnames:
            raise ValueError(f"'caption' column not found in {csv_path}")
        for row in r:
            prompts.append(row["caption"])
            if len(prompts) >= args.num_images:
                break

    if len(prompts) == 0:
        raise ValueError("No prompts loaded from CSV.")

    logger.info(f"Loaded {len(prompts)} prompts from {csv_path}")

    # Output dir
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (repo_root / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Images will be saved in: {outdir}")

    # Generation loop
    base_gen = torch.Generator(device=device).manual_seed(args.seed)
    logger.info("Starting generation...")

    for i, prompt in enumerate(prompts):
        g = base_gen.manual_seed(args.seed + i)
        img = generate_image(
            pipe=pipe,
            upscaler=upscaler,
            has_upscaler=has_upscaler,
            device=device,
            prompt=prompt,
            guidance_scale=args.guidance,
            num_steps=args.num_steps,
            height=args.height,
            width=args.width,
            negative_prompt=negative_prompt,
            use_upscaler=args.use_upscaler,
            generator=g,
            logger=logger,
        )

        save_path = outdir / f"exp_{i:05d}.png"
        img.save(save_path)
        logger.info(f"Saved: {save_path}")

    logger.info("=== Experiment generation complete ===")
    print("\n=== Experiment generation complete ===")
    print("Images saved in:", outdir)


if __name__ == "__main__":
    main()
