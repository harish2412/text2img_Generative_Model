import os, csv, argparse, torch, logging
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionUpscalePipeline,
)


def setup_logger(name: str) -> logging.Logger:
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


def read_captions(csv_path, limit, logger: logging.Logger):
    """Read up to `limit` captions from a CSV with a 'caption' column."""
    out = []
    logger.info(f"Loading captions from {csv_path} (limit={limit})")
    with open(csv_path) as f:
        r = csv.DictReader(f)
        if "caption" not in r.fieldnames:
            raise ValueError(f"'caption' column not found in {csv_path}")
        for row in r:
            out.append(row["caption"])
            if limit and len(out) >= limit:
                break
    logger.info(f"Loaded {len(out)} captions.")
    return out


def main():
    logger = setup_logger("gen_baseline")

    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True,
                    help="CSV with at least a 'caption' column.")
    ap.add_argument("--outdir", default="../sample_outputs/milestone1_output")
    ap.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--num", type=int, default=10,
                    help="Number of images to generate.")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)

    ap.add_argument("--use_upscaler", action="store_true",
                    help="Enable SD x4 upscaler for higher-res outputs.")
    ap.add_argument("--upscaler_id", type=str,
                    default="stabilityai/stable-diffusion-x4-upscaler")
    ap.add_argument("--up_steps", type=int, default=50)
    ap.add_argument("--up_guidance", type=float, default=0.0)
    args = ap.parse_args()

    logger.info("=== gen_baseline.py run configuration ===")
    logger.info(str(args))

    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"Using device: {device}, dtype={dtype}")

    # -----------------------------
    # Base Stable Diffusion model
    # -----------------------------
    logger.info(f"Loading Stable Diffusion model: {args.model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("Enabled xFormers memory efficient attention.")
    except Exception as e:
        logger.info(f"Could not enable xFormers attention: {e}")

    upscaler = None
    if args.use_upscaler:
        logger.info(f"Loading x4 upscaler: {args.upscaler_id}")
        try:
            upscaler = StableDiffusionUpscalePipeline.from_pretrained(
                args.upscaler_id,
                torch_dtype=dtype,
            ).to(device)
            upscaler.enable_attention_slicing()
            logger.info("Upscaler loaded successfully.")
        except Exception as e:
            logger.info(f"Could not load upscaler ({e}). Continuing without it.")
            upscaler = None

    prompts = read_captions(args.val_csv, args.num, logger)
    if len(prompts) == 0:
        raise ValueError("No captions loaded from val_csv. Check the file / column names.")

    base_gen = torch.Generator(device=device).manual_seed(args.seed)

    logger.info(f"Starting generation of {len(prompts)} images...")
    for i, p in enumerate(tqdm(prompts, desc="Generating")):
        g = base_gen.manual_seed(args.seed + i)

        with torch.autocast(
            device_type=device,
            dtype=torch.float16,
            enabled=(device == "cuda"),
        ):
            out = pipe(
                p,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=g,
            )
        img = out.images[0]

        if upscaler is not None:
            with torch.autocast(
                device_type=device,
                dtype=torch.float16,
                enabled=(device == "cuda"),
            ):
                up_out = upscaler(
                    prompt=p,
                    image=img,
                    num_inference_steps=args.up_steps,
                    guidance_scale=args.up_guidance,
                )
            img = up_out.images[0]

        out_path = os.path.join(args.outdir, f"gen_{i:05d}.png")
        img.save(out_path)
        logger.info(f"Saved {out_path}")

    logger.info(f"Saved {len(prompts)} images to {args.outdir}")
    logger.info("Generation complete.")


if __name__ == "__main__":
    main()