import argparse
import csv
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm


def load_captions(csv_path, max_images=None):
    captions = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        if "caption" not in reader.fieldnames:
            raise ValueError("CSV must have a 'caption' column.")
        for row in reader:
            captions.append(row["caption"])
            if max_images is not None and len(captions) >= max_images:
                break
    return captions


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve repo root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent

    # Paths
    csv_path = (repo_root / args.csv_path).resolve()
    weights_path = (repo_root / args.unet_weights).resolve()
    outdir = (repo_root / args.outdir).resolve()
    os.makedirs(outdir, exist_ok=True)

    # Load captions
    captions = load_captions(csv_path, max_images=args.num_images)
    print(f"Loaded {len(captions)} captions.")

    # Load SD1.5 base pipeline
    print("Loading Stable Diffusion v1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Match Model B scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Load finetuned UNet
    print(f"Loading finetuned UNet from: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    pipe.unet.load_state_dict(state_dict)

    # Move to GPU
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    # Generation params
    batch_size = args.batch_size
    gs = args.guidance_scale
    steps = args.num_inference_steps
    neg = args.negative_prompt

    print(f"Generating {len(captions)} images → {outdir}")

    idx = 0
    for i in tqdm(range(0, len(captions), batch_size)):
        batch = captions[i : i + batch_size]
        neg_batch = [neg] * len(batch) if neg else None

        with torch.autocast("cuda", enabled=(device == "cuda")):
            images = pipe(
                batch,
                negative_prompt=neg_batch,
                guidance_scale=gs,
                num_inference_steps=steps,
            ).images

        for img in images:
            img.save(outdir / f"modelC_4k_{idx:04d}.png")
            idx += 1

    print(f"Done! Saved {idx} images in {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 1000 images using Model C (4k finetune)")

    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--csv_path", type=str, default="data/manifests/coco_train1k.csv")
    parser.add_argument("--unet_weights", type=str, default="checkpoints/modelC_finetune_4k/unet_attention_finetuned.pt")
    parser.add_argument("--outdir", type=str, default="sample_outputs/modelC_finetuned_4k")
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--negative_prompt", type=str, default="blurry, distorted, low quality, artifacts, extra limbs")

    args = parser.parse_args()
    main(args)
