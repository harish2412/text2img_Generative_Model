"""
Simple demo script to generate one image from:
 - Model A (baseline)
 - Model B (sampling improved)
 - Model C (attention-finetuned)

Images are saved to: sample_outputs/demo_group_presentation/
"""

from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from PIL import Image


def load_pipelines(repo_root: Path, device: str = "cuda"):
    model_id = "runwayml/stable-diffusion-v1-5"

    print("\nLoading Model A (Baseline, DDIM)...")
    pipe_a = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    pipe_a.scheduler = DDIMScheduler.from_config(pipe_a.scheduler.config)

    print("Loading Model B (DPM++ + CFG 8 + negative prompt)...")
    pipe_b = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    pipe_b.scheduler = DPMSolverMultistepScheduler.from_config(pipe_b.scheduler.config)

    print("Loading Model C (Fine-tuned attention)...")
    pipe_c = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    pipe_c.scheduler = DPMSolverMultistepScheduler.from_config(pipe_c.scheduler.config)

    ckpt = repo_root / "checkpoints" / "modelC_finetune_4k" / "unet_attention_finetuned.pt"
    print(f"Loading fine-tuned UNet from: {ckpt}")
    state_dict = torch.load(ckpt, map_location="cpu")
    pipe_c.unet.load_state_dict(state_dict, strict=False)

    return pipe_a, pipe_b, pipe_c


def main():
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = repo_root / "sample_outputs" / "demo_group_presentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the three models
    pipe_a, pipe_b, pipe_c = load_pipelines(repo_root, device=device)

    # New demo prompt
    prompt = "A group of four people doing a presentation"
    negative_prompt = (
        "blurry, distorted, bad anatomy, extra limbs, low quality, bad resolution"
    )

    num_steps = 40
    cfg_a = 7.5   
    cfg_bc = 8.0  
    seed = 42

    print(f"\nUsing seed = {seed}")
    print(f"Prompt = {prompt}\n")

    # ---------------- MODEL A ----------------
    print("Generating with Model A (Baseline)...")
    gen_a = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(device):
        out_a = pipe_a(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=cfg_a,
            generator=gen_a,
        )
    img_a = out_a.images[0]
    img_a.save(out_dir / "demo_A_baseline.png")

    # ---------------- MODEL B ----------------
    print("Generating with Model B (Improved Sampling)...")
    gen_b = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(device):
        out_b = pipe_b(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=cfg_bc,
            generator=gen_b,
        )
    img_b = out_b.images[0]
    img_b.save(out_dir / "demo_B_sampling.png")

    # ---------------- MODEL C ----------------
    print("Generating with Model C (Fine-tuned Attention)...")
    gen_c = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(device):
        out_c = pipe_c(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=cfg_bc,
            generator=gen_c,
        )
    img_c = out_c.images[0]
    img_c.save(out_dir / "demo_C_finetuned.png")

    print(f"\n Demo complete! Images saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
