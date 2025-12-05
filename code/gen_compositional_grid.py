from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from pathlib import Path
import torch

# ------------ CONFIG ------------

OUT_DIR = Path("../sample_outputs/compositional_benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = Path("../checkpoints/modelC_finetune_4k/unet_attention_finetuned.pt")

# 10 compositional prompts
PROMPTS = [
    "A red car next to a blue bicycle on a snowy street",
    "A teddy bear eating ramen in space, digital art",
    "Two golden retrievers sitting on a green sofa in a living room",
    "A person wearing a yellow raincoat holding a transparent umbrella in the city",
    "A bowl of fruit with apples, bananas, and grapes on a wooden table",
    "A train passing over a bridge above a river at sunset",
    "A cat and a dog looking out of a window together",
    "A man in a black suit riding a skateboard in Times Square",
    "A lighthouse on a cliff with stormy waves crashing below",
    "A robot chef cooking pasta in a modern kitchen"
]

NUM_STEPS = 40
GUIDANCE_A = 7.5
GUIDANCE_B = 8.0
GUIDANCE_C = 7.5

NEGATIVE_PROMPT_B = (
    "blurry, low quality, distorted, deformed, extra limbs, "
    "text, watermark, logo, bad anatomy"
)


# ------------ HELPERS ------------

def slugify(text: str) -> str:
    return (
        text.lower()
        .replace(",", "")
        .replace("'", "")
        .replace("\"", "")
        .replace("  ", " ")
        .replace(" ", "_")
    )[:60]


def make_triplet_grid(img_a, img_b, img_c):
    w, h = img_a.size
    grid = Image.new("RGB", (w * 3, h))
    grid.paste(img_a, (0, 0))
    grid.paste(img_b, (w, 0))
    grid.paste(img_c, (2 * w, 0))
    return grid


# ------------ MAIN ------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Using device: {device}")

    # Load A
    print("Loading Model A (baseline, DDIM)...")
    pipe_a = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype
    ).to(device)

    # For A we keep default scheduler (usually DDIM in your earlier experiments)

    # Load B (same as A but with DPM-Solver++, CFG=8, neg prompt)
    print("Loading Model B (DPM-Solver++, neg prompt)...")
    pipe_b = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype
    ).to(device)
    pipe_b.scheduler = DPMSolverMultistepScheduler.from_config(pipe_b.scheduler.config)

    # Load C (same as B + finetuned UNet)
    print("Loading Model C (DPM-Solver++ + finetuned UNet)...")
    pipe_c = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype
    ).to(device)
    pipe_c.scheduler = DPMSolverMultistepScheduler.from_config(pipe_c.scheduler.config)

    print(f"Loading finetuned UNet from: {CHECKPOINT_PATH}")
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    missing, unexpected = pipe_c.unet.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing keys in UNet state_dict")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys in UNet state_dict")

    # Generate for each prompt
    for i, prompt in enumerate(PROMPTS, start=1):
        print(f"\n=== [{i}/{len(PROMPTS)}] Prompt: {prompt}")
        slug = slugify(prompt)

        # Fixed seed per prompt so A/B/C are comparable
        generator = torch.Generator(device=device).manual_seed(42)

        # A
        img_a = pipe_a(
            prompt=prompt,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_A,
            generator=generator,
        ).images[0]

        # B
        img_b = pipe_b(
            prompt=prompt,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_B,
            negative_prompt=NEGATIVE_PROMPT_B,
            generator=generator,
        ).images[0]

        # C
        img_c = pipe_c(
            prompt=prompt,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_C,
            generator=generator,
        ).images[0]

        # Save individual images
        img_a.save(OUT_DIR / f"{slug}_A.png")
        img_b.save(OUT_DIR / f"{slug}_B.png")
        img_c.save(OUT_DIR / f"{slug}_C.png")

        # Save grid
        grid = make_triplet_grid(img_a, img_b, img_c)
        grid.save(OUT_DIR / f"{slug}_ABC_grid.png")

        print(f"Saved A/B/C and grid for: {slug}")

    print("\nDone! All compositional grids saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
