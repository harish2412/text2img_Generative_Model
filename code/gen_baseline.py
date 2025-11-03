import os, csv, argparse, torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline

def read_captions(csv_path, limit):
    out = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(row["caption"])
            if limit and len(out) >= limit:
                break
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--outdir", default="../outputs/baseline")
    ap.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--num", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)

    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    prompts = read_captions(args.val_csv, args.num)

    base_gen = torch.Generator(device=device).manual_seed(args.seed)

    for i, p in enumerate(tqdm(prompts, desc="Generating")):
        g = base_gen.manual_seed(args.seed + i)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            out = pipe(
                p,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=g,
            )
        img = out.images[0]
        img.save(os.path.join(args.outdir, f"gen_{i:05d}.png"))