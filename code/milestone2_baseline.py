import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------------------------
# 1. Load Stable Diffusion (with built-in tokenizer + text encoder)
# -------------------------------------------------
print("Loading Stable Diffusion pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to(device)

pipe.enable_attention_slicing()
pipe.unet.to(device)
pipe.vae.to(device)

pipe.safety_checker = lambda images, **kwargs: (images, [False])

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder.to(device)

# ------------------------------
# 2. Encode prompt → embeddings
# ------------------------------
def get_text_embeddings(prompt):
    # conditional embedding
    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    cond = text_encoder(inputs.input_ids)[0]

    # unconditional embedding
    uncond_inputs = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    ).to(device)

    uncond = text_encoder(uncond_inputs.input_ids)[0]

    return uncond, cond

# -------------------------------------------------
# 3. Generate a single image
# -------------------------------------------------
def generate_image(prompt, guidance_scale=7.5, num_steps=30):
    print(f"\nGenerating: {prompt}")

    uncond, cond = get_text_embeddings(prompt)

    with torch.autocast("cuda", dtype=torch.float16):
        out = pipe(
            prompt_embeds=cond,
            negative_prompt_embeds=uncond,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            output_type="pil"
        )

    return out.images[0]

# -------------------------------------------------
# 4. Run Milestone 2 baseline generations
# -------------------------------------------------
output_dir = "sample_outputs/milestone2_output/"
os.makedirs(output_dir, exist_ok=True)

prompts = [
    "A red sports car speeding on a highway",
    "A mountain landscape above the clouds",
    "A cyberpunk street scene with neon lights",
    "A magical forest with glowing mushrooms",
    "A spaceship landing on Mars during sunset",
    "A cozy living room with a fireplace",
    "A bowl of fresh fruit on a wooden table",
    "A brilliantly colored bird sitting on a tree branch",
    "A sushi platter with artistic presentation",
    "A futuristic city skyline at night"
]

print("\nStarting Milestone 2 baseline generation...\n")

for i, prompt in enumerate(prompts):
    img = generate_image(prompt)
    save_path = os.path.join(output_dir, f"sample_{i+1}.png")
    img.save(save_path)
    print(f"Saved: {save_path}")

print("\n=== Milestone 2 Complete ===")
print("Images saved in:", output_dir)
