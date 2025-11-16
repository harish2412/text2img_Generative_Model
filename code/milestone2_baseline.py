import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# 1. Load CLIP Text Encoder
# -----------------------------
print("Loading CLIP tokenizer + encoder...")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

# -----------------------------
# 2. Load Stable Diffusion
# -----------------------------
print("Loading Stable Diffusion pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to(device)

pipe.enable_attention_slicing()  # helps reduce memory usage

# -----------------------------
# 3. Helper function: Generate
# -----------------------------
def generate_image(prompt, guidance_scale=7.5, num_steps=30):
    print(f"\nGenerating for prompt: {prompt}")

    # Encode text → CLIP embedding
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    text_embeddings = text_encoder(**inputs).last_hidden_state

    # Run diffusion sampling
    with torch.autocast("cuda"):
        image = pipe(
            prompt_embeds=text_embeddings,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale
        ).images[0]

    return image

# -----------------------------
# 4. Run baseline generation
# -----------------------------
output_dir = "../sample_outputs/milestone2/"
os.makedirs(output_dir, exist_ok=True)

prompts = [
    "A cute dog playing in snow",
    "A futuristic city skyline at night",
    "A bowl of fresh fruit on a wooden table",
    "A mountain landscape with clouds",
    "A red sports car speeding on a highway"
]

print("\nStarting baseline generation...")
for i, prompt in enumerate(prompts):
    img = generate_image(prompt)
    save_path = os.path.join(output_dir, f"sample_{i+1}.png")
    img.save(save_path)
    print(f"Saved: {save_path}")

print("\n=== Milestone 2 Complete ===")
print("Generated images saved in: sample_outputs/milestone2/")