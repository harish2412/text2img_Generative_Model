import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


# -----------------------------
# Dataset
# -----------------------------
class CocoCSVDataset(Dataset):
    """
    Simple dataset reading a CSV with columns:
      - image_path (or image)
      - caption
    If image paths are relative, they are resolved against image_root.
    """

    def __init__(self, csv_path: Path, image_root: Optional[Path] = None, image_size: int = 512):
        import csv

        self.samples = []
        self.image_root = image_root
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                # explicit 3-channel normalize
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ValueError(f"No header row found in CSV: {csv_path}")

            # Support either 'image_path' or 'image' as the image column
            if "image_path" in reader.fieldnames:
                image_col = "image_path"
            elif "image" in reader.fieldnames:
                image_col = "image"
            else:
                raise ValueError(
                    "CSV must have an 'image_path' or 'image' column plus a 'caption' column."
                )

            if "caption" not in reader.fieldnames:
                raise ValueError("CSV must have a 'caption' column.")

            for row in reader:
                img_path = Path(row[image_col])
                if self.image_root is not None and not img_path.is_absolute():
                    img_path = self.image_root / img_path
                self.samples.append((img_path, row["caption"]))

        if len(self.samples) == 0:
            raise ValueError(f"No samples loaded from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {
            "pixel_values": image,
            "caption": caption,
        }


# -----------------------------
# Training config
# -----------------------------
@dataclass
class TrainConfig:
    model_id: str
    csv_path: Path
    image_root: Optional[Path]
    output_dir: Path
    resolution: int = 512
    train_batch_size: int = 2
    num_epochs: int = 3
    lr: float = 1e-4
    max_train_steps: Optional[int] = None
    seed: int = 42


def select_trainable_params(unet: UNet2DConditionModel):
    """
    Freeze everything in the UNet, then unfreeze only attention-related params.
    """
    for p in unet.parameters():
        p.requires_grad_(False)

    trainable_params = []
    for name, param in unet.named_parameters():
        # Train only attention blocks (attn1 and attn2)
        if "attn1" in name or "attn2" in name:
            param.requires_grad_(True)
            trainable_params.append(param)

    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable attention parameters found in UNet. "
            "Check name filters in select_trainable_params()."
        )

    print(
        f"Number of trainable attention params: "
        f"{sum(p.numel() for p in trainable_params)}"
    )
    return trainable_params


def train_finetune(cfg: TrainConfig):
    """
    Lightweight fine-tuning of Stable Diffusion:
    - Freeze VAE + text encoder
    - Freeze most of UNet
    - Train only attention-related parameters in UNet
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CocoCSVDataset(cfg.csv_path, cfg.image_root, image_size=cfg.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=1,  # safer on shared HPC
        pin_memory=True,
    )

    vae = AutoencoderKL.from_pretrained(cfg.model_id, subfolder="vae").to(device)
    text_encoder = CLIPTextModel.from_pretrained(cfg.model_id, subfolder="text_encoder").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model_id, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(cfg.model_id, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.model_id, subfolder="scheduler")

    # Freeze VAE + text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Select only attention params in UNet for training
    trainable_params = select_trainable_params(unet)
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)

    # Use full precision for stability
    weight_dtype = torch.float32
    vae.to(device=device, dtype=weight_dtype)
    text_encoder.to(device=device, dtype=weight_dtype)
    unet.to(device=device, dtype=weight_dtype)

    steps_per_epoch = math.ceil(len(dataset) / cfg.train_batch_size)
    total_steps = cfg.num_epochs * steps_per_epoch
    if cfg.max_train_steps is not None:
        total_steps = min(total_steps, cfg.max_train_steps)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Seeding
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    global_step = 0

    unet.train()
    text_encoder.eval()
    vae.eval()

    print(f"Starting attention-only finetuning on {device}")
    print(f"Dataset size: {len(dataset)}, total_steps (cap): {total_steps}")

    for epoch in range(cfg.num_epochs):
        for batch in dataloader:
            if cfg.max_train_steps is not None and global_step >= total_steps:
                break

            # Encode images to latents
            with torch.no_grad():
                pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            num_train_timesteps = (
                noise_scheduler.config.num_train_timesteps
                if hasattr(noise_scheduler, "config")
                else noise_scheduler.num_train_timesteps
            )
            timesteps = torch.randint(
                0,
                num_train_timesteps,
                (bsz,),
                device=device,
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Text encoding
            text_inputs = tokenizer(
                batch["caption"],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Predict noise
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            loss = nn.functional.mse_loss(noise_pred.float(), noise.float())

            # NaN guard
            if torch.isnan(loss):
                print(f"Encountered NaN loss at global_step={global_step}, skipping batch.")
                optimizer.zero_grad()
                continue

            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % 50 == 0:
                print(
                    f"Epoch {epoch+1}, step {global_step}/{total_steps}, "
                    f"loss={loss.item():.6f}"
                )

            if cfg.max_train_steps is not None and global_step >= total_steps:
                break

        if cfg.max_train_steps is not None and global_step >= total_steps:
            break

    print("Training complete, saving finetuned UNet weights...")
    out_path = os.path.join(cfg.output_dir, "unet_attention_finetuned.pt")
    torch.save(unet.state_dict(), out_path)
    print(f"Saved finetuned UNet to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight attention-only finetuning for Stable Diffusion v1.5 on a COCO CSV."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base SD model id.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV with 'image_path'/'image' and 'caption' columns.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Optional root directory to resolve relative image paths.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/modelC_finetune",
        help="Directory to save finetuned UNet (relative to repo root).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Training image resolution.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Batch size.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of epochs over the dataset.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2500,
        help="Optional cap on total training steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    # Resolve paths relative to repo root (one level above this script)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent

    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = (repo_root / csv_path).resolve()

    image_root = None
    if args.image_root is not None:
        image_root_path = Path(args.image_root)
        if not image_root_path.is_absolute():
            image_root = (repo_root / image_root_path).resolve()
        else:
            image_root = image_root_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()

    cfg = TrainConfig(
        model_id=args.model_id,
        csv_path=csv_path,
        image_root=image_root,
        output_dir=output_dir,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        max_train_steps=args.max_train_steps,
        seed=args.seed,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train_finetune(cfg)
