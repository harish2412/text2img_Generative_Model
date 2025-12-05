import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ------------------ DEVICE & MODEL ------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading CLIP model (openai/clip-vit-base-patch32)...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()


# ------------------ CAPTION LOADING ------------------

def load_captions(csv_path, max_n=None):
    csv_path = Path(csv_path)
    captions = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            captions.append(row["caption"])
            if max_n is not None and len(captions) >= max_n:
                break
    return captions


# ------------------ CLIP SIMILARITY ------------------

def compute_clip_score_for_model(images_dir, captions, batch_size=32):
    images_dir = Path(images_dir)
    image_paths = sorted(images_dir.glob("*.png"))[:len(captions)]

    scores = []
    for i in range(0, len(image_paths), batch_size):
        batch_imgs = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch_size]]
        batch_txts = captions[i:i + batch_size]

        inputs = processor(
            text=batch_txts,
            images=batch_imgs,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            out = model(**inputs)
            img_emb = out.image_embeds  # (B, D)
            txt_emb = out.text_embeds   # (B, D)

        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        sim = (img_emb * txt_emb).sum(dim=-1)  # cosine similarity per pair

        scores.extend(sim.cpu().tolist())

    scores = np.array(scores)
    return float(scores.mean()), float(scores.std())


# ------------------ CLIP DIVERSITY ------------------

def compute_clip_diversity(images_dir, max_n=200, batch_size=32):
    images_dir = Path(images_dir)
    image_paths = sorted(images_dir.glob("*.png"))[:max_n]

    if len(image_paths) < 2:
        return float("nan")

    embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_imgs = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch_size]]
        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.get_image_features(**inputs)

        feats = out / out.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu())

    embs = torch.cat(embs, dim=0)  # (N, D)

    # Cosine similarity matrix
    sim = embs @ embs.T  # (N, N)

    n = sim.shape[0]
    triu_indices = torch.triu_indices(n, n, offset=1)
    pairwise = 1.0 - sim[triu_indices[0], triu_indices[1]]

    return float(pairwise.mean().item())


# ------------------ MAIN ------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute CLIP similarity (caption-image) and CLIP diversity (image-image) for models A/B/C."
    )

    parser.add_argument(
        "--val_csv",
        type=str,
        required=True,
        help="Path to COCO captions CSV (e.g., coco_train1k.csv).",
    )
    parser.add_argument(
        "--modelA_dir",
        type=str,
        required=True,
        help="Directory with Model A generated PNG images.",
    )
    parser.add_argument(
        "--modelB_dir",
        type=str,
        required=True,
        help="Directory with Model B generated PNG images.",
    )
    parser.add_argument(
        "--modelC_dir",
        type=str,
        required=True,
        help="Directory with Model C generated PNG images.",
    )
    parser.add_argument(
        "--max_n_captions",
        type=int,
        default=1000,
        help="Max number of (image, caption) pairs to use for CLIP similarity.",
    )
    parser.add_argument(
        "--diversity_max_n",
        type=int,
        default=200,
        help="Max number of images per model for CLIP diversity.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="Output JSON file to write all metrics.",
    )

    args = parser.parse_args()

    # Load captions
    print(f"Loading captions from: {args.val_csv}")
    captions = load_captions(args.val_csv, max_n=args.max_n_captions)
    print(f"Using {len(captions)} captions for CLIP similarity.")

    results = {}

    for name, img_dir in [
        ("A", args.modelA_dir),
        ("B", args.modelB_dir),
        ("C", args.modelC_dir),
    ]:
        print(f"\n=== Evaluating Model {name} ===")
        print(f"Images dir: {img_dir}")

        mean_sim, std_sim = compute_clip_score_for_model(img_dir, captions)
        print(f"Model {name} CLIP similarity: mean={mean_sim:.4f}, std={std_sim:.4f}")

        diversity = compute_clip_diversity(img_dir, max_n=args.diversity_max_n)
        print(f"Model {name} CLIP diversity (mean 1 - cos): {diversity:.4f}")

        results[name] = {
            "clip_similarity_mean": mean_sim,
            "clip_similarity_std": std_sim,
            "clip_diversity_mean_1_minus_cos": diversity,
        }

    output = {
        "val_csv": args.val_csv,
        "max_n_captions": args.max_n_captions,
        "diversity_max_n": args.diversity_max_n,
        "models": results,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved CLIP metrics JSON to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
