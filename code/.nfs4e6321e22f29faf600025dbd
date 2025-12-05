import argparse
import csv
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ------------------ HELPERS ------------------


def load_captions(csv_path: Path, max_samples: Optional[int] = None) -> List[str]:
    captions: List[str] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        if "caption" not in reader.fieldnames:
            raise ValueError(f"CSV {csv_path} must have a 'caption' column.")
        for row in reader:
            captions.append(row["caption"])
            if max_samples is not None and len(captions) >= max_samples:
                break
    return captions


def load_image_paths(image_dir: Path, max_samples: Optional[int] = None) -> List[Path]:
    # Sort by name so index alignment across A/B/C is consistent
    paths = sorted(image_dir.glob("*.png"))
    if max_samples is not None:
        paths = paths[:max_samples]
    return paths


def compute_clip_scores_for_model(
    model_name: str,
    image_dir: Path,
    captions: List[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
    batch_size: int = 16,
) -> Tuple[float, float]:
    """
    Compute CLIP similarity between each (image, caption) pair for a given model folder.
    Returns (mean, std) of cosine similarity scores.
    """
    image_paths = load_image_paths(image_dir, max_samples=len(captions))
    if len(image_paths) == 0:
        raise ValueError(f"No .png images found in {image_dir} for {model_name}")

    if len(image_paths) != len(captions):
        print(
            f"[WARN] {model_name}: number of images ({len(image_paths)}) "
            f"!= number of captions ({len(captions)}). Truncating to min."
        )
        n = min(len(image_paths), len(captions))
        image_paths = image_paths[:n]
        captions = captions[:n]

    print(f"\n=== {model_name} ({len(image_paths)} samples) ===")

    scores = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_caps = captions[i : i + batch_size]

        images = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = clip_processor(
            text=batch_caps,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            out = clip_model(**inputs)
            img_emb = out.image_embeds  # (B, D)
            txt_emb = out.text_embeds   # (B, D)

        # Normalize
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        # Cosine sim per pair
        sim = (img_emb * txt_emb).sum(dim=-1)  # (B,)
        scores.extend(sim.cpu().tolist())

    scores = np.array(scores)
    mean = float(scores.mean())
    std = float(scores.std())
    print(f"{model_name}: mean CLIP similarity = {mean:.4f} ± {std:.4f}")
    return mean, std


# ------------------ MAIN ------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare CLIP ViT-B/32 vs ViT-L/14 text encoders on "
            "Model A/B/C generations using image-caption similarity."
        )
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        required=True,
        help="Path to COCO CSV with a 'caption' column (e.g., coco_train1k.csv).",
    )
    parser.add_argument(
        "--modelA_dir",
        type=str,
        default="../sample_outputs/milestone1_output",
        help="Directory with Model A images.",
    )
    parser.add_argument(
        "--modelB_dir",
        type=str,
        default="../sample_outputs/milestone2_output",
        help="Directory with Model B images.",
    )
    parser.add_argument(
        "--modelC_dir",
        type=str,
        default="../sample_outputs/modelC_finetuned_4k",
        help="Directory with Model C images.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Max number of (image, caption) pairs to evaluate per model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for CLIP forward passes.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="Where to save the JSON summary.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent  # text2img/

    # Allow both absolute + relative paths
    val_csv = Path(args.val_csv)
    if not val_csv.is_absolute():
        val_csv = (repo_root / val_csv).resolve()

    model_a_dir = Path(args.modelA_dir)
    if not model_a_dir.is_absolute():
        model_a_dir = (repo_root / model_a_dir).resolve()

    model_b_dir = Path(args.modelB_dir)
    if not model_b_dir.is_absolute():
        model_b_dir = (repo_root / model_b_dir).resolve()

    model_c_dir = Path(args.modelC_dir)
    if not model_c_dir.is_absolute():
        model_c_dir = (repo_root / model_c_dir).resolve()

    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = (repo_root / out_json).resolve()

    print(f"Loading captions from: {val_csv}")
    captions = load_captions(val_csv, max_samples=args.max_samples)
    print(f"Loaded {len(captions)} captions.\n")

    results: Dict[str, Dict[str, Dict[str, float]]] = {
        "clip_vit_b32": {},
        "clip_vit_l14": {},
    }

    # ---------- CLIP ViT-B/32 ----------
    print("===== Evaluating with CLIP ViT-B/32 =====")
    clip_b32 = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    proc_b32 = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    b32_A = compute_clip_scores_for_model(
        "Model A (baseline) [B/32]",
        model_a_dir,
        captions,
        clip_b32,
        proc_b32,
        device,
        batch_size=args.batch_size,
    )
    b32_B = compute_clip_scores_for_model(
        "Model B (improved) [B/32]",
        model_b_dir,
        captions,
        clip_b32,
        proc_b32,
        device,
        batch_size=args.batch_size,
    )
    b32_C = compute_clip_scores_for_model(
        "Model C (finetuned 4k) [B/32]",
        model_c_dir,
        captions,
        clip_b32,
        proc_b32,
        device,
        batch_size=args.batch_size,
    )

    results["clip_vit_b32"]["A"] = {"mean": b32_A[0], "std": b32_A[1]}
    results["clip_vit_b32"]["B"] = {"mean": b32_B[0], "std": b32_B[1]}
    results["clip_vit_b32"]["C"] = {"mean": b32_C[0], "std": b32_C[1]}

    # ---------- CLIP ViT-L/14 ----------
    print("\n\n===== Evaluating with CLIP ViT-L/14 =====")
    clip_l14 = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    proc_l14 = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    l14_A = compute_clip_scores_for_model(
        "Model A (baseline) [L/14]",
        model_a_dir,
        captions,
        clip_l14,
        proc_l14,
        device,
        batch_size=args.batch_size,
    )
    l14_B = compute_clip_scores_for_model(
        "Model B (improved) [L/14]",
        model_b_dir,
        captions,
        clip_l14,
        proc_l14,
        device,
        batch_size=args.batch_size,
    )
    l14_C = compute_clip_scores_for_model(
        "Model C (finetuned 4k) [L/14]",
        model_c_dir,
        captions,
        clip_l14,
        proc_l14,
        device,
        batch_size=args.batch_size,
    )

    results["clip_vit_l14"]["A"] = {"mean": l14_A[0], "std": l14_A[1]}
    results["clip_vit_l14"]["B"] = {"mean": l14_B[0], "std": l14_B[1]}
    results["clip_vit_l14"]["C"] = {"mean": l14_C[0], "std": l14_C[1]}

    # ---------- Summary print ----------
    print("\n\n================ SUMMARY ================")
    print(f"Samples per model: {len(captions)}")
    print("\nCLIP ViT-B/32:")
    print(f"  Model A: {b32_A[0]:.4f} ± {b32_A[1]:.4f}")
    print(f"  Model B: {b32_B[0]:.4f} ± {b32_B[1]:.4f}")
    print(f"  Model C: {b32_C[0]:.4f} ± {b32_C[1]:.4f}")

    print("\nCLIP ViT-L/14:")
    print(f"  Model A: {l14_A[0]:.4f} ± {l14_A[1]:.4f}")
    print(f"  Model B: {l14_B[0]:.4f} ± {l14_B[1]:.4f}")
    print(f"  Model C: {l14_C[0]:.4f} ± {l14_C[1]:.4f}")
    print("=========================================")

    # ---------- Save JSON ----------
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "val_csv": str(val_csv),
        "n_samples": len(captions),
        "metrics": results,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved text-encoder comparison JSON to: {out_json}")


if __name__ == "__main__":
    main()
