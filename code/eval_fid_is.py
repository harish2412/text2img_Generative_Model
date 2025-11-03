import os, csv, argparse, torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

TF_FID = transforms.Compose([
    transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(299),
    transforms.PILToTensor(),
])

def load_real_from_csv(csv_path, n):
    paths = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            p = row["image_path"]
            if os.path.exists(p):
                paths.append(p)
            if n and len(paths) >= n:
                break
    return paths

def iter_images(paths):
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            yield TF_FID(im)
        except Exception:
            continue

def iter_folder(folder, n=None):
    files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if n:
        files = files[:n]
    for p in files:
        try:
            im = Image.open(p).convert("RGB")
            yield TF_FID(im)
        except Exception:
            continue

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid = FrechetInceptionDistance(normalize=True).to(device)
    isc = InceptionScore().to(device)

    real_paths = load_real_from_csv(args.val_csv, args.n)
    with torch.no_grad():
        for t in tqdm(iter_images(real_paths), total=len(real_paths), desc="Real"):
            fid.update(t.unsqueeze(0).to(device), real=True)

        gen_files = [
            f for f in os.listdir(args.gen_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        gen_files = sorted(gen_files)[:args.n]

        for t in tqdm(iter_folder(args.gen_dir, args.n), total=len(gen_files), desc="Gen"):
            img = t.unsqueeze(0).to(device)
            fid.update(img, real=False)
            isc.update(img)

        fid_score = fid.compute().item()
        is_mean, is_std = isc.compute()

    print(f"FID: {fid_score:.4f}")
    print(f"Inception Score: {is_mean.item():.4f} ± {is_std.item():.4f}")
