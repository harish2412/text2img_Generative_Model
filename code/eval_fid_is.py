import os, csv, argparse, torch, logging
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from datetime import datetime
from pathlib import Path

TF_FID = transforms.Compose([
    transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(299),
    transforms.PILToTensor(),
])


def setup_logger(name: str) -> logging.Logger:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(sh)

    logger.info(f"Logging to {log_file}")
    return logger


def load_real_from_csv(csv_path, n, logger: logging.Logger):
    """Load up to n image paths from a CSV with column 'image_path'."""
    paths = []
    logger.info(f"Loading real image paths from {csv_path} (limit={n})")
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            p = row["image_path"]
            if os.path.exists(p):
                paths.append(p)
            if n and len(paths) >= n:
                break
    logger.info(f"Loaded {len(paths)} real image paths.")
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
    logger = setup_logger("eval_fid_is")

    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()

    logger.info("=== eval_fid_is.py run configuration ===")
    logger.info(str(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device for metrics: {device}")

    fid = FrechetInceptionDistance(normalize=True).to(device)
    isc = InceptionScore().to(device)

    real_paths = load_real_from_csv(args.val_csv, args.n, logger)
    if len(real_paths) == 0:
        raise ValueError("No valid real images found from CSV.")

    with torch.no_grad():
        logger.info("Updating FID with real images...")
        for t in tqdm(iter_images(real_paths), total=len(real_paths), desc="Real"):
            fid.update(t.unsqueeze(0).to(device), real=True)

        gen_files = [
            f for f in os.listdir(args.gen_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        gen_files = sorted(gen_files)[:args.n]

        if len(gen_files) == 0:
            raise ValueError("No generated images found in gen_dir.")

        logger.info(f"Found {len(gen_files)} generated images in {args.gen_dir}")
        logger.info("Updating FID+IS with generated images...")
        for t in tqdm(iter_folder(args.gen_dir, args.n), total=len(gen_files), desc="Gen"):
            img = t.unsqueeze(0).to(device)
            fid.update(img, real=False)
            isc.update(img)

        fid_score = fid.compute().item()
        is_mean, is_std = isc.compute()

    logger.info(f"FID: {fid_score:.4f}")
    logger.info(f"Inception Score: {is_mean.item():.4f} ± {is_std.item():.4f}")

    print(f"FID: {fid_score:.4f}")
    print(f"Inception Score: {is_mean.item():.4f} ± {is_std.item():.4f}")
