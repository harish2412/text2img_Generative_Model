import json, random, csv, os, argparse, logging
from pathlib import Path
from datetime import datetime


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


def build(csv_path, img_root, ann_json, k, min_len=5, seed=42):
    """
    Build a CSV of (image_path, caption) pairs from COCO-style annotation JSON.

    Args:
        csv_path: Output CSV path.
        img_root: Root directory containing the images.
        ann_json: COCO annotation JSON file.
        k: Maximum number of pairs to keep (None for all).
        min_len: Minimum caption length in words.
        seed: RNG seed for shuffling.
    """
    logger = setup_logger("data_prep_coco")

    logger.info("Starting COCO data prep...")
    logger.info(f"images_root={img_root}")
    logger.info(f"annotations={ann_json}")
    logger.info(f"out_csv={csv_path}, k={k}, min_len={min_len}, seed={seed}")

    with open(ann_json, "r") as f:
        ann = json.load(f)

    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    img_root = os.path.abspath(img_root)

    pairs = []
    added = 0
    skipped_short = 0
    skipped_missing = 0

    for c in ann["annotations"]:
        cap = (c.get("caption") or "").strip()
        if len(cap.split()) < min_len:
            skipped_short += 1
            continue

        fname = id2file.get(c["image_id"])
        if not fname:
            skipped_missing += 1
            continue

        img = os.path.join(img_root, fname)
        if os.path.exists(img):
            pairs.append((img, cap))
            added += 1
        else:
            skipped_missing += 1

    random.seed(seed)
    random.shuffle(pairs)

    if k is not None:
        pairs = pairs[:k]

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "caption"])
        w.writerows(pairs)

    logger.info(f"Found {added} valid pairs in dataset.")
    logger.info(f"Skipped {skipped_short} captions (too short).")
    logger.info(f"Skipped {skipped_missing} due to missing images/ids.")
    logger.info(f"Wrote {len(pairs)} rows -> {csv_path}")
    logger.info("Data prep complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--shuffle_seed", type=int, default=42)
    a = ap.parse_args()
    build(a.out_csv, a.images_root, a.annotations, a.k, a.min_len, a.shuffle_seed)