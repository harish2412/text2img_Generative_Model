import json, random, csv, os, argparse
from pathlib import Path

def build(csv_path, img_root, ann_json, k, min_len=5, seed=42):
    with open(ann_json, "r") as f:
        ann = json.load(f)

    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    img_root = os.path.abspath(img_root)

    pairs = []
    added = 0
    for c in ann["annotations"]:
        cap = (c.get("caption") or "").strip()
        if len(cap.split()) < min_len:
            continue
        fname = id2file.get(c["image_id"])
        if not fname:
            continue
        img = os.path.join(img_root, fname)
        if os.path.exists(img):
            pairs.append((img, cap))
            added += 1

    random.seed(seed)
    random.shuffle(pairs)

    if k is not None:
        pairs = pairs[:k]

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "caption"])
        w.writerows(pairs)

    print(f"Found {added} valid pairs in dataset.")
    print(f"Wrote {len(pairs)} rows -> {csv_path}")

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