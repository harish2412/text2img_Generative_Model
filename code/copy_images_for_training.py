import csv
import os
import shutil
from pathlib import Path


def main():
    """
    Copy images referenced in a COCO CSV (image_path column)
    into a single folder for LoRA training.
    """
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent

    CSV_PATH = repo_root / "data" / "manifests" / "coco_train2k.csv"
    OUT_DIR = repo_root / "training_data" / "coco2k"

    print(f"Reading image paths from: {CSV_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Copying images into: {OUT_DIR}")

    copied = 0

    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        if "image_path" not in reader.fieldnames:
            raise ValueError(f"'image_path' column not found in {CSV_PATH}")
        for row in reader:
            src = Path(row["image_path"]).expanduser()
            if not src.exists():
                continue
            dst = OUT_DIR / src.name
            if not dst.exists():
                shutil.copy(src, dst)
                copied += 1

    print(f"Copied {copied} images into {OUT_DIR}")


if __name__ == "__main__":
    main()
