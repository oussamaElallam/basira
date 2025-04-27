"""Prepare MIMIC-CXR dataset for Basira.
Requires that you have already requested access on PhysioNet and set up HF_AUTH_TOKEN.
The script streams the report CSV from 🤗 Hub (MITAI/mimic_cxr_reports) and pairs it with
local JPEG/PNG images you downloaded from PhysioNet mirror.

Usage (after downloading images with official script):
    python src/data/prepare_mimic.py --images /path/to/mimic/images --out data/mimic_cxr --limit 50000

It will create parquet shards of (image_path, report) ready for training.
"""
import argparse
from pathlib import Path
import datasets
from PIL import Image

def iter_reports(limit=None):
    ds = datasets.load_dataset("MITAI/mimic_cxr_reports", split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        text = (row.get("findings") or "").strip()
        if not text:
            continue
        yield row["study_id"], text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True, help="Root dir of downloaded MIMIC images")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=50000)
    args = p.parse_args()

    images_root = args.images
    args.out.mkdir(parents=True, exist_ok=True)

    imgs, reports = [], []
    for study_id, text in iter_reports(args.limit):
        pngs = list(images_root.glob(f"**/{study_id}*.png"))
        if not pngs:
            continue
        img_path = pngs[0]
        try:
            Image.open(img_path)
        except Exception:
            continue
        imgs.append(str(img_path))
        reports.append(text)
    datasets.Dataset.from_dict({"image_path": imgs, "report": reports}).to_parquet(args.out / "dataset.parquet")
    print(f"Saved {len(imgs)} samples to {args.out}")

if __name__ == "__main__":
    main()
