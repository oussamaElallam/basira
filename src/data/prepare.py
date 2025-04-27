"""Download and prepare radiology datasets for Basira.
Currently supports: IU-Xray (default).  MIMIC-CXR stub.
Usage:
    python src/data/prepare.py --dataset iu_xray --out data/iu_xray
"""
import argparse
from pathlib import Path

import datasets
from PIL import Image

def prepare_iu_xray(out_dir: Path):
    ds = datasets.load_dataset("iu_xray", split="train+validation+test")
    out_dir.mkdir(parents=True, exist_ok=True)
    images, reports = [], []
    for row in ds:
        # only take PA view first image
        img = row["image"][0] if isinstance(row["image"], list) else row["image"]
        findings = row["findings"].strip()
        if not findings:
            continue
        # save image
        img_path = out_dir / f"{row['uid']}.png"
        img.resize((224, 224)).save(img_path)
        images.append(str(img_path))
        reports.append(findings)
    datasets.Dataset.from_dict({"image_path": images, "report": reports}).to_parquet(out_dir / "dataset.parquet")
    print(f"Saved {len(images)} samples to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iu_xray", "mimic_cxr"], default="iu_xray")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset == "iu_xray":
        prepare_iu_xray(args.out)
    else:
        raise NotImplementedError("MIMIC-CXR prep coming soon.")

if __name__ == "__main__":
    main()
