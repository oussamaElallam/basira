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
    ds = datasets.load_dataset("Jyothirmai/iu-xray-dataset", split="train")
    out_dir.mkdir(parents=True, exist_ok=True)
    images, reports = [], []
    uid_counter = 0
    for row in ds:
        # dataset stores local file path string; open image
        img_field = row.get("image") or row.get("image_path")
        if isinstance(img_field, list):
            img_field = img_field[0]
        if img_field is None:
            continue
        try:
            img = Image.open(img_field).convert("RGB") if isinstance(img_field, str) else img_field.convert("RGB")
        except Exception:
            continue
        text = None
        for key in ["findings", "report", "caption", "sentence", "text"]:
            if key in row and isinstance(row[key], str) and row[key].strip():
                text = row[key].strip()
                break
        if not text:
            continue
        # save image
        uid = str(row.get("uid") or row.get("id") or uid_counter)
        img_path = out_dir / f"{uid}.png"
        img.resize((224, 224)).save(img_path)
        images.append(str(img_path))
        reports.append(text)
        uid_counter += 1
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
