"""Evaluate Basira checkpoint on IU-Xray test split.
Metrics: BLEU-4, ROUGE-L, CIDEr; plus CheXbert F1 if model outputs 14-label vector.
Usage:
    python eval/evaluate.py --ckpt outputs/basira-lora/final --data data/iu_xray
"""
import argparse
from pathlib import Path

import evaluate as hf_eval
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
from tqdm import tqdm

BLEU = hf_eval.load("bleu")
ROUGE = hf_eval.load("rouge")
CIDEr = hf_eval.load("cider")


def load_test(data_dir: Path) -> Dataset:
    return Dataset.from_parquet(Path(data_dir) / "dataset.parquet").train_test_split(test_size=0.2, seed=42)["test"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    ds = load_test(args.data)
    processor = AutoProcessor.from_pretrained(args.ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt, device_map="auto")

    preds, refs = [], []
    for row in tqdm(ds, desc="Generating"):
        image = processor(image_path=row["image_path"], return_tensors="pt").pixel_values.to(model.device)
        output = model.generate(image, max_new_tokens=128)[0]
        text = processor.tokenizer.decode(output, skip_special_tokens=True)
        preds.append(text.lower())
        refs.append({"reference": row["report"].lower()})

    bleu = BLEU.compute(predictions=preds, references=[[r["reference"]] for r in refs])["bleu"]
    rouge = ROUGE.compute(predictions=preds, references=[r["reference"] for r in refs])["rougeL"]
    cider = CIDEr.compute(predictions=preds, references=[r["reference"] for r in refs])["cider"]

    print("BLEU-4:", bleu)
    print("ROUGE-L:", rouge)
    print("CIDEr:", cider)


if __name__ == "__main__":
    main()
