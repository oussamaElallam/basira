"""LoRA fine-tuning script for Basira using Paligemma2-3B.
Minimal: single-GPU, resume-friendly.
"""
import argparse
from pathlib import Path
import torch
import datasets
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoProcessor, TrainingArguments, Trainer

MODEL_NAME = "google/paligemma2-3b"

def load_data(path: Path):
    ds = datasets.Dataset.from_parquet(str(path / "dataset.parquet"))
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer

    def preprocess(batch):
        image = Image.open(batch["image_path"]).convert("RGB")
        batch["pixel_values"] = processor(image, return_tensors="pt").pixel_values[0]
        batch["labels"] = tokenizer(batch["report"], truncation=True, max_length=256).input_ids
        return batch
    ds = ds.map(preprocess, batched=False)
    return ds, tokenizer, processor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/basira-lora"))
    parser.add_argument("--stream", action="store_true", help="Use streaming dataset (no parquet)")
    args = parser.parse_args()

    if args.stream:
        ds_stream = datasets.load_dataset("MITAI/mimic_cxr_reports", split="train", streaming=True)
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        tokenizer = processor.tokenizer
        def gen():
            for row in ds_stream:
                txt = row.get("findings") or ""
                img_path = row.get("image_path") or ""
                if not txt or not img_path:
                    continue
                yield {"image_path": img_path, "report": txt}
        ds = datasets.Dataset.from_generator(gen)
    else:
        ds, tokenizer, processor = load_data(args.data)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
    model = get_peft_model(model, peft_cfg)

    training_args = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=ds)
    trainer.train()
    model.save_pretrained(args.out / "final")

if __name__ == "__main__":
    main()
