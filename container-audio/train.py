import os

import numpy as np
import pandas as pd
import torch
import librosa
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)

MODEL_ID = os.environ.get("BASE_MODEL_ID", "MelodyMachine/Deepfake-audio-detection-V2")
DATASET_PATH = os.environ.get("DATASET_PATH", "./audio-deepfake-detection-dataset")
OUTPUT_DIR = "./results"
SAVE_DIR = "./fine_tuned_deepfake_detector"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if os.environ.get("DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)


def prepare_dataset(path: str) -> Dataset:
    data = []
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        label = 0 if folder_name == "real_samples" else 1
        for f in os.listdir(folder_path):
            if f.lower().endswith((".wav", ".mp3")):
                data.append({"file_path": os.path.join(folder_path, f), "label": label})
    return Dataset.from_pandas(pd.DataFrame(data))


def preprocess_function(examples):
    audio_arrays = [librosa.load(p, sr=16_000)[0] for p in examples["file_path"]]
    inputs = processor(
        audio_arrays,
        sampling_rate=16_000,
        max_length=16_000,
        truncation=True,
        padding="max_length",
    )
    inputs["labels"] = examples["label"]
    return inputs


def main():
    raw_dataset = prepare_dataset(DATASET_PATH)
    splits = raw_dataset.train_test_split(test_size=0.2, seed=42)
    encoded = splits.map(
        preprocess_function,
        batched=True,
        batch_size=16,
        remove_columns=splits["train"].column_names,
    )

    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID, num_labels=2).to(DEVICE)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        num_train_epochs=8,
        remove_unused_columns=False,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        processing_class=processor,
        compute_metrics=lambda p: {
            "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))
        },
    )

    trainer.train()
    trainer.save_model(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)
    print(f"Model saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
