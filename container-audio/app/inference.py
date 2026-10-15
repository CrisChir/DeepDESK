import io
from functools import lru_cache

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from .config import DEVICE, LABELS, MAX_LENGTH, MODEL_PATH, SAMPLING_RATE


@lru_cache(maxsize=1)
def load_model():
    processor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    return processor, model


def preprocess_audio(audio_bytes: bytes):
    processor, _ = load_model()
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLING_RATE)
    if len(audio) > MAX_LENGTH:
        audio = audio[:MAX_LENGTH]
    else:
        audio = np.pad(audio, (0, MAX_LENGTH - len(audio)))
    inputs = processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    return {k: v.to(DEVICE) for k, v in inputs.items()}


def predict(audio_bytes: bytes) -> dict:
    _, model = load_model()
    inputs = preprocess_audio(audio_bytes)
    with torch.no_grad():
        logits = model(**inputs).logits
        scores = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1).item()
    return {
        "prediction": LABELS[prediction],
        "confidence": f"{scores[0][prediction].item():.2%}",
    }
