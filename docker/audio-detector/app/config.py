import os

import torch

API_KEY = os.environ.get("API_KEY", "")
API_KEY_NAME = "X-API-KEY"

MODEL_ID = os.environ.get("MODEL_ID", "Bgeorge/deepfake-audio-detector-wav2vec2")
USE_LOCAL_WEIGHTS = os.environ.get("USE_LOCAL_WEIGHTS", "false").lower() == "true"
LOCAL_CHECKPOINT_PATH = os.environ.get("MODEL_PATH", "/models/fine_tuned")
MODEL_PATH = LOCAL_CHECKPOINT_PATH if USE_LOCAL_WEIGHTS else MODEL_ID

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if os.environ.get("DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

SAMPLING_RATE = 16_000
MAX_LENGTH = 16_000
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "20"))
ALLOWED_EXTENSIONS = (".wav", ".mp3", ".flac")
LABELS = ["REAL", "FAKE/DEEPFAKE"]
