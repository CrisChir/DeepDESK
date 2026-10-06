# DeepDESK Docker Container — Usage Guide

A single Docker container packaging two deepfake detectors with both an interactive
Gradio web interface and a REST API:

| Detector | Task | Paper | Code origin (MIT) |
|---|---|---|---|
| UniversalFakeDetect (CLIP ViT-L/14 + linear head) | AI-generated images (GAN, diffusion) | Ojha, Li & Lee, CVPR 2023 ([arXiv:2302.10174](https://arxiv.org/abs/2302.10174)) | [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) |
| Efficient ViT (EfficientNet-B0 + ViT) | Face-swap deepfakes in videos (DFDC) | Coccomini et al., ICIAP 2021 ([arXiv:2107.02612](https://arxiv.org/abs/2107.02612)) | [davide-coccomini/...-Video-Deepfake-Detection](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection) |

Both models run inside one process; the container auto-detects CUDA and falls back to CPU.

---

## 1. Directory layout

```
docker/
├── Dockerfile          # CPU image (default)
├── Dockerfile.gpu      # CUDA 12.1 runtime image
├── requirements.txt
├── .dockerignore
└── app/
    ├── main.py                 # FastAPI + Gradio server
    ├── detectors.py            # loading, preprocessing, inference
    └── models/
        ├── clip_univfd.py      # UniversalFakeDetect model (from upstream)
        └── efficient_vit.py    # Efficient ViT model (from upstream)
```

## 2. Build

```bash
cd docker
docker build -t deepdesk:latest .
```

GPU image (requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) on the host):

```bash
docker build -f Dockerfile.gpu -t deepdesk:gpu .
```

## 3. Run

```bash
docker run --rm -p 7860:7860 -v deepdesk-weights:/data/weights deepdesk:latest
```

- `-v deepdesk-weights:/data/weights` persists the model checkpoints across restarts
  (on first start the container downloads approximately 1.0 GB of weights; without the
  volume they are re-downloaded on every restart).
- GPU:

```bash
docker run --rm --gpus all -p 7860:7860 -v deepdesk-weights:/data/weights deepdesk:gpu
```

Startup order: weights download → CLIP/EfficientNet backbone load → both detectors ready.
First start takes a few minutes; subsequent starts are fast with a mounted volume.

### Endpoints after startup

| URL | Description |
|---|---|
| `http://localhost:7860/` | Gradio web interface (three tabs) |
| `http://localhost:7860/docs` | Interactive OpenAPI/Swagger documentation |
| `http://localhost:7860/health` | Health check (device + loaded models) |

## 4. Web interface

Open `http://localhost:7860/`. Three tabs:

1. Image — AI-generated (UniversalFakeDetect): upload or capture an image, receive
   REAL/FAKE probabilities.
2. Frame — face-swap (Efficient ViT): same, for a single video frame; the detector crops
   the face first.
3. Video — face-swap (Efficient ViT): upload a video; 30 equidistant frames are scored and
   averaged into a video-level verdict, with a per-frame score chart.

## 5. REST API

All API endpoints accept `multipart/form-data` file uploads and return JSON. Verdicts use
`label` (REAL/FAKE) with `threshold = 0.5` on the sigmoid probability `p_fake`.

### 5.1 Health

```bash
curl http://localhost:7860/health
```

```json
{"status": "ok", "device": "cuda", "models": ["universal-fake-detect", "video-deepfake-efficientvit"]}
```

### 5.2 AI-generated image (UniversalFakeDetect)

```bash
curl -s -X POST http://localhost:7860/api/v1/images/universal-fake-detect \
  -F "file=@photo.png"
```

```json
{
  "model": "universal-fake-detect",
  "p_fake": 0.9312,
  "p_real": 0.0688,
  "label": "FAKE",
  "threshold": 0.5
}
```

### 5.3 Single frame, face-swap (Efficient ViT)

```bash
curl -s -X POST http://localhost:7860/api/v1/images/video-deepfake-efficientvit \
  -F "file=@frame.jpg"
```

```json
{
  "model": "video-deepfake-efficientvit",
  "p_fake": 0.8721,
  "p_real": 0.1279,
  "label": "FAKE",
  "face_detected": true,
  "threshold": 0.5
}
```

### 5.4 Video, face-swap (Efficient ViT)

```bash
curl -s -X POST http://localhost:7860/api/v1/videos/video-deepfake-efficientvit \
  -F "file=@interview.mp4"
```

```json
{
  "model": "video-deepfake-efficientvit",
  "p_fake": 0.7903,
  "p_real": 0.2097,
  "label": "FAKE",
  "face_detected": true,
  "frames_scored": 30,
  "frame_scores": [0.81, 0.79, "..."],
  "threshold": 0.5
}
```

### 5.5 Python client example

```python
import requests

r = requests.post(
    "http://localhost:7860/api/v1/images/universal-fake-detect",
    files={"file": open("photo.png", "rb")},
)
print(r.json()["label"], r.json()["p_fake"])
```

### 5.6 HTTP status codes

| Code | Meaning |
|---|---|
| 200 | Success |
| 400 | Unreadable image or video file |
| 422 | Missing file field |

## 6. Model weights

Downloaded automatically on first start into `/data/weights` (override with the
`WEIGHTS_DIR` environment variable):

| File | Size | Source |
|---|---|---|
| `fc_weights.pth` (linear head) | ~3 KB | https://huggingface.co/siddharthksah/deepsafe-weights/resolve/main/universalfakedetect/fc_weights.pth |
| `efficient_vit.pth` | ~54 MB | http://datino.isti.cnr.it/efficientvit_deepfake/efficient_vit.pth (fallback: [Google Drive](https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1)) |
| CLIP ViT-L/14 backbone | ~890 MB | downloaded automatically by the OpenAI CLIP package |
| EfficientNet-B0 backbone | ~20 MB | downloaded automatically by `efficientnet_pytorch` |

Offline hosts: pre-download the files, place them in a directory, and mount it as
`-v /path/to/weights:/data/weights`; the container skips downloads when the files exist.

## 7. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `WEIGHTS_DIR` | `/data/weights` | Checkpoint directory |

## 8. Performance notes

- CPU: UniversalFakeDetect processes one image in roughly 0.5–2 s; the video endpoint
  takes several minutes for a 30-frame video. A GPU is strongly recommended for videos.
- GPU: both detectors run in a few tens of milliseconds per frame.
- Requests run sequentially per worker; increase uvicorn workers only with a GPU and
  sufficient memory (each worker loads its own copy of both models).

## 9. Limitations and responsible use

- Scores are sigmoid probabilities, not calibrated confidence; values near 0.5 are
  uncertain, and accuracy is well below 100 percent on every benchmark domain.
- UniversalFakeDetect was trained on ProGAN images; it generalizes across generator
  families but is degraded by heavy JPEG compression or blur.
- Efficient ViT was trained on DFDC face swaps; faceless content is classified on the
  full frame and is less reliable.
- These detectors provide investigative signals, not definitive verdicts. Do not use
  them as sole evidence for the authenticity of visual content, and do not base
  consequential decisions on a single score.

## 10. Licenses and attribution

Both underlying codebases are MIT licensed and are credited here:

- UniversalFakeDetect — MIT License, Copyright (c) 2025 Wisconsin AI and Vision Lab (WAIV).
  [LICENSE](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/LICENSE)
- Efficient ViT video deepfake detection — MIT License, Copyright (c) Davide Coccomini.
  [LICENSE](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/blob/main/LICENSE)

The model definition files in `app/models/` are derived from those repositories; the
derivation is documented in each file's header. If you use the models in research, please
cite the respective papers (BibTeX entries in the notebooks' README).
