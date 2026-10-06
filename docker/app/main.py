"""DeepDESK server: REST API (FastAPI) + web interface (Gradio) in one container.

Models:
- UniversalFakeDetect (images): https://github.com/WisconsinAIVision/UniversalFakeDetect
- Efficient ViT (videos): https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection

Run: uvicorn app.main:app --host 0.0.0.0 --port 7860
- Web interface: http://<host>:7860/
- API docs (Swagger): http://<host>:7860/docs
"""
import io
import os
import tempfile

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from .detectors import (
    UniversalFakeDetector,
    VideoDeepfakeDetector,
    download_weights,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
download_weights()

univfd = UniversalFakeDetector(DEVICE)
evit = VideoDeepfakeDetector(DEVICE)
print("Both detectors loaded.")

# ==================== FastAPI application ====================

app = FastAPI(
    title="DeepDESK — Deepfake Detection API",
    description=(
        "REST API for two deepfake detectors: UniversalFakeDetect "
        "(AI-generated images, CVPR 2023) and Efficient ViT "
        "(video deepfakes, ICIAP 2021). Interactive documentation "
        "is served automatically at /docs."
    ),
    version="1.0.0",
)


def _read_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file.")


def _read_video(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        return tmp.name


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models": ["universal-fake-detect", "video-deepfake-efficientvit"],
    }


@app.post("/api/v1/images/universal-fake-detect")
def api_universal_image(file: UploadFile = File(...)):
    """Classify an image as REAL or AI-generated (GAN/diffusion families)."""
    image = _read_image(file)
    p_fake = univfd.predict(image)
    return {
        "model": "universal-fake-detect",
        "p_fake": p_fake,
        "p_real": 1.0 - p_fake,
        "label": "FAKE" if p_fake > 0.5 else "REAL",
        "threshold": 0.5,
    }


@app.post("/api/v1/images/video-deepfake-efficientvit")
def api_evit_image(file: UploadFile = File(...)):
    """Classify a single frame as REAL or face-swapped (DFDC training domain)."""
    image = np.array(_read_image(file))
    p_fake, face_found = evit.predict_image(image)
    return {
        "model": "video-deepfake-efficientvit",
        "p_fake": p_fake,
        "p_real": 1.0 - p_fake,
        "label": "FAKE" if p_fake > 0.5 else "REAL",
        "face_detected": bool(face_found),
        "threshold": 0.5,
    }


@app.post("/api/v1/videos/video-deepfake-efficientvit")
def api_evit_video(file: UploadFile = File(...)):
    """Classify a video: 30 equidistant frames, per-frame scores averaged."""
    path = _read_video(file)
    try:
        p_fake, face_found, frame_scores = evit.predict_video(path)
    finally:
        os.unlink(path)
    if p_fake is None:
        raise HTTPException(status_code=400, detail="Could not read any frames.")
    return {
        "model": "video-deepfake-efficientvit",
        "p_fake": p_fake,
        "p_real": 1.0 - p_fake,
        "label": "FAKE" if p_fake > 0.5 else "REAL",
        "face_detected": bool(face_found),
        "frames_scored": len(frame_scores),
        "frame_scores": frame_scores,
        "threshold": 0.5,
    }


# ==================== Gradio interface ====================

def ui_universal_image(image):
    if image is None:
        return {"REAL": 0.0, "FAKE": 0.0}, ""
    p_fake = univfd.predict(Image.fromarray(image) if isinstance(image, np.ndarray) else image)
    note = "Scores near 0.5 indicate uncertainty. Trained on ProGAN; generalizes across generators."
    return {"REAL": 1.0 - p_fake, "FAKE": p_fake}, note


def ui_evit_image(image):
    if image is None:
        return {"REAL": 0.0, "FAKE": 0.0}, ""
    p_fake, found = evit.predict_image(image)
    note = "" if found else "No face detected; the full frame was classified. "
    note += "Scores near 0.5 indicate uncertainty. Trained on DFDC face crops."
    return {"REAL": 1.0 - p_fake, "FAKE": p_fake}, note


def ui_evit_video(video_path):
    if video_path is None:
        return {"REAL": 0.0, "FAKE": 0.0}, None, ""
    p_fake, found, frame_scores = evit.predict_video(video_path)
    if p_fake is None:
        return {"REAL": 0.0, "FAKE": 0.0}, None, "Could not read any frames from the video."
    df = pd.DataFrame({
        "frame": list(range(1, len(frame_scores) + 1)),
        "p_fake": frame_scores,
    })
    note = "" if found else "No face detected; full frames were classified. "
    note += f"Video score = mean of {len(frame_scores)} frame scores."
    return {"REAL": 1.0 - p_fake, "FAKE": p_fake}, df, note


with gr.Blocks(title="DeepDESK — Deepfake Detection") as demo:
    gr.Markdown(
        """
        # DeepDESK — Deepfake Detection
        Two research-grade detectors with a REST API and this web interface.
        - **Universal Fake Detector** (CVPR 2023): AI-generated images (GANs, diffusion).
        - **Efficient ViT** (ICIAP 2021): face-swap deepfakes in videos (DFDC).
        A probability above 0.5 for FAKE means the content is predicted to be manipulated.
        """
    )
    with gr.Tab("Image — AI-generated (UniversalFakeDetect)"):
        with gr.Row():
            with gr.Column():
                in_img_ufd = gr.Image(type="numpy", label="Input image",
                                       sources=["upload", "webcam"])
                btn_ufd = gr.Button("Classify image", variant="primary")
            with gr.Column():
                out_ufd = gr.Label(label="Verdict", num_top_classes=2)
                note_ufd = gr.Markdown()
    with gr.Tab("Frame — face-swap (Efficient ViT)"):
        with gr.Row():
            with gr.Column():
                in_img_evit = gr.Image(type="numpy", label="Input frame",
                                       sources=["upload", "webcam"])
                btn_evit_img = gr.Button("Classify frame", variant="primary")
            with gr.Column():
                out_evit_img = gr.Label(label="Verdict", num_top_classes=2)
                note_evit_img = gr.Markdown()
    with gr.Tab("Video — face-swap (Efficient ViT)"):
        in_vid = gr.Video(label="Input video")
        btn_vid = gr.Button("Classify video", variant="primary")
        out_vid = gr.Label(label="Verdict (video level)", num_top_classes=2)
        chart_vid = gr.LinePlot(x="frame", y="p_fake", y_lim=[0, 1],
                                title="Per-frame p(fake)", height=300)
        note_vid = gr.Markdown()

    btn_ufd.click(ui_universal_image, inputs=in_img_ufd, outputs=[out_ufd, note_ufd])
    btn_evit_img.click(ui_evit_image, inputs=in_img_evit, outputs=[out_evit_img, note_evit_img])
    btn_vid.click(ui_evit_video, inputs=in_vid, outputs=[out_vid, chart_vid, note_vid])

gr.mount_gradio_app(app, demo, path="/")
