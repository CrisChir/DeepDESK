import os

import gradio as gr

from .inference import predict

GRADIO_USER = os.environ.get("GRADIO_USER", "")
GRADIO_PASSWORD = os.environ.get("GRADIO_PASSWORD", "")
AUTH = (GRADIO_USER, GRADIO_PASSWORD) if GRADIO_USER and GRADIO_PASSWORD else None

LABELS = ["REAL", "FAKE/DEEPFAKE"]


def classify(audio_path):
    if not audio_path:
        raise gr.Error("Încărcați un fișier audio (WAV / MP3 / FLAC).")
    with open(audio_path, "rb") as f:
        result = predict(f.read())
    verdict = result["prediction"]
    confidence = result["confidence"]
    color = "#2e7d32" if verdict == "REAL" else "#c62828"
    return (
        f'<h2 style="color:{color};margin:0">{verdict}</h2>',
        f"**Confidence:** {confidence}",
    )


demo = gr.Interface(
    fn=classify,
    inputs=gr.Audio(type="filepath", label="Fișier audio (max. analizată: prima secundă)"),
    outputs=[
        gr.HTML(label="Predicție"),
        gr.Markdown(label="Detalii"),
    ],
    title="Deepfake Audio Detector",
    description="Încărcați un fișier audio pentru a determina dacă vocea este REALĂ sau generată AI (FAKE/DEEPFAKE). Model: Bgeorge/deepfake-audio-detector-wav2vec2",
    examples=None,
    allow_flagging="never",
)

demo.auth = AUTH
