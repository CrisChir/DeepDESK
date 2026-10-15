import gradio as gr
from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile, status
from fastapi.security.api_key import APIKeyHeader

from .config import ALLOWED_EXTENSIONS, API_KEY, API_KEY_NAME, MAX_UPLOAD_MB
from .inference import predict
from .ui import demo

app = FastAPI(title="Secure Deepfake Detection API")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )


@app.get("/")
async def home():
    return {
        "message": "API is Online. Use /docs for the API or /gradio for the web UI.",
    }


app = gr.mount_gradio_app(app, demo, path="/gradio")


@app.post("/predict", dependencies=[Depends(get_api_key)])
async def classify_audio(file: UploadFile = File(...)) -> dict:
    if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Invalid file type. Upload WAV, MP3 or FLAC.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file received.")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.")
    try:
        return predict(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
