"""Clienți HTTP subțiri pentru fiecare container backend.

Fiecare funcție ridică BackendError cu un mesaj lizibil atunci când
backend-ul nu este accesibil sau răspunde cu o eroare, astfel încât
interfața Gradio poate afișa un avertisment clar în loc de un stack trace.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

try:  # import de tip pachet (python -m app.main)
    from .config import (
        API_TIMEOUT_SECONDS,
        DEEPDESK_AUDIO_API_KEY,
        DEEPDESK_AUDIO_URL,
        DEEPDESK_IMAGES_URL,
        GEND_URL,
        HEALTH_TIMEOUT_SECONDS,
        RPPG_URL,
    )
except ImportError:  # import de tip script (python app/main.py)
    from config import (  # type: ignore
        API_TIMEOUT_SECONDS,
        DEEPDESK_AUDIO_API_KEY,
        DEEPDESK_AUDIO_URL,
        DEEPDESK_IMAGES_URL,
        GEND_URL,
        HEALTH_TIMEOUT_SECONDS,
        RPPG_URL,
    )


class BackendError(Exception):
    """Eroare de comunicare cu un backend, cu mesaj prietenos pentru UI."""


_TIMEOUT = httpx.Timeout(connect=10.0, read=API_TIMEOUT_SECONDS, write=600.0, pool=10.0)
_FILES_MIME = "application/octet-stream"


def _post_file(
    service: str,
    url: str,
    field: str,
    path: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """POST multipart cu un fișier; returnează JSON-ul sau ridică BackendError."""
    src = Path(path)
    if not src.is_file():
        raise BackendError(f"{service}: fișierul încărcat nu a fost găsit pe disc.")
    try:
        with src.open("rb") as fh:
            files = {field: (src.name, fh, _FILES_MIME)}
            resp = httpx.post(url, files=files, data=data, headers=headers, timeout=_TIMEOUT)
    except httpx.ConnectError as exc:
        raise BackendError(
            f"{service}: nu se poate conecta la {url}. Este containerul pornit? ({exc})"
        ) from exc
    except httpx.ReadTimeout as exc:
        raise BackendError(
            f"{service}: timeout — analiza a durat mai mult de {API_TIMEOUT_SECONDS:.0f}s. "
            "Creșteți API_TIMEOUT_SECONDS sau folosiți un clip mai scurt."
        ) from exc
    if resp.status_code >= 400:
        detail = ""
        try:
            payload = resp.json()
            detail = str(payload.get("detail") or payload)[:400]
        except Exception:
            detail = resp.text[:400]
        raise BackendError(f"{service}: HTTP {resp.status_code} — {detail}")
    try:
        return resp.json()
    except Exception as exc:
        raise BackendError(f"{service}: răspuns invalid (non-JSON): {exc}") from exc


def _get_json(service: str, url: str, timeout: float) -> Dict[str, Any]:
    try:
        resp = httpx.get(url, timeout=timeout)
    except httpx.ConnectError as exc:
        raise BackendError(f"{service}: nu se poate conecta la {url}. ({exc})") from exc
    if resp.status_code >= 400:
        raise BackendError(f"{service}: HTTP {resp.status_code}")
    return resp.json()


# ============================================================================
# Starea serviciilor (tab-ul Showcase)
# ============================================================================

_HEALTH_ENDPOINTS = [
    ("deepdesk-audio", DEEPDESK_AUDIO_URL, "/"),
    ("deepdesk-images-video", DEEPDESK_IMAGES_URL, "/health"),
    ("gend-app", GEND_URL, "/api/v1/health"),
    ("clinical-vitals-api", RPPG_URL, "/health"),
]


def _probe_one(service: str, base_url: str, path: str) -> Dict[str, Any]:
    try:
        payload = _get_json(service, base_url.rstrip("/") + path, HEALTH_TIMEOUT_SECONDS)
        return {"service": service, "url": base_url, "online": True, "info": payload}
    except Exception as exc:
        return {"service": service, "url": base_url, "online": False, "info": {"error": str(exc)}}


def service_health() -> List[Dict[str, Any]]:
    """Interoghează în paralel starea tuturor backend-urilor."""
    with ThreadPoolExecutor(max_workers=len(_HEALTH_ENDPOINTS)) as pool:
        futures = [pool.submit(_probe_one, s, u, p) for s, u, p in _HEALTH_ENDPOINTS]
        return [f.result() for f in futures]


# ============================================================================
# DeepDESK — imagini și video (port 7860)
# ============================================================================

def deepdesk_image_universal(path: str) -> Dict[str, Any]:
    """UniversalFakeDetect: imagine → REAL / generată de AI."""
    return _post_file(
        "DeepDESK imagini",
        DEEPDESK_IMAGES_URL.rstrip("/") + "/api/v1/images/universal-fake-detect",
        "file",
        path,
    )


def deepdesk_frame_evit(path: str) -> Dict[str, Any]:
    """Efficient ViT pe un cadru unic → REAL / face-swap."""
    return _post_file(
        "DeepDESK cadre",
        DEEPDESK_IMAGES_URL.rstrip("/") + "/api/v1/images/video-deepfake-efficientvit",
        "file",
        path,
    )


def deepdesk_video_evit(path: str) -> Dict[str, Any]:
    """Efficient ViT pe videoclip: 30 de cadre echidistante."""
    return _post_file(
        "DeepDESK video",
        DEEPDESK_IMAGES_URL.rstrip("/") + "/api/v1/videos/video-deepfake-efficientvit",
        "file",
        path,
    )


# ============================================================================
# DeepDESK — audio (port 8000, autentificare X-API-KEY)
# ============================================================================

def deepdesk_audio(path: str, api_key: str) -> Dict[str, Any]:
    """wav2vec 2.0: REAL / FAKE-DEEPFAKE cu nivel de încredere."""
    key = (api_key or "").strip() or DEEPDESK_AUDIO_API_KEY.strip()
    if not key:
        raise BackendError(
            "DeepDESK audio: lipsește cheia API. Setați variabila de mediu "
            "DEEPDESK_AUDIO_API_KEY sau introduceți cheia în interfață."
        )
    return _post_file(
        "DeepDESK audio",
        DEEPDESK_AUDIO_URL.rstrip("/") + "/predict",
        "file",
        path,
        headers={"X-API-KEY": key},
    )


# ============================================================================
# GenD (port 7860 în container, API la /api/v1)
# ============================================================================

GEND_DEFAULT_MODELS = [
    "yermandy/GenD_CLIP_L_14",
    "yermandy/GenD_PE_L",
    "yermandy/GenD_DINOv3_L",
]
GEND_GRAPH_STYLES = ["Organic", "Radar 12 sectors", "None"]


def gend_models() -> Dict[str, Any]:
    """Lista modelelor GenD disponibile pe backend."""
    return _get_json("GenD", GEND_URL.rstrip("/") + "/api/v1/models", HEALTH_TIMEOUT_SECONDS)


def gend_analyze(
    path: str,
    *,
    model_source: str = "Hugging Face",
    hf_model: str = "yermandy/GenD_CLIP_L_14",
    local_ckpt: str = "",
    face_thresh: float = 0.5,
    scale: float = 1.3,
    target_size: int = -1,
    max_faces: int = -1,
    stride: int = 1,
    max_frames: int = -1,
    graph_style: str = "Organic",
    legend: bool = True,
    verdict_threshold: float = 0.5,
    include_base64: bool = False,
) -> Dict[str, Any]:
    """Analiză GenD imagine/videoclip; scoruri, verdict, grafic."""
    data = {
        "model_source": model_source,
        "hf_model": hf_model,
        "local_ckpt": local_ckpt,
        "face_thresh": str(face_thresh),
        "scale": str(scale),
        "target_size": str(int(target_size)),
        "max_faces": str(int(max_faces)),
        "stride": str(int(stride)),
        "max_frames": str(int(max_frames)),
        "graph_style": graph_style,
        "legend": "true" if legend else "false",
        "verdict_threshold": str(verdict_threshold),
        "include_base64": "true" if include_base64 else "false",
    }
    return _post_file(
        "GenD",
        GEND_URL.rstrip("/") + "/api/v1/analyze",
        "file",
        path,
        data=data,
    )


def gend_download_file(relative_url: str, dest_dir: str) -> str:
    """Descarcă un fișier generat de GenD (media adnotată) în dest_dir."""
    url = GEND_URL.rstrip("/") + relative_url
    try:
        resp = httpx.get(url, timeout=_TIMEOUT)
    except httpx.HTTPError as exc:
        raise BackendError(f"GenD: descărcarea {url} a eșuat ({exc})") from exc
    if resp.status_code >= 400:
        raise BackendError(f"GenD: HTTP {resp.status_code} la descărcarea {url}")
    dest = Path(dest_dir) / Path(relative_url).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return str(dest)


def save_data_uri_png(data_uri: str, dest_dir: str, stem: str) -> Optional[str]:
    """Decodează un PNG furnizat ca data-URI (base64) într-un fișier local."""
    if not data_uri or not data_uri.startswith("data:image"):
        return None
    try:
        raw = data_uri.split(",", 1)[1]
        dest = Path(dest_dir) / f"{stem}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base64.b64decode(raw))
        return str(dest)
    except Exception as exc:
        raise BackendError(f"Nu am putut decoda graficul base64: {exc}") from exc


# ============================================================================
# simple_mcd_rppg — clinical-vitals-api (port 7861)
# ============================================================================

def rppg_predict(path: str) -> Dict[str, Any]:
    """Puls (BPM), forma de undă rPPG, grafic și 9 indicatori vitali."""
    return _post_file(
        "MCD-RPPG",
        RPPG_URL.rstrip("/") + "/predict",
        "video",
        path,
    )
