"""Inspectarea media încărcat: rezoluție, FPS, durată, dimensiune."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _suffix(path: str) -> str:
    return Path(path).suffix.lower() if path else ""


def is_image(path: Optional[str]) -> bool:
    return bool(path) and _suffix(path) in IMAGE_EXTS


def is_video(path: Optional[str]) -> bool:
    return bool(path) and _suffix(path) in VIDEO_EXTS


def image_info(path: str) -> Dict[str, Any]:
    """Rezoluția și mărimea unei imagini (PIL)."""
    from PIL import Image

    with Image.open(path) as img:
        width, height = img.size
        mode = img.mode
    info = {
        "type": "image",
        "width": width,
        "height": height,
        "resolution": f"{width} × {height}",
        "size_mb": Path(path).stat().st_size / (1024 * 1024),
        "extra": {"format": Path(path).suffix.upper().lstrip("."), "mode": mode},
    }
    return info


def video_info(path: str) -> Dict[str, Any]:
    """Rezoluția, FPS, numărul de cadre și durata unui videoclip (OpenCV)."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError("nu a putut fi deschis cu OpenCV")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    duration = frames / fps if fps > 0 else 0.0
    return {
        "type": "video",
        "width": width,
        "height": height,
        "resolution": f"{width} × {height}",
        "fps": round(fps, 2),
        "frames": frames,
        "duration_s": round(duration, 1),
        "size_mb": Path(path).stat().st_size / (1024 * 1024),
        "extra": {"format": Path(path).suffix.upper().lstrip(".")},
    }


def inspect_media(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Detectează tipul media și returnează informațiile complete."""
    if not path:
        return None
    if is_image(path):
        return image_info(path)
    if is_video(path):
        return video_info(path)
    return {
        "type": "file",
        "resolution": "n/a",
        "size_mb": Path(path).stat().st_size / (1024 * 1024),
        "extra": {"format": Path(path).suffix.upper().lstrip(".") or "n/a"},
    }


def media_info_markdown(path: Optional[str]) -> str:
    """Bloc Markdown cu proprietățile media încărcat (rezoluție, FPS etc.)."""
    if not path:
        return "_Nu a fost încărcat niciun fișier._"
    try:
        info = inspect_media(path)
    except Exception as exc:
        return f"_Informațiile media nu au putut fi citite: {exc}_"
    if info is None:
        return "_Tip de fișier nesuportat._"
    lines = []
    lines.append(f"**Fișier:** `{Path(path).name}`")
    if info.get("resolution") and info["resolution"] != "n/a":
        lines.append(f"**Rezoluție:** {info['resolution']} px")
    if info["type"] == "video":
        lines.append(f"**FPS:** {info['fps']}")
        lines.append(f"**Cadre:** {info['frames']}")
        lines.append(f"**Durată:** {info['duration_s']} s")
    lines.append(f"**Dimensiune:** {info['size_mb']:.1f} MB")
    fmt = info.get("extra", {}).get("format")
    if fmt:
        lines.append(f"**Format:** {fmt}")
    return "\n\n".join(lines)
