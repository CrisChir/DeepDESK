"""Showcase Hub — front-end Gradio unificat pentru containerele de modele.

Interfață cu taburi care apelază exclusiv API-urile REST ale containerelor
backend (DeepDESK imagini/video, DeepDESK audio, GenD, MCD-RPPG).
Pornire: python app/main.py (sau containerul Docker).
"""

from __future__ import annotations

import base64
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

try:  # import de tip pachet (python -m app.main)
    from .catalog import catalog_cards_html
    from .clients import (
        BackendError,
        GEND_DEFAULT_MODELS,
        GEND_GRAPH_STYLES,
        deepdesk_audio,
        deepdesk_frame_evit,
        deepdesk_image_universal,
        deepdesk_video_evit,
        gend_analyze,
        gend_download_file,
        gend_models,
        rppg_predict,
        save_data_uri_png,
        service_health,
    )
    from .config import HUB_PASSWORD, HUB_USER, HOST, PORT
    from .media_info import media_info_markdown
except ImportError:  # import de tip script (python app/main.py)
    from catalog import catalog_cards_html  # type: ignore
    from clients import (  # type: ignore
        BackendError,
        GEND_DEFAULT_MODELS,
        GEND_GRAPH_STYLES,
        deepdesk_audio,
        deepdesk_frame_evit,
        deepdesk_image_universal,
        deepdesk_video_evit,
        gend_analyze,
        gend_download_file,
        gend_models,
        rppg_predict,
        save_data_uri_png,
        service_health,
    )
    from config import HUB_PASSWORD, HUB_USER, HOST, PORT  # type: ignore
    from media_info import media_info_markdown  # type: ignore


TMP_DIR = Path(tempfile.gettempdir()) / "showcase-hub"
TMP_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_FILE_TYPES = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
VIDEO_FILE_TYPES = [".mp4", ".avi", ".mov", ".mkv", ".webm"]


# ============================================================================
# Utilitare UI
# ============================================================================

def _error_md(exc: Exception) -> str:
    """Bloc Markdown de eroare, afișat în locul rezultatului."""
    return (
        "<div style='border: 1px solid #fecaca; background: #fef2f2; color: #991b1b; "
        "border-radius: 10px; padding: 12px 16px;'>"
        f"⚠️ <b>Eroare:</b> {exc}"
        "</div>"
    )


def _ok_label(p_fake: Optional[float]) -> Dict[str, float]:
    p_fake = 0.0 if p_fake is None else float(p_fake)
    p_real = max(0.0, 1.0 - p_fake)
    return {"REAL": p_real, "FAKE": p_fake}


def _target_size_choice_to_int(choice: str) -> int:
    """'Original (-1)' → -1; '224' → 224 etc."""
    choice = choice.strip()
    if choice.startswith("Original") or choice in ("", "-1"):
        return -1
    try:
        return int(float(choice.split()[0]))
    except ValueError:
        return -1


def _verdict_md(label: str, score: Optional[float], extra: str = "") -> str:
    """Rezumat verdict ca Markdown."""
    color = "#16a34a" if label.upper() == "REAL" else "#dc2626"
    score_txt = f" · p(fake) = {score:.3f}" if score is not None else ""
    md = (
        f"<span style='font-size: 1.3rem; font-weight: 700; color: {color};'>"
        f"{label}</span>{score_txt}"
    )
    if extra:
        md += f"<br/><span style='color: #6b7280; font-size: 0.9rem;'>{extra}</span>"
    return md


def _save_base64_png(data_str: Any, out_dir: Path, name: str = "plot") -> Optional[str]:
    """Decodifică un PNG transmis ca base64 (cu sau fără prefix data URI) și îl salvează local.

    Serverul rPPG returnează `ppg_plot_png_base64` ca base64 simplu, fără
    prefixul „data:image/png;base64,”; această funcție gestionează ambele formate.
    """
    try:
        s = str(data_str).strip()
        if s.startswith("data:") and "," in s:
            s = s.split(",", 1)[1]
        payload = base64.b64decode(s, validate=False)
        if not payload.startswith(b"\x89PNG"):
            return None
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.png"
        path.write_bytes(payload)
        return str(path)
    except Exception:
        return None


def _bpm_from_waveform(wave: List[Any], fs: float) -> Optional[float]:
    """BPM recalculat în hub direct din forma de undă BVP (FFT, banda 0.7–4 Hz)."""
    try:
        import numpy as np

        x = np.asarray([float(v) for v in wave], dtype=float)
        n = x.size
        if n < 10 * max(1.0, fs):
            return None
        x = x - x.mean()
        window = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(x * window))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        band = (freqs >= 0.7) & (freqs <= 4.0)
        if not band.any():
            return None
        peak_freq = freqs[band][int(np.argmax(spectrum[band]))]
        return float(peak_freq * 60.0)
    except Exception:
        return None


# ============================================================================
# Callbacks — Showcase (starea serviciilor)
# ============================================================================

def refresh_status() -> str:
    rows = []
    for svc in service_health():
        if svc["online"]:
            info = svc["info"]
            device = info.get("device") or info.get("status") or "ok"
            extra = ""
            if "models" in info:
                extra = " · " + ", ".join(info["models"])
            badge = "🟢 **Online**"
            detail = f"{device}{extra}"
        else:
            badge = "🔴 **Offline**"
            detail = str(svc["info"].get("error", "inaccesibil"))[:120]
        rows.append(f"| {svc['service']} | `{svc['url']}` | {badge} | {detail} |")
    table = (
        "| Serviciu | Adresă internă | Stare | Detalii |\n|---|---|---|---|\n"
        + "\n".join(rows)
    )
    return table


# ============================================================================
# Callbacks — DeepDESK imagini & video
# ============================================================================

def analyze_universal_image(image_path: Optional[str], progress=gr.Progress()):
    if not image_path:
        return {}, "Încărcați o imagine mai întâi."
    try:
        progress(0.3, desc="Trimit imaginea către UniversalFakeDetect…")
        r = deepdesk_image_universal(image_path)
    except BackendError as exc:
        return {}, _error_md(exc)
    note = (
        f"Scor p(fake) = {r.get('p_fake', 0):.4f} · prag {r.get('threshold', 0.5)}. "
        "Scoruri aproape de 0.5 indică incertitudine. Antrenat pe ProGAN; "
        "generalizează între generatoare, dar este degradat de compresie JPEG intensă."
    )
    return _ok_label(r.get("p_fake")), note


def analyze_evit_frame(image_path: Optional[str], progress=gr.Progress()):
    if not image_path:
        return {}, "Încărcați un cadru mai întâi."
    try:
        progress(0.3, desc="Trimit cadrul către Efficient ViT…")
        r = deepdesk_frame_evit(image_path)
    except BackendError as exc:
        return {}, _error_md(exc)
    note = "" if r.get("face_detected") else "Nu a fost detectată nicio față; a fost clasificat cadrul întreg. "
    note += (
        f"Scor p(fake) = {r.get('p_fake', 0):.4f} · prag {r.get('threshold', 0.5)}. "
        "Antrenat pe crop-uri de fețe DFDC."
    )
    return _ok_label(r.get("p_fake")), note


def analyze_evit_video(video_path: Optional[str], progress=gr.Progress()):
    empty_df = pd.DataFrame({"frame": [], "p_fake": []})
    if not video_path:
        return {}, empty_df, "Încărcați un videoclip mai întâi."
    try:
        progress(0.3, desc="Trimit videoclipul către DeepDESK (30 de cadre echidistante)…")
        r = deepdesk_video_evit(video_path)
    except BackendError as exc:
        return {}, empty_df, _error_md(exc)
    scores = r.get("frame_scores") or []
    df = pd.DataFrame({
        "frame": list(range(1, len(scores) + 1)),
        "p_fake": [float(s) for s in scores],
    })
    note = "" if r.get("face_detected") else "Nu a fost detectată nicio față; cadrele întregi au fost clasificate. "
    note += f"Scor video = media celor {r.get('frames_scored', len(scores))} de cadre."
    return _ok_label(r.get("p_fake")), df, note


# ============================================================================
# Callbacks — DeepDESK audio
# ============================================================================

def analyze_audio(audio_path: Optional[str], api_key: str):
    if not audio_path:
        return {}, "Încărcați o înregistrare sau un fișier audio mai întâi."
    try:
        r = deepdesk_audio(audio_path, api_key)
    except BackendError as exc:
        return {}, _error_md(exc)
    raw_label = str(r.get("prediction") or r.get("label") or "?")
    conf_raw = str(r.get("confidence") or "")
    try:
        conf = float(conf_raw.replace("%", "").strip()) / 100.0
    except ValueError:
        conf = 0.0
    is_fake = "FAKE" in raw_label.upper()
    label_dict = {raw_label: conf, ("REAL" if is_fake else "FAKE/DEEPFAKE"): max(0.0, 1.0 - conf)}
    note = (
        f"Verdict: **{raw_label}** cu încredere **{conf_raw or 'n/a'}**. "
        "Predicția se bazează pe prima secundă de audio (16 kHz). "
        "Performanța pe sisteme TTS never văzute nu este garantată."
    )
    return label_dict, note


# ============================================================================
# Callbacks — GenD
# ============================================================================

def gend_refresh_models() -> Tuple[Any, str]:
    try:
        payload = gend_models()
        models = payload.get("hugging_face") or GEND_DEFAULT_MODELS
        styles = payload.get("graph_styles") or GEND_GRAPH_STYLES
        info = (
            f"Backend GenD online. **Modele disponibile:** {', '.join(models)}. "
            f"Stiluri de grafic: {', '.join(styles)}."
        )
    except Exception as exc:
        models, styles = GEND_DEFAULT_MODELS, GEND_GRAPH_STYLES
        info = _error_md(BackendError(f"nu am putut citi lista de modele GenD: {exc}"))
    default_model = models[-1] if models else GEND_DEFAULT_MODELS[0]
    return gr.update(choices=models, value=default_model), info


def analyze_gend(
    file_path: Optional[str],
    model_source: str,
    hf_model: str,
    local_ckpt: str,
    face_thresh: float,
    scale: float,
    target_size_choice: str,
    max_faces: int,
    stride: int,
    max_frames: int,
    graph_style: str,
    legend: bool,
    verdict_threshold: float,
    progress=gr.Progress(),
):
    empty_plot = pd.DataFrame({"cadru": [], "p_fake": []})
    if not file_path:
        return "Încărcați o imagine sau un videoclip.", empty_plot, None, None, None
    if model_source == "Local Checkpoint" and not local_ckpt.strip():
        return "Pentru «Local Checkpoint» este necesară calea către fișierul .ckpt.", empty_plot, None, None, None
    try:
        progress(0.15, desc="Trimit fișierul către GenD…")
        r = gend_analyze(
            file_path,
            model_source=model_source,
            hf_model=hf_model,
            local_ckpt=local_ckpt.strip(),
            face_thresh=float(face_thresh),
            scale=float(scale),
            target_size=_target_size_choice_to_int(target_size_choice),
            max_faces=int(max_faces),
            stride=int(stride),
            max_frames=int(max_frames),
            graph_style=graph_style,
            legend=bool(legend),
            verdict_threshold=float(verdict_threshold),
            include_base64=True,
        )
    except BackendError as exc:
        return _error_md(exc), empty_plot, None, None, None

    verdict = r.get("verdict") or {}
    summary = _verdict_md(
        verdict.get("label", "?"),
        verdict.get("p_fake", r.get("avg_p_fake")),
        extra=(
            f"{r.get('media_type', '?')} · {r.get('num_frames', '?')} cadre · "
            f"{r.get('num_faces', '?')} fețe · medie p(fake) = "
            f"{float(r.get('avg_p_fake') or 0):.3f} · mediană = "
            f"{float(r.get('median_p_fake') or 0):.3f} · "
            f"{r.get('processing_seconds', '?')}s"
        ),
    )

    scores = r.get("p_fake_scores") or []
    plot_df = pd.DataFrame({
        "cadru": list(range(1, len(scores) + 1)),
        "p_fake": [float(s) for s in scores],
    })

    graph_path = None
    graph_block = r.get("graph") or {}
    if graph_block.get("base64_png"):
        try:
            graph_path = save_data_uri_png(graph_block["base64_png"], str(TMP_DIR / "gend"), "gend_graph")
        except BackendError:
            graph_path = None
        if graph_path is None:
            graph_path = _save_base64_png(graph_block["base64_png"], TMP_DIR / "gend", "gend_graph")

    annotated_path = None
    annotated_block = r.get("annotated_media") or {}
    if annotated_block.get("url"):
        try:
            annotated_path = gend_download_file(annotated_block["url"], str(TMP_DIR / "gend"))
        except BackendError:
            annotated_path = None

    return summary, plot_df, graph_path, annotated_path, annotated_path


# ============================================================================
# Callbacks — rPPG (puls din semnalul BVP)
# ============================================================================

def analyze_rppg(video_path: Optional[str], progress=gr.Progress()):
    empty_wave = pd.DataFrame({"timp (s)": [], "BVP": []})
    empty_vitals = pd.DataFrame(columns=["Indicator", "Valoare", "Unitate"])
    if not video_path:
        return "—", "", None, empty_wave, empty_vitals, "Încărcați un videoclip facial mai întâi."
    try:
        progress(0.2, desc="Trimit videoclipul către MCD-RPPG (poate dura câteva minute pe CPU)…")
        r = rppg_predict(video_path)
    except BackendError as exc:
        return "—", _error_md(exc), None, empty_wave, empty_vitals, ""

    fs = float(r.get("sampling_rate_fps") or 30)
    wave = r.get("ppg_waveform") or []
    wave_df = pd.DataFrame({
        "timp (s)": [i / fs for i in range(len(wave))],
        "BVP": [float(v) for v in wave],
    })

    bpm = r.get("heart_rate_bpm")
    bpm_hub = _bpm_from_waveform(wave, fs)
    hr_md = (
        f"<span style='font-size: 2rem; font-weight: 700; color: #2563eb;'>"
        f"{bpm:.1f} BPM</span><br/>"
        f"<span style='color: #6b7280; font-size: 0.9rem;'>"
        "puls estimat din semnalul rPPG (FFT pe unda BVP)</span>"
        if isinstance(bpm, (int, float))
        else "n/a"
    )

    metrics_lines = [
        f"**{r.get('frames_processed', '?')}** cadre procesate · **{fs:.1f}** FPS eșantionare",
        f"Sursă puls: `{r.get('heart_rate_source', 'stage1_ppg_fft')}`",
    ]
    if bpm_hub is not None:
        metrics_lines.append(
            f"BPM recalculat în hub din forma de undă (FFT, 0.7–4 Hz): **{bpm_hub:.1f}**"
        )
    metrics_md = "<br/>".join(metrics_lines)

    plot_path = _save_base64_png(r.get("ppg_plot_png_base64"), TMP_DIR / "rppg", "ppg_wave")

    vitals_rows = [
        {"Indicator": key, "Valoare": entry.get("value"), "Unitate": entry.get("unit")}
        for key, entry in (r.get("vitals") or {}).items()
    ]
    vitals_df = pd.DataFrame(vitals_rows, columns=["Indicator", "Valoare", "Unitate"])

    note = (
        f"{r.get('notes', '')} "
        "Rezultat de cercetare, nu un diagnostic medical. Indicatorii din secțiunea "
        "secundară sunt estimări ale modelului Stage 2, cu relevanță limitată."
    )
    return hr_md, metrics_md, plot_path, wave_df, vitals_df, note


# ============================================================================
# Callbacks — Analiză integrată video
# ============================================================================

def _run_deepdesk_video(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return deepdesk_video_evit(path), None
    except BackendError as exc:
        return None, str(exc)


def _run_gend_video(path: str, model: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return gend_analyze(
            path,
            hf_model=model,
            stride=5,
            max_frames=60,
            graph_style="Organic",
            legend=True,
        ), None
    except BackendError as exc:
        return None, str(exc)


def _run_rppg(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return rppg_predict(path), None
    except BackendError as exc:
        return None, str(exc)


def analyze_integrated(video_path: Optional[str], gend_model: str, progress=gr.Progress()):
    empty_plot = pd.DataFrame({"cadru": [], "p_fake": []})
    empty_wave = pd.DataFrame({"timp (s)": [], "BVP": []})
    empty_vitals = pd.DataFrame(columns=["Indicator", "Valoare", "Unitate"])
    if not video_path:
        return "—", empty_plot, "—", None, "—", "", None, empty_wave, empty_vitals, "Încărcați un videoclip mai întâi."

    progress(0.1, desc="Rulez în paralel toate cele 3 analizoare de video…")
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_dd = pool.submit(_run_deepdesk_video, video_path)
        f_gd = pool.submit(_run_gend_video, video_path, gend_model)
        f_rp = pool.submit(_run_rppg, video_path)
        dd_res, dd_err = f_dd.result()
        gd_res, gd_err = f_gd.result()
        rp_res, rp_err = f_rp.result()
    progress(0.95, desc="Randez rezultatele…")

    # --- DeepDESK Efficient ViT ---
    if dd_res is not None:
        scores = dd_res.get("frame_scores") or []
        dd_plot = pd.DataFrame({"cadru": list(range(1, len(scores) + 1)), "p_fake": [float(s) for s in scores]})
        dd_md = _verdict_md(dd_res.get("label", "?"), dd_res.get("p_fake"),
                            extra=f"Efficient ViT · {dd_res.get('frames_scored', len(scores))} cadre")
    else:
        dd_md, dd_plot = _error_md(BackendError(dd_err)), empty_plot

    # --- GenD ---
    gd_graph = None
    if gd_res is not None:
        verdict = gd_res.get("verdict") or {}
        gd_md = _verdict_md(
            verdict.get("label", "?"),
            verdict.get("p_fake", gd_res.get("avg_p_fake")),
            extra=f"GenD · {gd_res.get('num_faces', 0)} fețe · {gd_res.get('processing_seconds', '?')}s",
        )
        graph_block = gd_res.get("graph") or {}
        if graph_block.get("base64_png"):
            gd_graph = _save_base64_png(graph_block["base64_png"], TMP_DIR / "integrated", "gend_graph")
    else:
        gd_md, gd_graph = _error_md(BackendError(gd_err)), None

    # --- rPPG: puls din semnal + forma de undă ---
    rp_plot = None
    if rp_res is not None:
        bpm = rp_res.get("heart_rate_bpm")
        rp_hr_md = (
            f"<span style='font-size: 1.6rem; font-weight: 700; color: #2563eb;'>{bpm:.1f} BPM</span>"
            if isinstance(bpm, (int, float))
            else "n/a"
        )
        fs = float(rp_res.get("sampling_rate_fps") or 30)
        wave = rp_res.get("ppg_waveform") or []
        rp_wave = pd.DataFrame({"timp (s)": [i / fs for i in range(len(wave))], "BVP": [float(v) for v in wave]})
        bpm_hub = _bpm_from_waveform(wave, fs)
        metrics_lines = [
            f"{rp_res.get('frames_processed', '?')} cadre · {fs:.1f} FPS · "
            f"sursă: `{rp_res.get('heart_rate_source', 'stage1_ppg_fft')}`",
        ]
        if bpm_hub is not None:
            metrics_lines.append(f"BPM recalculat în hub din forma de undă: **{bpm_hub:.1f}**")
        rp_metrics = "<br/>".join(metrics_lines)
        rp_plot = _save_base64_png(rp_res.get("ppg_plot_png_base64"), TMP_DIR / "integrated", "ppg_wave")
        vitals_rows = [
            {"Indicator": k, "Valoare": v.get("value"), "Unitate": v.get("unit")}
            for k, v in (rp_res.get("vitals") or {}).items()
        ]
        rp_vitals = pd.DataFrame(vitals_rows, columns=["Indicator", "Valoare", "Unitate"])
    else:
        rp_hr_md = _error_md(BackendError(rp_err))
        rp_metrics, rp_plot, rp_wave, rp_vitals = "", None, empty_wave, empty_vitals

    note = "Fiecare secțiune afișează erori independent — un backend indisponibil nu blochează celelalte."
    return dd_md, dd_plot, gd_md, gd_graph, rp_hr_md, rp_metrics, rp_plot, rp_wave, rp_vitals, note


# ============================================================================
# Construcția interfeței
# ============================================================================

def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

    with gr.Blocks(theme=theme, title="Showcase Hub — Modele AI") as demo:
        gr.Markdown(
            """
            # 🧠 Showcase Hub — front-end unificat pentru containerele de modele
            Toate analizele de mai jos sunt executate de containerele backend
            (DeepDESK, GenD, MCD-RPPG); acest hub doar le apelează prin API-urile REST.
            """
        )

        # ------------------------------------------------------------------
        # Tab 1 — Showcase
        # ------------------------------------------------------------------
        with gr.Tab("🏠 Showcase"):
            gr.HTML(catalog_cards_html())
            with gr.Row():
                status_btn = gr.Button("🔄 Verifică starea serviciilor", variant="primary")
            status_table = gr.Markdown(refresh_status())

        # ------------------------------------------------------------------
        # Tab 2 — DeepDESK imagini & video
        # ------------------------------------------------------------------
        with gr.Tab("🖼️ DeepDESK — imagini & video"):
            gr.Markdown("### Detectorul de imagini generate de AI și de deepfake video")
            with gr.Tab("Imagini generate de AI — UniversalFakeDetect"):
                with gr.Row():
                    with gr.Column():
                        ufd_img = gr.Image(type="filepath", sources=["upload", "webcam"], label="Imagine")
                        ufd_info = gr.Markdown("_Nu a fost încărcat niciun fișier._")
                        ufd_btn = gr.Button("Clasifică imaginea", variant="primary")
                    with gr.Column():
                        ufd_label = gr.Label(label="Verdict", num_top_classes=2)
                        ufd_note = gr.Markdown()
            with gr.Tab("Cadru face-swap — Efficient ViT"):
                with gr.Row():
                    with gr.Column():
                        evit_img = gr.Image(type="filepath", sources=["upload", "webcam"], label="Cadru")
                        evit_info = gr.Markdown("_Nu a fost încărcat niciun fișier._")
                        evit_btn = gr.Button("Clasifică cadrul", variant="primary")
                    with gr.Column():
                        evit_label = gr.Label(label="Verdict", num_top_classes=2)
                        evit_note = gr.Markdown()
            with gr.Tab("Video face-swap — Efficient ViT"):
                with gr.Row():
                    with gr.Column():
                        evit_vid = gr.Video(sources=["upload"], label="Videoclip")
                        evit_vid_info = gr.Markdown("_Nu a fost încărcat niciun fișier._")
                        evit_vid_btn = gr.Button("Clasifică videoclipul", variant="primary")
                    with gr.Column():
                        evit_vid_label = gr.Label(label="Verdict (nivel videoclip)", num_top_classes=2)
                        evit_vid_chart = gr.LinePlot(
                            x="frame", y="p_fake", y_lim=[0, 1],
                            title="p(fake) per cadru", height=300,
                        )
                        evit_vid_note = gr.Markdown()

        # ------------------------------------------------------------------
        # Tab 3 — DeepDESK audio
        # ------------------------------------------------------------------
        with gr.Tab("🎙️ DeepDESK — audio"):
            gr.Markdown("### Detectorul de voce sintetică / TTS (wav2vec 2.0)")
            with gr.Row():
                with gr.Column():
                    audio_in = gr.Audio(
                        type="filepath", sources=["upload", "microphone"],
                        label="Fișier audio (WAV / MP3 / FLAC) sau înregistrare",
                    )
                    audio_key = gr.Textbox(
                        type="password",
                        label="Cheie API (X-API-KEY) — opțional dacă e setată în .env",
                        placeholder="openssl rand -hex 32",
                    )
                    audio_btn = gr.Button("Clasifică audio", variant="primary")
                with gr.Column():
                    audio_label = gr.Label(label="Verdict", num_top_classes=2)
                    audio_note = gr.Markdown()

        # ------------------------------------------------------------------
        # Tab 4 — GenD
        # ------------------------------------------------------------------
        with gr.Tab("🛡️ GenD — deepfake generalizat"):
            gr.Markdown("### GenD (WACV 2026) — imagini și videoclipuri, cu grafice de scoruri")
            with gr.Row():
                with gr.Column():
                    gend_file = gr.File(
                        label="Imagine sau videoclip",
                        file_types=IMAGE_FILE_TYPES + VIDEO_FILE_TYPES,
                    )
                    gend_file_info = gr.Markdown("_Nu a fost încărcat niciun fișier._")
                    gend_models_btn = gr.Button("🔄 Reîncarcă lista de modele")
                    gend_models_info = gr.Markdown()
                    gend_model_source = gr.Radio(
                        ["Hugging Face", "Local Checkpoint"], value="Hugging Face", label="Sursa modelului"
                    )
                    gend_hf_model = gr.Dropdown(
                        choices=GEND_DEFAULT_MODELS, value=GEND_DEFAULT_MODELS[0], label="Model Hugging Face"
                    )
                    gend_local_ckpt = gr.Textbox(
                        label="Cale checkpoint local (doar pentru «Local Checkpoint»)",
                        placeholder="runs/.../best_mAP.ckpt",
                    )
                    with gr.Accordion("Parametri avansați", open=False):
                        gend_face_thresh = gr.Slider(0.1, 0.9, 0.5, 0.05, label="Prag detector de fețe")
                        gend_scale = gr.Slider(1.0, 2.0, 1.3, 0.05, label="Scalare aliniere față")
                        gend_target_size = gr.Dropdown(
                            choices=["Original (-1)", "160", "224", "256", "320", "384"],
                            value="Original (-1)",
                            label="Rezoluția crop-ului de față (px)",
                        )
                        gend_max_faces = gr.Slider(-1, 10, -1, 1, label="Fețe maxime pe cadru (-1 = toate)")
                        gend_stride = gr.Slider(1, 20, 1, 1, label="Stride cadre video (1 = toate)")
                        gend_max_frames = gr.Slider(-1, 300, -1, 10, label="Cadre maxime (-1 = fără limită)")
                        gend_verdict_threshold = gr.Slider(0.1, 0.9, 0.5, 0.05, label="Prag verdict FAKE")
                    with gr.Accordion("Grafic", open=True):
                        gend_graph_style = gr.Radio(GEND_GRAPH_STYLES, value="Organic", label="Stil grafic")
                        gend_legend = gr.Checkbox(value=True, label="Afișează legenda")
                    gend_btn = gr.Button("Analizează", variant="primary")
                with gr.Column():
                    gend_summary = gr.Markdown("—")
                    gend_score_plot = gr.LinePlot(
                        x="cadru", y="p_fake", y_lim=[0, 1],
                        title="p(fake) per față / cadru", height=280,
                    )
                    gend_graph_img = gr.Image(label="Grafic de scoruri (backend)", type="filepath")
                    gend_annotated_img = gr.Image(label="Media adnotată (imagini)", type="filepath")
                    gend_annotated_file = gr.File(label="Media adnotată (descărcare)")

        # ------------------------------------------------------------------
        # Tab 5 — rPPG: puls și forma de undă
        # ------------------------------------------------------------------
        with gr.Tab("❤️ MCD-RPPG — puls și forma de undă rPPG"):
            gr.Markdown(
                "### Estimarea pulsului din videoclip facial (rPPG)\n"
                "Rezultat de cercetare — **nu** un diagnostic medical. "
                "Recomandat: față bine iluminată, 15+ secunde de video."
            )
            with gr.Row():
                with gr.Column():
                    rppg_vid = gr.Video(sources=["upload"], label="Videoclip facial")
                    rppg_vid_info = gr.Markdown("_Nu a fost încărcat niciun fișier._")
                    rppg_btn = gr.Button("Estimează pulsul din rPPG", variant="primary")
                with gr.Column():
                    rppg_hr = gr.Markdown("—")
                    rppg_metrics = gr.Markdown()
            rppg_plot = gr.Image(
                label="Forma de undă rPPG (grafic generat de backend)", type="filepath", height=320
            )
            rppg_wave_plot = gr.LinePlot(
                x="timp (s)", y="BVP",
                title="Forma de undă BVP — semnalul rPPG (din răspunsul JSON)",
                height=280,
            )
            with gr.Accordion("Estimări secundare — modelul Stage 2 (relevanță limitată)", open=False):
                rppg_vitals = gr.Dataframe(
                    headers=["Indicator", "Valoare", "Unitate"],
                    label="Estimări de cercetare, nu un diagnostic medical",
                    interactive=False,
                    wrap=True,
                )
            rppg_note = gr.Markdown()

        # ------------------------------------------------------------------
        # Tab 6 — Analiză integrată
        # ------------------------------------------------------------------
        with gr.Tab("🧪 Analiză integrată video"):
            gr.Markdown(
                "### Rulează același videoclip prin toate cele 3 analizoare video, în paralel\n"
                "DeepDESK (face-swap) + GenD (deepfake generalizat) + MCD-RPPG (puls din rPPG)."
            )
            with gr.Row():
                with gr.Column():
                    int_vid = gr.Video(sources=["upload"], label="Videoclip facial")
                    int_vid_info = gr.Markdown("_Nu a fost încărcat niciun fișier._")
                    int_gend_model = gr.Dropdown(
                        choices=GEND_DEFAULT_MODELS, value=GEND_DEFAULT_MODELS[0],
                        label="Model GenD folosit",
                    )
                    int_btn = gr.Button("Rulează analiza integrată", variant="primary")
                    int_note = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### DeepDESK — Efficient ViT")
                    int_dd_md = gr.Markdown("—")
                    int_dd_plot = gr.LinePlot(x="cadru", y="p_fake", y_lim=[0, 1], title="p(fake) per cadru", height=260)
                with gr.Column():
                    gr.Markdown("#### GenD")
                    int_gd_md = gr.Markdown("—")
                    int_gd_graph = gr.Image(label="Grafic GenD", type="filepath")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### MCD-RPPG — puls și forma de undă rPPG")
                    int_rp_hr = gr.Markdown("—")
                    int_rp_metrics = gr.Markdown()
                    int_rp_plot = gr.Image(label="Forma de undă rPPG (backend)", type="filepath")
                    int_rp_wave = gr.LinePlot(
                        x="timp (s)", y="BVP",
                        title="Forma de undă BVP — semnalul rPPG (din răspunsul JSON)",
                        height=260,
                    )
                with gr.Column():
                    gr.Markdown("#### Estimări secundare")
                    with gr.Accordion("Indicatori vitali — modelul Stage 2 (relevanță limitată)", open=True):
                        int_rp_vitals = gr.Dataframe(
                            headers=["Indicator", "Valoare", "Unitate"],
                            interactive=False, wrap=True,
                        )

        # ------------------------------------------------------------------
        # Legături eveniment → callback
        # ------------------------------------------------------------------
        status_btn.click(refresh_status, outputs=status_table)

        # DeepDESK imagini / video
        ufd_img.change(media_info_markdown, inputs=ufd_img, outputs=ufd_info)
        ufd_btn.click(analyze_universal_image, inputs=ufd_img, outputs=[ufd_label, ufd_note])
        evit_img.change(media_info_markdown, inputs=evit_img, outputs=evit_info)
        evit_btn.click(analyze_evit_frame, inputs=evit_img, outputs=[evit_label, evit_note])
        evit_vid.change(media_info_markdown, inputs=evit_vid, outputs=evit_vid_info)
        evit_vid_btn.click(
            analyze_evit_video, inputs=evit_vid, outputs=[evit_vid_label, evit_vid_chart, evit_vid_note]
        )

        # DeepDESK audio
        audio_btn.click(analyze_audio, inputs=[audio_in, audio_key], outputs=[audio_label, audio_note])

        # GenD
        gend_file.change(media_info_markdown, inputs=gend_file, outputs=gend_file_info)
        gend_models_btn.click(gend_refresh_models, outputs=[gend_hf_model, gend_models_info])
        gend_btn.click(
            analyze_gend,
            inputs=[
                gend_file, gend_model_source, gend_hf_model, gend_local_ckpt,
                gend_face_thresh, gend_scale, gend_target_size, gend_max_faces,
                gend_stride, gend_max_frames, gend_graph_style, gend_legend,
                gend_verdict_threshold,
            ],
            outputs=[
                gend_summary, gend_score_plot, gend_graph_img,
                gend_annotated_img, gend_annotated_file,
            ],
        )

        # rPPG
        rppg_vid.change(media_info_markdown, inputs=rppg_vid, outputs=rppg_vid_info)
        rppg_btn.click(
            analyze_rppg, inputs=rppg_vid,
            outputs=[rppg_hr, rppg_metrics, rppg_plot, rppg_wave_plot, rppg_vitals, rppg_note],
        )

        # Analiză integrată
        int_vid.change(media_info_markdown, inputs=int_vid, outputs=int_vid_info)
        int_btn.click(
            analyze_integrated, inputs=[int_vid, int_gend_model],
            outputs=[
                int_dd_md, int_dd_plot, int_gd_md, int_gd_graph,
                int_rp_hr, int_rp_metrics, int_rp_plot, int_rp_wave, int_rp_vitals, int_note,
            ],
        )

        # La încărcarea paginii: starea serviciilor + lista modelelor GenD
        demo.load(refresh_status, outputs=status_table)
        demo.load(gend_refresh_models, outputs=[gend_hf_model, gend_models_info])

    return demo


def main() -> None:
    demo = build_ui()
    auth = None
    if HUB_USER and HUB_PASSWORD:
        auth = (HUB_USER, HUB_PASSWORD)
    demo.queue(max_size=16).launch(
        server_name=HOST,
        server_port=PORT,
        auth=auth,
        show_error=True,
    )


if __name__ == "__main__":
    main()
