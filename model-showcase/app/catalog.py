"""Catalogul static al modelelor expuse în hub (tab-ul Showcase)."""

from typing import Any, Dict, List

try:  # import de tip pachet (python -m app.main)
    from .config import DEEPDESK_IMAGES_URL, DEEPDESK_AUDIO_URL, GEND_URL, RPPG_URL, PUBLIC_LINKS
except ImportError:  # import de tip script (python app/main.py)
    from config import DEEPDESK_IMAGES_URL, DEEPDESK_AUDIO_URL, GEND_URL, RPPG_URL, PUBLIC_LINKS  # type: ignore


CATALOG: List[Dict[str, Any]] = [
    {
        "key": "audio_wav2vec",
        "service": "deepdesk-audio",
        "name": "Detector audio deepfake",
        "model": "wav2vec 2.0 fine-tunat",
        "task": ("Detectarea vocii sintetice / TTS. Analizează prima secundă de audio "
                 "(16 kHz) și returnează REAL sau FAKE/DEEPFAKE cu nivel de încredere. "
                 "Acuratețe 0.99 pe setul de test."),
        "paper": "https://huggingface.co/Bgeorge/deepfake-audio-detector-wav2vec2",
        "paper_label": "Model Hugging Face",
        "internal_url": DEEPDESK_AUDIO_URL,
        "public_url": PUBLIC_LINKS["deepdesk_audio"] + "/gradio",
        "docs_url": PUBLIC_LINKS["deepdesk_audio"] + "/docs",
        "accent": "#7c5cff",
        "emoji": "🎙️",
    },
    {
        "key": "universal_fake_detect",
        "service": "deepdesk-images-video",
        "name": "UniversalFakeDetect",
        "model": "CLIP ViT-L/14 + strat liniar",
        "task": ("Clasificare REAL / FAKE pentru imagini generate de AI (familii GAN "
                 "și difuziune). Generalizează între generatoare; sensibil la "
                 "compresie JPEG intensă."),
        "paper": "https://arxiv.org/abs/2302.10174",
        "paper_label": "Ojha et al., CVPR 2023",
        "internal_url": DEEPDESK_IMAGES_URL,
        "public_url": PUBLIC_LINKS["deepdesk_images"],
        "docs_url": PUBLIC_LINKS["deepdesk_images"] + "/docs",
        "accent": "#e2554d",
        "emoji": "🖼️",
    },
    {
        "key": "efficient_vit",
        "service": "deepdesk-images-video",
        "name": "Efficient ViT (face-swap)",
        "model": "EfficientNet-B0 + ViT",
        "task": ("Detectarea schimbării de față (face-swap) în cadre și videoclipuri, "
                 "antrenat pe DFDC. La video: 30 de cadre echidistante, verdict la "
                 "nivel de clip + grafic al scorurilor pe cadre."),
        "paper": "https://arxiv.org/abs/2107.02612",
        "paper_label": "Coccomini et al., ICIAP 2021",
        "internal_url": DEEPDESK_IMAGES_URL,
        "public_url": PUBLIC_LINKS["deepdesk_images"],
        "docs_url": PUBLIC_LINKS["deepdesk_images"] + "/docs",
        "accent": "#f59e0b",
        "emoji": "🎥",
    },
    {
        "key": "gend",
        "service": "gend-app",
        "name": "GenD",
        "model": "CLIP-L/14 · PE-L · DINOv3-L (doar parametrii LayerNorm)",
        "task": ("Deepfake detection care generalizează pe 14 benchmark-uri (2019–2025), "
                 "WACV 2026. Detectează fețele, aliniază-le, scor pe fiecare față, "
                 "verdict + grafice circulare (Organic / Radar 12 sectoare)."),
        "paper": "https://arxiv.org/abs/2508.06248",
        "paper_label": "Yermakov et al., WACV 2026",
        "internal_url": GEND_URL,
        "public_url": PUBLIC_LINKS["gend"] + "/ui",
        "docs_url": PUBLIC_LINKS["gend"] + "/docs",
        "accent": "#0ea5a4",
        "emoji": "🛡️",
    },
    {
        "key": "rppg",
        "service": "clinical-vitals-api",
        "name": "MCD-RPPG — Semne vitale",
        "model": "PhysNet3D spatiotemporal (10 ROI) + cap de regresie pentru vitale",
        "task": ("Estimarea semnelor vitale dintr-un videoclip facial: puls (BPM prin "
                 "FFT pe forma de undă rPPG) + 9 indicatori clinici. Rezultat "
                 "cercetare, NU diagnostic medical."),
        "paper": "https://github.com/CrisChir/simple_mcd_rppg",
        "paper_label": "Repository GitHub",
        "internal_url": RPPG_URL,
        "public_url": PUBLIC_LINKS["rppg_api"] + "/docs",
        "docs_url": PUBLIC_LINKS["rppg_api"] + "/docs",
        "accent": "#2563eb",
        "emoji": "❤️",
    },
]


def catalog_cards_html() -> str:
    """Carduri HTML pentru tab-ul Showcase."""
    cards = []
    for m in CATALOG:
        cards.append(
            f"""
            <div style="flex: 1 1 340px; min-width: 320px; border: 1px solid #e5e7eb;
                        border-radius: 14px; padding: 18px 20px; margin: 8px 0;
                        border-top: 4px solid {m['accent']}; background: #ffffff;">
              <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 4px;">
                {m['emoji']} {m['name']}
              </div>
              <div style="color: #6b7280; font-size: 0.85rem; margin-bottom: 10px;">
                Serviciu: <code>{m['service']}</code> &nbsp;·&nbsp; {m['model']}
              </div>
              <div style="font-size: 0.92rem; line-height: 1.45; margin-bottom: 12px;">
                {m['task']}
              </div>
              <div style="font-size: 0.85rem;">
                <a href="{m['paper']}" target="_blank">{m['paper_label']}</a>
                &nbsp;·&nbsp;
                <a href="{m['public_url']}" target="_blank">UI nativă</a>
                &nbsp;·&nbsp;
                <a href="{m['docs_url']}" target="_blank">API docs</a>
              </div>
            </div>
            """
        )
    body = "".join(cards)
    return (
        "<div style='display: flex; flex-wrap: wrap; gap: 4px; align-items: stretch;'>"
        + body
        + "</div>"
    )
