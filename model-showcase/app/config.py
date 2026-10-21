"""Configurația centrală a hub-ului Showcase.

Toate valorile pot fi suprascrise prin variabile de mediu, astfel încât
aceeași imagine Docker funcționează atât în rețeaua docker-compose
(adrese implicite: numele serviciilor), cât și împotriva unor servicii
externe (adrese publice sau de pe altă mașină).
"""

import os


def _url(name: str, default: str) -> str:
    value = os.getenv(name, "").strip().rstrip("/")
    return value or default


# ---- Adresele containerelor backend (nume de servicii docker-compose) ----
DEEPDESK_IMAGES_URL = _url("DEEPDESK_IMAGES_URL", "http://deepdesk-images-video:7860")
DEEPDESK_AUDIO_URL = _url("DEEPDESK_AUDIO_URL", "http://deepdesk-audio:8000")
GEND_URL = _url("GEND_URL", "http://gend-app:7860")
RPPG_URL = _url("RPPG_URL", "http://clinical-vitals-api:7861")

# ---- Cheia API pentru endpoint-ul /predict al detectorului audio ----
DEEPDESK_AUDIO_API_KEY = os.getenv("DEEPDESK_AUDIO_API_KEY", "")

# ---- Comportament HTTP ----
# Analiza unui videoclip pe CPU poate dura mai multe minute.
API_TIMEOUT_SECONDS = float(os.getenv("API_TIMEOUT_SECONDS", "900"))
HEALTH_TIMEOUT_SECONDS = float(os.getenv("HEALTH_TIMEOUT_SECONDS", "6"))

# ---- Serverul hub-ului ----
HOST = os.getenv("HUB_HOST", "0.0.0.0")
PORT = int(os.getenv("HUB_PORT", "7870"))
HUB_USER = os.getenv("HUB_USER", "")
HUB_PASSWORD = os.getenv("HUB_PASSWORD", "")

# Linkurile afișate în UI către interfețele native ale backend-urilor
# (adrese văzute din browserul utilizatorului, configurabile pentru
# deploymenturi în care containerele sunt expuse pe alt host/port).
PUBLIC_LINKS = {
    "deepdesk_images": _url("PUBLIC_DEEPDESK_IMAGES_URL", "http://localhost:7860"),
    "deepdesk_audio": _url("PUBLIC_DEEPDESK_AUDIO_URL", "http://localhost:8000"),
    "gend": _url("PUBLIC_GEND_URL", "http://localhost:7862"),
    "rppg_api": _url("PUBLIC_RPPG_API_URL", "http://localhost:7861"),
}
