# DeepDESK — Ghid detaliat de instalare și utilizare: Deepfake Audio

Acest ghid descrie complet instalarea, configurarea, utilizarea și întreținerea containerului
de detecție audio deepfake din proiectul DeepDESK. Componenta se află în folderul
[`container-audio/`](../container-audio) de la rădăcina repository-ului.

Componenta oferă:
- un **API REST** securizat (`POST /predict`, autentificare `X-API-KEY`),
- o **interfață web Gradio** (`/gradio`) pentru utilizare direct din browser,
- un **CLI** pentru inferență locală,
- un **pipeline de retrainuire** (`train.py`).

Modelul utilizat: [Bgeorge/deepfake-audio-detector-wav2vec2](https://huggingface.co/Bgeorge/deepfake-audio-detector-wav2vec2)
(wav2vec 2.0 fine-tunat; clasificare binară REAL / FAKE-DEEPFAKE; acuratețe 0.99 pe setul de test).

---

## Cuprins
1. [Prezentare generală](#1-prezentare-generală)
2. [Arhitectura componentei](#2-arhitectura-componentei)
3. [Cerințe de sistem](#3-cerințe-de-sistem)
4. [Instalare pas cu pas](#4-instalare-pas-cu-pas)
5. [Configurare completă](#5-configurare-completă)
6. [Utilizarea interfeței web Gradio](#6-utilizarea-interfeței-web-gradio)
7. [Utilizarea API-ului REST](#7-utilizarea-api-ului-rest)
8. [Utilizarea CLI](#8-utilizarea-cli)
9. [Rulare pe GPU](#9-rulare-pe-gpu)
10. [Retrainuirea și publicarea modelului](#10-retrainuirea-și-publicarea-modelului)
11. [Securitate și bune practici](#11-securitate-și-bune-practici)
12. [Depanare avansată](#12-depanare-avansată)

---

## 1. Prezentare generală

Componenta analizează o **secundă de audio** (resample la 16 kHz, truncation/padding la 16.000
eșantioane) și returnează un verdict: `REAL` sau `FAKE/DEEPFAKE`, împreună cu nivelul de încredere
(softmax). Modelul a fost fine-tunat pe datasetul Kaggle
[`audio-deepfake-detection-dataset`](https://www.kaggle.com/datasets/adarshsingh0903/audio-deepfake-detection-dataset)
(3.557 exemple de antrenare / 890 de test, split 80/20).

| Etalon | Valoare |
|---|---|
| Acuratețe test | 0.99 |
| Precizie Real / Fake | 0.98 / 1.00 |
| Recall Real / Fake | 1.00 / 0.98 |
| F1 (macro / weighted) | 0.99 |

## 2. Arhitectura componentei

```
container-audio/
├── app/
│   ├── __init__.py
│   ├── config.py        # configurare prin variabile de mediu
│   ├── inference.py     # încărcare model (o singură dată) + preprocesare + predicție
│   ├── main.py          # API FastAPI cu Gradio montat la /gradio
│   ├── ui.py            # interfața Gradio
│   └── cli.py           # inferență din linia de comandă
├── train.py             # pipeline de fine-tuning
├── requirements.txt     # dependențe CPU
├── requirements-gpu.txt # dependențe CUDA 12.1
├── Dockerfile           # imagine CPU (python:3.12-slim + ffmpeg)
├── Dockerfile.gpu       # imagine GPU (nvidia/cuda:12.1.1-runtime)
├── docker-compose.yml   # serviciu CPU + serviciu GPU (profil)
├── .dockerignore
└── .env.example         # șablon de configurare
```

Un singur container deservind două canale: API-ul programatic (pentru integrări) și interfața
Gradio (pentru utilizatori). Modelul se încarcă în memorie o singură dată (`lru_cache`), la primul
request, indiferent de canal.

## 3. Cerințe de sistem

| Componentă | Minim | Recomandat |
|---|---|---|
| CPU | 2 nuclee | 4+ nuclee |
| RAM | 4 GB | 8 GB |
| GPU | — (CPU funcționează) | NVIDIA cu ≥4 GB VRAM |
| Docker | 24+ | ultima versiune stabilă |
| Docker Compose | v2 | v2 |
| Internet | necesar la prima pornire (descărcarea modelului, ~380 MB) | opțional ulterior |

Pentru GPU: driver NVIDIA + `nvidia-container-toolkit` instalate pe host.

## 4. Instalare pas cu pas

### 4.1. Clonarea repository-ului

```bash
git clone https://github.com/CrisChir/DeepDESK.git
cd DeepDESK/container-audio
```

### 4.2. Configurarea inițială

```bash
cp .env.example .env
```

Editați `.env` și setați obligatoriu `API_KEY`. Generați o cheie puternică:

```bash
openssl rand -hex 32
```

Opțional, protejați interfața web cu `GRADIO_USER` / `GRADIO_PASSWORD`.

### 4.3. Construirea și pornirea (CPU)

```bash
docker compose up -d --build
```

### 4.4. Verificarea instalării

```bash
# Starea serviciului
curl http://localhost:8000/
# {"message":"API is Online. Use /docs for the API or /gradio for the web UI."}

# Interfața web
open http://localhost:8000/gradio

# Documentația Swagger a API-ului
open http://localhost:8000/docs

# Log-uri (dacă ceva nu funcționează)
docker compose logs -f deepfake-api
```

Notă: la prima pornire, modelul (~380 MB) se descarcă în volumul `hf-cache/`. Pornirile
următoare pornesc imediat, din cache.

### 4.5. Oprirea și curățarea

```bash
docker compose down            # oprire
docker compose down --rmi local # oprire + ștergerea imaginii construite local
```

## 5. Configurare completă

Toate setările se fac prin variabile de mediu (fișierul `.env`):

| Variabilă | Implicit | Descriere |
|---|---|---|
| `API_KEY` | — (obligatoriu) | Cheia autentificării pentru `POST /predict` |
| `GRADIO_USER` | gol | Utilizator pentru UI Gradio (gol = UI publică) |
| `GRADIO_PASSWORD` | gol | Parola pentru UI Gradio |
| `MODEL_ID` | `Bgeorge/deepfake-audio-detector-wav2vec2` | Repo Hugging Face folosit |
| `USE_LOCAL_WEIGHTS` | `false` | `true` = checkpoint montat local, nu de pe HF |
| `MODEL_PATH` | `/models/fine_tuned` | Calea checkpoint-ului local |
| `DISABLE_CUDNN` | `0` | `1` pentru GPU-uri Pascal vechi (Tesla P40) |
| `MAX_UPLOAD_MB` | `20` | Limita de dimensiune a upload-ului |

Exemplu de utilizare a checkpoint-ului local în loc de repo-ul HF: montați
`./fine_tuned_deepfake_detector:/models/fine_tuned:ro` (deja definit în compose) și setați
`USE_LOCAL_WEIGHTS=true`.

## 6. Utilizarea interfeței web Gradio

- **URL:** `http://localhost:8000/gradio`
- Dacă `GRADIO_USER`/`GRADIO_PASSWORD` sunt setate, se cere autentificare la deschidere.
- Pași: încărcați fișierul audio (WAV/MP3/FLAC) → apăsați **Submit** → citiți verdictul:
  - `REAL` (verde) — voce autentică,
  - `FAKE/DEEPFAKE` (roșu) — voce generată AI,
  - împreună cu nivelul de încredere (ex. `100.00%`).
- Interfața poate face și înregistrare direct din microfon (componenta `gr.Audio`).

Interfața folosește aceeași funcție de inferență ca și API-ul; rezultatele sunt identice.

## 7. Utilizarea API-ului REST

### Verificare stare (public)

```bash
curl http://localhost:8000/
```

### Predicție (autentificată)

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-KEY: <cheia din .env>" \
  -F "file=@audio42.wav"
```

Răspuns de succes (200):

```json
{"prediction": "FAKE/DEEPFAKE", "confidence": "100.00%"}
```

### Coduri de răspuns

| Cod | Situație | Acțiune |
|---|---|---|
| 200 | predicție reușită | citiți `prediction` și `confidence` |
| 400 | fișier gol, tip neacceptat sau peste `MAX_UPLOAD_MB` | verificați fișierul |
| 401 | `X-API-KEY` lipsă sau greșit | corectați header-ul |
| 500 | eroare internă de inferență | verificați log-urile (`docker compose logs`) |

### Exemplu Python (client)

```python
import requests

resp = requests.post(
    "http://localhost:8000/predict",
    headers={"X-API-KEY": "cheia-voastră"},
    files={"file": open("audio42.wav", "rb")},
)
resp.raise_for_status()
print(resp.json())   # {'prediction': 'FAKE/DEEPFAKE', 'confidence': '100.00%'}
```

## 8. Utilizarea CLI

Inferență locală, fără server (rulează în container):

```bash
docker compose run --rm deepfake-api python -m app.cli /cale/către/fișier.wav
```

Ieșire:

```
========================================
FILE:      audio42.wav
PREDICTION: FAKE/DEEPFAKE
CONFIDENCE: 100.00%
========================================
```

## 9. Rulare pe GPU

```bash
docker compose -f docker-compose.yml --profile gpu up -d --build
```

- Imaginea `Dockerfile.gpu` folosește PyTorch CUDA 12.1 (`requirements-gpu.txt`).
- Necesar pe host: driver NVIDIA + `nvidia-container-toolkit` (`docker run --gpus all` trebuie să funcționeze).
- Verificați GPU-ul în container: `docker compose exec deepfake-api-gpu python -c "import torch; print(torch.cuda.is_available())"`.
- Pe Tesla P40 (Pascal): `DISABLE_CUDNN=1`; pe GPU-uri moderne lăsați `0`.
- Consum de memorie: ~3 GB VRAM pentru inferență; ~8-10 GB pentru antrenare.

## 10. Retrainuirea și publicarea modelului

### 10.1. Pregătirea datasetului

Descărcați datasetul Kaggle:

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('adarshsingh0903/audio-deepfake-detection-dataset'))"
```

Structura așteptată: un folder cu `real_samples/` (eticheta 0) și folderele cu probe sintetice (eticheta 1).

### 10.2. Antrenare

```bash
pip install -r requirements-gpu.txt datasets accelerate scikit-learn kagglehub
export DATASET_PATH=/cale/către/dataset
python train.py
```

Hiperparametri (identici cu antrenarea publicată): 8 epoci, batch 8, fp16, evaluare per epocă,
split 80/20 seed 42. Checkpoint-ul final: `./fine_tuned_deepfake_detector`.

### 10.3. Publicarea pe Hugging Face

```bash
hf auth login               # token de tip Write
hf upload Bgeorge/deepfake-audio-detector-wav2vec2 ./fine_tuned_deepfake_detector \
    --exclude "training_args.bin"
```

### 10.4. Utilizarea noului model

După upload, containerul îl va folosi automat la următoarea pornire (nici o modificare de cod —
`MODEL_ID` rămâne neschimbat). Pentru a reveni la vechiul model, fixați `MODEL_ID` pe versiunea
dorită.

## 11. Securitate și bune practici

- Generați `API_KEY` cu `openssl rand -hex 32`; **nu** o comitați în git (`.env` este în `.dockerignore`).
- Protejați interfața Gradio cu `GRADIO_USER`/`GRADIO_PASSWORD` dacă serviciul este expus în rețea.
- Pentru expunere publică, montați în spatele unui reverse proxy cu HTTPS (nginx/Traefik/Caddy).
- Rotați cheia API periodic; fiecare client trebuie să folosească propria cheie dacă este posibil.
- Verdicturile sunt probabilistice: nu le folosi ca unică decizie automată în contexte critice
  (detalii în model card-ul de pe Hugging Face).

## 12. Depanare avansată

| Simptom | Cauză probabilă | Soluție |
|---|---|---|
| Prima pornire durează mult | modelul (~380 MB) se descarcă | așteptați; verificați `hf-cache/` |
| `401 Invalid or missing API Key` | header lipsă/greșit | `X-API-KEY` identic cu `.env` |
| `/gradio` cere login | credențiale Gradio setate | folosiți `.env` |
| `400 Invalid file type` | extensie neacceptată | WAV, MP3 sau FLAC |
| `500` cu eroare audioread | decodare eșuată | convertiți: `ffmpeg -i in.ogg -ar 16000 -ac 1 out.wav` |
| Containerul moare la pornire pe GPU | toolkit lipsă | instalați `nvidia-container-toolkit`, restart Docker |
| Crash cuDNN pe Tesla P40 | Pascal incompatibil cu cuDNN nou | `DISABLE_CUDNN=1` |
| OOM în timpul antrenării | VRAM insuficient | reduceți `per_device_train_batch_size` la 4 |
| Predicții mereu identice | modelul nu s-a încărcat corect | verificați log-urile la pornire |

Pentru orice problemă persistentă: `docker compose logs deepfake-api` este prima sursă de diagnostic.
