# DeepDESK

**DeepDESK** este o suită completă de detecție deepfake, care pachetizează trei detectoare
specializate în containere Docker ușor de implementat, fiecare cu API REST securizat și
interfață web Gradio:

| Componentă | Sarcină | Model | Localizare |
|---|---|---|---|
| **Audio deepfake** | Detectarea vocii sintetice/TTS (REAL vs. FAKE) | wav2vec 2.0 fine-tunat — [Bgeorge/deepfake-audio-detector-wav2vec2](https://huggingface.co/Bgeorge/deepfake-audio-detector-wav2vec2) | [`container-audio/`](container-audio/) |
| **Imagini generate de AI** | Clasificare REAL/FAKE pentru imagini GAN/difuziune | UniversalFakeDetect (CLIP ViT-L/14 + strat liniar) — [Ojha et al., CVPR 2023](https://arxiv.org/abs/2302.10174) | [`docker/`](docker/) |
| **Deepfake video (face-swap)** | Detectarea schimbării de față în cadre și videoclipuri (DFDC) | Efficient ViT (EfficientNet-B0 + ViT) — [Coccomini et al., ICIAP 2021](https://arxiv.org/abs/2107.02612) | [`docker/`](docker/) |

Suita poate fi rulată în întregime prin **Showcase Hub** ([`model-showcase/`](model-showcase/)) —
un front-end Gradio containerizat care unifică cele trei detectoare de mai sus cu încă două
modele din repository-urile înrudite: **GenD** (deepfake detection generalizat, WACV 2026 —
[GenD](https://github.com/CrisChir/GenD)) și **MCD-RPPG** (estimarea pulsului și a semnelor
vitale din videoclip facial — [simple_mcd_rppg](https://github.com/CrisChir/simple_mcd_rppg)).
Ghidul complet de instalare a întregii suita de la zero, pe un calculator nou, este
[docs/INSTALARE_COMPLETA.md](docs/INSTALARE_COMPLETA.md).

---

## Cuprins

1. [Structura repository-ului](#structura-repository-ului)
2. [Cerințe de sistem](#cerințe-de-sistem)
3. [Instalare rapidă](#instalare-rapidă)
4. [Componenta audio (container-audio)](#componenta-audio-container-audio)
5. [Componenta imagini și video (docker)](#componenta-imagini-și-video-docker)
6. [API REST](#api-rest)
7. [Configurare](#configurare)
8. [Retrainuirea modelelor](#retrainuirea-modelelor)
9. [Documentație detaliată](#documentație-detaliată)
10. [Performanță și limitări](#performanță-și-limitări)
11. [Securitate](#securitate)
12. [Licențe și atribuire](#licențe-și-atribuire)

---

## Structura repository-ului

```
DeepDESK/
├── container-audio/            # detecția audio deepfake (wav2vec 2.0)
│   ├── app/
│   │   ├── config.py           # configurare prin variabile de mediu
│   │   ├── inference.py        # încărcare model + preprocesare + predicție
│   │   ├── main.py             # API FastAPI + Gradio montat la /gradio
│   │   ├── ui.py               # interfața web Gradio
│   │   └── cli.py              # inferență din linia de comandă
│   ├── train.py                # pipeline de fine-tuning
│   ├── Dockerfile              # imagine CPU (python:3.12-slim + ffmpeg)
│   ├── Dockerfile.gpu          # imagine CUDA 12.1
│   ├── docker-compose.yml      # servicii CPU + GPU (profil "gpu")
│   ├── requirements*.txt
│   └── .env.example            # șablon de configurare
├── docker/                     # detecția de imagini și video deepfake
│   ├── app/
│   │   ├── main.py             # server FastAPI + Gradio (trei taburi)
│   │   ├── detectors.py        # încărcare, preprocesare, inferență
│   │   └── models/             # definițiile modelelor (derivate din codul original)
│   ├── Dockerfile / Dockerfile.gpu
│   └── requirements.txt
├── model-showcase/             # Showcase Hub — front-end unificat (DeepDESK + GenD + MCD-RPPG)
│   ├── app/                    # interfața Gradio și clienții HTTP ai hub-ului
│   ├── docker-compose.yml      # hub + toate cele 4 backend-uri (profiluri)
│   ├── Dockerfile / requirements.txt / .env.example
│   └── README.md
├── docs/
│   ├── INSTALARE_COMPLETA.md   # instalarea întregii suite de la zero (hub + toate backend-urile)
│   ├── INSTALLARE_SI_UTILIZARE.md   # ghid complet pentru detectoarele de imagini/video
│   └── GHID_AUDIO_DEEPFAKE.md       # ghid complet pentru detectorul audio
└── notebooks/                  # notebook-uri Jupyter interactive
    ├── universal-fake-detect/UniversalFakeDetect_inference.ipynb
    └── video-deepfake-efficientvit/VideoDeepfakeDetection_gradio.ipynb
```

## Cerințe de sistem

| Componentă | Audio (container-audio) | Imagini/Video (docker) |
|---|---|---|
| CPU | 2 nuclee (minim) | orice x86-64 |
| RAM | 4 GB minim / 8 GB recomandat | 8 GB minim / 16 GB recomandat |
| GPU | opțional (CUDA 12.1, ≥4 GB VRAM) | recomandat (≥8 GB VRAM, obligatoriu pentru video) |
| Docker | 24+ cu Compose v2 | 20.10+ |
| Disk | ~1 GB (imagine + model ~380 MB) | ~5 GB (imagini + greutăți ~1 GB) |

Pentru GPU pe oricare componentă: driver NVIDIA + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

## Instalare rapidă

```bash
git clone https://github.com/CrisChir/DeepDESK.git
cd DeepDESK
```

Pentru instalarea întregii suite (DeepDESK + GenD + MCD-RPPG + Showcase Hub) pe un
calculator nou, urmați [docs/INSTALARE_COMPLETA.md](docs/INSTALARE_COMPLETA.md).

## Componenta audio (container-audio)

Detectorul audio analizează o secundă de audio (16 kHz) și returnează
`REAL` sau `FAKE/DEEPFAKE` cu nivel de încredere. Acuratețe pe setul de test: **0.99**
(precizie 0.98/1.00, recall 1.00/0.98, F1 0.99).

```bash
cd container-audio
cp .env.example .env      # setați API_KEY (și opțional GRADIO_USER/GRADIO_PASSWORD)
docker compose up -d --build
```

| URL | Descriere |
|---|---|
| `http://localhost:8000/gradio` | interfața web (upload sau înregistrare din microfon) |
| `http://localhost:8000/predict` | API REST (header `X-API-KEY`) |
| `http://localhost:8000/docs` | documentația Swagger |

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-KEY: <cheia din .env>" \
  -F "file=@audio42.wav"
# {"prediction": "FAKE/DEEPFAKE", "confidence": "100.00%"}
```

Inferență locală fără server:

```bash
docker compose run --rm deepfake-api python -m app.cli /cale/fișier.wav
```

Ghid complet: [docs/GHID_AUDIO_DEEPFAKE.md](docs/GHID_AUDIO_DEEPFAKE.md).

## Componenta imagini și video (docker)

Un container cu două detectoare, pe trei taburi Gradio:

1. **Image — AI-generated (UniversalFakeDetect):** clasifică o imagine ca REAL sau generată
   de AI (GAN/difuziune).
2. **Frame — face-swap (Efficient ViT):** clasifică un cadru unic; fața este decupată automat.
3. **Video — face-swap (Efficient ViT):** evaluează 30 de cadre echidistante și returnează un
   verdict la nivel de videoclip, cu grafic al scorurilor pe cadre.

```bash
cd docker
docker build -t deepdesk:latest .
docker run --rm -p 7860:7860 -v deepdesk-weights:/data/weights deepdesk:latest

# variantă GPU
docker build -f Dockerfile.gpu -t deepdesk:gpu .
docker run --rm --gpus all -p 7860:7860 -v deepdesk-weights:/data/weights deepdesk:gpu
```

| URL | Descriere |
|---|---|
| `http://localhost:7860/` | interfața web Gradio (trei taburi) |
| `http://localhost:7860/docs` | documentația OpenAPI/Swagger |
| `http://localhost:7860/health` | verificare stare (dispozitiv + modele încărcate) |

Ghid complet (inclusiv referința integrală a API-ului și link-urile de descărcare a
greutăților pentru rulare offline): [docs/INSTALLARE_SI_UTILIZARE.md](docs/INSTALLARE_SI_UTILIZARE.md).

## API REST

**Audio** (`container-audio`, port 8000):

| Endpoint | Autentificare | Descriere |
|---|---|---|
| `GET /` | public | verificare stare serviciu |
| `POST /predict` | `X-API-KEY` | predicție REAL/FAKE-DEEPFAKE cu încredere |
| `/gradio` | opțional (GRADIO_USER/PASSWORD) | interfața web |

**Imagini și video** (`docker`, port 7860):

| Endpoint | Descriere |
|---|---|
| `GET /health` | stare serviciu |
| `POST /api/v1/images/universal-fake-detect` | imagine → REAL/FAKE (generată AI) |
| `POST /api/v1/images/video-deepfake-efficientvit` | cadru → REAL/FAKE (face-swap) |
| `POST /api/v1/videos/video-deepfake-efficientvit` | videoclip → REAL/FAKE (media pe 30 de cadre) |

## Configurare

**Audio** — fișierul `container-audio/.env`:

| Variabilă | Implicit | Descriere |
|---|---|---|
| `API_KEY` | — (obligatoriu) | cheia pentru `POST /predict` (`openssl rand -hex 32`) |
| `GRADIO_USER` / `GRADIO_PASSWORD` | gol | autentificarea UI Gradio |
| `MODEL_ID` | `Bgeorge/deepfake-audio-detector-wav2vec2` | modelul folosit |
| `USE_LOCAL_WEIGHTS` / `MODEL_PATH` | `false` / `/models/fine_tuned` | checkpoint local în loc de HF |
| `DISABLE_CUDNN` | `0` | `1` pentru GPU-uri Pascal vechi (Tesla P40) |
| `MAX_UPLOAD_MB` | `20` | limita de dimensiune a upload-ului |

**Imagini/video** — variabilă de mediu `WEIGHTS_DIR` (implicit `/data/weights`);
dispozitivul CUDA/CPU este detectat automat.

## Retrainuirea modelelor

Doar detectorul audio are pipeline de retrainuire inclus (`container-audio/train.py`):

```bash
export DATASET_PATH=/cale/către/dataset     # real_samples/ + foldere fake
python train.py                              # 8 epoci, batch 8, fp16
hf upload Bgeorge/deepfake-audio-detector-wav2vec2 ./fine_tuned_deepfake_detector
```

Detaliile complete (pregătirea datasetului Kaggle, publicarea pe Hugging Face) sunt în
[docs/GHID_AUDIO_DEEPFAKE.md](docs/GHID_AUDIO_DEEPFAKE.md) §10.

## Documentație detaliată

- [docs/INSTALARE_COMPLETA.md](docs/INSTALARE_COMPLETA.md) — instalarea întregii suite
  (DeepDESK + GenD + MCD-RPPG + Showcase Hub) de la zero, pe un calculator nou: cerințe,
  Git/Docker, clonare, configurare `.env`, pornire, verificare și depanare.
- [docs/GHID_AUDIO_DEEPFAKE.md](docs/GHID_AUDIO_DEEPFAKE.md) — ghidul complet al componentei
  audio: instalare pas cu pas, configurare, API, CLI, GPU, retrainuire, securitate, depanare.
- [docs/INSTALLARE_SI_UTILIZARE.md](docs/INSTALLARE_SI_UTILIZARE.md) — ghidul complet al
  detectoarelor de imagini/video: referință integrală a API-ului, greutăți, performanță, licențe.
- [container-audio/README.md](container-audio/README.md) și [docker/README.md](docker/README.md) —
  readme-uri specifice fiecărei componente.
- [model-showcase/README.md](model-showcase/README.md) — documentația front-end-ului
  Showcase Hub (taburi, variabile de mediu, profiluri Docker Compose).

## Performanță și limitări

- **Audio:** acuratețe 0.99 pe setul de test al datasetului Kaggle de antrenare; predicțiile se
  bazează pe prima secundă de audio; performanța pe sisteme TTS never văzute nu este garantată.
- **Imagini:** UniversalFakeDetect generalizează între familii de generatoare, dar este degradat
  de compresie JPEG intensă sau neclaritate.
- **Video:** Efficient ViT a fost antrenat pe schimbări de față DFDC; conținutul fără fețe este
  clasificat pe cadrul întreg și este mai puțin fiabil.
- Scorurile sunt **semnale de investigație, nu verdicte definitive** — nu le folosiți ca unică
  probă privind autenticitatea unui conținut și nu bazați decizii cu consecințe pe un singur scor.

## Securitate

- Generați cheia `API_KEY` cu `openssl rand -hex 32`; nu o comitați în git (`.env` este ignorat).
- Protejați interfața Gradio (audio: `GRADIO_USER`/`GRADIO_PASSWORD`) dacă serviciul este expus.
- Pentru expunere publică, montați în spatele unui reverse proxy cu HTTPS (nginx/Traefik/Caddy).
- Rotați cheile periodic; ideal, o cheie per client.

## Licențe și atribuire

- **Modelul audio** — wav2vec 2.0 (Baevski et al., NeurIPS 2020), fine-tunat în acest proiect și
  publicat la [Bgeorge/deepfake-audio-detector-wav2vec2](https://huggingface.co/Bgeorge/deepfake-audio-detector-wav2vec2).
- **UniversalFakeDetect** — licență MIT, Copyright (c) 2025 Wisconsin AI and Vision Lab (WAIV)
  ([LICENSE](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/LICENSE)).
- **Efficient ViT** — licență MIT, Copyright (c) Davide Coccomini
  ([LICENSE](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/blob/main/LICENSE)).

Fișierele de definiție ale modelelor din `docker/app/models/` sunt derivate din aceste
repository-uri; derivarea este documentată în antetul fiecărui fișier. Înregistrările BibTeX
pentru citare se află în `notebooks/README.md`.
