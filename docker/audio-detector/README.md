# Audio Deepfake Detector — Container

Serviciu REST + interfață web Gradio pentru detecția fișierelor audio deepfake (voce sintetică/TTS),
bazat pe modelul fine-tunat `Bgeorge/deepfake-audio-detector-wav2vec2` (wav2vec 2.0, clasificare
binară: REAL / FAKE-DEEPFAKE).

## Cuprins
1. [Structura proiectului](#structura-proiectului)
2. [Cerințe preliminare](#cerințe-preliminare)
3. [Instalare și pornire rapidă](#instalare-și-pornire-rapidă)
4. [Configurare](#configurare)
5. [Interfața web (Gradio)](#interfața-web-gradio)
6. [Utilizarea API-ului](#utilizarea-api-ului)
7. [Utilizarea CLI (inferență locală)](#utilizarea-cli-inferență-locală)
8. [Rulare pe GPU](#rulare-pe-gpu)
9. [Retrainuirea modelului](#retrainuirea-modelului)
10. [Depanare](#depanare)

## Structura proiectului

```
audio-detector/
├── app/
│   ├── __init__.py
│   ├── config.py        # configurare prin variabile de mediu
│   ├── inference.py     # încărcare model + preprocesare + predicție
│   ├── main.py          # API FastAPI cu Gradio montat la /gradio
│   ├── ui.py            # interfața Gradio
│   └── cli.py           # inferență din linia de comandă
├── train.py             # pipeline de fine-tuning (opțional)
├── requirements.txt
├── requirements-gpu.txt
├── Dockerfile           # imagine CPU
├── Dockerfile.gpu       # imagine CUDA 12.1
├── docker-compose.yml
├── .dockerignore
└── .env.example
```

## Cerințe preliminare

- Docker 24+ și Docker Compose v2
- (opțional, pentru GPU) NVIDIA Container Toolkit instalat pe host
- Acces la internet la prima pornire (modelul se descarcă de pe Hugging Face și se păstrează în cache pe volum)

## Instalare și pornire rapidă

1. Copiați configurația implicită și setați cheia API:

```bash
cd docker/audio-detector
cp .env.example .env
# editați .env: API_KEY=<o cheie puternică, generată aleatoriu>
# opțional: GRADIO_USER / GRADIO_PASSWORD pentru a proteja interfața web
```

2. Construiți și porniți:

```bash
docker compose up -d --build
```

3. Verificați că serviciul rulează:

```bash
curl http://localhost:8000/
# {"message":"API is Online. Use /docs for the API or /gradio for the web UI."}
```

La prima pornire modelul (~380 MB) se descarcă în volumul `hf-cache`; pornirile ulterioare sunt instantanee.

## Configurare

Toate setările se fac prin variabile de mediu (fișierul `.env`):

| Variabilă | Implicit | Descriere |
|---|---|---|
| `API_KEY` | — (obligatoriu) | Cheia autentificării pentru `POST /predict` |
| `GRADIO_USER` | gol | Utilizator pentru autentificarea UI Gradio (gol = UI publică) |
| `GRADIO_PASSWORD` | gol | Parola pentru UI Gradio |
| `MODEL_ID` | `Bgeorge/deepfake-audio-detector-wav2vec2` | Repo Hugging Face folosit |
| `USE_LOCAL_WEIGHTS` | `false` | `true` pentru checkpoint montat local |
| `MODEL_PATH` | `/models/fine_tuned` | Calea checkpoint-ului local (dacă `USE_LOCAL_WEIGHTS=true`) |
| `DISABLE_CUDNN` | `0` | `1` doar pentru GPU-uri Pascal vechi (ex. Tesla P40) |
| `MAX_UPLOAD_MB` | `20` | Dimensiunea maximă a fișierului uploadat |

## Interfața web (Gradio)

Interfața Gradio este montată în aceeași aplicație FastAPI, pe același port:

- **URL:** `http://localhost:8000/gradio`
- Autentificare: dacă `GRADIO_USER` și `GRADIO_PASSWORD` sunt setate în `.env`, accesul cere
  utilizator și parolă; altfel interfața este publică.
- Utilizare: încărcați un fișier audio (WAV/MP3/FLAC) și apăsați **Submit**. Rezultatul afișează
  verdictul (`REAL` în verde / `FAKE/DEEPFAKE` în roșu) și nivelul de încredere.
- Interfața folosește intern aceeași funcție de inferență ca API-ul; nu necesită `API_KEY`
  (protejată separat prin `GRADIO_USER`/`GRADIO_PASSWORD` dacă este nevoie).

## Utilizarea API-ului

### Endpoint public

`GET /` — verificare stare serviciu (nu necesită autentificare).

### Endpoint de predicție

`POST /predict` — necesită header-ul `X-API-KEY`. Formular multipart cu câmpul `file`
(WAV, MP3 sau FLAC; se analizează prima secundă de audio, resample la 16 kHz).

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-KEY: <cheia din .env>" \
  -F "file=@audio42.wav"
```

Răspuns de succes (200):

```json
{"prediction": "FAKE/DEEPFAKE", "confidence": "100.00%"}
```

Coduri de eroare:
- `401` — cheie API lipsă sau invalidă
- `400` — fișier gol, tip neacceptat sau peste limita de dimensiune
- `500` — eroare internă de inferență (detaliul este returnat în `detail`)

Interfața Swagger este disponibilă la `http://localhost:8000/docs` (pentru testarea autentificată,
adăugați cheia în „Authorize").

## Utilizarea CLI (inferență locală)

Fără a porni serverul, direct în container:

```bash
docker compose run --rm deepfake-api python -m app.cli /cale/către/fișier.wav
```

Exemplu de ieșire:

```
========================================
FILE:      audio42.wav
PREDICTION: FAKE/DEEPFAKE
CONFIDENCE: 100.00%
========================================
```

## Rulare pe GPU

1. Construiți imaginea GPU și porniți cu profilul dedicat:

```bash
docker compose -f docker-compose.yml --profile gpu up -d --build
```

(`Dockerfile.gpu` instalează PyTorch varianta CUDA 12.1; cerințele sunt în `requirements-gpu.txt`.)

2. Necesar pe host: driver NVIDIA + `nvidia-container-toolkit`.

Notă: pe Tesla P40 (Pascal) setați `DISABLE_CUDNN=1` în `.env`; pe GPU-uri moderne (T4, A10, A100, RTX) lăsați `0`.

## Retrainuirea modelului

Scriptul `train.py` reproduce fine-tuning-ul (datasetul Kaggle `audio-deepfake-detection-dataset`,
8 epoci, batch 8, fp16). Rulează-l de preferință pe GPU, în afara containerului API:

```bash
pip install -r requirements-gpu.txt kagglehub datasets accelerate scikit-learn
export DATASET_PATH=/cale/către/dataset   # folderul cu real_samples/ și folderele fake
python train.py
```

Checkpoint-ul rezultat (`./fine_tuned_deepfake_detector`) se poate urca pe Hugging Face:

```bash
hf upload Bgeorge/deepfake-audio-detector-wav2vec2 ./fine_tuned_deepfake_detector
```

sau se montează local în container cu `USE_LOCAL_WEIGHTS=true`.

## Depanare

| Simptom | Cauză probabilă | Soluție |
|---|---|---|
| Prima pornire durează mult | modelul se descarcă | așteptați; verificăți volumul `hf-cache` |
| `401 Invalid or missing API Key` | header lipsă/greșit | trimiteți `X-API-KEY` identic cu `.env` |
| `/gradio` cere login | `GRADIO_USER`/`GRADIO_PASSWORD` setate | folosiți credențialele din `.env` |
| `500` cu „audioread“/decoding error | format neacceptat | convertiți în WAV 16 kHz mono |
| OOM pe GPU | batch/alt proces ocupă VRAM | opriți alte procese; modelul cere ~3 GB |
| Crash pe Pascal (P40) | cuDNN incompatibil | `DISABLE_CUDNN=1` |
