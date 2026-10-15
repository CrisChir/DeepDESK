# Audio Deepfake Detector — Container

Serviciu REST + interfață web Gradio pentru detecția fișierelor audio deepfake (voce sintetică/TTS),
bazat pe modelul fine-tunat `Bgeorge/deepfake-audio-detector-wav2vec2` (wav2vec 2.0, clasificare
binară: REAL / FAKE-DEEPFAKE).

Ghidul detaliat de instalare și utilizare se află în [docs/GHID_AUDIO_DEEPFAKE.md](../docs/GHID_AUDIO_DEEPFAKE.md).

## Structura proiectului

```
container-audio/
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

## Pornire rapidă

```bash
cd container-audio
cp .env.example .env      # setați API_KEY (și opțional GRADIO_USER/GRADIO_PASSWORD)
docker compose up -d --build
```

- Interfață web: `http://localhost:8000/gradio`
- API: `POST http://localhost:8000/predict` (header `X-API-KEY`)
- Swagger: `http://localhost:8000/docs`

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-KEY: <cheia din .env>" \
  -F "file=@audio42.wav"
# {"prediction": "FAKE/DEEPFAKE", "confidence": "100.00%"}
```

Pentru configurare completă, rulare pe GPU, retrainuire, securitate și depanare, consultați
[ghidul detaliat](../docs/GHID_AUDIO_DEEPFAKE.md).
