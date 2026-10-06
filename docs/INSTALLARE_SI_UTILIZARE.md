# DeepDESK — Document complet de instalare și utilizare

**Repository:** https://github.com/CrisChir/DeepDESK

DeepDESK pachetizează două detectoare de deepfake cu interfață web Gradio și API REST:

| Detector | Sarcină | Lucrare științifică | Cod sursă (licență MIT) |
|---|---|---|---|
| UniversalFakeDetect (CLIP ViT-L/14 + strat liniar) | Imagini generate de AI (GAN, difuziune) | Ojha, Li & Lee, CVPR 2023 ([arXiv:2302.10174](https://arxiv.org/abs/2302.10174)) | [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) |
| Efficient ViT (EfficientNet-B0 + ViT) | Deepfake-uri cu schimbare de față în videoclipuri (DFDC) | Coccomini et al., ICIAP 2021 ([arXiv:2107.02612](https://arxiv.org/abs/2107.02612)) | [davide-coccomini/...-Video-Deepfake-Detection](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection) |

---

## 1. Structura repository-ului

```
DeepDESK/
├── docker/                     # container Docker cu ambele detectoare + API + Gradio
│   ├── Dockerfile              # imagine CPU (implicit)
│   ├── Dockerfile.gpu          # imagine CUDA 12.1
│   ├── requirements.txt
│   ├── README.md               # ghid de utilizare a containerului
│   └── app/
│       ├── main.py             # server FastAPI + interfață Gradio
│       ├── detectors.py        # încărcare, preprocesare, inferență
│       └── models/             # definițiile modelelor, derivate din codul original
└── notebooks/                  # notebook-uri Jupyter interactive
    ├── README.md               # proveniența codului, licențele, linkuri greutăți
    ├── universal-fake-detect/UniversalFakeDetect_inference.ipynb
    └── video-deepfake-efficientvit/VideoDeepfakeDetection_gradio.ipynb
```

## 2. Cerințe de sistem

| Componentă | Minimum | Recomandat |
|---|---|---|
| CPU | orice x86-64 | 8 nuclee |
| RAM | 8 GB | 16 GB |
| GPU | opțional | CUDA 12.1, ≥ 8 GB VRAM (obligatoriu pentru videoclipuri la viteză utilă) |
| Disk | 5 GB liberi (imagini + greutăți) | 10 GB |
| Docker | 20.10+ | 24+ (cu nvidia-container-toolkit pentru GPU) |

## 3. Instalare

### 3.1. Varianta Docker (recomandată)

```bash
git clone https://github.com/CrisChir/DeepDESK.git
cd DeepDESK/docker
docker build -t deepdesk:latest .
```

Imagine GPU (necesită [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) pe gazdă):

```bash
docker build -f Dockerfile.gpu -t deepdesk:gpu .
```

### 3.2. Varianta fără Docker (instalare locală)

```bash
git clone https://github.com/CrisChir/DeepDESK.git
cd DeepDESK/docker
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
mkdir -p /data/weights
export WEIGHTS_DIR=/data/weights
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### 3.3. Notebook-uri

Notebook-urile din `notebooks/` sunt autonome: instalează dependențele, clonează
repository-urile originale, descarcă greutățile și lansează interfața Gradio la execuție.
Rulează-le în Jupyter, JupyterLab, VS Code sau Google Colab.

## 4. Descărcarea modelelor (greutăți)

Greutățile se descarcă **automat** la prima pornire în `WEIGHTS_DIR` (implicit `/data/weights`).
Dacă dorești descărcarea manuală (de exemplu pe un server fără acces la internet), folosește
linkurile de mai jos și plasează fișierele în directorul de greutăți:

| Fișier | Dimensiune | Link de descărcare |
|---|---|---|
| `fc_weights.pth` (cap liniar UniversalFakeDetect) | ~3 KB | [Hugging Face](https://huggingface.co/siddharthksah/deepsafe-weights/resolve/main/universalfakedetect/fc_weights.pth) — distribuția originală: [pagina proiectului](https://utkarshojha.github.io/universal-fake-detection/) |
| `efficient_vit.pth` (Efficient ViT, varianta B0) | ~54 MB | [Server CNR-ISTI](http://datino.isti.cnr.it/efficientvit_deepfake/efficient_vit.pth) — rezervă: [Google Drive](https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1) |
| `cross_efficient_vit.pth` (varianta Cross ViT, opțional) | — | [Server CNR-ISTI](http://datino.isti.cnr.it/efficientvit_deepfake/cross_efficient_vit.pth) |
| Backbone CLIP ViT-L/14 | ~890 MB | descărcat automat de pachetul OpenAI CLIP |
| Backbone EfficientNet-B0 | ~20 MB | descărcat automat de pachetul `efficientnet_pytorch` |

Pentru pornirea offline, montează directorul cu greutăți:

```bash
docker run --rm -p 7860:7860 -v /cale/catre/greutati:/data/weights deepdesk:latest
```

## 5. Pornirea serviciului

```bash
# CPU, cu volum persistent pentru greutăți
docker run --rm -p 7860:7860 -v deepdesk-weights:/data/weights deepdesk:latest

# GPU
docker run --rm --gpus all -p 7860:7860 -v deepdesk-weights:/data/weights deepdesk:gpu
```

Prima pornire durează câteva minute (descărcarea a aproximativ 1 GB de greutăți);
pornirile ulterioare cu volum montat sunt rapide.

| URL | Descriere |
|---|---|
| `http://localhost:7860/` | interfața web Gradio (trei taburi) |
| `http://localhost:7860/docs` | documentația interactivă OpenAPI/Swagger |
| `http://localhost:7860/health` | verificare stare (dispozitiv + modele încărcate) |

## 6. Interfața web

Deschide `http://localhost:7860/`. Trei taburi:

1. **Image — AI-generated (UniversalFakeDetect):** încarcă sau capturează o imagine;
   primești probabilități REAL/FAKE.
2. **Frame — face-swap (Efficient ViT):** același lucru pentru un cadru unic;
   detectorul decupează mai întâi fața.
3. **Video — face-swap (Efficient ViT):** încarcă un videoclip; 30 de cadre echidistante
   sunt evaluate și mediate într-un verdict la nivel de videoclip, cu grafic al
   scorurilor pe cadre.

## 7. API REST — referință integrală

Toate endpoint-urile acceptă încărcări `multipart/form-data` și returnează JSON.
Verdictul se află în câmpul `label` (REAL/FAKE), aplicând pragul 0.5 pe probabilitatea
sigmoidă `p_fake`.

### 7.1. GET /health

Starea serviciului.

```bash
curl http://localhost:7860/health
```

```json
{"status": "ok", "device": "cuda", "models": ["universal-fake-detect", "video-deepfake-efficientvit"]}
```

### 7.2. POST /api/v1/images/universal-fake-detect

Classifică o imagine ca REAL sau generată de AI (familii GAN/difuziune).

| Parametru | Tip | Descriere |
|---|---|---|
| `file` | fișier (obligatoriu) | imagine PNG/JPG/JPEG/BMP |

```bash
curl -s -X POST http://localhost:7860/api/v1/images/universal-fake-detect \
  -F "file=@poza.png"
```

```json
{
  "model": "universal-fake-detect",
  "p_fake": 0.9312,
  "p_real": 0.0688,
  "label": "FAKE",
  "threshold": 0.5
}
```

### 7.3. POST /api/v1/images/video-deepfake-efficientvit

Classifică un cadru unic ca REAL sau cu fața interschimbată (domeniul de antrenare: DFDC).
Răspunsul include `face_detected` — dacă nu s-a detectat nicio față, cadrul întreg a fost
clasificat și verdictul este mai puțin fiabil.

| Parametru | Tip | Descriere |
|---|---|---|
| `file` | fișier (obligatoriu) | imagine PNG/JPG/JPEG/BMP |

```bash
curl -s -X POST http://localhost:7860/api/v1/images/video-deepfake-efficientvit \
  -F "file=@cadru.jpg"
```

```json
{
  "model": "video-deepfake-efficientvit",
  "p_fake": 0.8721,
  "p_real": 0.1279,
  "label": "FAKE",
  "face_detected": true,
  "threshold": 0.5
}
```

### 7.4. POST /api/v1/videos/video-deepfake-efficientvit

Classifică un videoclip: 30 de cadre echidistante sunt evaluate individual, iar scorul
videoclipului este media scorurilor pe cadre.

| Parametru | Tip | Descriere |
|---|---|---|
| `file` | fișier (obligatoriu) | videoclip (de ex. MP4) |

```bash
curl -s -X POST http://localhost:7860/api/v1/videos/video-deepfake-efficientvit \
  -F "file=@interviu.mp4"
```

```json
{
  "model": "video-deepfake-efficientvit",
  "p_fake": 0.7903,
  "p_real": 0.2097,
  "label": "FAKE",
  "face_detected": true,
  "frames_scored": 30,
  "frame_scores": [0.81, 0.79, "..."],
  "threshold": 0.5
}
```

### 7.5. Coduri de stare HTTP

| Cod | Semnificație |
|---|---|
| 200 | succes |
| 400 | fișier imagine/videoclip ilizibil sau fără cadre |
| 422 | câmpul `file` lipsește din cerere |

### 7.6. Exemplu de client Python

```python
import requests

BASE = "http://localhost:7860"

# Imagine — UniversalFakeDetect
r = requests.post(f"{BASE}/api/v1/images/universal-fake-detect",
                  files={"file": open("poza.png", "rb")})
print(r.json()["label"], r.json()["p_fake"])

# Videoclip — Efficient ViT
r = requests.post(f"{BASE}/api/v1/videos/video-deepfake-efficientvit",
                  files={"file": open("interviu.mp4", "rb")})
data = r.json()
print(data["label"], data["p_fake"], f"({data['frames_scored']} cadre)")
```

### 7.7. Exemplu de integrare JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);
const res = await fetch("http://localhost:7860/api/v1/images/universal-fake-detect", {
  method: "POST",
  body: formData,
});
const verdict = await res.json();
console.log(verdict.label, verdict.p_fake);
```

## 8. Configurare

| Variabilă de mediu | Implicit | Rol |
|---|---|---|
| `WEIGHTS_DIR` | `/data/weights` | directorul cu punctele de control |

Dispozitivul (CUDA/CPU) este detectat automat; nu necesită configurare.

## 9. Note de performanță

- CPU: UniversalFakeDetect procesează o imagine în circa 0,5–2 s; endpoint-ul de
  videoclip poate dura câteva minute. Pentru videoclipuri se recomandă ferm un GPU.
- GPU: ambele detectoare rulează în zeci de milisecunde pe cadru.
- Cererile sunt procesate secvențial per worker; mărește numărul de workere uvicorn
  doar cu GPU și memorie suficientă (fiecare worker își încarcă propria copie a modelelor).

## 10. Limitări și utilizare responsabilă

- Scorurile sunt probabilități sigmoide, nu niveluri de încredere calibrate; valorile
  apropiate de 0.5 sunt incerte, iar acuratețea este sub 100% pe toate domeniile de test.
- UniversalFakeDetect a fost antrenat pe imagini ProGAN; generalizează între familii de
  generatoare, dar este degradat de compresie JPEG intensă sau de neclaritate.
- Efficient ViT a fost antrenat pe schimbări de față DFDC; conținutul fără fețe este
  clasificat pe cadrul întreg și este mai puțin fiabil.
- Detectoarele oferă semnale de investigație, nu verdicte definitive. Nu le folosi ca
  unică probă privind autenticitatea unui conținut vizual și nu baza decizii cu consecințe
  pe un singur scor.

## 11. Licențe și atribuire

Ambele coduri de bază sunt licențiate MIT și sunt creditate în acest proiect:

- UniversalFakeDetect — licență MIT, Copyright (c) 2025 Wisconsin AI and Vision Lab (WAIV).
  ([LICENSE](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/LICENSE))
- Efficient ViT — licență MIT, Copyright (c) Davide Coccomini.
  ([LICENSE](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/blob/main/LICENSE))

Fișierele de definiție ale modelelor din `docker/app/models/` sunt derivate din aceste
repository-uri; derivarea este documentată în antetul fiecărui fișier. Dacă folosești
modelele în cercetare, te rugăm să citezi lucrările respective (înregistrările BibTeX se
află în `notebooks/README.md`).
