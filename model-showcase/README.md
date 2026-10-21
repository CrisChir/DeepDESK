# Showcase Hub — front-end unificat pentru containerele de modele

Interfață Gradio **containerizată** care apelează exclusiv API-urile REST ale
containerelor backend și oferă într-un singur loc toate funcționalitățile:
afișare, încărcare imagine/video/audio, analiză cu grafice, informații de
rezoluție și un showcase complet al modelelor.

| Backend | Repo | Endpoint-uri folosite |
|---|---|---|
| DeepDESK — imagini & video | [CrisChir/DeepDESK](https://github.com/CrisChir/DeepDESK) (`docker/`) | `/health`, `/api/v1/images/universal-fake-detect`, `/api/v1/images/video-deepfake-efficientvit`, `/api/v1/videos/video-deepfake-efficientvit` |
| DeepDESK — audio | [CrisChir/DeepDESK](https://github.com/CrisChir/DeepDESK) (`container-audio/`) | `GET /`, `POST /predict` (antet `X-API-KEY`) |
| GenD — deepfake generalizat (WACV 2026) | [CrisChir/GenD](https://github.com/CrisChir/GenD) | `/api/v1/health`, `/api/v1/models`, `/api/v1/analyze`, `/api/v1/files/...` |
| MCD-RPPG — semne vitale | [CrisChir/simple_mcd_rppg](https://github.com/CrisChir/simple_mcd_rppg) (`sliding_window_model/`) | `/health`, `/predict` |

Hub-ul nu încarcă niciun model propriu-zis — este un client HTTP (Gradio +
httpx); întreaga inferență rulează în containerele backend.

## Funcționalități

1. **Showcase** — carduri pentru toate cele 5 modele (detalii, referințe,
   linkuri către UI-urile native și documentația API) + tabel de stare
   alimentat din health check-urile backend-urilor.
2. **DeepDESK imagini & video** — trei subtaburi (imagini AI, cadru
   face-swap, video face-swap), verdict REAL/FAKE cu scoruri și grafic
   p(fake) per cadru.
3. **DeepDESK audio** — upload sau înregistrare din microfon, verdict cu
   nivel de încredere; cheia API se poate da în interfață sau prin `.env`.
4. **GenD** — imagine sau videoclip, alegerea modelului (lista se preia
   automat de la backend), parametri avansați (prag detector fețe, scalare
   aliniere, **rezoluția crop-ului de față**, fețe maxime, stride, cadre
   maxime, prag verdict), grafic de scoruri generat de backend + media
   adnotată descărcabilă.
5. **MCD-RPPG** — puls în BPM (FFT pe semnalul rPPG, calculat de backend și
   recalculat independent în hub din forma de undă), graficul forma de undă
   rPPG (PNG de la backend) și grafic al datelor JSON; estimările
   secundare de semne vitale (modelul Stage 2) sunt afișate separat,
   marcate ca rezultat de cercetare cu relevanță limitată.
6. **Analiză integrată video** — trimite același clip în paralel către toate
   cele 3 analizoare video și afișează verdicturile alăturate; erorile
   fiecărui backend sunt afișate independent.
7. **Rezoluții** — pentru orice fișier încărcat hub-ul afișează rezoluția,
   FPS, numărul de cadre, durata și dimensiunea; pentru GenD rezoluția
   crop-ului de față este un parametru de analiză.

## Structura proiectului

```
model-showcase/
├── app/
│   ├── main.py          # interfața Gradio (toate tab-urile)
│   ├── clients.py       # clienți HTTP pentru cele 4 containere backend
│   ├── catalog.py       # catalogul de modele pentru tab-ul Showcase
│   ├── config.py        # configurare prin variabile de mediu
│   └── media_info.py    # rezoluție / FPS / durată pentru fișierele încărcate
├── docker-compose.yml   # hub + toate backend-urile (profiluri)
├── Dockerfile           # imaginea hub-ului (python:3.12-slim, CPU)
├── requirements.txt
├── .env.example
└── README.md
```

## Cerințe

- Docker 24+ cu Compose v2
- ~12 GB spațiu liber (imaginile backend + greutățile modelelor)
- GPU opțional (CUDA) — backend-urile detectează automat dispozitivul;
  hub-ul rulează pe CPU
- Pentru GPU pe backend-uri: driver NVIDIA + nvidia-container-toolkit

## Instalare și pornire

### 1. Clonați toate repository-urile alături

```bash
mkdir ai-showcase && cd ai-showcase
git clone https://github.com/CrisChir/DeepDESK.git   # include model-showcase/
git clone https://github.com/CrisChir/GenD.git
git clone https://github.com/CrisChir/simple_mcd_rppg.git
```

Front-end-ul `model-showcase` este deja inclus în repository-ul DeepDESK;
nu trebuie descărcat separat.

Structura finală:

```
ai-showcase/
├── DeepDESK/
│   └── model-showcase/   (acest proiect)
├── GenD/
└── simple_mcd_rppg/
```

### 2. Configurați

```bash
cd DeepDESK/model-showcase
cp .env.example .env

# tokenul Hugging Face (necesar pentru modelul gated GenD_DINOv3_L):
echo "HF_TOKEN=<tokenul_dumneavoastra_hf>" >> .env

# cheia API pentru detectorul audio (POST /predict):
echo "DEEPDESK_AUDIO_API_KEY=$(openssl rand -hex 32)" >> .env
```

> Securitate: `HF_TOKEN` este un secret personal. Nu îl comitați în git și
> nu îl partajați; dacă a fost expus public, revocați-l din setările
> Hugging Face (Settings → Access Tokens).

### 3. Porniți

```bash
# Varianta completă: hub + toate backend-urile
docker compose --profile full up --build

# Sau selectiv, pe grupuri
docker compose --profile deepdesk up --build   # imagini+video și audio
docker compose --profile gend up --build       # doar GenD + hub
docker compose --profile rppg up --build       # doar vitale + hub

# Doar hub-ul (backend-urile sunt deja pornite sau sunt externe)
docker compose up --build
```

Interfața: **http://localhost:7870**

### 4. Verificați

- Tab-ul **Showcase** afișează starea fiecărui backend (🟢/🔴).
- Primele rulări GenD descarcă modelul de pe Hugging Face (1–2 GB, cache-ul
  persistă în volumul `gend-hf-cache`).

## Descărcarea modelelor este automată — nu dezarhivați niciun model manual

Toate containerele își obțin greutățile singure, la prima pornire sau la
prima analiză; nu există niciun pas manual de descărcare/dezarhivare:

| Backend | Ce se descarcă automat | Unde este cache-ul |
|---|---|---|
| GenD | modelele de pe Hugging Face (`GenD_CLIP_L_14`, `GenD_PE_L`, `GenD_DINOv3_L`), la prima analiză | volumul `gend-hf-cache` (`/cache/huggingface`) |
| DeepDESK imagini/video | greutățile UniversalFakeDetect și Efficient ViT | volumul `deepdesk-weights` (`/data/weights`) |
| DeepDESK audio | modelul wav2vec 2.0 de pe Hugging Face | în container, la pornire |
| MCD-RPPG | checkpoint-urile sunt deja în repo; doar `face_landmarker.task` (~10 MB) se descarcă la pornirea containerului | în container |

> Notă: dacă serverul `datino.isti.cnr.it` este temporar indisponibil,
> `efficient_vit.pth` trebuie descărcat manual (vezi mesajul de eroare din
> jurnalul containerului `deepdesk-images-video`) și copiat în volumul
> `deepdesk-weights`, la calea `/data/weights/efficient_vit.pth`.

## Configurare (variabile de mediu)

| Variabilă | Implicit | Descriere |
|---|---|---|
| `DEEPDESK_IMAGES_URL` | `http://deepdesk-images-video:7860` | adresa API-ului DeepDESK imagini/video |
| `DEEPDESK_AUDIO_URL` | `http://deepdesk-audio:8000` | adresa API-ului audio |
| `GEND_URL` | `http://gend-app:7860` | adresa API-ului GenD |
| `RPPG_URL` | `http://clinical-vitals-api:7861` | adresa API-ului de vitale |
| `DEEPDESK_AUDIO_API_KEY` | — | cheia pentru `POST /predict` (audio) |
| `HF_TOKEN` | — | token Hugging Face (GenD_DINOv3_L este gated) |
| `API_TIMEOUT_SECONDS` | `900` | timeout-ul cererilor de analiză |
| `HUB_USER` / `HUB_PASSWORD` | — | autentificare opțională pentru hub |
| `HUB_PORT` | `7870` | portul public al hub-ului |
| `PUBLIC_*_URL` | `http://localhost:<port>` | linkuri afișate în UI (văzute din browser) |

Pentru backend-uri **externe** rețelei compose (de exemplu un server GPU cu
adresă publică), setați în `.env`:

```env
GEND_URL=http://gpu-server.example.com:7862
PUBLIC_GEND_URL=http://gpu-server.example.com:7862
```

## Note și depanare

- **Porturi publice**: hub `7870`, DeepDESK imagini/video `7860`,
  DeepDESK audio `8000`, GenD `7862`, API vitale `7861`.
- **Apple Silicon**: pentru `clinical-vitals-api` decomentați
  `platform: linux/amd64` în `docker-compose.yml` dacă build-ul nativ eșuează.
- **Timeout la GenD**: prima analiză descarcă modelul; creșteți
  `API_TIMEOUT_SECONDS` sau reîncercați după descărcarea completă.
- **401 la audio**: cheia din `.env` / interfață nu coincide cu `API_KEY`
  din containerul `deepdesk-audio`.
- **Backend offline**: tab-ul afectat afișează un avertisment clar; restul
  hub-ului funcționează în continuare.

## Securitate

- Nu expuneți hub-ul sau backend-urile direct pe internet fără HTTPS;
  folosiți un reverse proxy (nginx / Traefik / Caddy).
- Setați `HUB_USER`/`HUB_PASSWORD` dacă hub-ul este accesibil în afara
  localhost-ului.
- Rotiți periodic `DEEPDESK_AUDIO_API_KEY`; ideal, o cheie per client.

## Avertisment privind rezultatele

Scorurile tuturor detectoarelor sunt **semnale de investigație, nu verdicte
definitive**, iar estimările rPPG sunt rezultat de cercetare, **nu diagnostic
medical**. Nu bazați decizii cu consecințe pe un singur scor.
