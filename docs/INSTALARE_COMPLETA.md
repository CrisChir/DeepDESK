# Instalare completă de la zero — DeepDESK + GenD + MCD-RPPG + Showcase Hub

Acest ghid descrie instalarea integrală a suitei, pe un calculator nou, pornind de la
zero (fără Git, fără Docker, fără niciun repository clonat anterior). Pașii au fost
verificați practic pe macOS (Apple Silicon); acolo unde există diferențe pentru
Windows sau Linux, acestea sunt menționate explicit.

Componentele instalate:

| Componentă | Rol | Port public | Repository |
|---|---|---|---|
| **Showcase Hub** | front-end Gradio unificat, apelează toate backend-urile prin REST | `7870` | `DeepDESK/model-showcase/` |
| **deepdesk-images-video** | UniversalFakeDetect + Efficient ViT (imagini și video) | `7860` | `DeepDESK/docker/` |
| **deepdesk-audio** | detector de voce sintetică / TTS (wav2vec 2.0) | `8000` | `DeepDESK/container-audio/` |
| **gend-app** | GenD — deepfake detection generalizat (WACV 2026) | `7862` | [GenD](https://github.com/CrisChir/GenD) |
| **clinical-vitals-api** | MCD-RPPG — puls și semne vitale din videoclip facial | `7861` | [simple_mcd_rppg](https://github.com/CrisChir/simple_mcd_rppg) (`sliding_window_model/`) |

---

## 1. Cerințe de sistem

- **Sistem de operare:** macOS (verificat pe Apple Silicon), Linux sau Windows 10/11
  cu WSL2.
- **Memorie RAM:** minimum 8 GB; recomandat 16 GB (analizele video pe CPU consumă
  mult).
- **Spațiu pe disc:** aproximativ 20 GB liberi (imaginile Docker, greutățile
  modelelor și cache-ul Hugging Face).
- **Internet:** necesar la prima pornire și la primele analize (modelele se
  descarcă automat, 1–2 GB pentru GenD).
- **Software instalat pe parcurs:** Git și Docker Desktop (sau Docker Engine) 24+
  cu Compose v2.

## 2. Pasul 1 — Instalarea Git

**macOS** — deschideți Terminal (Aplicații → Utilitare → Terminal) și rulați:

```bash
xcode-select --install
```

Confirmă fereastra de instalare și așteptați finalizarea.

**Windows** — descărcați installerul de la <https://git-scm.com/download/win> și
rulați-l cu opțiunile implicite.

**Linux (Debian/Ubuntu):**

```bash
sudo apt update && sudo apt install -y git
```

**Verificare (toate sistemele):**

```bash
git --version
```

## 3. Pasul 2 — Instalarea Docker

**macOS și Windows** — descărcați Docker Desktop de la
<https://www.docker.com/products/docker-desktop/>:

- macOS: montați imaginea `.dmg` și trageți Docker în Applications; la prima
  pornire acceptați termenii.
- Windows: instalați cu backend WSL2 (installerul vă ghidează).
- Pe Mac-uri cu 8–16 GB RAM: Settings → Resources → Memory — alocați cel puțin
  8 GB Docker-ului.
- Pe Mac-uri Apple Silicon: Settings → General — activați „Use Rosetta for
  x86_64/amd64 emulation on Apple silicon" (necesar pentru serviciul rPPG,
  a se vedea secțiunea 9.6).

**Linux** — instalați Docker Engine și pluginul Compose v2 conform
<https://docs.docker.com/engine/install/> (secțiunea pentru distribuția
dumneavoastră).

**Verificare:**

```bash
docker --version
docker compose version
docker run --rm hello-world
```

Dacă ultima comandă afișează „Hello from Docker!", instalarea este corectă.

## 4. Pasul 3 — Clonarea repository-urilor

Toate cele trei repository-uri trebuie clonate **alături**, în același dosar:

```bash
mkdir ai-showcase && cd ai-showcase
git clone https://github.com/CrisChir/DeepDESK.git
git clone https://github.com/CrisChir/GenD.git
git clone https://github.com/CrisChir/simple_mcd_rppg.git
```

Structura finală obligatorie:

```
ai-showcase/
├── DeepDESK/
│   └── model-showcase/    ← front-end-ul (docker-compose.yml se află aici)
├── GenD/
└── simple_mcd_rppg/
```

Verificare:

```bash
ls ~/calea/către/ai-showcase
```

trebuie să afișeze cele trei dosare: `DeepDESK`, `GenD`, `simple_mcd_rppg`.

## 5. Pasul 4 — Configurarea fișierului `.env`

```bash
cd DeepDESK/model-showcase
cp .env.example .env

# tokenul Hugging Face — necesar DOAR pentru modelul gated yermandy/GenD_DINOv3_L:
echo "HF_TOKEN=<tokenul_dumneavoastra_hf>" >> .env

# cheia API a detectorului audio:
echo "DEEPDESK_AUDIO_API_KEY=$(openssl rand -hex 32)" >> .env
```

Observații:

- Tokenul Hugging Face se creează la <https://huggingface.co/settings/tokens>.
  Pentru modelul `yermandy/GenD_DINOv3_L` (gated) este necesar și acceptarea
  condițiilor pe pagina modelului, fiind autentificat.
- Ambele valori sunt secrete personale. Fișierul `.env` este ignorat de git
  (`.gitignore`); nu îl publicați niciodată.

## 6. Pasul 5 — Construirea și pornirea

Din dosarul `DeepDESK/model-showcase`:

```bash
docker compose --profile full up -d --build
```

- Prima construire durează 10–20 de minute (descărcarea imaginilor de bază,
  instalarea pachetelor Python).
- Pornirea completă pornește hub-ul și toate cele patru backend-uri.
- Variante selective:

```bash
docker compose --profile deepdesk up -d --build   # imagini/video + audio + hub
docker compose --profile gend up -d --build       # doar GenD + hub
docker compose --profile rppg up -d --build       # doar vitale + hub
docker compose up -d --build                       # doar hub-ul
```

## 7. Pasul 6 — Verificarea instalării

```bash
docker compose --profile full ps
```

Toate cele cinci servicii trebuie să afișeze starea **Up**:

```
NAME                    ... STATUS
clinical-vitals-api     ... Up
deepdesk-audio          ... Up
deepdesk-images-video   ... Up
gend-app                ... Up
showcase-hub            ... Up (healthy)
```

Apoi deschideți în browser **http://localhost:7870**, tabul **Showcase**, și apăsați
**Verifică starea serviciilor**: toate cele patru backend-uri trebuie să apară
online.

## 8. Pasul 7 — Primul test

1. **DeepDESK imagini & video** — încărcați o imagine; verdictul apare în câteva
   secunde.
2. **DeepDESK audio** — încărcați un fișier WAV; se cere cheia API (cea generată
   la pasul 4).
3. **GenD** — la prima analiză se descarcă modelul (1–2 GB, durată mai lungă);
   cache-ul persistă în volumul `gend-hf-cache`, deci descărcarea are loc o
   singură dată.
4. **MCD-RPPG** — încărcați un videoclip facial de minim 15 secunde; hub-ul
   afișează pulsul în BPM (calculat din semnalul rPPG) și forma de undă BVP.
5. **Analiză integrată** — trimite același videoclip tuturor celor trei
   analizoare video în paralel.

---

## 9. Probleme cunoscute și soluții

### 9.1 `deepdesk-images-video` se repornește la nesfârșit: „Could not download efficient_vit.pth"

Cauza: serverul `datino.isti.cnr.it` (sursa oficială a greutăților Efficient ViT)
refuză ocazional conexiunile. Fișierul se descarcă manual, o singură dată:

1. Descărcați în browser `efficient_vit.pth` din directorul Google Drive
   <https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1>
   (ajunge în Downloads).
2. Copiați-l în container:

```bash
cd DeepDESK/model-showcase
docker compose stop deepdesk-images-video
docker cp ~/Downloads/efficient_vit.pth deepdesk-images-video:/data/weights/
docker compose start deepdesk-images-video
docker compose logs --tail 20 deepdesk-images-video
```

Jurnalul trebuie să conțină liniile `Device: cpu` și `Weights ready.` Liniile
`RuntimeError` mai vechi din jurnal aparțin repornirilor anterioare și pot fi
ignorate. Alternativ, fișierul poate fi montat direct dintr-un dosar local
în volumul `deepdesk-weights`.

### 9.2 `clinical-vitals-api`: „Stage 1 checkpoint not found"

Checkpoint-urile (`spatiotemporal_physnet_best_shuffle.pth`,
`clinical_vitals_stage2_best.pth`) fac parte din repository-ul
`simple_mcd_rppg`; în configurația actuală ele sunt montate automat în
container, deci eroarea nu ar trebui să apară. Dacă totuși apare (de exemplu
după o modificare locală a `docker-compose.yml`), copiați manual fișierele:

```bash
cd DeepDESK/model-showcase
docker compose stop clinical-vitals-api
docker cp ../../simple_mcd_rppg/sliding_window_model/models/spatiotemporal_physnet_best_shuffle.pth clinical-vitals-api:/app/sliding_window_model/models/
docker cp ../../simple_mcd_rppg/sliding_window_model/models/clinical_vitals_stage2_best.pth clinical-vitals-api:/app/sliding_window_model/models/
docker compose start clinical-vitals-api
```

### 9.3 `gend-app` rulează comanda greșită (`python app/run.py`) sau „Container name already in use"

Cauza: o clonă GenD învechită sau un container vechi rămas pe sistem.

```bash
docker rm -f gend-app
cd ../../GenD
git pull
cd ../DeepDESK/model-showcase
docker compose up -d --build gend-app
```

Verificați în `docker compose --profile full ps` că linia `gend-app` afișează
comanda `python app/api.py`.

### 9.4 GenD: „HTTP 500" la prima analiză, cu menționarea `cdn.hf.co`

Eroare tranzitorie de rețea la Hugging Face. Reluați analiza; dacă persistă,
verificați tokenul din `.env` și posibilitatea ca modelul gated să necesite
acceptarea condițiilor pe pagina acestuia.

### 9.5 Audio: „HTTP 401"

Cheia `DEEPDESK_AUDIO_API_KEY` din `.env` / interfață nu coincide cu variabila
`API_KEY` cu care a fost pornit containerul `deepdesk-audio`. Regenerați cheia,
actualizați `.env` și reporniți:

```bash
docker compose up -d --force-recreate deepdesk-audio
```

### 9.6 Apple Silicon — avertisment de platformă la `clinical-vitals-api`

La pornire, Docker poate afișa mesajul:

```
! clinical-vitals-api  The requested image's platform (linux/amd64) does not match
  the host platform (linux/arm64)
```

Acesta este un **avertisment, nu o eroare**: imaginea de bază
`pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` este publicată doar pentru
`linux/amd64`, iar pe Apple Silicon Docker o rulează prin emulare (Rosetta/QEMU).
Verificați starea serviciului:

```bash
docker compose --profile full ps
```

Dacă `clinical-vitals-api` este **Up**, nu trebuie făcut nimic — analizele rPPG
vor fi doar mai lente decât pe un computer cu procesor nativ amd64. Dacă
serviciul nu pornește (se repornește la nesfârșit sau nu apare deloc):

1. Decomentați linia `# platform: linux/amd64` din secțiunea serviciului
   `clinical-vitals-api` din `docker-compose.yml` (ștergeți doar `# ` din fața ei).
2. Activați în Docker Desktop → Settings → General opțiunea
   „Use Rosetta for x86_64/amd64 emulation on Apple silicon".
3. Reluați construirea:

```bash
docker compose --profile full up -d --build clinical-vitals-api
```

### 9.7 Comenzi de diagnostic

```bash
docker compose --profile full ps            # starea serviciilor
docker compose logs --tail 40 <serviciu>    # jurnalul unui serviciu
docker compose restart <serviciu>           # repornirea unui serviciu
docker compose --profile full stop          # oprirea completă
```

## 10. Porniri ulterioare și actualizări

După instalarea inițială, pornirea zilnică nu mai reconstruiește imaginile:

```bash
cd DeepDESK/model-showcase
docker compose --profile full up -d
```

Actualizarea suitei la ultimele versiuni:

```bash
cd ai-showcase/DeepDESK && git pull
cd ../GenD && git pull
cd ../simple_mcd_rppg && git pull
cd ../DeepDESK/model-showcase
docker compose --profile full up -d --build
```

Atenție: comanda `docker compose down -v` șterge și volumele — deci și
cache-urile modelelor (GenD, greutăți DeepDESK). Evitați `-v` dacă nu doriți
descărcarea modelelor de la zero.

## 11. Securitate

- Nu publicați și nu commit-uiți niciodată `.env`, `HF_TOKEN` sau cheile API.
- Dacă un token a fost expus, revocați-l din setările Hugging Face și
  regenerați-l.
- Dacă hub-ul este accesibil în afara `localhost`, setați `HUB_USER` /
  `HUB_PASSWORD` în `.env` și montați un reverse proxy cu HTTPS.
- Rezultatele detectoarelor sunt semnale de investigație, nu verdicte
  definitive; estimările rPPG sunt rezultat de cercetare, nu diagnostic medical.
