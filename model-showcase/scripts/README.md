# Comenzi rapide pentru macOS — pornire și oprire

Două scripturi destinate utilizatorilor care nu folosesc linia de comandă.
Se execută prin dublu-click în Finder:

| Script | Efect |
|---|---|
| `porneste-showcase-hub.command` | pornește automat Docker Desktop dacă nu rulează, pornește hub-ul și toate cele 4 backend-uri, așteaptă interfața web și deschide singur browserul la adresa hub-ului |
| `opreste-showcase-hub.command` | oprește toate serviciile, păstrând intacte datele și modelele (nu folosește `-v`) |

Ambele afișează mesaje clare de stare și de eroare; fereastra Terminal rămâne
deschisă până la apăsarea tastei Enter.

## Instalare pe Desktop (o singură dată)

Din dosarul în care a fost clonat DeepDESK:

```bash
cd model-showcase/scripts
cp porneste-showcase-hub.command opreste-showcase-hub.command ~/Desktop/
chmod +x ~/Desktop/porneste-showcase-hub.command ~/Desktop/opreste-showcase-hub.command
```

Dacă proiectul a fost descărcat din GitHub ca arhivă ZIP, macOS atașează
fișierelor un atribut de carantină; eliminați-l pentru a evita blocarea:

```bash
xattr -d com.apple.quarantine ~/Desktop/*.command 2>/dev/null
```

## Prima utilizare

1. La primul dublu-click pe fiecare script, macOS poate afișa o avertizare de
   securitate. Rezolvarea: click-dreapta pe pictogramă → Open → Open
   (o singură dată per fișier).
2. Recomandat, pentru pornire complet automată: Docker Desktop →
   Settings → General → bifați „Start Docker Desktop when you sign in to your
   computer".

## Cum își găsesc scripturile proiectul

Fiecare script caută automat dosarul `model-showcase` (cel care conține
`docker-compose.yml`) în următoarele locații, în ordine:

1. valoarea variabilei de mediu `SHOWCASE_HUB_DIR` (dacă este setată);
2. `~/ProjectDeep/DeepDESK/model-showcase`
3. `~/ai-showcase/DeepDESK/model-showcase`
4. `~/DeepDESK/model-showcase`
5. dosarul părinte al locului în care se află scriptul (util dacă este rulat
   direct din clona repository-ului).

Dacă proiectul se află în altă parte, editați lista `CAUTARI` din antetul
scriptului sau setați `SHOWCASE_HUB_DIR` înainte de rulare:

```bash
export SHOWCASE_HUB_DIR="/Users/nume/utilizator/calea/mea/model-showcase"
```

## Personalizarea pictogramelor

Pictogramele pot fi schimbate fără niciun instrument suplimentar: deschideți
imaginea dorită (ideal PNG pătrat, 512 × 512) în Preview → Cmd+A → Cmd+C,
apoi în Finder selectați scriptul → Cmd+I → click pe iconița din colțul
stânga-sus → Cmd+V.

## Note tehnice

- Scripturile sunt specifice macOS (comanda `open`). Pe Linux sau Windows,
  porniți suita cu `docker compose --profile full up -d` direct.
- Pornirea zilnică nu reconstruiește imaginile și nu descarcă modelele —
  durează câteva secunde. Reconstruirea se face numai după actualizări:
  `docker compose --profile full up -d --build`.
- Scriptul de oprire folosește `stop`, niciodată `down -v`: volumele cu
  modele (GenD, greutăți DeepDESK) nu sunt niciodată șterse.
