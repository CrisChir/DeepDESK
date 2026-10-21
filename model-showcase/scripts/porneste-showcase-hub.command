#!/bin/bash
# ============================================================
# Showcase Hub — PORNIRE (macOS, dublu-click in Finder)
#
# Ce face:
#   1. cauta automat dosarul proiectului (lista CAUTARI de mai jos);
#   2. daca Docker Desktop nu ruleaza, il porneste si asteapta (max 3 min);
#   3. porneste hub-ul si toate cele 4 backend-uri (fara --build);
#   4. asteapta interfata web, apoi deschide browserul la adresa hub-ului.
#
# Instalare pe Desktop: vezi scripts/README.md.
# Daca proiectul este in alta parte: editati lista CAUTARI din acest
# script sau setati variabila SHOWCASE_HUB_DIR inainte de rulare.
# ============================================================

echo "== Showcase Hub — pornire =="

# ---------- 1. Localizarea proiectului ----------
CAUTARI=(
    "${SHOWCASE_HUB_DIR:-}"
    "$HOME/ProjectDeep/DeepDESK/model-showcase"
    "$HOME/ai-showcase/DeepDESK/model-showcase"
    "$HOME/DeepDESK/model-showcase"
    "$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"
)
DIR=""
for c in "${CAUTARI[@]}"; do
    if [ -n "$c" ] && [ -f "$c/docker-compose.yml" ]; then
        DIR="$c"
        break
    fi
done

if [ -z "$DIR" ]; then
    echo ""
    echo "EROARE: Nu am gasit dosarul proiectului (model-showcase)."
    echo "Am cautat in:"
    printf '   %s\n' "${CAUTARI[@]}"
    echo ""
    echo "Solutie: editati lista CAUTARI din acest script sau setati"
    echo "variabila SHOWCASE_HUB_DIR cu dosarul corect al proiectului."
    read -p "Apasati Enter pentru a inchide fereastra... " _ ; exit 1
fi
echo "Proiect: $DIR"
cd "$DIR" || exit 1

# ---------- 2. Verificarea instalarii Docker ----------
if ! command -v docker >/dev/null 2>&1; then
    echo ""
    echo "EROARE: Docker nu este instalat pe acest calculator."
    echo "Instalati Docker Desktop de la https://www.docker.com/products/docker-desktop/"
    read -p "Apasati Enter pentru a inchide fereastra... " _ ; exit 1
fi

# ---------- 3. Pornirea automata a Docker, daca nu ruleaza ----------
if ! docker info >/dev/null 2>&1; then
    echo "Docker Desktop nu ruleaza. Il pornesc automat (dureaza 1-2 minute)..."
    open -a Docker
    printf "Astept pornirea Docker"
    for i in $(seq 1 36); do
        if docker info >/dev/null 2>&1; then printf " GATA\n"; break; fi
        printf "."
        sleep 5
    done
    if ! docker info >/dev/null 2>&1; then
        echo ""
        echo "EROARE: Docker nu a reusit sa porneasca in 3 minute."
        echo "Deschideti manual Docker din Applications, asteptati sa scrie 'Running', apoi reluati."
        read -p "Apasati Enter pentru a inchide fereastra... " _ ; exit 1
    fi
else
    echo "Docker: pornit."
fi

# ---------- 4. Pornirea serviciilor ----------
echo "Pornesc aplicatia (dureaza cateva secunde)..."
docker compose --profile full up -d
if [ $? -ne 0 ]; then
    echo ""
    echo "EROARE: Serviciile nu au pornit. Transmiteți urmatorul mesaj tehnic persoanei care a instalat aplicatia:"
    docker compose --profile full ps
    read -p "Apasati Enter pentru a inchide fereastra... " _ ; exit 1
fi

# ---------- 5. Asteptarea interfetei web ----------
HUB_PORT=$(grep -E '^HUB_PORT=' .env 2>/dev/null | tail -1 | cut -d= -f2)
HUB_PORT=${HUB_PORT:-7870}
printf "Astept interfata web"
for i in $(seq 1 24); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$HUB_PORT" 2>/dev/null)
    if [ "$code" = "200" ]; then printf " GATA\n"; break; fi
    printf "."
    sleep 5
done

# ---------- 6. Deschiderea browserului ----------
open "http://localhost:$HUB_PORT"

echo ""
echo "== Aplicatia ruleaza. == Browserul s-a deschis la adresa:"
echo "   http://localhost:$HUB_PORT"
echo "Puteti folosi linistit aplicatia; aceasta fereastra se poate inchide."
read -p "Apasati Enter pentru a inchide fereastra... " _
