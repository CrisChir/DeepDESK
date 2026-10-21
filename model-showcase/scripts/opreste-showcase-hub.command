#!/bin/bash
# ============================================================
# Showcase Hub — OPRIRE (macOS, dublu-click in Finder)
#
# Opreste toate serviciile, pastrand intacte datele si modelele.
# NU foloseste -v: volumele (modelele GenD, greutatile DeepDESK)
# raman pe disc.
#
# Instalare pe Desktop: vezi scripts/README.md.
# Daca proiectul este in alta parte: editati lista CAUTARI din acest
# script sau setati variabila SHOWCASE_HUB_DIR inainte de rulare.
# ============================================================

echo "== Showcase Hub — oprire =="

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

# ---------- 2. Verificarea ca Docker ruleaza ----------
if ! command -v docker >/dev/null 2>&1; then
    echo "Docker nu este instalat pe acest calculator."
    read -p "Apasati Enter pentru a inchide fereastra... " _ ; exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "Docker nu ruleaza — aplicatia este deja oprita. Nu este nevoie de nimic."
    read -p "Apasati Enter pentru a inchide fereastra... " _ ; exit 0
fi

# ---------- 3. Oprirea serviciilor ----------
echo "Opresc toate serviciile..."
docker compose --profile full stop
echo ""
echo "== Aplicatia a fost oprita. Modelele si datele au ramas salvate. =="
read -p "Apasati Enter pentru a inchide fereastra... " _
