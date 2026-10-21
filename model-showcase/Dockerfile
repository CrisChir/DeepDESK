# ============================================================
# Showcase Hub — front-end Gradio unificat
# Imagine CPU ușoară: Gradio + clienți HTTP către backend-uri.
#
# Build:
#   docker build -t showcase-hub:latest .
# Run:
#   docker run --rm -p 7870:7870 showcase-hub:latest
# ============================================================

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    HUB_HOST=0.0.0.0 \
    HUB_PORT=7870

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 7870

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import httpx; httpx.get('http://127.0.0.1:7870/', timeout=4)" || exit 1

CMD ["python", "app/main.py"]
