FROM python:3.10-slim

# git is required to install the OpenAI CLIP package from source
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/deepdesk

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir git+https://github.com/openai/CLIP.git

COPY app ./app

# Weights are downloaded to this volume on first start (mount a volume to persist).
RUN mkdir -p /data/weights
ENV WEIGHTS_DIR=/data/weights

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
