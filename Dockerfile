# Hugging Face Docker Space — Yield Copilot dashboard
# https://huggingface.co/docs/hub/spaces-sdks-docker
#
# Serves dashboard.server:app on 0.0.0.0:7860 with models/v0-ens3 and the
# validation stream. No raw WM-811K dataset required (stream maps are bundled).

FROM python:3.11-slim-bookworm

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAM_CSV=/app/streams/stream_val_ens3.csv \
    WAFER_MODEL_DIR=/app/models/v0-ens3 \
    WAFER_CLASSIFIER=cnn \
    # AMD Token Factory — set at host/Space runtime (never bake secrets into the image)
    AMD_BASE_URL= \
    AMD_MODEL= \
    AMD_API_KEY=

WORKDIR /app

# System libs for opencv-headless / pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user requirements-space.txt /app/requirements-space.txt
# Torch + torchvision from the same CPU index (ultralytics needs matching torchvision ops).
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch>=2.2,<3" "torchvision>=0.17,<1" \
 && pip install --no-cache-dir -r /app/requirements-space.txt

# App code + demo assets only (see .dockerignore)
COPY --chown=user wafer /app/wafer
COPY --chown=user dashboard /app/dashboard
COPY --chown=user models/v0-ens3 /app/models/v0-ens3
COPY --chown=user models/yolo-cls-exp /app/models/yolo-cls-exp
COPY --chown=user streams/stream_val_ens3.csv /app/streams/stream_val_ens3.csv
COPY --chown=user streams/stream_val_yolo.csv /app/streams/stream_val_yolo.csv
COPY --chown=user docs/results/report /app/docs/results/report
COPY --chown=user docs/results/v0-ens3_val /app/docs/results/v0-ens3_val
COPY --chown=user pyproject.toml /app/pyproject.toml
COPY --chown=user LICENSE /app/LICENSE

USER user
EXPOSE 7860

CMD ["uvicorn", "dashboard.server:app", "--host", "0.0.0.0", "--port", "7860"]
