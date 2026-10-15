# GPU image for inference, evaluation, training, and the web app. Build from the repository root:
#   docker build -f docker/Dockerfile -t three-lidars .
#   docker run --rm --gpus all -v $PWD/weights:/app/weights -v $PWD/examples:/data \
#       three-lidars predict /data/05554_Pinus_sylvestris_ULS.laz
# Web app (or `docker compose -f docker/compose.yaml up -d`):
#   docker run -d --gpus all -p 127.0.0.1:8000:8000 -v $PWD/weights:/app/weights -v $PWD/mvtree_data:/data \
#       three-lidars serve --host 0.0.0.0 --data-dir /data
# The PyTorch CUDA 12.8 wheels bundle the CUDA runtime, so the base image needs
# no CUDA toolkit; the host needs an NVIDIA driver >= 570 and the NVIDIA
# Container Toolkit. Without --gpus the image runs on CPU.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    MVTREE_WEIGHTS=/app/weights

# libgomp: LightGBM; libgl/glib are not needed (no OpenCV).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Dependencies first so code edits do not invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra geometry --extra train

COPY mvtree ./mvtree
COPY scripts ./scripts
COPY configs ./configs
COPY splits ./splits
COPY results ./results
COPY data/metadata ./data/metadata
COPY examples ./examples
COPY weights/manifest.json ./weights/manifest.json
RUN uv sync --frozen --extra geometry --extra train

EXPOSE 8000
ENTRYPOINT ["mvtree"]
CMD ["--help"]
