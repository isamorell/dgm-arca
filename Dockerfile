# ARCA · imagen única para servir la API (todas las fases) y, si tenéis una máquina con GPU y
# Docker, también para entrenar. En las sesiones de la DGX no hay Docker: ver docs/dgx.md.
#
# La imagen base solo aporta las librerías mínimas de CUDA: las ruedas de torch para
# Linux que fija uv.lock (índice cu128) ya incluyen su propio runtime de CUDA. Lo único
# que necesita la máquina anfitriona es el driver de NVIDIA y nvidia-container-toolkit.
#
#   docker compose build
#   docker compose run --rm check-gpu
#   docker compose run --rm smoke
#   docker compose up api

FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/data/hf_cache \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv se copia desde su imagen oficial: sin pip, sin conda, versión fijada.
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

WORKDIR /app

# Primero solo los ficheros de dependencias, para que esta capa (la pesada) se cachee
# mientras el código cambia.
COPY pyproject.toml uv.lock README.md ./
RUN uv python install 3.11 \
    && uv sync --frozen --no-install-project --extra train --extra rag --extra agent

# Después el resto del proyecto. El entorno vive en /opt/venv, fuera de /app, para que
# montar el código con un volumen no lo tape.
COPY . .
RUN uv sync --frozen --extra train --extra rag --extra agent

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000
CMD ["arca-api"]
