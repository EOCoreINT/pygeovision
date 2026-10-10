# ─── PyGeoVision v2.0 Production Docker Image ──────────────────────────────
# Multi-stage build: base → deps → test → production → gpu
#
# Usage:
#   docker build -t pygeovision:2.1.8 .
#   docker build --target gpu -t pygeovision:2.1.8-gpu .
#   docker run -p 8080:8080 -v $(pwd)/models:/models pygeovision:2.1.8
#   docker run pygeovision:2.1.8 pygeovision status

# ── Stage 1: base system ─────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    PGV_LOG_LEVEL=INFO

# System dependencies for geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev libproj-dev libgeos-dev libsqlite3-dev \
        libexpat1-dev git curl build-essential proj-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: Python dependencies ─────────────────────────────────────────────
FROM base AS deps

COPY pyproject.toml ./
COPY pygeovision/_version.py ./pygeovision/
COPY pygeovision/__init__.py ./pygeovision/

RUN pip install --upgrade pip && \
    pip install ".[geo,serve]" PyGeoFetch && \
    pip cache purge

# ── Stage 3: full source ──────────────────────────────────────────────────────
FROM deps AS source

COPY pygeovision/ ./pygeovision/
COPY tests/        ./tests/

# ── Stage 4: test ─────────────────────────────────────────────────────────────
FROM source AS test

RUN pip install ".[dev]" && \
    python -m pytest tests/ -q --tb=short \
        --ignore=tests/test_data_layer.py && \
    echo "✓ All tests passed"

# ── Stage 5: production (CPU) ─────────────────────────────────────────────────
FROM deps AS production

COPY pygeovision/ ./pygeovision/
COPY projects/    ./projects/
COPY pyproject.toml ./

# Non-root user
RUN useradd -m -u 1000 pgvuser && \
    mkdir -p /models /data /results && \
    chown -R pgvuser:pgvuser /app /models /data /results
USER pgvuser

VOLUME ["/models", "/data", "/results"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default: inference API server
ENTRYPOINT ["pygeovision"]
CMD ["serve", "start", "--host", "0.0.0.0", "--port", "8080"]

# ── Stage 6: GPU (CUDA 12.1 + cuDNN 8) ───────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS gpu

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive \
    PGV_GPU_DEVICE=cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        libgdal-dev libproj-dev libgeos-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install ".[geo,serve]" PyGeoFetch && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip cache purge

COPY pygeovision/ ./pygeovision/
COPY projects/    ./projects/

RUN useradd -m -u 1000 pgvuser && \
    mkdir -p /models /data /results && \
    chown -R pgvuser:pgvuser /app /models /data /results
USER pgvuser

VOLUME ["/models", "/data", "/results"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["pygeovision"]
CMD ["serve", "start", "--host", "0.0.0.0", "--port", "8080"]
