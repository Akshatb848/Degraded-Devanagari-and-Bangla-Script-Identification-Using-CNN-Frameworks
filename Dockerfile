# AIOCR API Service Dockerfile
# Multi-stage build for minimal production image

# === Stage 1: Builder ===
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for CV2, PostgreSQL, and ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# === Stage 2: Runtime ===
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-ben \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN groupadd -r aiocr && useradd -r -g aiocr aiocr

# Create required directories
RUN mkdir -p /app/saved_models /app/logs /app/temp && \
    chown -R aiocr:aiocr /app

# Copy application code
COPY --chown=aiocr:aiocr app/ ./app/
COPY --chown=aiocr:aiocr agents/ ./agents/
COPY --chown=aiocr:aiocr models/ ./models/
COPY --chown=aiocr:aiocr services/ ./services/
COPY --chown=aiocr:aiocr workers/ ./workers/
COPY --chown=aiocr:aiocr scripts/ ./scripts/

USER aiocr

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
