# AIOCR Deployment Guide

## Prerequisites

- Docker 24+ and Docker Compose v2
- Python 3.11+
- 8GB RAM minimum (16GB recommended for LLM inference)
- NVIDIA GPU (optional but recommended for ESRGAN + TrOCR)

---

## Quick Start (Local Development)

```bash
# 1. Clone and enter repo
git clone <repo-url>
cd Degraded-Devanagari-and-Bangla-Script-Identification-Using-CNN-Frameworks

# 2. Set up environment
cp .env.example .env
# Edit .env: add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)

# 3. Start all services
docker compose up -d

# 4. Check service health
curl http://localhost:8000/health

# 5. Open API docs
open http://localhost:8000/docs
```

---

## Step-by-Step Setup

### 1. Train CNN Models

The system uses pre-trained CNN models for script detection.
Train them from the Ekush dataset:

```bash
# Install dependencies
pip install -r requirements.txt

# Train the custom CNN (matches notebook architecture, ~99% accuracy)
python scripts/train_model.py \
  --data-dir /path/to/processed_data \
  --model custom_cnn \
  --epochs 20

# Train all architectures for ensemble
python scripts/train_model.py \
  --data-dir /path/to/processed_data \
  --model all

# Models saved to saved_models/
```

Dataset structure required:
```
processed_data/
├── Bangla/      (3,814+ images)
└── Devanagari/  (3,814+ images)
```

### 2. Ingest Indic Corpus (RAG)

```bash
# Start ChromaDB first
docker compose up chromadb -d

# Ingest Devanagari text corpus
python scripts/ingest_corpus.py \
  --corpus-dir /path/to/hindi-texts \
  --script devanagari

# Ingest Bengali text corpus
python scripts/ingest_corpus.py \
  --corpus-dir /path/to/bengali-texts \
  --script bangla
```

### 3. Run with Docker Compose

```bash
# Start full stack
docker compose up -d

# View logs
docker compose logs -f api
docker compose logs -f worker

# Monitor Celery tasks
open http://localhost:5555  # Flower dashboard

# Scale workers for heavy load
docker compose up -d --scale worker=4
```

### 4. API Usage

#### Detect Script Only
```bash
curl -X POST http://localhost:8000/api/v1/detect-script/ \
  -F "file=@document.png"
```

Response:
```json
{
  "request_id": "...",
  "script": "devanagari",
  "confidence": 0.99,
  "model_used": "ensemble",
  "processing_time_ms": 45.2
}
```

#### Restore Degraded Image
```bash
curl -X POST http://localhost:8000/api/v1/restore-image/?return_base64=true \
  -F "file=@degraded.png"
```

#### OCR Only (Fast Path)
```bash
curl -X POST http://localhost:8000/api/v1/ocr/ \
  -F "file=@document.png" \
  -G -d "apply_correction=true"
```

#### Full 7-Agent Pipeline
```bash
curl -X POST http://localhost:8000/api/v1/full-pipeline/ \
  -F "file=@document.png" \
  -G -d "enable_restoration=true&enable_rag=true&include_annotated_image=true"
```

Response:
```json
{
  "request_id": "abc-123",
  "status": "completed",
  "script": "devanagari",
  "raw_text": "भारत का इतिहस",
  "corrected_text": "भारत का इतिहास",
  "overall_confidence": 0.96,
  "text_regions": [...],
  "bounding_boxes": [...],
  "language": "Hindi/Sanskrit/Marathi",
  "reasoning": "Corrected 'इतिहस' → 'इतिहास' (missing matra)",
  "corrections_made": ["इतिहस → इतिहास: Missing aa-matra"],
  "agent_statuses": [
    {"agent_name": "ScriptDetectionAgent", "status": "completed", "processing_time_ms": 45},
    {"agent_name": "ImageRestorationAgent", "status": "completed", "processing_time_ms": 230},
    ...
  ],
  "annotated_image_base64": "<base64-image>",
  "processing_time_ms": 3420.5
}
```

#### Async Processing (Large Documents)
```bash
# Submit job
curl -X POST "http://localhost:8000/api/v1/full-pipeline/async" \
  -F "file=@large_document.pdf"

# Returns: {"job_id": "xyz-789", "status": "pending", ...}

# Poll status
curl http://localhost:8000/api/v1/full-pipeline/status/xyz-789
```

---

## Kubernetes Deployment

```bash
# Create namespace
kubectl apply -f k8s/service.yaml  # Contains namespace definition

# Create secrets (fill in values first)
cp k8s/secrets.yaml.example k8s/secrets.yaml
# Edit k8s/secrets.yaml with base64-encoded values
kubectl apply -f k8s/secrets.yaml

# Apply configuration
kubectl apply -f k8s/configmap.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify deployment
kubectl get pods -n aiocr
kubectl get services -n aiocr

# View API logs
kubectl logs -n aiocr -l component=api -f
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (for LLM correction) | Anthropic Claude API key |
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `REDIS_URL` | Yes | Redis connection string |
| `MODEL_DIR` | Yes | Path to saved CNN models |
| `OPENAI_API_KEY` | No | Fallback LLM (OpenAI GPT-4) |
| `CHROMA_HOST` | No | ChromaDB host (for RAG) |
| `SENTRY_DSN` | No | Error monitoring |

---

## Architecture Components

```
┌─────────────────────────────────────────────────────────┐
│                        AIOCR Stack                       │
├─────────────────────────────────────────────────────────┤
│  Nginx (port 80)          → Load balancer / proxy       │
│  FastAPI (port 8000)      → REST API                    │
│  Celery Workers           → Async task processing       │
│  Flower (port 5555)       → Task monitoring             │
├─────────────────────────────────────────────────────────┤
│  PostgreSQL (port 5432)   → Metadata storage            │
│  Redis (port 6379)        → Cache + task queue          │
│  ChromaDB (port 8001)     → Vector store (RAG)          │
└─────────────────────────────────────────────────────────┘
```

---

## Running Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test file
pytest tests/test_agents.py -v
pytest tests/test_api.py -v
```

---

## Monitoring

- **API metrics**: `GET /metrics` (Prometheus format)
- **Task dashboard**: `http://localhost:5555` (Flower)
- **API docs**: `http://localhost:8000/docs`
- **Health check**: `GET /health`
