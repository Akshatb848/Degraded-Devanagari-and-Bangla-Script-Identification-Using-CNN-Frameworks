# Manuscript Intelligence: Devanagari & Bangla OCR

An OCR system for degraded Indic manuscripts. It cleans up the scanned
image, works out whether the script is **Devanagari** or **Bangla**, runs OCR
in the right language, and can use an LLM to correct the OCR output.

The repo has two parts:

1. **Research**: `Degraded_Script_Classifier.ipynb`, a Keras CNN that tells
   Devanagari from Bangla character images (see [Research results](#research-results)).
2. **Application**: a Streamlit demo app, plus a FastAPI service backed by a
   LangGraph multi-agent pipeline, a Celery worker for async jobs, and
   Docker / Kubernetes manifests.

> **Model status:** this repository contains **no trained weights**. Without
> them the apps run in a clearly labelled fallback mode. Script detection
> returns `unknown`, the API sets `model_status: "untrained_fallback"`, and
> Streamlit shows a warning banner. The [Training](#training-the-cnn) section
> shows how to produce weights.

---

## Architecture

```
                        ┌──────────────────────── Streamlit app (streamlit_app.py) ───────────────────────┐
 image upload ────────► │ 1 Restoration      2 Script detection     3 OCR             4 LLM reconstruction │
                        │ OpenCV denoise,     CNN if weights exist, Tesseract          Claude / OpenAI       │
                        │ threshold, deskew   else heuristic/unknown hin+san / ben     (optional API keys)   │
                        └──────────────────────────── core/ ──────────────────────────────────────────────────┘

                        ┌──────────────────────── FastAPI service (app/) ─────────────────────────────────────┐
 POST /api/v1/...  ───► │ LangGraph orchestrator (agents/orchestrator.py), 7 agents in sequence:              │
                        │ script detection → image restoration → text detection → character recognition →     │
                        │ LLM correction → knowledge retrieval (RAG) → output formatting                       │
                        │ /full-pipeline/async ──► Redis ──► Celery worker (workers/celery_app.py)            │
                        └──────────────────────────────────────────────────────────────────────────────────────┘
```

Each agent tries its stronger back-end first and falls back if that back-end
is not installed. For example, text detection tries YOLO, then docTR, then
OpenCV connected components. OCR tries TrOCR, then Tesseract. RAG uses
ChromaDB and falls back to basic dictionary validation. The heavy back-ends are
optional (`requirements-extras.txt`).

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, plus `model_status` of the CNN classifier |
| `POST /api/v1/detect-script/` | Devanagari / Bangla classification (`model_status`, `warning` fields) |
| `POST /api/v1/restore-image/` | Denoise / CLAHE / deskew / optional super-resolution |
| `POST /api/v1/ocr/` | Text detection + recognition (+ optional LLM correction) |
| `POST /api/v1/full-pipeline/` | All 7 agents, synchronous |
| `POST /api/v1/full-pipeline/async` | Queue the pipeline on Celery; poll `GET /api/v1/full-pipeline/status/{job_id}` |

Interactive API docs are at `http://localhost:8000/docs`.

---

## Running locally

Requires Python 3.11–3.13. For OCR, install Tesseract with the Hindi,
Sanskrit and Bengali language packs (`packages.txt` lists the Debian
packages: `tesseract-ocr tesseract-ocr-hin tesseract-ocr-ben`). Without
Tesseract the apps still start, but OCR returns no text.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # runtime deps + pytest
cp .env.example .env                       # optional: ANTHROPIC_API_KEY / OPENAI_API_KEY
```

| Component | Command |
|---|---|
| **Streamlit app** | `streamlit run streamlit_app.py` → http://localhost:8501 |
| **API** | `uvicorn app.main:app --reload --port 8000` → http://localhost:8000/docs |
| **Worker** (needs Redis) | `redis-server` then `celery -A workers.celery_app worker --loglevel=info --queues=pipeline,detection` |

The API and worker read Redis/Celery URLs from the environment (defaults:
`redis://localhost:6379/0..2`). Redis is only needed for the `/async`
endpoint and the worker. Everything else runs without it.

All three commands above were run end to end against this branch, including
an async job going through Redis → Celery → status endpoint.

### Docker

```bash
cp .env.example .env
docker compose up --build        # api, worker, redis, postgres, chromadb, flower, nginx
```

To make the images load trained weights, build them with
`--build-arg WITH_TENSORFLOW=true`. The Docker setup itself was not
re-verified for this README (no Docker daemon was available). Kubernetes
manifests are in `k8s/`. See [DEPLOYMENT.md](DEPLOYMENT.md).

### Requirements files

| File | Contents |
|---|---|
| `requirements.txt` | Runtime for Streamlit, API and worker (no TensorFlow) |
| `requirements-train.txt` | + TensorFlow, to train or load the CNN |
| `requirements-extras.txt` | Optional agent back-ends: TrOCR/torch, YOLO, docTR, Real-ESRGAN, ChromaDB (unpinned, not part of CI) |
| `requirements-dev.txt` | + pytest, pytest-asyncio, httpx |

---

## Tests

```bash
pytest -q
```

This runs 30 tests (API endpoints, schemas, agents, model-status reporting).
One test needs TensorFlow and is skipped with a reason when it is not
installed. Install `requirements-train.txt` to run it.

---

## Training the CNN

`scripts/train_model.py` trains the notebook's custom CNN (or VGG16,
DenseNet121 or ResNet50 transfer-learning variants) from a folder with one
sub-folder per class:

```
data/
├── Bangla/        *.jpg|png
└── Devanagari/    *.jpg|png
```

```bash
pip install -r requirements-train.txt
python scripts/train_model.py --data-dir data --output-dir saved_models --model custom_cnn --epochs 20
```

This writes `saved_models/script_classifier.keras`, using an 80/20
train/validation split with augmentation. It was smoke-tested on a small
synthetic dataset, not retrained on Ekush. To use the weights:

- **Streamlit** picks up `saved_models/script_classifier.keras` or
  `models/script_classifier.{keras,h5}` automatically.
- **API / worker**: set `MODEL_DIR=saved_models` (default
  `/app/saved_models` in Docker). `/health` then reports
  `model_status: {"cnn_classifier": "trained"}`.

Model input is 64×64 RGB scaled to [0, 1], with classes `Bangla=0` and
`Devanagari=1` (Keras `flow_from_directory` order).

---

## Research results

From the executed outputs in `Degraded_Script_Classifier.ipynb`:

| Model | Data | Reported accuracy |
|---|---|---|
| Custom CNN (3× Conv2D/MaxPool → Dense 128 → Dropout 0.5 → softmax, 64×64 RGB, augmentation, early stopping) | 7,628 images, 3,814 per class (Bangla / Devanagari), Ekush-derived | **99.08%** (`Test accuracy: 0.9908`) |

How to read this number: in the notebook, the "test" generator reads the
same `/content/processed_data` folder as the training generator (only
without augmentation). So 99.08% is accuracy on the training images, not on
a held-out test set. A proper held-out evaluation is still to be done.

> **Note:** [README_DSC.md](README_DSC.md) lists a comparison of VGG-16
> (99.34%), DenseNet-121, AlexNet and ResNet-50. **Those models are not
> trained in the committed notebook, and those figures are not reproduced
> by anything in this repository.** `scripts/train_model.py` can train
> VGG16, DenseNet121 and ResNet50 variants, but no results from it are
> committed.

---

## Project structure

```
├── Degraded_Script_Classifier.ipynb  research notebook (custom CNN)
├── streamlit_app.py                  Streamlit UI (4-step pipeline)
├── core/                             Streamlit pipeline: preprocessing, classifier, Tesseract OCR, LLM corrector
├── app/                              FastAPI service: main.py, routes/, schemas, config
├── agents/                           LangGraph orchestrator + 7 agents
├── models/cnn_classifier.py          CNN loader / inference used by the API & worker
├── services/                         Redis job-status cache, image helpers
├── workers/celery_app.py             Celery tasks for async pipeline jobs
├── scripts/train_model.py            CNN training
├── scripts/ingest_corpus.py          ChromaDB corpus ingestion for RAG (needs requirements-extras)
├── tests/                            pytest suite
├── Dockerfile, docker/, docker-compose.yml, k8s/   deployment
└── README_DSC.md, DEPLOYMENT.md      original project notes, deployment guide
```
