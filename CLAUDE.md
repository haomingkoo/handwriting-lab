# CLAUDE.md

Handwritten digit recognition app: draw a digit, get a live MNIST prediction from a PyTorch CNN served over FastAPI. Part of the kooexperience.com portfolio (`handwriting.kooexperience.com`).

## Tech Stack

- **Backend**: FastAPI + Uvicorn, served at `/api/v1/model/*`
- **Model**: PyTorch CNN (`src/mnist/modeling/models.py::Net`) — 2 conv layers + 2 FC layers, trained on MNIST
- **Frontend**: Custom vanilla HTML/JS (`src/mnist_fastapi/web/index.html`), served by FastAPI at `/`
- **Legacy frontend**: Streamlit app (`streamlit_app.py`, `src/mnist_streamlit/`) — not used for Railway deploy
- **Training**: Hydra + MLflow (`src/train_model.py`, `conf/`)
- **Package manager**: `uv`
- **Deploy**: Railway (Docker via `inference.Containerfile`), GitLab CI (`.gitlab-ci.yml`) handles training/model-registry sync to GCP/Cloud Run

## Commands

```bash
# Install all dependency groups
uv sync --group dev --group training --group backend

# Run tests
uv run pytest src/tests/

# Lint
uv run ruff check src/

# Run FastAPI app locally (serves UI at / and API at /api/v1/model/*)
uv run --group backend uvicorn mnist_fastapi.main:APP --app-dir src --reload

# Retrain model locally and stage artifacts/model.pth
SKIP_PROCESS_DATA=true NO_CUDA=true ./scripts/retrain_local_model.sh
```

## Architecture

- `src/mnist/` — shared library: data prep (`data_prep/`), model definition + load/predict utils (`modeling/`)
- `src/mnist_fastapi/` — inference service
  - `main.py` — app setup, CORS, in-memory rate limiter, serves web UI + `/healthz`
  - `config.py` — `Settings` (pydantic-settings), all env-configurable (CORS origins, rate limits, model path, upload guardrails)
  - `deps.py` — loads model + device once at startup (singleton, not per-request)
  - `v1/routers/model.py` — `POST /predict`, `POST /batch`, `GET /version`
  - `web/index.html` — canvas drawing UI, calls the API directly
- `src/train_model.py` — Hydra-driven training entry point, logs to MLflow
- `artifacts/model.pth` — baked model checkpoint for single-service Railway deploy
- `reports/evaluation_latest.json` — latest eval metrics (confusion matrix, per-class), served to the UI's Evaluation tab
- `docs/training_pipeline.md` — served to the UI's Pipeline Guide tab

## Key env vars (see README.md for full list)

`PRED_MODEL_PATH`, `USE_CUDA`, `USE_MPS`, `CORS_ALLOW_ORIGINS`, `RATE_LIMIT_ENABLED`, `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`

## Notes

- `pyproject.toml` project name is still `assignment3` (this started as a training assignment) — actual product is "Handwriting Lab".
- Tests are thin: `src/tests/test_dummy.py` is a CI placeholder; `src/tests/test_data_prep.py` covers data prep. No tests for the FastAPI routes or model inference yet.
