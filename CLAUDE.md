# CLAUDE.md

MNIST digit recognizer: FastAPI + PyTorch CNN with a vanilla JS canvas UI, live at handwriting.kooexperience.com.

## Commands

```bash
uv sync --group dev --group training --group backend
uv run pytest src/tests/          # data-prep tests need the training group
uv run ruff check src/
uv run --group backend uvicorn mnist_fastapi.main:APP --app-dir src --reload
SKIP_PROCESS_DATA=true NO_CUDA=true ./scripts/retrain_local_model.sh   # retrain, writes artifacts/model.pth
```

## Deploy

- Railway project `profound-essence` auto-deploys every push to `main`; there are no watch paths, so docs-only merges rebuild too.
- Build is `inference.Containerfile` (set in `railway.toml`), `uv sync --frozen --group backend`, so `uv.lock` must stay in sync with `pyproject.toml`.
- No GitHub CI. `.gitlab-ci.yml` is a leftover course template and does not run.

## Gotchas

- The served model is the committed `artifacts/model.pth`, baked into the image. Retraining only ships once that file is committed.
- `docs/training_pipeline.md` and `reports/evaluation_latest.json` are served to the UI's Pipeline Guide and Evaluation tabs, so editing them changes the live site.
- The Streamlit app (`streamlit_app.py`, `src/mnist_streamlit/`, `streamlit.Containerfile`) is legacy and not deployed.
- `pyproject.toml` name `assignment3` is historical; the product is Handwriting Lab.
- Tests cover data prep only; nothing tests the API routes or inference.
- The UI self-hosts its font from `src/mnist_fastapi/web/fonts`; do not add third-party font or script hosts (privacy fix #2).
