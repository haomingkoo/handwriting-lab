# Handwriting Model Pipeline

## 1. Data Processing
`src/process_data.py` reads raw MNIST-style folders and writes processed images + CSV manifests.

Outputs:
- `train.csv` with filepaths and labels
- `test.csv` with filepaths and labels
- processed image files under `data/processed/...`

## 2. Model Training
`src/train_model.py` loads processed data, trains the classifier, evaluates each epoch, and writes:
- checkpoint artifacts (`checkpoint_model.pt`)
- final model weights (`data/model.pth`)

Training uses Hydra config from `conf/train_model.yaml` and can be tracked in MLflow when enabled.

## 3. Robustness Augmentation
To improve predictions on rotated or upside-down handwriting, training supports augmentation:
- `enable_train_augmentation`
- `train_rotation_degrees`
- `train_rotation_prob`
- `train_invert_prob`

These are applied at training time so the model sees multiple variants over epochs.

## 4. Artifact Promotion
Local retrain workflow:
`scripts/retrain_local_model.sh`

This script can:
- process raw data
- train locally with augmentation
- stage `model.pth` for deployment (`data/local-model-export/model.pth`)
- auto-copy Railway artifact (`artifacts/model.pth`)
- optionally upload to GCS

## 5. Inference Deployment
FastAPI (`src/mnist_fastapi`) loads the committed model weights at startup and serves:
- `POST /api/v1/model/predict`
- `POST /api/v1/model/batch`
- `GET /api/v1/model/version`
- `GET /` (custom handwriting web UI)

The custom web frontend calls these endpoints directly.

Recommended public-demo deployment is a single FastAPI service using
`inference.Containerfile`. That service hosts both:
- the browser UI at `/`
- the inference API under `/api/v1/model/*`

The web UI and the API share one origin, so other websites cannot call the API from a browser unless they are on the allowed list.

## 6. UI Prediction Flow
1. User draws digit in browser canvas
2. UI converts image to grayscale 28x28 PNG
3. UI posts image to FastAPI endpoint
4. Backend predicts digit and returns JSON
5. UI renders prediction and diagnostics

## 7. Evaluation Report
Run:
`uv run --group training python scripts/evaluate_model.py`

Output:
- `reports/evaluation_latest.json`

The web app reads this file and displays:
- confusion matrix table
- per-class precision/recall/F1 table
- top misclassification pairs

## 8. Security Controls
Runtime protections currently in the inference service:
- state-dict-only model loading for safer checkpoint handling
- upload validation on inference endpoints (type, size, batch count)
- browser calls from other websites are limited to an allowed list
- per-IP app-level throttling on inference `POST` routes

These controls are suitable for a public demo, but infrastructure-side protections
such as CDN/WAF rate limiting and TLS enforcement should still be enabled in the
hosting platform.
