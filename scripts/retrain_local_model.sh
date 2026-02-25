#!/usr/bin/env bash
set -euo pipefail

# Local retraining helper for robustness against rotated/inverted handwriting.
# This script keeps the workflow aligned with train_model.py + Hydra config.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RAW_DATA_DIR="${RAW_DATA_DIR:-${ROOT_DIR}/data/mnist-pngs-data-aisg}"
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-${ROOT_DIR}/data/processed/mnist-pngs-data-aisg-processed}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
NO_CUDA="${NO_CUDA:-true}"
NO_MPS="${NO_MPS:-false}"
SETUP_MLFLOW="${SETUP_MLFLOW:-false}"
MLFLOW_TRACKING_URI_LOCAL="${MLFLOW_TRACKING_URI_LOCAL:-./mlruns}"
MLFLOW_EXP_NAME_LOCAL="${MLFLOW_EXP_NAME_LOCAL:-mnist-local-retrain}"
LOG_DIR_LOCAL="${LOG_DIR_LOCAL:-${ROOT_DIR}/logs}"
CHECKPOINT_DIR_LOCAL="${CHECKPOINT_DIR_LOCAL:-${ROOT_DIR}/models_local}"
RUN_EVALUATION_AFTER_TRAIN="${RUN_EVALUATION_AFTER_TRAIN:-true}"
EVAL_REPORT_PATH="${EVAL_REPORT_PATH:-${ROOT_DIR}/reports/evaluation_latest.json}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
TRAINED_MODEL_SOURCE="${TRAINED_MODEL_SOURCE:-}"

ROTATION_DEGREES="${ROTATION_DEGREES:-45}"
ROTATION_PROB="${ROTATION_PROB:-0.7}"
AFFINE_PROB="${AFFINE_PROB:-0.6}"
AFFINE_TRANSLATE_X="${AFFINE_TRANSLATE_X:-0.12}"
AFFINE_TRANSLATE_Y="${AFFINE_TRANSLATE_Y:-0.12}"
AFFINE_SCALE_MIN="${AFFINE_SCALE_MIN:-0.9}"
AFFINE_SCALE_MAX="${AFFINE_SCALE_MAX:-1.1}"
AFFINE_SHEAR_DEGREES="${AFFINE_SHEAR_DEGREES:-10}"
PERSPECTIVE_PROB="${PERSPECTIVE_PROB:-0.25}"
PERSPECTIVE_DISTORTION_SCALE="${PERSPECTIVE_DISTORTION_SCALE:-0.2}"
INVERT_PROB="${INVERT_PROB:-0.05}"

SKIP_PROCESS_DATA="${SKIP_PROCESS_DATA:-false}"

RUN_NAME="${RUN_NAME:-local_rotation_retrain}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/outputs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"

MODEL_EXPORT_DIR="${MODEL_EXPORT_DIR:-${ROOT_DIR}/data/local-model-export}"
MODEL_EXPORT_PATH="${MODEL_EXPORT_PATH:-${MODEL_EXPORT_DIR}/model.pth}"
STAGE_FOR_RAILWAY="${STAGE_FOR_RAILWAY:-true}"
RAILWAY_ARTIFACT_DIR="${RAILWAY_ARTIFACT_DIR:-${ROOT_DIR}/artifacts}"
RAILWAY_MODEL_PATH="${RAILWAY_MODEL_PATH:-${RAILWAY_ARTIFACT_DIR}/model.pth}"

UPLOAD_TO_GCS="${UPLOAD_TO_GCS:-false}"
GCS_TARGET_PATH="${GCS_TARGET_PATH:-gs://aiap21-aut0/haoming_koo/inference/model.pth}"

echo "=== Local Retrain Configuration ==="
echo "ROOT_DIR=${ROOT_DIR}"
echo "RAW_DATA_DIR=${RAW_DATA_DIR}"
echo "PROCESSED_DATA_DIR=${PROCESSED_DATA_DIR}"
echo "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
echo "NO_CUDA=${NO_CUDA}"
echo "NO_MPS=${NO_MPS}"
echo "SETUP_MLFLOW=${SETUP_MLFLOW}"
echo "MLFLOW_TRACKING_URI_LOCAL=${MLFLOW_TRACKING_URI_LOCAL}"
echo "MLFLOW_EXP_NAME_LOCAL=${MLFLOW_EXP_NAME_LOCAL}"
echo "LOG_DIR_LOCAL=${LOG_DIR_LOCAL}"
echo "CHECKPOINT_DIR_LOCAL=${CHECKPOINT_DIR_LOCAL}"
echo "RUN_EVALUATION_AFTER_TRAIN=${RUN_EVALUATION_AFTER_TRAIN}"
echo "EVAL_REPORT_PATH=${EVAL_REPORT_PATH}"
echo "SKIP_TRAIN=${SKIP_TRAIN}"
echo "TRAINED_MODEL_SOURCE=${TRAINED_MODEL_SOURCE}"
echo "ROTATION_DEGREES=${ROTATION_DEGREES}"
echo "ROTATION_PROB=${ROTATION_PROB}"
echo "AFFINE_PROB=${AFFINE_PROB}"
echo "AFFINE_TRANSLATE_X=${AFFINE_TRANSLATE_X}"
echo "AFFINE_TRANSLATE_Y=${AFFINE_TRANSLATE_Y}"
echo "AFFINE_SCALE_MIN=${AFFINE_SCALE_MIN}"
echo "AFFINE_SCALE_MAX=${AFFINE_SCALE_MAX}"
echo "AFFINE_SHEAR_DEGREES=${AFFINE_SHEAR_DEGREES}"
echo "PERSPECTIVE_PROB=${PERSPECTIVE_PROB}"
echo "PERSPECTIVE_DISTORTION_SCALE=${PERSPECTIVE_DISTORTION_SCALE}"
echo "INVERT_PROB=${INVERT_PROB}"
echo "RUN_DIR=${RUN_DIR}"
echo "MODEL_EXPORT_PATH=${MODEL_EXPORT_PATH}"
echo "STAGE_FOR_RAILWAY=${STAGE_FOR_RAILWAY}"
echo "RAILWAY_MODEL_PATH=${RAILWAY_MODEL_PATH}"

if [[ "${SKIP_PROCESS_DATA}" != "true" ]]; then
    echo "=== Processing raw data ==="
    uv run --group training python src/process_data.py \
        raw_data_dir="${RAW_DATA_DIR}" \
        processed_data_dir="${PROCESSED_DATA_DIR}"
fi

if [[ "${SKIP_TRAIN}" != "true" ]]; then
    echo "=== Training model with augmentation ==="
    uv run --group training python src/train_model.py \
        setup_mlflow="${SETUP_MLFLOW}" \
        mlflow_tracking_uri="${MLFLOW_TRACKING_URI_LOCAL}" \
        mlflow_exp_name="${MLFLOW_EXP_NAME_LOCAL}" \
        mlflow_run_name="${RUN_NAME}" \
        data_dir_path="${PROCESSED_DATA_DIR}" \
        epochs="${TRAIN_EPOCHS}" \
        no_cuda="${NO_CUDA}" \
        no_mps="${NO_MPS}" \
        log_dir="${LOG_DIR_LOCAL}" \
        model_checkpoint_dir_path="${CHECKPOINT_DIR_LOCAL}" \
        enable_train_augmentation=true \
        train_rotation_degrees="${ROTATION_DEGREES}" \
        train_rotation_prob="${ROTATION_PROB}" \
        train_affine_prob="${AFFINE_PROB}" \
        train_affine_translate_x="${AFFINE_TRANSLATE_X}" \
        train_affine_translate_y="${AFFINE_TRANSLATE_Y}" \
        train_affine_scale_min="${AFFINE_SCALE_MIN}" \
        train_affine_scale_max="${AFFINE_SCALE_MAX}" \
        train_affine_shear_degrees="${AFFINE_SHEAR_DEGREES}" \
        train_perspective_prob="${PERSPECTIVE_PROB}" \
        train_perspective_distortion_scale="${PERSPECTIVE_DISTORTION_SCALE}" \
        train_invert_prob="${INVERT_PROB}" \
        hydra.run.dir="${RUN_DIR}"
else
    echo "=== SKIP_TRAIN=true, reusing existing model artifact ==="
fi

if [[ -n "${TRAINED_MODEL_SOURCE}" ]]; then
    TRAINED_MODEL_PATH="${TRAINED_MODEL_SOURCE}"
else
    TRAINED_MODEL_PATH=""
    for candidate in \
        "${RUN_DIR}/data/model.pth" \
        "${ROOT_DIR}/data/model.pth"
    do
        if [[ -f "${candidate}" ]]; then
            TRAINED_MODEL_PATH="${candidate}"
            break
        fi
    done
fi

if [[ -z "${TRAINED_MODEL_PATH}" || ! -f "${TRAINED_MODEL_PATH}" ]]; then
    echo "Could not find trained model. Checked:"
    echo "  - ${RUN_DIR}/data/model.pth"
    echo "  - ${ROOT_DIR}/data/model.pth"
    echo "You can pass TRAINED_MODEL_SOURCE=/absolute/path/to/model.pth"
    exit 1
fi
echo "Using trained model from: ${TRAINED_MODEL_PATH}"

mkdir -p "${MODEL_EXPORT_DIR}"
cp "${TRAINED_MODEL_PATH}" "${MODEL_EXPORT_PATH}"
echo "Model staged at: ${MODEL_EXPORT_PATH}"

if [[ "${STAGE_FOR_RAILWAY}" == "true" ]]; then
    mkdir -p "${RAILWAY_ARTIFACT_DIR}"
    cp "${MODEL_EXPORT_PATH}" "${RAILWAY_MODEL_PATH}"
    echo "Railway artifact staged at: ${RAILWAY_MODEL_PATH}"
fi

if [[ "${RUN_EVALUATION_AFTER_TRAIN}" == "true" ]]; then
    echo "=== Evaluating model and writing report ==="
    EVAL_FLAGS=()
    if [[ "${NO_MPS}" == "false" ]]; then
        EVAL_FLAGS+=(--use-mps)
    fi
    if [[ "${NO_CUDA}" == "false" ]]; then
        EVAL_FLAGS+=(--use-cuda)
    fi

    uv run --group training python scripts/evaluate_model.py \
        --model-path "${MODEL_EXPORT_PATH}" \
        --data-dir "${PROCESSED_DATA_DIR}" \
        --split test \
        --output-json "${EVAL_REPORT_PATH}" \
        "${EVAL_FLAGS[@]}"
fi

if [[ "${UPLOAD_TO_GCS}" == "true" ]]; then
    echo "=== Uploading model to GCS ==="
    gcloud storage cp "${MODEL_EXPORT_PATH}" "${GCS_TARGET_PATH}"
    echo "Uploaded to: ${GCS_TARGET_PATH}"
fi
