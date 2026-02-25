#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Deployment script for the handwriting Streamlit app.
# Supports:
# - local direct hosting
# - optional socat bridge mode (devbox -> cpubox)
# - URL path prefixes such as /dev/handwriting for website integration
# ---------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

APP_NAME="${APP_NAME:-mnist-handwriting}"
IMAGE_NAME="${IMAGE_NAME:-mnist-streamlit}"
CONTAINERFILE="${CONTAINERFILE:-${ROOT_DIR}/streamlit.Containerfile}"

HOST_PORT="${HOST_PORT:-8123}"
CONTAINER_PORT="${CONTAINER_PORT:-8080}"
BACKEND_URL="${MNIST_API_BASE_URL:-http://10.0.0.3:8081}"

# Streamlit expects this value without a leading slash.
APP_BASE_PATH="${APP_BASE_PATH:-dev/handwriting}"
APP_BASE_PATH="${APP_BASE_PATH#/}"
APP_BASE_PATH="${APP_BASE_PATH%/}"

PUBLIC_URL="${PUBLIC_URL:-}"

ENABLE_SOCAT_BRIDGE="${ENABLE_SOCAT_BRIDGE:-false}"
SOCAT_TARGET_HOST="${SOCAT_TARGET_HOST:-10.0.0.3}"
SOCAT_LOG_FILE="${SOCAT_LOG_FILE:-${ROOT_DIR}/socat.log}"

ENABLE_BOOT_AUTOSTART="${ENABLE_BOOT_AUTOSTART:-false}"

if [[ ! -f "${CONTAINERFILE}" ]]; then
    echo "Missing containerfile: ${CONTAINERFILE}"
    exit 1
fi

echo "Building ${IMAGE_NAME} from ${CONTAINERFILE}"
docker build -f "${CONTAINERFILE}" -t "${IMAGE_NAME}" "${ROOT_DIR}"

echo "Replacing container ${APP_NAME}"
docker stop "${APP_NAME}" 2>/dev/null || true
docker rm "${APP_NAME}" 2>/dev/null || true

docker_run_cmd=(
    docker run -d
    --name "${APP_NAME}"
    --restart unless-stopped
    -e "MNIST_API_BASE_URL=${BACKEND_URL}"
    -e "STREAMLIT_SERVER_PORT=${CONTAINER_PORT}"
    -e "STREAMLIT_SERVER_ADDRESS=0.0.0.0"
    -e "STREAMLIT_SERVER_HEADLESS=true"
    -p "${HOST_PORT}:${CONTAINER_PORT}"
)

if [[ -n "${APP_BASE_PATH}" ]]; then
    docker_run_cmd+=(-e "STREAMLIT_SERVER_BASE_URL_PATH=${APP_BASE_PATH}")
fi

docker_run_cmd+=("${IMAGE_NAME}")
"${docker_run_cmd[@]}"

sleep 3
if ! docker ps --format "{{.Names}}" | grep -q "^${APP_NAME}$"; then
    echo "Container failed to start. Check: docker logs ${APP_NAME}"
    exit 1
fi

if [[ "${ENABLE_SOCAT_BRIDGE}" == "true" ]]; then
    echo "Starting socat bridge on :${HOST_PORT} -> ${SOCAT_TARGET_HOST}:${HOST_PORT}"
    pkill -f "socat.*TCP-LISTEN:${HOST_PORT}" 2>/dev/null || true
    nohup socat "TCP-LISTEN:${HOST_PORT},reuseaddr,fork" \
        "TCP:${SOCAT_TARGET_HOST}:${HOST_PORT}" > "${SOCAT_LOG_FILE}" 2>&1 &
fi

if [[ -n "${PUBLIC_URL}" ]]; then
    if [[ -n "${APP_BASE_PATH}" ]]; then
        echo "${PUBLIC_URL%/}/${APP_BASE_PATH}"
    else
        echo "${PUBLIC_URL%/}"
    fi
else
    if [[ -n "${APP_BASE_PATH}" ]]; then
        echo "http://localhost:${HOST_PORT}/${APP_BASE_PATH}"
    else
        echo "http://localhost:${HOST_PORT}"
    fi
fi

if [[ "${ENABLE_BOOT_AUTOSTART}" == "true" ]]; then
    BASHRC_FILE="${HOME}/.bashrc"
    START_MARKER_BEGIN="# BEGIN_MNIST_HANDWRITING_AUTOSTART"
    START_MARKER_END="# END_MNIST_HANDWRITING_AUTOSTART"

    if ! grep -q "${START_MARKER_BEGIN}" "${BASHRC_FILE}" 2>/dev/null; then
        {
            echo "${START_MARKER_BEGIN}"
            echo "docker start ${APP_NAME} 2>/dev/null || true"
            if [[ "${ENABLE_SOCAT_BRIDGE}" == "true" ]]; then
                echo "pkill -f 'socat.*TCP-LISTEN:${HOST_PORT}' 2>/dev/null || true"
                echo "nohup socat TCP-LISTEN:${HOST_PORT},reuseaddr,fork TCP:${SOCAT_TARGET_HOST}:${HOST_PORT} > ${SOCAT_LOG_FILE} 2>&1 &"
            fi
            echo "${START_MARKER_END}"
        } >> "${BASHRC_FILE}"
    fi
fi
