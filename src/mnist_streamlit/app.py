"""
Streamlit UI for MNIST inference (FastAPI backend).

What this app does
- Upload single or multiple images (PNG/JPG/JPEG by default).
- Sends images to FastAPI using multipart/form-data.
- Shows:
  - the request payload (filename, content-type, size)
  - the raw JSON returned by the backend
  - a clean results table with meaningful column headers

Backend endpoints expected
- POST {BASE_URL}/api/v1/model/predict   (field name: image_file)
- POST {BASE_URL}/api/v1/model/batch     (field name: image_files, repeated)
- GET  {BASE_URL}/api/v1/model/version
- GET  {BASE_URL}/healthz

How to configure BASE_URL (pick one)
1) Environment variable (recommended locally):
   export MNIST_API_BASE_URL="http://127.0.0.1:8000"

2) Streamlit secrets (recommended for Streamlit Cloud):
   .streamlit/secrets.toml
   [app]
   api_base_url = "https://your-backend"

Notes on “model name / uuid”
- Streamlit does NOT decide the model UUID.
- The backend decides it and returns it via GET /api/v1/model/version.
- In your container run command, you typically pass:
  -e PRED_MODEL_PATH=...
  -e PRED_MODEL_UUID=...
"""

from __future__ import annotations

import io
import os
from typing import Any

import requests
import streamlit as st
from PIL import Image


# =============================================================================
# 1) Backend configuration
# =============================================================================
def _get_base_url() -> str:
    """
    Resolve FastAPI base URL.

    Priority:
    1) Env var MNIST_API_BASE_URL
    2) Streamlit secrets: st.secrets["app"]["api_base_url"]
    3) Local default http://127.0.0.1:8000

    If you are confused where to change things, change MNIST_API_BASE_URL.
    """
    env_url = os.getenv("MNIST_API_BASE_URL")
    if env_url:
        return env_url.rstrip("/")

    # secrets.toml is optional. If not present, Streamlit can raise.
    try:
        return str(st.secrets["app"]["api_base_url"]).rstrip("/")
    except Exception:
        return "http://127.0.0.1:8000"


BASE_URL = _get_base_url()

# These should match your FastAPI routes exactly
PREDICT_URL = f"{BASE_URL}/api/v1/model/predict"
BATCH_URL = f"{BASE_URL}/api/v1/model/batch"
VERSION_URL = f"{BASE_URL}/api/v1/model/version"
HEALTH_URL = f"{BASE_URL}/healthz"

# Change accepted image types here if you want more formats.
# Keep this aligned with the README text below.
ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg"]


# =============================================================================
# 2) Backend helper functions
# =============================================================================
def _check_api_health() -> tuple[bool, str]:
    """Fast connectivity probe. If this fails, fix backend/ports before UI."""
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        if r.status_code == 200:
            return True, "ok"
        return False, f"status_code={r.status_code}"
    except requests.RequestException as e:
        return False, f"{type(e).__name__}: {e}"


def _get_model_uuid() -> str:
    """
    Fetch model identifier from backend.

    Where does this come from?
    - From the backend's /version endpoint response.
    - Backend usually reads PRED_MODEL_UUID from environment variables.
    """
    try:
        r = requests.get(VERSION_URL, timeout=5)
        r.raise_for_status()
        payload: dict[str, Any] = r.json()
        return str(payload.get("data", {}).get("model_uuid", "unknown"))
    except Exception:
        return "unknown"


def _render_image_preview(uploaded_file) -> None:
    """Show a small preview of an uploaded image."""
    try:
        img = Image.open(io.BytesIO(uploaded_file.getvalue()))
        st.image(img, caption=uploaded_file.name, width=140)
    except Exception:
        st.warning("Unable to preview image. File may not be a valid image.")


def _file_payload(uploaded_file) -> dict[str, Any]:
    """
    Build a human-friendly view of what we will send.

    We do NOT show raw bytes. We show metadata only.
    """
    return {
        "filename": uploaded_file.name,
        "content_type": uploaded_file.type,
        "size_bytes": len(uploaded_file.getvalue()),
    }


def _predict_single(uploaded_file) -> dict[str, Any]:
    """
    Send one file to POST /predict.

    IMPORTANT: field name must match the FastAPI param name.
    If backend has: def predict(image_file: UploadFile = File(...))
    then key must be "image_file".
    """
    files = {
        "image_file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    r = requests.post(PREDICT_URL, files=files, timeout=30)
    r.raise_for_status()
    return r.json()


def _predict_batch(uploaded_files) -> dict[str, Any]:
    """
    Send multiple files to POST /batch.

    IMPORTANT: field name must match the FastAPI param name.
    If backend has: def batch(image_files: list[UploadFile] = File(...))
    then you send repeated parts with key "image_files".
    """
    files = [
        (
            "image_files",
            (
                f.name,
                f.getvalue(),
                f.type or "application/octet-stream",
            ),
        )
        for f in uploaded_files
    ]
    r = requests.post(BATCH_URL, files=files, timeout=60)
    r.raise_for_status()
    return r.json()


def _result_to_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert backend JSON to a list-of-dicts so Streamlit shows meaningful headers.

    Expected backend shape:
    {
      "data": [
        {"filename": "...", "prediction": "..."},
        ...
      ]
    }

    If your backend returns something else, adjust this function only.
    """
    rows: list[dict[str, Any]] = []
    for item in result.get("data", []):
        rows.append(
            {
                "file_name": item.get("filename", ""),
                "predicted_digit": item.get("prediction", ""),
            }
        )
    return rows


# =============================================================================
# 3) Streamlit UI
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="MNIST Inference", layout="centered")

    st.title("MNIST Digit Classifier")
    st.caption("Educational Streamlit UI calling a FastAPI inference service.")

    tab_readme, tab_single, tab_batch = st.tabs(
        ["README", "Single inference", "Batch inference"]
    )

    # -------------------------------------------------------------------------
    # README
    # -------------------------------------------------------------------------
    with tab_readme:
        st.markdown(
            f"""
### Purpose
This application demonstrates an end-to-end machine learning inference workflow:
- Upload an image in the browser
- Send it to a FastAPI backend as `multipart/form-data`
- Receive a structured JSON response
- Display predictions in a readable way

The Streamlit app is a **client UI only**.  
The model is loaded and executed **only in the FastAPI backend**.

---

### Accepted input files
The UI accepts the following formats:
- {", ".join([f"`.{x}`" for x in ALLOWED_IMAGE_TYPES])}

Recommended characteristics (for best accuracy):
- A single digit per image
- High contrast
- Square or near-square aspect ratio

---

### Backend endpoints used
- `POST /api/v1/model/predict` — single image inference
- `POST /api/v1/model/batch` — batch image inference
- `GET  /api/v1/model/version` — model identifier (e.g., UUID)
- `GET  /healthz` — health check

---

### How a request flows
1. You upload an image
2. Streamlit sends it as `multipart/form-data`
3. FastAPI receives it as an `UploadFile`
4. The backend runs the model to predict the digit
5. The backend returns JSON, and Streamlit renders it
"""
        )

    # -------------------------------------------------------------------------
    # SINGLE INFERENCE
    # -------------------------------------------------------------------------
    with tab_single:
        st.subheader("Single inference")

        uploaded = st.file_uploader(
            "Upload one MNIST image",
            type=ALLOWED_IMAGE_TYPES,
            accept_multiple_files=False,
        )

        if uploaded is not None:
            st.success("File uploaded successfully.")
            _render_image_preview(uploaded)

            with st.expander("Request payload", expanded=True):
                st.json({"image_file": _file_payload(uploaded)})

            if st.button("Run inference", type="primary"):
                with st.spinner("Calling backend..."):
                    try:
                        result = _predict_single(uploaded)
                        rows = _result_to_rows(result)

                        if rows:
                            st.success(f"Prediction: **{rows[0]['predicted_digit']}**")
                        else:
                            st.warning("No prediction returned in result['data'].")

                        with st.expander("Raw JSON response", expanded=True):
                            st.json(result)

                    except requests.HTTPError as e:
                        st.error(f"Backend returned HTTP error: {e}")
                        st.code(getattr(e.response, "text", ""), language="json")
                    except Exception as e:
                        st.error(f"Request failed: {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # BATCH INFERENCE
    # -------------------------------------------------------------------------
    with tab_batch:
        st.subheader("Batch inference")

        uploaded_many = st.file_uploader(
            "Upload multiple MNIST images",
            type=ALLOWED_IMAGE_TYPES,
            accept_multiple_files=True,
        )

        if uploaded_many:
            st.success(f"{len(uploaded_many)} files uploaded.")

            with st.expander("Request payload", expanded=False):
                st.json({"image_files": [_file_payload(f) for f in uploaded_many]})

            if st.button("Run batch inference"):
                with st.spinner("Calling backend batch endpoint..."):
                    try:
                        result = _predict_batch(uploaded_many)
                        rows = _result_to_rows(result)

                        st.success("Batch inference completed.")
                        st.dataframe(rows, use_container_width=True, hide_index=True)

                        with st.expander("Raw JSON response", expanded=True):
                            st.json(result)

                    except requests.HTTPError as e:
                        st.error(f"Backend returned HTTP error: {e}")
                        st.code(getattr(e.response, "text", ""), language="json")
                    except Exception as e:
                        st.error(f"Request failed: {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # Diagnostics (bottom)
    # -------------------------------------------------------------------------
    st.divider()
    with st.expander("System diagnostics (debug)"):
        ok, msg = _check_api_health()
        st.write(f"BASE_URL: `{BASE_URL}`")
        st.write(f"Health: {'OK' if ok else 'FAIL'} ({msg})")
        st.write(f"Model UUID: `{_get_model_uuid()}`")


if __name__ == "__main__":
    main()