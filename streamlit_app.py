from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------
_CLOUD_RUN_URL = "https://haoming-koo-mnist-inference-202744597330.asia-southeast1.run.app"
SERVICE_URL = os.getenv("MNIST_API_BASE_URL", _CLOUD_RUN_URL).rstrip("/")
PREDICT_PATH = "/api/v1/model/predict"


def _resolve_local_model_path() -> str:
    """Resolve local model path from env or common Railway/repo artifact locations."""
    env_path = os.getenv("MNIST_LOCAL_MODEL_PATH", "").strip()
    if env_path:
        return env_path

    candidate_paths = (
        Path("/app/artifacts/model.pth"),
        Path(__file__).resolve().parent / "artifacts" / "model.pth",
    )
    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)

    return ""


LOCAL_MODEL_PATH = _resolve_local_model_path()
LOCAL_DEVICE_PREF = os.getenv("MNIST_LOCAL_DEVICE", "auto").strip().lower()
USE_LOCAL_INFERENCE = bool(LOCAL_MODEL_PATH)
_DISABLE_AUTH = os.getenv("MNIST_DISABLE_AUTH", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
USE_AUTH = (
    (SERVICE_URL == _CLOUD_RUN_URL.rstrip("/"))
    and not _DISABLE_AUTH
    and not USE_LOCAL_INFERENCE
)


def get_cloud_run_id_token(audience: str) -> str:
    """Fetch an ID token for authenticated Cloud Run calls."""
    from google.auth.transport.requests import Request
    from google.oauth2 import id_token

    return id_token.fetch_id_token(Request(), audience)


def _theme_var(name: str, default: str) -> str:
    """Load theme overrides from env so UI can match external site style."""
    return os.getenv(name, default)


def apply_custom_theme() -> None:
    """Inject CSS aligned with your existing /dev deployment theme."""
    ui_bg = _theme_var("MNIST_UI_BG", "#080808")
    ui_surface = _theme_var("MNIST_UI_SURFACE", "#111111")
    ui_surface_2 = _theme_var("MNIST_UI_SURFACE_2", "#0d0d0d")
    ui_border = _theme_var("MNIST_UI_BORDER", "#1e1e1e")
    ui_text = _theme_var("MNIST_UI_TEXT", "#f0f0f0")
    ui_muted = _theme_var("MNIST_UI_MUTED", "#6b7280")
    ui_accent = _theme_var("MNIST_UI_ACCENT", "#3b82f6")
    ui_accent_dim = _theme_var("MNIST_UI_ACCENT_DIM", "rgba(59,130,246,.12)")

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {{
  --app-bg: {ui_bg};
  --surface: {ui_surface};
  --surface-2: {ui_surface_2};
  --border: {ui_border};
  --text: {ui_text};
  --muted: {ui_muted};
  --accent: {ui_accent};
  --accent-dim: {ui_accent_dim};
}}

.stApp {{
  background:
    radial-gradient(circle at 60% 10%, rgba(59,130,246,.08) 0%, transparent 55%),
    radial-gradient(rgba(255,255,255,.045) 1px, transparent 1px),
    var(--app-bg);
  background-size: 100% 100%, 28px 28px, 100% 100%;
  color: var(--text);
  font-family: "Inter", system-ui, sans-serif;
}}

[data-testid="stAppViewContainer"] {{
  color: var(--text);
}}

[data-testid="stMainBlockContainer"] {{
  max-width: 1200px;
}}

h1, h2, h3 {{
  color: var(--text);
  font-family: "Inter", system-ui, sans-serif;
  font-weight: 700;
  letter-spacing: -0.02em;
}}

p, li, div[data-testid="stMarkdownContainer"], label, span, div {{
  color: var(--text);
  font-family: "Inter", system-ui, sans-serif;
}}

div[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0c0c0c 0%, #080808 100%);
  border-right: 1px solid var(--border);
}}

div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: linear-gradient(180deg, var(--surface) 0%, var(--surface-2) 100%);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 0.7rem;
}}

button[kind="primary"] {{
  background: linear-gradient(135deg, rgba(59,130,246,.26), rgba(59,130,246,.10));
  border: 1px solid rgba(59,130,246,.38) !important;
  color: #fff;
  transition: border-color .2s, background .2s;
}}

button[kind="primary"]:hover {{
  border-color: rgba(59,130,246,.58) !important;
}}

button[kind="secondary"] {{
  border: 1px solid var(--border) !important;
  background: var(--surface) !important;
  color: var(--muted) !important;
}}

input, textarea, [data-baseweb="select"] > div {{
  background: #0b0b0b !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}}

[data-testid="stMetricValue"] {{
  color: var(--text);
}}

[data-testid="stMetricLabel"] {{
  color: var(--muted);
}}

.mnist-subtle {{
  color: var(--muted);
}}

.mnist-chip {{
  display: inline-block;
  font-family: "Inter", system-ui, sans-serif;
  font-size: 12px;
  letter-spacing: .08em;
  text-transform: uppercase;
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.2rem 0.6rem;
  margin-right: 0.4rem;
  margin-bottom: 0.4rem;
  background: #0c0c0c;
  color: var(--muted);
}}
</style>
""",
        unsafe_allow_html=True,
    )


def _preprocess_drawing(raw_rgba_array) -> tuple[Image.Image, Image.Image]:
    """Convert canvas image to model input."""
    raw_image = Image.fromarray(raw_rgba_array.astype("uint8"), "RGBA")
    processed = raw_image.convert("L")
    processed = ImageOps.invert(processed)
    processed = processed.resize((28, 28), Image.Resampling.LANCZOS)
    return raw_image, processed


def _build_local_net():
    """Define the same architecture used during training for local inference mode."""
    import torch

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
            self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = torch.nn.Dropout(0.25)
            self.dropout2 = torch.nn.Dropout(0.5)
            self.fc1 = torch.nn.Linear(9216, 128)
            self.fc2 = torch.nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = torch.nn.functional.relu(x)
            x = self.conv2(x)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = torch.nn.functional.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            return torch.nn.functional.log_softmax(x, dim=1)

    return Net()


@st.cache_resource(show_spinner=False)
def _load_local_model(model_path: str, device_pref: str):
    """Load model checkpoint once for local inference."""
    import torch

    model_file = Path(model_path).resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Local model file not found: {model_file}")

    if device_pref == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device_pref == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_pref == "cpu":
        device = torch.device("cpu")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    checkpoint = torch.load(str(model_file), map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(
            "Unsupported checkpoint format. Expected a state_dict or dict with 'model_state_dict'."
        )
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise TypeError("Checkpoint state_dict contains non-tensor values.")

    model = _build_local_net()
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, str(device), str(model_file)


def _predict_locally(processed_img: Image.Image) -> dict:
    """Predict directly in Streamlit process, avoiding external API/services."""
    import torch

    model, device_str, _ = _load_local_model(LOCAL_MODEL_PATH, LOCAL_DEVICE_PREF)
    device = torch.device(device_str)

    # Model expects [N, C, H, W] float tensor in [0, 1].
    pixel_data = (np.asarray(processed_img, dtype=np.float32) / 255.0)
    image_tensor = torch.from_numpy(pixel_data).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = int(model(image_tensor).argmax(dim=1).item())

    return {"data": [{"filename": "drawing.png", "prediction": str(prediction)}]}


def _predict_with_api(processed_img: Image.Image) -> dict:
    """Call prediction endpoint and return parsed JSON payload."""
    image_bytes = io.BytesIO()
    processed_img.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    headers = {"accept": "application/json"}
    if USE_AUTH:
        try:
            headers["Authorization"] = f"Bearer {get_cloud_run_id_token(SERVICE_URL)}"
        except Exception as error:
            raise RuntimeError(
                "Cloud Run auth token could not be generated. "
                "Set MNIST_API_BASE_URL to a local FastAPI URL "
                "(for example http://127.0.0.1:8081), or configure ADC with "
                "`gcloud auth application-default login`, or run with "
                "`MNIST_DISABLE_AUTH=true` if your endpoint is public."
            ) from error

    response = requests.post(
        f"{SERVICE_URL}{PREDICT_PATH}",
        headers=headers,
        files={"image_file": ("drawing.png", image_bytes, "image/png")},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _predict(processed_img: Image.Image) -> dict:
    """Unified prediction path for API mode and single-service local mode."""
    if USE_LOCAL_INFERENCE:
        return _predict_locally(processed_img)
    return _predict_with_api(processed_img)


def render_predict_page() -> None:
    """Interactive canvas for handwriting prediction."""
    st.header("Handwriting Playground")
    st.markdown(
        "<span class='mnist-chip'>draw digit</span>"
        "<span class='mnist-chip'>predict</span>"
        "<span class='mnist-chip'>inspect preprocessing</span>",
        unsafe_allow_html=True,
    )

    if "last_drawing" not in st.session_state:
        st.session_state.last_drawing = None
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    controls_col, preview_col = st.columns([3, 2], gap="large")

    with controls_col:
        stroke_width = st.slider("Stroke width", min_value=10, max_value=35, value=20)

        if st.button("Clear canvas"):
            st.session_state.last_drawing = None
            st.session_state.canvas_key += 1
            st.rerun()

        canvas = st_canvas(
            fill_color="white",
            stroke_width=stroke_width,
            stroke_color="black",
            background_color="white",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )

        if canvas.image_data is not None:
            st.session_state.last_drawing = canvas.image_data

        if st.button("Predict digit", type="primary"):
            if st.session_state.last_drawing is None:
                st.warning("Draw a digit before predicting.")
                st.stop()

            raw_img, processed_img = _preprocess_drawing(st.session_state.last_drawing)
            st.session_state["latest_raw_img"] = raw_img
            st.session_state["latest_processed_img"] = processed_img

            try:
                payload = _predict(processed_img)
                prediction = payload["data"][0]["prediction"]
                st.success(f"Predicted digit: {prediction}")
                st.session_state["latest_payload"] = payload
            except requests.HTTPError as error:
                st.error(
                    f"Inference API error ({error.response.status_code}): "
                    f"{error.response.text}"
                )
            except Exception as error:
                st.error(f"Prediction failed: {type(error).__name__}: {error}")

    with preview_col:
        st.subheader("Preview")
        if st.session_state.get("latest_raw_img") is not None:
            st.image(st.session_state["latest_raw_img"], caption="Canvas image", width=190)
            st.image(
                st.session_state["latest_processed_img"],
                caption="Model input (28x28, grayscale)",
                width=190,
            )

        if st.session_state.get("latest_payload"):
            with st.expander("Raw response JSON", expanded=False):
                st.json(st.session_state["latest_payload"])

    with st.expander("Diagnostics", expanded=False):
        if USE_LOCAL_INFERENCE:
            st.write("Inference mode: `local`")
            st.write(f"Local model path: `{LOCAL_MODEL_PATH}`")
            try:
                _, local_device, resolved_model_path = _load_local_model(
                    LOCAL_MODEL_PATH, LOCAL_DEVICE_PREF
                )
                st.write(f"Resolved model path: `{resolved_model_path}`")
                st.write(f"Local device: `{local_device}`")
            except Exception as error:
                st.error(f"Local model failed to load: {type(error).__name__}: {error}")
        else:
            st.write("Inference mode: `api`")
            st.write(f"SERVICE_URL: `{SERVICE_URL}`")
            st.write(f"Cloud Run auth enabled: `{USE_AUTH}`")
        st.markdown(
            "<p class='mnist-subtle'>"
            "For rotated or upside-down handwriting, retrain with augmentation in "
            "`scripts/retrain_local_model.sh`."
            "</p>",
            unsafe_allow_html=True,
        )


def render_pipeline_page() -> None:
    """In-app guide for training and deployment pipeline."""
    st.header("Training Pipeline")

    guide_path = Path(__file__).resolve().parent / "docs" / "training_pipeline.md"
    if guide_path.exists():
        st.markdown(guide_path.read_text(encoding="utf-8"))
    else:
        st.warning(f"Guide file not found: {guide_path}")

    st.subheader("Retrain command (rotation-aware)")
    st.code(
        "\n".join(
            [
                "ROTATION_DEGREES=60 ROTATION_PROB=0.8 INVERT_PROB=0.1 \\",
                "TRAIN_EPOCHS=6 ./scripts/retrain_local_model.sh",
            ]
        ),
        language="bash",
    )


def render_deploy_page() -> None:
    """Deployment notes for local /dev route and website integration."""
    st.header("Deploy To /dev Path")
    st.markdown(
        """
Use `setup-streamlit.sh` with a URL base path so the app can live under your site,
for example `https://handwriting.kooexperience.com/`.
"""
    )

    st.code(
        "\n".join(
            [
                "APP_BASE_PATH= \\",
                "PUBLIC_URL=https://handwriting.kooexperience.com \\",
                "MNIST_API_BASE_URL=http://10.0.0.3:8081 \\",
                "./setup-streamlit.sh",
            ]
        ),
        language="bash",
    )

    st.subheader("Railway (Simple + Cheap)")
    st.markdown(
        """
Use one Railway service (Streamlit only) with local in-app inference.
No second backend service needed.
"""
    )
    st.code(
        "\n".join(
            [
                "# train + auto-stage for Railway",
                "SKIP_PROCESS_DATA=true NO_CUDA=true NO_MPS=false ./scripts/retrain_local_model.sh",
                "",
                "# optional Railway env vars",
                "# MNIST_LOCAL_MODEL_PATH auto-detects /app/artifacts/model.pth",
                "MNIST_LOCAL_DEVICE=cpu",
            ]
        ),
        language="bash",
    )

    st.subheader("Optional bridge mode (devbox -> cpubox)")
    st.code(
        "\n".join(
            [
                "ENABLE_SOCAT_BRIDGE=true \\",
                "SOCAT_TARGET_HOST=10.0.0.3 \\",
                "HOST_PORT=8123 \\",
                "./setup-streamlit.sh",
            ]
        ),
        language="bash",
    )

    st.subheader("Theme alignment with your /dev website")
    st.markdown(
        """
Default theme now mirrors your `/dev` token set (`#080808`, `#111111`, `#3b82f6`, Inter).
You can still override with env vars:
- `MNIST_UI_BG`
- `MNIST_UI_SURFACE`
- `MNIST_UI_SURFACE_2`
- `MNIST_UI_BORDER`
- `MNIST_UI_TEXT`
- `MNIST_UI_MUTED`
- `MNIST_UI_ACCENT`
- `MNIST_UI_ACCENT_DIM`
"""
    )


def render_evaluation_page() -> None:
    """Render local evaluation report with confusion matrix and metric tables."""
    st.header("Evaluation Report")
    st.caption("Confusion matrix, per-class metrics, and top misclassifications.")

    report_path = Path(
        os.getenv("MNIST_EVAL_REPORT_PATH", "reports/evaluation_latest.json")
    )
    if not report_path.exists():
        st.warning(f"Evaluation report not found: {report_path}")
        st.markdown(
            "Run this after training to generate the report:"
        )
        st.code(
            "\n".join(
                [
                    "SKIP_PROCESS_DATA=true NO_CUDA=true NO_MPS=false \\",
                    "./scripts/retrain_local_model.sh",
                ]
            ),
            language="bash",
        )
        st.code(
            "uv run --group training python scripts/evaluate_model.py",
            language="bash",
        )
        return

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as error:
        st.error(f"Failed to read report JSON: {type(error).__name__}: {error}")
        return

    metric_cols = st.columns(5)
    metric_cols[0].metric("Accuracy", f"{100 * report.get('accuracy', 0):.2f}%")
    metric_cols[1].metric("Macro Precision", f"{100 * report.get('macro_precision', 0):.2f}%")
    metric_cols[2].metric("Macro Recall", f"{100 * report.get('macro_recall', 0):.2f}%")
    metric_cols[3].metric("Macro F1", f"{100 * report.get('macro_f1', 0):.2f}%")
    metric_cols[4].metric("Samples", str(report.get("num_samples", 0)))

    st.markdown(
        f"Report: `{report_path}`  \n"
        f"Generated (UTC): `{report.get('generated_at_utc', 'n/a')}`  \n"
        f"Model: `{report.get('model_path', 'n/a')}`  \n"
        f"Device: `{report.get('device', 'n/a')}`"
    )

    st.subheader("Confusion Matrix")
    confusion = report.get("confusion_matrix", [])
    if confusion:
        labels = [str(i) for i in range(10)]
        cm_df = pd.DataFrame(confusion, index=labels, columns=labels)
        st.dataframe(cm_df, use_container_width=True)
        st.caption("Rows = actual label, columns = predicted label.")
    else:
        st.warning("Confusion matrix is empty in report.")

    st.subheader("Per-Class Metrics")
    per_class = report.get("per_class_metrics", [])
    if per_class:
        per_class_df = pd.DataFrame(per_class)
        st.dataframe(per_class_df, use_container_width=True, hide_index=True)
        st.bar_chart(
            per_class_df.set_index("label")[["precision", "recall", "f1"]],
            use_container_width=True,
        )
    else:
        st.warning("Per-class metrics are empty in report.")

    st.subheader("Top Misclassifications")
    misclf = report.get("top_misclassifications", [])
    if misclf:
        st.dataframe(pd.DataFrame(misclf), use_container_width=True, hide_index=True)
    else:
        st.info("No misclassifications found.")


def main() -> None:
    """Main Streamlit entrypoint."""
    st.set_page_config(page_title="Handwriting Lab", layout="wide")
    apply_custom_theme()

    with st.sidebar:
        st.title("Handwriting Lab")
        st.markdown("<p class='mnist-subtle'>mnist + mlops + dev site</p>", unsafe_allow_html=True)
        page = st.radio(
            "Navigate",
            ("Predict", "Evaluation Report", "Pipeline Guide", "Deploy /dev"),
            index=0,
        )
        st.divider()
        if USE_LOCAL_INFERENCE:
            st.write("Inference: `local`")
            st.write(f"Model: `{LOCAL_MODEL_PATH}`")
        else:
            st.write("Inference: `api`")
            st.write(f"API: `{SERVICE_URL}`")

    if page == "Predict":
        render_predict_page()
    elif page == "Evaluation Report":
        render_evaluation_page()
    elif page == "Pipeline Guide":
        render_pipeline_page()
    else:
        render_deploy_page()


if __name__ == "__main__":
    main()
