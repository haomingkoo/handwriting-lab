"""Main module for initialising and defining the FastAPI application."""

from __future__ import annotations

import logging
from pathlib import Path

import fastapi
import mnist
from fastapi import responses
from fastapi.middleware.cors import CORSMiddleware

import mnist_fastapi

LOGGER = logging.getLogger(__name__)
LOGGER.info("Setting up logging configuration.")
mnist.general_utils.setup_logging(
    logging_config_path=mnist_fastapi.config.SETTINGS.LOGGER_CONFIG_PATH
)

API_V1_STR = mnist_fastapi.config.SETTINGS.API_V1_STR
APP = fastapi.FastAPI(
    title=mnist_fastapi.config.SETTINGS.API_NAME,
    openapi_url=f"{API_V1_STR}/openapi.json",
)

API_ROUTER = fastapi.APIRouter()
API_ROUTER.include_router(
    mnist_fastapi.v1.routers.model.ROUTER,
    prefix="/model",
    tags=["model"],
)
APP.include_router(API_ROUTER, prefix=API_V1_STR)

def _csv_to_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or ["*"]


cors_origins = _csv_to_list(mnist_fastapi.config.SETTINGS.CORS_ALLOW_ORIGINS)
cors_methods = _csv_to_list(mnist_fastapi.config.SETTINGS.CORS_ALLOW_METHODS)
cors_headers = _csv_to_list(mnist_fastapi.config.SETTINGS.CORS_ALLOW_HEADERS)
cors_allow_credentials = bool(mnist_fastapi.config.SETTINGS.CORS_ALLOW_CREDENTIALS)
if cors_allow_credentials and cors_origins == ["*"]:
    LOGGER.warning(
        "CORS_ALLOW_CREDENTIALS=true with wildcard origin is unsafe; forcing credentials off."
    )
    cors_allow_credentials = False

APP.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=cors_methods,
    allow_headers=cors_headers,
)

APP_ROOT = Path(__file__).resolve().parents[2]
WEB_INDEX_PATH = APP_ROOT / "src" / "mnist_fastapi" / "web" / "index.html"
REPORT_PATH = APP_ROOT / "reports" / "evaluation_latest.json"
PIPELINE_DOC_PATH = APP_ROOT / "docs" / "training_pipeline.md"


@APP.get("/", include_in_schema=False)
def serve_web():
    """Serve the custom non-Streamlit frontend."""
    if not WEB_INDEX_PATH.exists():
        raise fastapi.HTTPException(status_code=404, detail="Web app not found.")
    return responses.FileResponse(WEB_INDEX_PATH)


@APP.get("/reports/evaluation_latest.json", include_in_schema=False)
def serve_evaluation_report():
    """Serve latest local evaluation report for frontend metrics page."""
    if not REPORT_PATH.exists():
        raise fastapi.HTTPException(status_code=404, detail="Evaluation report not found.")
    return responses.FileResponse(REPORT_PATH, media_type="application/json")


@APP.get("/docs/training_pipeline.md", include_in_schema=False)
def serve_pipeline_doc():
    """Serve the training pipeline markdown used by the guide tab."""
    if not PIPELINE_DOC_PATH.exists():
        raise fastapi.HTTPException(status_code=404, detail="Pipeline guide not found.")
    return responses.FileResponse(PIPELINE_DOC_PATH, media_type="text/markdown")


@APP.get("/healthz", status_code=fastapi.status.HTTP_200_OK)
def health_check():
    """Health check endpoint used for readiness/liveness probing."""
    return {"status": "ok", "service": "mnist-fastapi"}


app = APP
