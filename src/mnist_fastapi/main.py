"""Main module for initialising and defining the FastAPI application."""

from __future__ import annotations

import logging
from pathlib import Path

import fastapi
import mnist
from fastapi import responses
from fastapi.middleware.cors import CORSMiddleware

import mnist_fastapi
from mnist_fastapi.rate_limit import InMemoryRateLimiter

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

rate_limit_path_prefixes = tuple(
    _csv_to_list(mnist_fastapi.config.SETTINGS.RATE_LIMIT_PATH_PREFIXES)
)
rate_limit_methods = {
    method.upper() for method in _csv_to_list(mnist_fastapi.config.SETTINGS.RATE_LIMIT_METHODS)
}
rate_limiter = InMemoryRateLimiter(
    limit=int(mnist_fastapi.config.SETTINGS.RATE_LIMIT_REQUESTS),
    window_seconds=int(mnist_fastapi.config.SETTINGS.RATE_LIMIT_WINDOW_SECONDS),
)


def _get_request_ip(request: fastapi.Request) -> str:
    """Resolve client IP from common proxy headers or socket peer."""
    for header_name in ("cf-connecting-ip", "x-real-ip", "x-forwarded-for"):
        raw_value = (request.headers.get(header_name) or "").strip()
        if not raw_value:
            continue
        if header_name == "x-forwarded-for":
            return raw_value.split(",")[0].strip()
        return raw_value

    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@APP.middleware("http")
async def enforce_rate_limit(request: fastapi.Request, call_next):
    """Throttle expensive inference routes per client IP."""
    if not mnist_fastapi.config.SETTINGS.RATE_LIMIT_ENABLED:
        return await call_next(request)

    method = request.method.upper()
    path = request.url.path
    if method == "OPTIONS":
        return await call_next(request)
    if method not in rate_limit_methods:
        return await call_next(request)
    if not any(path.startswith(prefix) for prefix in rate_limit_path_prefixes):
        return await call_next(request)

    client_ip = _get_request_ip(request)
    limit_key = f"{client_ip}:{path}:{method}"
    result = rate_limiter.check(limit_key)
    rate_limit_headers = {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
        "X-RateLimit-Window": str(
            mnist_fastapi.config.SETTINGS.RATE_LIMIT_WINDOW_SECONDS
        ),
    }
    if not result.allowed:
        rate_limit_headers["Retry-After"] = str(result.retry_after_seconds)
        return responses.JSONResponse(
            status_code=fastapi.status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": (
                    "Rate limit exceeded. Please wait a moment before retrying."
                )
            },
            headers=rate_limit_headers,
        )

    response = await call_next(request)
    for header_name, header_value in rate_limit_headers.items():
        response.headers[header_name] = header_value
    return response

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
