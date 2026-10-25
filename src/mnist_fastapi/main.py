"""Main module for initialising and defining the FastAPI application."""

from __future__ import annotations

import ipaddress
import logging
import os
from pathlib import Path

import fastapi
import mnist
from fastapi import responses
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import mnist_fastapi
from mnist_fastapi.rate_limit import InMemoryRateLimiter

LOGGER = logging.getLogger(__name__)
LOGGER.info("Setting up logging configuration.")
mnist.general_utils.setup_logging(
    logging_config_path=mnist_fastapi.config.SETTINGS.LOGGER_CONFIG_PATH
)

API_V1_STR = mnist_fastapi.config.SETTINGS.API_V1_STR
IS_PRODUCTION = bool(os.environ.get("RAILWAY_ENVIRONMENT")) or (
    os.environ.get("ENV", "").lower() == "production"
)
APP = fastapi.FastAPI(
    title=mnist_fastapi.config.SETTINGS.API_NAME,
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else f"{API_V1_STR}/openapi.json",
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


# https://www.cloudflare.com/ips/
CLOUDFLARE_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "173.245.48.0/20", "103.21.244.0/22", "103.22.200.0/22", "103.31.4.0/22",
        "141.101.64.0/18", "108.162.192.0/18", "190.93.240.0/20", "188.114.96.0/20",
        "197.234.240.0/22", "198.41.128.0/17", "162.158.0.0/15", "104.16.0.0/13",
        "104.24.0.0/14", "172.64.0.0/13", "131.0.72.0/22", "2400:cb00::/32",
        "2606:4700::/32", "2803:f800::/32", "2405:b500::/32", "2405:8100::/32",
        "2a06:98c0::/29", "2c0f:f248::/32",
    )
)


def _is_cloudflare(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in network for network in CLOUDFLARE_NETWORKS)


def _get_request_ip(request: fastapi.Request) -> str:
    """Right-most XFF hop is Railway's; trust CF-Connecting-IP only if that hop is Cloudflare."""
    forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[-1].strip()
    peer = forwarded or (request.client.host if request.client else "")
    cf_ip = (request.headers.get("cf-connecting-ip") or "").strip()
    if cf_ip and _is_cloudflare(peer):
        return cf_ip
    return peer or "unknown"


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


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


@APP.middleware("http")
async def add_security_headers(request: fastapi.Request, call_next):
    """Add basic security headers to every response."""
    response = await call_next(request)
    for header_name, header_value in SECURITY_HEADERS.items():
        response.headers.setdefault(header_name, header_value)
    return response

APP_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = APP_ROOT / "src" / "mnist_fastapi" / "web"
WEB_INDEX_PATH = WEB_DIR / "index.html"
REPORT_PATH = APP_ROOT / "reports" / "evaluation_latest.json"
PIPELINE_DOC_PATH = APP_ROOT / "docs" / "training_pipeline.md"
FONTS_DIR = APP_ROOT / "src" / "mnist_fastapi" / "web" / "fonts"

APP.mount("/fonts", StaticFiles(directory=FONTS_DIR), name="fonts")


@APP.get("/", include_in_schema=False)
def serve_web():
    """Serve the custom non-Streamlit frontend."""
    if not WEB_INDEX_PATH.exists():
        raise fastapi.HTTPException(status_code=404, detail="Web app not found.")
    return responses.FileResponse(WEB_INDEX_PATH)


@APP.get("/privacy", include_in_schema=False)
def serve_privacy():
    """Serve the privacy notice."""
    return responses.FileResponse(WEB_DIR / "privacy.html")


@APP.get("/favicon.svg", include_in_schema=False)
def serve_favicon():
    """Serve the site icon."""
    return responses.FileResponse(WEB_DIR / "favicon.svg", media_type="image/svg+xml")


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
