"""Main module for initialising and defining the FastAPI application."""

import logging

import fastapi
import mnist
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

## Reference: https://fastapi.tiangolo.com/tutorial/bigger-applications/
API_ROUTER = fastapi.APIRouter()
API_ROUTER.include_router(
    mnist_fastapi.v1.routers.model.ROUTER, prefix="/model", tags=["model"]
)
APP.include_router(API_ROUTER, prefix=mnist_fastapi.config.SETTINGS.API_V1_STR)

ORIGINS = ["*"]

APP.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@APP.get("/healthz", status_code=fastapi.status.HTTP_200_OK)
def health_check():
    """Health check endpoint.

    Used for readiness/liveness probing.
    """
    return {
        "status": "ok",
        "service": "mnist-fastapi",
    }

app = APP