"""Configuration module for the FastAPI application.

This module defines runtime settings for the inference service.
Values are loaded from environment variables using Pydantic Settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the FastAPI application
    These values should be environment-configurable so that
    the same container/image can be deployed across environments.
    """
    # API metadata
    API_NAME: str = "Handwriting Lab API"
    API_V1_STR: str = "/api/v1"
    LOGGER_CONFIG_PATH: str = "./conf/logging.yaml"

    # Device configuration
    USE_CUDA: bool = False
    USE_MPS: bool = False

    # Model metadata (required)
    # These MUST be provided, otherwise FastAPI should fail on startup
    PRED_MODEL_UUID: str = "mnist-local-001"
    PRED_MODEL_PATH: str = "artifacts/model.pth"

    model_config = SettingsConfigDict(env_file=".env", extra ="ignore")

# Singleton settings object
SETTINGS = Settings()
