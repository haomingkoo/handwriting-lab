"""FastAPI dependencies and global variables."""

import mnist
import mnist_fastapi

## Load constant from config.py. DO NOT MODIFY the constannt name in config.py. You can set the value in config.py
# PRED_MODEL, DEVICE = mnist.modeling.utils.load_model(
#     <PRED_MODEL_PATH>,
#     <USE_CUDA>,
#     <USE_MPS>,
# )

# Load model and device once at application startup.
# This ensures:
# - the model is NOT reloaded per request
# - inference latency is stable
# - GPU / CPU memory is allocated once

PRED_MODEL, DEVICE = mnist.modeling.utils.load_model(
    path_to_model=mnist_fastapi.config.SETTINGS.PRED_MODEL_PATH,
    use_cuda=mnist_fastapi.config.SETTINGS.USE_CUDA,
    use_mps=mnist_fastapi.config.SETTINGS.USE_MPS,
)