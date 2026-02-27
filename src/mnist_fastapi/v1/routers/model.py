"""Module containing definitions and workflows for FastAPI's application endpoints."""

import io
import logging

import fastapi
import torch
import torchvision
from PIL import Image, ImageOps, UnidentifiedImageError

import mnist_fastapi
from mnist_fastapi.deps import PRED_MODEL, DEVICE

logger = logging.getLogger(__name__)


ROUTER = fastapi.APIRouter()
MAX_UPLOAD_BYTES = int(mnist_fastapi.config.SETTINGS.MAX_UPLOAD_BYTES)
MAX_BATCH_FILES = int(mnist_fastapi.config.SETTINGS.MAX_BATCH_FILES)
ALLOWED_IMAGE_CONTENT_TYPES = {
    item.strip().lower()
    for item in mnist_fastapi.config.SETTINGS.ALLOWED_IMAGE_CONTENT_TYPES.split(",")
    if item.strip()
}

# Skipped the below two using from above
#PRED_MODEL = <load from deps.py>
#DEVICE = <load from deps.py>


def _prepare_image_tensor(image_bytes: bytes):
    """Decode and normalize an uploaded image to model tensor format."""
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("L")
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

    image_tensor = torchvision.transforms.functional.to_tensor(image)

    # Auto-invert white-background sketches to MNIST-like black background.
    if float(image_tensor.mean().item()) > 0.5:
        image_tensor = 1.0 - image_tensor

    return image_tensor


def _read_validated_image_bytes(image_file: fastapi.UploadFile) -> bytes:
    """Validate content type and enforce max upload size."""
    content_type = (image_file.content_type or "").strip().lower()
    if ALLOWED_IMAGE_CONTENT_TYPES and content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Unsupported content type. "
                f"Allowed types: {', '.join(sorted(ALLOWED_IMAGE_CONTENT_TYPES))}"
            ),
        )

    contents = image_file.file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(contents) > MAX_UPLOAD_BYTES:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File is too large. Max allowed size is {MAX_UPLOAD_BYTES} bytes.",
        )

    return contents


def _predict_single_result(image_bytes: bytes, filename: str | None) -> dict[str, str | float]:
    """Run inference for a single image payload and return response object."""
    try:
        image_tensor = _prepare_image_tensor(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file.",
        ) from error

    with torch.no_grad():
        output = PRED_MODEL(image_tensor.unsqueeze(0).to(DEVICE))
    pred = output.argmax(dim=1, keepdim=True)
    confidence = float(output.exp().max().item())
    pred_str = str(int(pred[0]))

    logger.info(
        "Prediction for image filename %s: %s (confidence=%.4f)",
        filename,
        pred_str,
        confidence,
    )

    return {
        "filename": filename or "uploaded.png",
        "prediction": pred_str,
        "confidence": round(confidence, 4),
    }


## Create a routing method for "/predict" endpoint.
## Since the MNIST Classifier takes an image as an input, the user accessing the
## API for inference will be uploading a image file.

## Refer to https://fastapi.tiangolo.com/tutorial/request-files/#file-parameters-with-uploadfile

## Hint:
# @ROUTER.<http-request-method>(<endpoint>, <status-code>)
# def classify_image(image_file: <what-type?>):

@ROUTER.post("/predict", status_code=fastapi.status.HTTP_200_OK)
def predict(image_file: fastapi.UploadFile = fastapi.File(...)):
    """Run single-image inference on an uploaded MNIST image.

    Parameters
    ----------
    image_file : UploadFile
        Uploaded image file containing a handwritten digit.

    Returns
    -------
    dict
        Prediction result including filename and predicted digit.
    """
    try:
        logger.info("Received image for inference: %s", image_file.filename)
        contents = _read_validated_image_bytes(image_file)
        result_dict = {
            "data": [_predict_single_result(contents, image_file.filename)],
        }
    except fastapi.HTTPException:
        raise
    except Exception as error:
        logger.exception("Single-image inference failed: %s", error)
        raise fastapi.HTTPException(status_code=500, detail="Internal server error.")
    finally:
        image_file.file.close()

    return result_dict


## Create a batch inference endpoint
@ROUTER.post("/batch", status_code=fastapi.status.HTTP_200_OK)
def batch_predict(
    image_files: list[fastapi.UploadFile] = fastapi.File(...)
):
    """Run batch inference on multiple MNIST images."""
    if not image_files:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded.",
        )
    if len(image_files) > MAX_BATCH_FILES:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Too many files. Max allowed files per request is {MAX_BATCH_FILES}.",
        )

    results = []

    try:
        for image_file in image_files:
            try:
                logger.info("Received image for batch inference: %s", image_file.filename)
                contents = _read_validated_image_bytes(image_file)
                results.append(_predict_single_result(contents, image_file.filename))
            finally:
                image_file.file.close()
    except fastapi.HTTPException:
        raise
    except Exception as error:
        logger.exception("Batch inference failed: %s", error)
        raise fastapi.HTTPException(status_code=500, detail="Internal server error.")

    return {"data": results}


@ROUTER.get("/version", status_code=fastapi.status.HTTP_200_OK)
def model_version():
    """Get version (UUID) of predictive model used for the API.

    Returns
    -------
    dict
        Dictionary containing the UUID of the predictive model being
        served.
    """
    return {"data": {"model_uuid": mnist_fastapi.config.SETTINGS.PRED_MODEL_UUID}}
