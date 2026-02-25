"""Module containing definitions and workflows for FastAPI's application endpoints."""

import io
import logging

import fastapi
import torchvision
from PIL import Image, ImageOps

import mnist_fastapi
from mnist_fastapi.deps import PRED_MODEL, DEVICE

logger = logging.getLogger(__name__)


ROUTER = fastapi.APIRouter()

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
    result_dict = {"data": []}

    try:
        logger.info("Received image for inference: %s", image_file.filename)
        contents = image_file.file.read()
        image_tensor = _prepare_image_tensor(contents)

        # Model inference
        output = PRED_MODEL(image_tensor.unsqueeze(0).to(DEVICE))
        pred = output.argmax(dim=1, keepdim=True)
        confidence = float(output.exp().max().item())
        pred_str = str(int(pred[0]))

        ## what data would you think the user would like to see
        result_dict["data"].append(
            {
                "filename": image_file.filename or "uploaded.png",
                "prediction": pred_str,
                "confidence": round(confidence, 4),
            }
        )
        logger.info(
            "Prediction for image filename %s: %s (confidence=%.4f)",
            image_file.filename,
            pred_str,
            confidence,
        )

    except Exception as error:
        logger.error(error)
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
    results = []

    for image_file in image_files:
        prediction = predict(image_file)
        results.extend(prediction["data"])

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

