"""Module containing definitions and workflows for FastAPI's application endpoints."""

import logging
import os

import fastapi
import torchvision
from PIL import Image

import mnist_fastapi
from mnist_fastapi.deps import PRED_MODEL, DEVICE

logger = logging.getLogger(__name__)


ROUTER = fastapi.APIRouter()

# Skipped the below two using from above
#PRED_MODEL = <load from deps.py>
#DEVICE = <load from deps.py>

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
        
        # Persist file temporarily to disk
        contents = image_file.file.read()
        with open(image_file.filename, "wb") as buffer:
            buffer.write(contents)

        # Load and preprocess image
        image = Image.open(image_file.filename)
        image = torchvision.transforms.functional.to_grayscale(image)
        image = torchvision.transforms.functional.to_tensor(image)

        # Model inference
        output = PRED_MODEL(image.unsqueeze(0).to(DEVICE))
        pred = output.argmax(dim=1, keepdim=True)
        pred_str = str(int(pred[0]))

        ## what data would you think the user would like to see
        result_dict["data"].append(
            {
                "filename": image_file.filename,
                "prediction": pred_str,
            }
        )
        logger.info(
            "Prediction for image filename %s: %s", image_file.filename, pred_str
        )

    except Exception as error:
        logger.error(error)
        raise fastapi.HTTPException(status_code=500, detail="Internal server error.")

    finally:
        image_file.file.close()
        os.remove(image_file.filename)

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


