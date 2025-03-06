from fastapi import status, HTTPException, File, UploadFile
from PIL import Image
import io
import os

from cnnClassifier.pipeline.prediction import PredictPipeline


async def Predict(file: UploadFile = File(...)) -> dict:
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        return PredictPipeline().start(image)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error occurred while making prediction: {str(e)}",
        )


async def Train() -> dict:
    try:
        # os.system("")
        # Applying dvc tool
        os.system("dvc repro")
        return {"status": status.HTTP_200_OK, "description": "Training successfully completed"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error occurred while running training pipeline: {str(e)}",
        )
