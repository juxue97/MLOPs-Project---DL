from fastapi import status, HTTPException, File, UploadFile
from PIL import Image
import io

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
