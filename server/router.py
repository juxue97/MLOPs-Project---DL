from fastapi import APIRouter

from server.controller import Predict

routerPred = APIRouter(prefix="/v1", tags=["Prediction Pipeline"])


routerPred.add_api_route(
    path="/pred",
    endpoint=Predict,
    methods=["POST"],
    responses={
        200: {"description": "Successfully making prediction"},
        500: {"description": "Server error while making prediction"},
    }
)
