from fastapi import APIRouter

from server.controller import Predict, Train

routerTrain = APIRouter(prefix="/v1", tags=["Training Pipeline"])
routerPred = APIRouter(prefix="/v1", tags=["Prediction Pipeline"])

routerTrain.add_api_route(
    path="/train",
    endpoint=Train,
    methods=["GET"],
    responses={
        200: {"description": "Successfully model training"},
        500: {"description": "Server error while running training pipeline"},
    }
)


routerPred.add_api_route(
    path="/pred",
    endpoint=Predict,
    methods=["POST"],
    responses={
        200: {"description": "Successfully making prediction"},
        500: {"description": "Server error while making prediction"},
    }
)
