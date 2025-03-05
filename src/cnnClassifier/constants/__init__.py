from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv("./.env")

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

MLFLOW_TRACKING_USERNAME = os.getenv(
    "MLFLOW_TRACKING_USERNAME", "get-your-own")
MLFLOW_TRACKING_PASSWORD = os.getenv(
    "MLFLOW_TRACKING_PASSWORD", "get-your-own")
MLFLOW_URI = os.getenv(
    "MLFLOW_URI", "get-your-own"
)
