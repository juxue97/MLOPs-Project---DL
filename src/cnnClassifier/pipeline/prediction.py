import sys
import time
import threading
import torch
from PIL import Image
from torchvision import transforms
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from cnnClassifier.constants import MLFLOW_URI
from cnnClassifier.exception import CNNClassifierException

# Set MLflow Tracking URI
mlflow.set_tracking_uri(MLFLOW_URI)


class ModelLoader:
    model = None
    model_name = None
    model_version = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_used_time = time.time()
    unload_timeout = 300  # Unload after 5 minutes of inactivity

    @classmethod
    def load_model(cls):
        """Load model from MLflow if not already loaded"""
        if cls.model is None:
            print("Loading model from MLflow...")

            # Connect to MLflow
            client = MlflowClient()
            models = client.search_model_versions(
                "tags.environment = 'production'"
            )

            if not models:
                raise Exception("No production model found in MLflow")

            # Get the latest production model version
            latest_model = sorted(
                models, key=lambda m: int(m.version), reverse=True)[0]

            cls.model_name = latest_model.name
            cls.model_version = latest_model.version
            model_uri = f"models:/{cls.model_name}/{cls.model_version}"
            print(f"Using Model URI: {model_uri}")

            # Load model from MLflow
            cls.model = mlflow.pytorch.load_model(model_uri).to(cls.device)
            cls.model.eval()
            cls.last_used_time = time.time()
            print(
                f"Model {cls.model_name} (Version {cls.model_version}) loaded successfully!")

        return cls.model

    @classmethod
    def unload_model(cls):
        """Unload model if it has been idle beyond the timeout period"""
        while cls.model is not None:
            if time.time() - cls.last_used_time > cls.unload_timeout:
                print("Unloading model due to inactivity...")
                cls.model = None  # Free memory
            time.sleep(60)  # Check every 60 seconds


# Start a background thread to handle model unloading
threading.Thread(target=ModelLoader.unload_model, daemon=True).start()

# Define image transformations
image_transforms = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


class PredictPipeline:
    def __init__(self):
        pass

    @staticmethod
    def start(image: Image) -> dict:
        """Run inference and log to MLflow"""
        try:
            model = ModelLoader.load_model()  # Ensure model is loaded
            image_tensor = image_transforms(
                image).unsqueeze(0).to(ModelLoader.device)

            start_time = time.time()
            with torch.no_grad():
                output = model(image_tensor)
                prediction = torch.argmax(output, dim=1).item()
            latency = time.time() - start_time

            # Log inference to MLflow
            with mlflow.start_run():
                mlflow.log_param("image_size", image.size)
                mlflow.log_metric("inference_time", latency)
                mlflow.log_metric("prediction", prediction)

            return {"prediction": prediction, "latency": latency}

        except Exception as e:
            raise CNNClassifierException(e, sys) from e
