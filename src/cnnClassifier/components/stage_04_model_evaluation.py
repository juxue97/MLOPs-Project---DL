import sys
from pathlib import Path
from urllib.parse import urlparse
from cnnClassifier.utils.main_utils import save_json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import mlflow
import mlflow.pytorch
from tqdm import tqdm


from cnnClassifier.entity.config import ModelEvaluationConfigs
from cnnClassifier.logger import logging
from cnnClassifier.exception import CNNClassifierException


class ModelEvaluation:
    def __init__(self, modelEvaluationConfig: ModelEvaluationConfigs):
        try:
            self.modelEvaluationConfig = modelEvaluationConfig
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            raise CNNClassifierException(e, sys)

    def _load_model(self, path: Path) -> nn.Module:
        try:
            logging.info("Model Evaluation Pipeline: load trained model")
            model = torch.load(
                path,
                weights_only=False,
            )
            model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            raise CNNClassifierException(e, sys)

    def _valid_loader(self) -> None:
        try:
            logging.info(
                "Model Evaluation Pipeline: prepare valid data loader")
            transform = transforms.Compose([
                transforms.Resize(
                    self.modelEvaluationConfig.all_params.IMAGE_SIZE[:-1]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            valid_dataset = datasets.ImageFolder(
                root=self.modelEvaluationConfig.validation_data, transform=transform)
            self.valid_loader = DataLoader(
                valid_dataset, batch_size=self.modelEvaluationConfig.all_params.BATCH_SIZE, shuffle=False)
        except Exception as e:
            raise CNNClassifierException(e, sys)

    def _evaluate(self) -> tuple[float, float]:
        try:
            logging.info(
                "Model Evaluation Pipeline: evaluate model")
            total_loss = 0.0
            correct = 0
            total = 0

            progress_bar = tqdm(self.valid_loader,
                                desc="Evaluating", unit="batch")

            with torch.no_grad():
                for i, (images, labels) in enumerate(progress_bar, start=1):
                    images, labels = images.to(
                        self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                    progress_bar.set_postfix(
                        loss=total_loss / i, accuracy=100 * correct / total)
                    break

            avg_loss = total_loss / len(self.valid_loader)
            accuracy = 100 * correct / total

            logging.info(
                f"Validation Loss: {avg_loss:.4f} | Validation Accuracy: {accuracy:.2f}%")

            return avg_loss, accuracy
        except Exception as e:
            raise CNNClassifierException(e, sys)

    def _save_score(self, path: Path) -> None:
        try:
            logging.info(
                "Model Evaluation Pipeline: save model validation score")
            scores = {"loss": self.avgValidLoss,
                      "accuracy": self.validAccuracy}

            save_json(path=path, data=scores)
        except Exception as e:
            raise CNNClassifierException(e, sys)

    def log_into_mlflow(self):
        try:
            logging.info(
                "Model Evaluation Pipeline: log metrics, model, params onto mlflow")
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(
                mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                mlflow.log_params(self.modelEvaluationConfig.all_params)
                mlflow.log_metrics(
                    {"loss": self.avgValidLoss, "accuracy": self.validAccuracy})

                if tracking_url_type_store != "file":
                    mlflow.pytorch.log_model(
                        self.model, "model", registered_model_name="VGG16Model")
                else:
                    mlflow.pytorch.log_model(self.model, "model")
        except Exception as e:
            raise CNNClassifierException(e, sys) from e

    def initiate(self) -> None:
        try:
            logging.info("Initiating Model Evaluation Pipeline")
            self.model = self._load_model(
                path=self.modelEvaluationConfig.path_of_model
            )
            self._valid_loader()
            self.criterion = nn.CrossEntropyLoss()
            self.avgValidLoss, self.validAccuracy = self._evaluate()
            self._save_score(
                self.modelEvaluationConfig.score_file_path
            )

        except Exception as e:
            raise CNNClassifierException(e, sys) from e
