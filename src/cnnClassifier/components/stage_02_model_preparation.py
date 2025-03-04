from pathlib import Path
import sys
from typing import Optional

import torch
import torch.nn as nn
import torchvision

from cnnClassifier.entity.config import ModelPreparationConfigs
from cnnClassifier.exception import CNNClassifierException
from cnnClassifier.logger import logging


class ModelPreparation:
    def __init__(self, modelPreparationConfigs: ModelPreparationConfigs):
        try:
            self.modelPreparationConfigs = modelPreparationConfigs
        except Exception as e:
            raise CNNClassifierException(e, sys)

    @staticmethod
    def _save_model(path: Path, model: nn.Module) -> None:
        torch.save(model, path)

    def _download_base_model(self) -> None:
        try:
            logging.info("Model Preparation Pipeline: download base model")

            self.model = torchvision.models.vgg16(pretrained=True)

            self.model.avgpool = nn.AdaptiveAvgPool2d((7, 7))

            self._save_model(
                path=self.modelPreparationConfigs.base_model_path, model=self.model
            )

        except Exception as e:
            raise CNNClassifierException(e, sys)

    @staticmethod
    def _prepare_full_model(model: nn.Module,
                            classes: int,
                            freezeAll: Optional[bool],
                            freezeTill: Optional[int],
                            ) -> nn.Module:
        try:
            logging.info("Model Preparation Pipeline: prepare full model")

            if freezeAll:
                for param in model.parameters():
                    param.requires_grad = False

            elif (freezeTill is not None) and (freezeTill > 0):
                for param in list(model.parameters())[:-freezeTill]:
                    param.requires_grad = False

            model.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(25088, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, classes),
                nn.Softmax(dim=1)
            )

            logging.info(model)
            return model

        except Exception as e:
            raise CNNClassifierException(e, sys)

    def _update_base_model(self) -> None:
        try:
            logging.info("Model Preparation Pipeline: update base model")

            fullModel = self._prepare_full_model(
                model=self.model,
                classes=self.modelPreparationConfigs.params_classes,
                freezeAll=True,
                freezeTill=None,
            )

            self._save_model(
                path=self.modelPreparationConfigs.updated_base_model_path, model=fullModel
            )

        except Exception as e:
            raise CNNClassifierException(e, sys) from e

    def initiate(self):
        try:
            logging.info("Initiating Model Preparation Pipeline")
            self._download_base_model()
            self._update_base_model()

        except Exception as e:
            raise CNNClassifierException(e, sys) from e
