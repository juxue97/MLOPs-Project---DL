from pathlib import Path
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torchvision

from cnnClassifier.entity.config import ModelPreparationConfigs
from cnnClassifier.exception import CNNClassifierException
from cnnClassifier.logger import logging


class ModelPreparation:
    outputCnnDimension = (7, 7)

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

            self.model = torchvision.models.vgg16(
                weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
            )

            # this ensure the final size of the cnn layer will always be 7x7, regardless of size of input image
            self.model.avgpool = nn.AdaptiveAvgPool2d(self.outputCnnDimension)

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
                            outputCnnDimension: int,
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
                nn.Linear(outputCnnDimension, 4096),
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
            outputCnnDimension = self.outputCnnDimension[0] * \
                self.outputCnnDimension[1] * 512

            fullModel = self._prepare_full_model(
                model=self.model,
                classes=self.modelPreparationConfigs.params_classes,
                freezeAll=True,
                freezeTill=None,
                outputCnnDimension=outputCnnDimension,
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
