import sys

from cnnClassifier.entity.config import ModelTrainingConfigs
from cnnClassifier.exception import CNNClassifierException
from cnnClassifier.logger import logging


class ModelTraining:
    def __init__(self, modelTrainingConfigs: ModelTrainingConfigs):
        try:
            pass

        except Exception as e:
            raise CNNClassifierException(e, sys)
