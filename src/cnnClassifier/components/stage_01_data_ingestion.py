import sys

from cnnClassifier.entity.artifact import DataIngestionArtifact
from cnnClassifier.exception import CNNClassifierException
from cnnClassifier.logger import logging


class DataIngestion:
    def __init__(self):
        pass

    def initiate() -> DataIngestionArtifact:
        try:
            logging.info("Initiating Data Ingestion Pipeline")

            return DataIngestionArtifact()
        except Exception as e:
            raise CNNClassifierException(e, sys) from e
