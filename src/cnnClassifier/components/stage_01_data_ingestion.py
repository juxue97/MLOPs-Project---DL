import sys
import os
import zipfile
import gdown
import re

from cnnClassifier.entity.artifact import DataIngestionArtifact
from cnnClassifier.entity.config import DataIngestionConfigs
from cnnClassifier.exception import CNNClassifierException
from cnnClassifier.logger import logging


class DataIngestion:
    def __init__(self, dataIngestionConfig: DataIngestionConfigs):
        self.dataIngestionConfig = dataIngestionConfig

        # a file
        prefix = "https://drive.google.com/uc?id="
        dataID = self.dataIngestionConfig.dataset_URI.split("/")[-2]
        self.uri = prefix + dataID

    def _download_dataset(self):
        try:
            logging.info("Data Ingestion Pipeline: download dataset")
            gdown.download(
                url=self.uri, output=self.dataIngestionConfig.zip_data_file, quiet=False
            )
        except Exception as e:
            raise CNNClassifierException(e, sys)

    def _extract_zip_file(self):
        try:
            logging.info("Data Ingestion Pipeline: extract zip file")
            unzipPath = self.dataIngestionConfig.unzip_data_dir
            os.makedirs(unzipPath, exist_ok=True)
            with zipfile.ZipFile(self.dataIngestionConfig.zip_data_file, "r") as zipRef:
                zipRef.extractall(unzipPath)
        except Exception as e:
            raise CNNClassifierException(e, sys)

    def initiate(self) -> None:
        try:
            logging.info("Initiating Data Ingestion Pipeline")

            self._download_dataset()
            self._extract_zip_file()

        except Exception as e:
            raise CNNClassifierException(e, sys) from e
