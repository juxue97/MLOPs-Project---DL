import sys
from cnnClassifier.components.stage_01_data_ingestion import DataIngestion
from cnnClassifier.exception import CNNClassifierException


if __name__ == "__main__":
    try:

        dataIngestion = DataIngestion()
        dataIngestionArtifact = dataIngestion.initiate()

    except Exception as e:
        raise CNNClassifierException(e, sys) from e
