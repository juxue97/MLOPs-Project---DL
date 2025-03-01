from cnnClassifier.components.stage_01_data_ingestion import DataIngestion


if "__name__" == "__main__":
    dataIngestion = DataIngestion()
    dataIngestionArtifact = dataIngestion.initiate()
