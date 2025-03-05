import sys
from cnnClassifier.components.stage_01_data_ingestion import DataIngestion
from cnnClassifier.components.stage_02_model_preparation import ModelPreparation
from cnnClassifier.components.stage_03_model_training import ModelTraining
from cnnClassifier.components.stage_04_model_evaluation import ModelEvaluation
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.exception import CNNClassifierException
from cnnClassifier.logger import logging


class TrainPipeline:
    def __init__(self):
        configurationManager = ConfigurationManager()

        self.dataIngestionConfig = configurationManager.get_data_ingestion_configs()
        self.modelPreparationConfigs = configurationManager.get_model_preparation_configs()
        self.modelTrainingConfigs = configurationManager.get_model_training_configs()
        self.modelEvaluationConfig = configurationManager.get_model_evaluation_configs()

    def _start_data_ingestion(self) -> None:
        try:
            logging.info("Starting Process: Data Ingestion")
            dataIngestion = DataIngestion(
                dataIngestionConfig=self.dataIngestionConfig
            )
            dataIngestion.initiate()
            logging.info("Completed process: Data Ingestion")

        except Exception as e:
            raise CNNClassifierException(e, sys) from e

    def _start_model_preparation(self) -> None:
        try:
            logging.info("Starting Process: Base Model Preparation")
            modelPreparation = ModelPreparation(
                modelPreparationConfigs=self.modelPreparationConfigs
            )
            modelPreparation.initiate()
            logging.info("Completed process: Base Model Preparation")

        except Exception as e:
            raise CNNClassifierException(e, sys) from e

    def _start_model_training(self) -> None:
        try:
            logging.info("Starting Process: Model Training")
            modelTraining = ModelTraining(
                modelTrainingConfigs=self.modelTrainingConfigs
            )
            modelTraining.initiate()
            logging.info("Completed process: Model Training")
        except Exception as e:
            raise CNNClassifierException(e, sys) from e

    def _start_model_evaluation(self, log: bool = False) -> None:
        try:
            logging.info("Starting Process: Model Evaluation")
            modelEvaluation = ModelEvaluation(
                modelEvaluationConfig=self.modelEvaluationConfig
            )
            modelEvaluation.initiate()
            if log:
                modelEvaluation.log_into_mlflow()
            logging.info("Completed process: Model Evaluation")
        except Exception as e:
            raise CNNClassifierException(e, sys) from e

    def start(self) -> None:
        try:
            logging.info("Starting Training Pipeline")
            self._start_data_ingestion()
            self._start_model_preparation()
            self._start_model_training()
            self._start_model_evaluation(log=True)

            logging.info("Done Training Pipeline")

        except Exception as e:
            raise CNNClassifierException(e, sys) from e
