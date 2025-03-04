import os

from box import ConfigBox

from cnnClassifier.constants import *
from cnnClassifier.entity.config import DataIngestionConfigs, ModelPreparationConfigs, ModelTrainingConfigs
from cnnClassifier.utils.main_utils import create_directories, read_yaml


class ConfigurationManager:
    configFilePath = CONFIG_FILE_PATH
    paramFilePath = PARAMS_FILE_PATH

    def __init__(self):
        self.config = read_yaml(self.configFilePath)
        self.params = read_yaml(self.paramFilePath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_configs(self) -> DataIngestionConfigs:
        dataIngestionConfigs = self.config.data_ingestion

        create_directories([dataIngestionConfigs.root_dir])

        data_ingestion_config = DataIngestionConfigs(
            root_dir=dataIngestionConfigs.root_dir,
            dataset_URI=dataIngestionConfigs.dataset_URI,
            zip_data_file=dataIngestionConfigs.zip_data_file,
            unzip_data_dir=dataIngestionConfigs.unzip_data_dir,
        )

        return data_ingestion_config

    def get_model_preparation_configs(self) -> ModelPreparationConfigs:
        modelPreparationConfigs = self.config.model_preparation

        create_directories([modelPreparationConfigs.root_dir])

        model_preparation_config = ModelPreparationConfigs(
            root_dir=modelPreparationConfigs.root_dir,
            base_model_path=modelPreparationConfigs.base_model_path,
            updated_base_model_path=modelPreparationConfigs.updated_base_model_path,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )

        return model_preparation_config

    def get_model_training_configs(self) -> ModelTrainingConfigs:
        modelTrainingConfigs = self.config.model_training
        baseModelConfigs = self.config.model_preparation
        trainDataPath = os.path.join(
            self.config.data_ingestion.unzip_data_dir, modelTrainingConfigs.train_dataset_filename)
        testDataPath = os.path.join(
            self.config.data_ingestion.unzip_data_dir, modelTrainingConfigs.test_dataset_filename)
        validDataPath = os.path.join(
            self.config.data_ingestion.unzip_data_dir, modelTrainingConfigs.valid_dataset_filename)

        create_directories([modelTrainingConfigs.root_dir])

        model_training_configs = ModelTrainingConfigs(
            root_dir=modelTrainingConfigs.root_dir,
            trained_model_path=modelTrainingConfigs.trained_model_path,
            updated_base_model_path=baseModelConfigs.updated_base_model_path,
            train_data=trainDataPath,
            test_data=testDataPath,
            valid_data=validDataPath,
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_is_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
        )

        return model_training_configs
