from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfigs:
    root_dir: Path
    dataset_URI: str
    zip_data_file: Path
    unzip_data_dir: Path


@dataclass(frozen=True)
class ModelPreparationConfigs:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_classes: int
    params_image_size: list


@dataclass(frozen=True)
class ModelTrainingConfigs:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    train_data: Path
    test_data: Path
    valid_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_learning_rate: float


@dataclass(frozen=True)
class ModelEvaluationConfigs:
    path_of_model: Path
    validation_data: Path
    mlflow_uri: str
    all_params: dict
    score_file_path: Path
