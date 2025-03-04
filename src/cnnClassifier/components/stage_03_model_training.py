from pathlib import Path
import sys

import tensorflow as tf

from cnnClassifier.entity.config import ModelTrainingConfigs
from cnnClassifier.exception import CNNClassifierException
from cnnClassifier.logger import logging


class ModelTraining:
    def __init__(self, modelTrainingConfigs: ModelTrainingConfigs):
        try:
            self.modelTrainingConfigs = modelTrainingConfigs

        except Exception as e:
            raise CNNClassifierException(e, sys)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        logging.info(
            f"Model Training Pipeline: saving trained model to path {path}")

        model.save(path)

    def _get_base_model(self) -> tf.keras.Model:
        try:
            logging.info("Model Training Pipeline: get base model")
            model = tf.keras.models.load_model(
                filepath=self.modelTrainingConfigs.updated_base_model_path
            )
            return model

        except Exception as e:
            raise CNNClassifierException(e, sys)

    def _train_valid_generator(self):
        try:
            logging.info("Model Training Pipeline: create training generator")

            dataGeneratorKwargs = dict(rescale=1.0 / 255)

            dataFlowKwargs = dict(
                target_size=self.modelTrainingConfigs.params_image_size[:-1],
                batch_size=self.modelTrainingConfigs.params_batch_size,
                interpolation="bilinear",
            )

            # Training Data Generator
            if self.modelTrainingConfigs.params_is_augmentation:
                trainDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=40,
                    horizontal_flip=True,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    **dataGeneratorKwargs,
                )
            else:
                trainDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                    **dataGeneratorKwargs,
                )

            self.trainGenerator = trainDataGenerator.flow_from_directory(
                directory=self.modelTrainingConfigs.train_data,
                # subset="training", # only use if dataset is not pre-separated
                shuffle=True,
                **dataFlowKwargs,
            )

            # Validation & Test Data Generator
            validDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                **dataGeneratorKwargs
            )

            self.validGenerator = validDataGenerator.flow_from_directory(
                directory=self.modelTrainingConfigs.valid_data,
                # subset="validation", # only use if dataset is not pre-separated
                shuffle=False,
                **dataFlowKwargs,
            )

            # Test Data Generator (Optional - if needed for evaluation)
            self.testGenerator = validDataGenerator.flow_from_directory(
                directory=self.modelTrainingConfigs.test_data,
                shuffle=False,
                **dataFlowKwargs,
            )

        except Exception as e:
            raise CNNClassifierException(e, sys)

    def _train_model(self):
        try:
            logging.info("Model Training Pipeline: train model")
            stepsPerEpoch = self.trainGenerator.samples // self.trainGenerator.batch_size
            validation_steps = self.validGenerator.samples // self.validGenerator.batch_size

            self.model.fit(
                self.trainGenerator,
                epochs=self.modelTrainingConfigs.params_epochs,
                steps_per_epoch=stepsPerEpoch,
                validation_steps=validation_steps,
                validation_data=self.validGenerator,
            )

        except Exception as e:
            raise CNNClassifierException(e, sys)

    def initiate(self) -> None:
        try:
            logging.info("Initiating Model Training Pipeline")
            self.model = self._get_base_model()
            self._train_valid_generator()
            self._train_model()
            self.save_model(
                path=self.modelTrainingConfigs.trained_model_path, model=self.model
            )

        except Exception as e:
            raise CNNClassifierException(e, sys) from e
