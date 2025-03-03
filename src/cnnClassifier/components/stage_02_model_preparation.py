from pathlib import Path
import sys

import tensorflow as tf

from cnnClassifier.entity.config import ModelPreparationConfigs
from cnnClassifier.exception import CNNClassifierException
from cnnClassifier.logger import logging


class ModelPreparation:
    def __init__(self, modelPreparationConfigs: ModelPreparationConfigs):
        try:
            self.modelPreparationConfigs = modelPreparationConfigs
        except Exception as e:
            raise CNNClassifierException(e, sys)

    @staticmethod
    def _save_model(path: Path, model: tf.keras.Model) -> None:
        model.save(path)

    def _download_base_model(self) -> None:
        try:
            logging.info("Model Preparation Pipeline: download base model")
            self.model = tf.keras.applications.vgg16.VGG16(
                include_top=self.modelPreparationConfigs.params_include_top,
                weights=self.modelPreparationConfigs.params_weights,
                input_shape=self.modelPreparationConfigs.params_image_size,
            )

            self._save_model(
                path=self.modelPreparationConfigs.base_model_path, model=self.model
            )

        except Exception as e:
            raise CNNClassifierException(e, sys)

    @staticmethod
    def _prepare_full_model(model: tf.keras.Model,
                            classes: int,
                            freezeAll: int,
                            freezeTill: int,
                            learningRate: float
                            ) -> tf.keras.Model:
        try:
            if freezeAll:
                for layers in model.layers:
                    model.trainable = False
            elif (freezeTill is not None) and (freezeTill > 0):
                for layer in model.layers[:-freezeTill]:
                    model.trainable = False

            flatten = tf.keras.layers.Flatten()(model.output)
            prediction = tf.keras.layers.Dense(
                units=classes,
                activation="softmax"
            )(flatten)

            fullModel = tf.keras.models.Model(
                inputs=model.input,
                outputs=prediction,
            )

            fullModel.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"],
            )

            fullModel.summary()
            return fullModel

        except Exception as e:
            raise CNNClassifierException(e, sys)

    def _update_base_model(self) -> None:
        try:
            fullModel = self._prepare_full_model()

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
