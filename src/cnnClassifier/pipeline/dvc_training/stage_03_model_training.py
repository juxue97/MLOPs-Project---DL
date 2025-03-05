import sys

from cnnClassifier.pipeline.dvc_training import trainPipeline
from cnnClassifier.exception import CNNClassifierException
from cnnClassifier.logger import logging


if __name__ == "__main__":
    try:
        logging.info(">>>>>> Stage 3 <<<<<<")
        trainPipeline._start_model_training()
        logging.info(">>>>>> Stage 3 completed <<<<<<")

    except Exception as e:
        logging.exception(e)
        raise CNNClassifierException(e, sys) from e
