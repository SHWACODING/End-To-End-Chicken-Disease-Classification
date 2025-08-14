import os
import sys
from pathlib import Path

from CNNClassifier.config.configuration import ConfigurationManager
from CNNClassifier.components.model_trainer import Training
from CNNClassifier import logger


STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()

