import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from CNNClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    def train_valid_generator(self):
        # Use modern TensorFlow 2.x approach instead of deprecated ImageDataGenerator
        img_height, img_width = self.config.params_image_size[:-1]
        
        # Training dataset
        if self.config.params_is_augmentation:
            train_datagen = tf.keras.Sequential([
                tf.keras.layers.RandomRotation(0.4),
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomTranslation(0.2, 0.2),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomContrast(0.2),
            ])
        else:
            train_datagen = None

        # Load training data
        self.train_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=self.config.params_batch_size,
            shuffle=True
        )

        # Load validation data
        self.valid_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=self.config.params_batch_size,
            shuffle=False
        )

        # Apply data augmentation to training data if enabled
        if train_datagen:
            self.train_generator = self.train_generator.map(
                lambda x, y: (train_datagen(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Normalize pixel values
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        self.train_generator = self.train_generator.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        self.valid_generator = self.valid_generator.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Optimize performance
        self.train_generator = self.train_generator.prefetch(tf.data.AUTOTUNE)
        self.valid_generator = self.valid_generator.prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        # Calculate steps for training and validation
        # For tf.data.Dataset, we don't need to calculate steps manually
        # The fit method will automatically handle this
        
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            verbose=1
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
