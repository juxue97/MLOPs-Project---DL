from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from cnnClassifier.entity.config import ModelTrainingConfigs
from cnnClassifier.exception import CNNClassifierException
from cnnClassifier.logger import logging


class ModelTraining:
    def __init__(self, modelTrainingConfigs: ModelTrainingConfigs):
        try:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.modelTrainingConfigs = modelTrainingConfigs

        except Exception as e:
            raise CNNClassifierException(e, sys)

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        logging.info(
            f"Model Training Pipeline: saving trained model to path {path}")

        torch.save(model, path)

    def _get_base_model(self) -> nn.Module:
        try:
            logging.info("Model Training Pipeline: get base model")
            model = torch.load(
                self.modelTrainingConfigs.updated_base_model_path,
                weights_only=False,
            )
            model.to(self.device)

            return model

        except Exception as e:
            raise CNNClassifierException(e, sys)

    def _train_valid_generator(self):
        try:
            logging.info(
                "Model Training Pipeline: create training and validation dataloaders")

            # Define image transformations
            transform_train = transforms.Compose([
                transforms.Resize(
                    self.modelTrainingConfigs.params_image_size[:-1]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(40),
                transforms.RandomAffine(
                    degrees=0, shear=0.2, scale=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]) if self.modelTrainingConfigs.params_is_augmentation else transforms.Compose([
                transforms.Resize(
                    self.modelTrainingConfigs.params_image_size[:-1]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            transform_valid = transforms.Compose([
                transforms.Resize(
                    self.modelTrainingConfigs.params_image_size[:-1]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            # Load datasets
            train_dataset = datasets.ImageFolder(
                root=self.modelTrainingConfigs.train_data, transform=transform_train)
            valid_dataset = datasets.ImageFolder(
                root=self.modelTrainingConfigs.valid_data, transform=transform_valid)
            test_dataset = datasets.ImageFolder(
                root=self.modelTrainingConfigs.test_data, transform=transform_valid)

            # Create DataLoader
            self.trainLoader = DataLoader(
                train_dataset, batch_size=self.modelTrainingConfigs.params_batch_size, shuffle=True)
            self.validLoader = DataLoader(
                valid_dataset, batch_size=self.modelTrainingConfigs.params_batch_size, shuffle=False)
            self.testLoader = DataLoader(
                test_dataset, batch_size=self.modelTrainingConfigs.params_batch_size, shuffle=False)

        except Exception as e:
            raise CNNClassifierException(e, sys)

    def _train_model(self):
        try:
            logging.info("Model Training Pipeline: train model")
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(
            ), lr=self.modelTrainingConfigs.params_learning_rate)

            # Set to training phase
            self.model.train()

            for epoch in range(self.modelTrainingConfigs.params_epochs):
                logging.info(
                    f"Epoch {epoch + 1}/{self.modelTrainingConfigs.params_epochs}")

                total_loss = 0
                correct = 0
                total = 0

                progress_bar = tqdm(
                    self.trainLoader, desc=f"Training Epoch {epoch+1}", unit="batch")

                for images, labels in progress_bar:
                    # move data into device
                    images, labels = images.to(
                        self.device), labels.to(self.device)

                    # forward
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                    # backward
                    optimizer.zero_grad()
                    loss.backward()

                    # gradient descent
                    optimizer.step()

                    # generate metrics for progress bar to show total loss
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                    progress_bar.set_postfix(
                        loss=total_loss / (len(self.trainLoader)))

                    break  # remove this line for prod/fulltraining

                train_accuracy = 100 * correct / total
                logging.info(
                    f"Training Loss: {total_loss:.4f} | Training Accuracy: {train_accuracy:.2f}%")

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
