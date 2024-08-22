from src.exception import CustomException
from src.logger import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from dataclasses import dataclass
import os
import sys
import numpy as np
from src.utils import evaluate
from src.variables import AppWideVariables


@dataclass
class ModelTrainerConfig:
    variables = AppWideVariables().variables
    trained_model_path: str = os.path.join(variables.data_ingestion_variables.artifacts_folder_name, variables.model_training_variables.trained_model_save_folder_name)


class StudentDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target_feature = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx].astype(np.float32)
        y = self.target_feature.iloc[idx].astype(np.float32)
        return torch.tensor(X), torch.tensor(y)


class SingleLayerNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Sequential(nn.Linear(input_size, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 1))

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.fc1(x)


class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def init_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            num_columns = X_train.shape[1]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")

            train_dataset = StudentDataset(X_train, y_train)
            test_dataset = StudentDataset(X_test, y_test)

            batch_size = self.trainer_config.variables.model_training_variables.batch_size
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
            logging.info(f"Training and testing dataloader initialized")

            model = SingleLayerNN(input_size=num_columns).to(device)
            logging.info(f"Model initialized")

            loss_function = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

            epochs = self.trainer_config.variables.model_training_variables.epochs
            patience = self.trainer_config.variables.model_training_variables.patience
            best_score = None
            counter = 0

            tracked_scores_train = {'loss': [], 'rmse': [], 'r2': []}
            tracked_scores_test = {'loss': [], 'rmse': [], 'r2': []}

            os.makedirs(self.trainer_config.trained_model_path, exist_ok=True)

            for epoch in range(epochs):
                train_loss = 0.0
                all_targets_train = []
                all_predictions_train = []

                model.train()
                for _, (X, y) in enumerate(train_dataloader):
                    X, y = X.to(device), y.to(device)

                    y_pred = model(X).reshape(-1)

                    all_targets_train.extend(y.detach().cpu().numpy())
                    all_predictions_train.extend(y_pred.detach().cpu().numpy())

                    loss = loss_function(y_pred, y)
                    train_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                train_loss /= len(train_dataloader)
                rmse, r2 = evaluate(all_predictions_train, all_targets_train)

                tracked_scores_train['loss'].append(train_loss)
                tracked_scores_train['rmse'].append(rmse)
                tracked_scores_train['r2'].append(r2)

                test_loss = 0.0
                all_targets_test = []
                all_predictions_test = []
                model.eval()
                with torch.inference_mode():
                    for _, (X, y) in enumerate(test_dataloader):
                        X, y = X.to(device), y.to(device)
                        y_pred = model(X).reshape(-1)

                        all_targets_test.extend(y.cpu().numpy())
                        all_predictions_test.extend(y_pred.cpu().numpy())

                        loss = loss_function(y_pred, y)
                        test_loss += loss.item()

                test_loss /= len(test_dataloader)
                rmse, r2 = evaluate(all_predictions_test, all_targets_test)

                tracked_scores_test['loss'].append(test_loss)
                tracked_scores_test['rmse'].append(rmse)
                tracked_scores_test['r2'].append(r2)

                if best_score is None or best_score < r2:
                    best_score = r2
                    counter = 0
                    model_save_path = os.path.join(self.trainer_config.trained_model_path, f"model.pth")
                    torch.save(model.state_dict(), model_save_path)
                    logging.info(f"Model improved at Epoch {epoch + 1}. Model saved.")

                    if epoch == epochs - 1:
                        logging.info(f"Finished training for specified {epoch + 1} epochs.")
                        logging.info(f"Tracked metrics: {tracked_scores_test}")

                else:
                    counter += 1
                    if counter == patience:
                        logging.info(f"Stopped training after {epoch + 1} epochs.")
                        logging.info(f"Best scores are: {tracked_scores_test}")
                        break

        except Exception as e:
            raise CustomException(e, sys)
