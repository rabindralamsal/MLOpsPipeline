from src.exception import CustomException
from src.logger import logging
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig, DataTransformer
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def init_data_ingestion(self):
        logging.info("Initializing data ingestion")
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/krishnaik06/mlproject/main/notebook/data/stud.csv")
            logging.info('Dataset loaded.')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, header=True, index=False)

            logging.info('Train - test split initiated.')
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(self.ingestion_config.train_data_path, header=True, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, header=True, index=False)
            logging.info('Train - test split created and saved. Ingestion completed.')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.init_data_ingestion()
    data_transformation = DataTransformer()
    X_train, X_test, y_train, y_test, _ = data_transformation.init_data_transformation(train_data_path, test_data_path)
    model_train = ModelTrainer()
    model_train.init_model_trainer(X_train, X_test, y_train, y_test)


