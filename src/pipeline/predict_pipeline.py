import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.model_trainer import SingleLayerNN
import torch
from dataclasses import dataclass


@dataclass
class PredictPipelineConfig:
    data_transformer_path: str = os.path.join("artifacts", "data_transformer.pkl")
    model_path: str = os.path.join("artifacts", "model_checkpoints", "model.pth")



class PredictPipeline:
    def __init__(self):
        self.predict_pipeline_config = PredictPipelineConfig()

    def predict(self, features):
        try:
            transformer = load_object(self.predict_pipeline_config.data_transformer_path)
            transformed_data = torch.from_numpy(transformer.transform(features)).float()
            num_columns = transformed_data.shape[1]
            model = SingleLayerNN(input_size=num_columns).to("cpu")
            model.load_state_dict(torch.load(self.predict_pipeline_config.model_path, weights_only=False))
            model.eval()
            with torch.no_grad():
                predictions = model(transformed_data)
                return predictions

        except Exception as e:
            raise CustomException(e, sys)