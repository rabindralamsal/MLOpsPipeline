import os
import sys
import dill
from src.exception import CustomException
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pickle
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from pydantic import create_model
import pandas as pd

dtype_mapping = {
    'object': str,
    'int64': int,
    'float64': float
}

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate(predictions, targets):
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    return rmse, r2


def extract_save_columns(df: pd.DataFrame, yaml_file_path: str):
    columns = {col: str(df[col].dtype) for col in df.columns}

    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(columns, yaml_file, default_flow_style=False)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_yaml(file_path):
    try:
        with open(file_path) as f:
            content = yaml.safe_load(f)
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise CustomException(e, sys)


def yaml_to_pydantic(yaml_file_path: str, model_name):
    column_types_dict = load_yaml(yaml_file_path).to_dict()
    fields = {col: (dtype_mapping.get(dtype, str), ...) for col, dtype in column_types_dict.items()}
    return create_model(model_name, **fields)