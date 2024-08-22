from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, extract_save_columns
from dataclasses import dataclass
import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.variables import AppWideVariables


@dataclass
class DataTransformationConfig:
    variables = AppWideVariables().variables
    data_transformer_path = os.path.join(variables.data_ingestion_variables.artifacts_folder_name, variables.data_transformation_variables.data_transformer_name)


class DataTransformer:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_transformer(self, train_df):
        X = train_df
        try:
            categorical_features = [feature for feature in X.columns if X[feature].dtype == 'O']
            numerical_features = [feature for feature in X.columns if feature not in categorical_features]

            pipeline_for_numerical_data = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            pipeline_for_categorical_data = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder()),
                ]
            )

            logging.info("Data transformation (imputing, scaling, onehot, ...) process completed.")

            transformer = ColumnTransformer([
                ("numerical_pipeline", pipeline_for_numerical_data, numerical_features),
                ("categorical_pipeline", pipeline_for_categorical_data, categorical_features)
            ])

            return transformer

        except Exception as e:
            raise CustomException(e, sys)

    def init_data_transformation(self, train_data, test_data):
        try:
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)
            logging.info("Train and test data read completed.")

            to_predict_column = train_df.columns[-1]
            transformer = self.get_transformer(train_df.drop(columns=[to_predict_column]))
            logging.info("Data transformer loaded.")

            input_features_train = train_df.drop(columns=[to_predict_column], axis=1)
            to_predict_features_train = train_df[to_predict_column]
            input_features_test = test_df.drop(columns=[to_predict_column], axis=1)
            to_predict_features_test = test_df[to_predict_column]
            logging.info("Data train-test splits with input/target features separation completed.")

            input_features_train_processed = transformer.fit_transform(input_features_train)
            input_features_test_processed = transformer.transform(input_features_test)

            save_object(file_path=self.transformation_config.data_transformer_path, obj=transformer)
            logging.info('Data transformer saved.')

            extract_save_columns(input_features_train, os.path.join(self.transformation_config.variables.data_ingestion_variables.artifacts_folder_name, self.transformation_config.variables.endpoint_variables.columns_saved_file_name))
            print(self.transformation_config.variables.data_ingestion_variables.artifacts_folder_name)
            print(self.transformation_config.variables.endpoint_variables.columns_saved_file_name)
            print(os.path.join(self.transformation_config.variables.data_ingestion_variables.artifacts_folder_name, self.transformation_config.variables.endpoint_variables.columns_saved_file_name))
            logging.info('Columns and datatypes exported.')

            return (input_features_train_processed, input_features_test_processed, to_predict_features_train,
                    to_predict_features_test, self.transformation_config.data_transformer_path)

        except Exception as e:
            raise CustomException(e, sys)
