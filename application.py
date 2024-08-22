import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from fastapi import FastAPI
from pydantic import BaseModel
import os
from src.utils import yaml_to_pydantic
from src.variables import AppWideVariables

variables = AppWideVariables().variables

application = FastAPI()

Item = yaml_to_pydantic(os.path.join(variables.data_ingestion_variables.artifacts_folder_name, variables.endpoint_variables.columns_saved_file_name), "CustomPydanticModel")


@application.post("/")
async def endpoint(item: Item):
    inpt = item.model_dump()
    input_data = pd.DataFrame([inpt.values()], columns=inpt.keys())
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(input_data).item()
    return {'prediction': prediction}

