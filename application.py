import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from fastapi import FastAPI
from pydantic import BaseModel

application = FastAPI()


class Item(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int


@application.post("/")
async def endpoint(item: Item):
    inpt = item.model_dump()
    input_data = pd.DataFrame([inpt.values()], columns=inpt.keys())
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(input_data).item()
    return {'prediction': prediction}

