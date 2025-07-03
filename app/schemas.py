from pydantic import BaseModel
from typing import List


class PredictionRequest(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    prediction: int


class ExplanationResponse(BaseModel):
    shap_values: List
