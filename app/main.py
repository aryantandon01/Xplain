from fastapi import FastAPI, Query
from app.schemas import PredictionRequest, PredictionResponse, \
    ExplanationResponse
from app.model_loader import load_model
from app.explainers import explain
import mlflow

app = FastAPI()
model = load_model()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    pred = model.predict([request.features])[0]
    with mlflow.start_run(nested=True):
        mlflow.log_param("input_features", request.features)
        mlflow.log_metric("prediction", pred)
    return {"prediction": int(pred)}


@app.post("/explain", response_model=ExplanationResponse\
          , summary="Explain")
def explain_endpoint(request: PredictionRequest, explainer: str = Query("shap")):
    shap_vals = explain(model, request.features, explainer)
    with mlflow.start_run(nested=True):
        mlflow.log_param("input_features", request.features)
        mlflow.log_param("explainer", explainer)
    return {"shap_values": shap_vals}