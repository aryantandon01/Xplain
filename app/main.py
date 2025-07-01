from fastapi import FastAPI
from app.schemas import PredictionRequest, PredictionResponse, ExplanationResponse
from app.model_loader import load_model
from app.explainer import explain
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

@app.post("/explain", response_model=ExplanationResponse)
def explain_endpoint(request: PredictionRequest):
    shap_vals = explain(request.features)
    with mlflow.start_run(nested=True):
        mlflow.log_param("input_features", request.features)
        # Log as artifact (optional)
        # mlflow.log_artifact(...) if you save plot
    return {"shap_values": shap_vals}
