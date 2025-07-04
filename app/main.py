from fastapi import FastAPI, Query
from app.schemas import PredictionRequest, PredictionResponse, ExplanationResponse
from app.model_loader import load_model
from app.explainers import explain
import mlflow

app = FastAPI()


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest, model_type: str = Query("sklearn")  # default to sklearn
):
    model = load_model(model_type)

    if model_type == "pytorch":
        import torch

        input_tensor = torch.tensor([request.features], dtype=torch.float32)
        with torch.no_grad():
            pred = model(input_tensor).numpy()[0]
        pred_class = int(pred.argmax())  # assuming binary or multiclass
    elif model_type == "tensorflow":
        pred = model.predict([request.features])[0]
        pred_class = int(pred.argmax())
    else:  # sklearn
        pred_class = int(model.predict([request.features])[0])

    with mlflow.start_run(nested=True):
        mlflow.log_param("input_features", request.features)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("prediction", pred_class)

    return {"prediction": pred_class}


@app.post("/explain", response_model=ExplanationResponse, summary="Explain")
def explain_endpoint(
    request: PredictionRequest,
    explainer: str = Query("shap"),
    model_type: str = Query("sklearn"),
):
    model = load_model(model_type)
    shap_vals = explain(model, request.features, explainer, model_type)

    with mlflow.start_run(nested=True):
        mlflow.log_param("input_features", request.features)
        mlflow.log_param("explainer", explainer)
        mlflow.log_param("model_type", model_type)

    return {"shap_values": shap_vals}
