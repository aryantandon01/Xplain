# app/explainers/shap_explainer.py
import shap
import numpy as np

# background data: use small random sample or fixed dummy
background = np.random.rand(10, 4)  # shape should match your model input

def shap_explain(model, input_features, model_type="sklearn"):
    X = np.array([input_features])

    if model_type == "sklearn":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    elif model_type in ["tensorflow", "pytorch"]:
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Return simplified output: feature index and value
    return list(zip([f"f{i+1}" for i in range(len(input_features))], shap_values[0][0]))
