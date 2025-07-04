# app/explainers/shap_explainer.py
import shap
import numpy as np

# create background data as numpy; we'll convert if needed
background_np = np.random.rand(10, 4)


def shap_explain(model, input_features, model_type="sklearn"):
    X = np.array([input_features])

    if model_type == "sklearn":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

    elif model_type == "tensorflow":
        # background can stay numpy
        explainer = shap.DeepExplainer(model, background_np)
        shap_values = explainer.shap_values(X)

    elif model_type == "pytorch":
        import torch

        background_tensor = torch.tensor(background_np, dtype=torch.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        explainer = shap.DeepExplainer(model, background_tensor)
        shap_values = explainer.shap_values(X_tensor)
        # Convert to numpy if needed
        shap_values = [
            s.detach().numpy() if hasattr(s, "detach") else s for s in shap_values
        ]

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Return list of (feature_name, shap value)
    return list(zip([f"f{i+1}" for i in range(len(input_features))], shap_values[0][0]))
