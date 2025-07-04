from .shap_explainer import shap_explain
from .lime_explainer import lime_explain


def explain(model, input_features, explainer="shap", model_type="sklearn"):
    if explainer == "lime":
        return lime_explain(model, input_features)  # only works for sklearn for now
    else:
        return shap_explain(model, input_features, model_type=model_type)
