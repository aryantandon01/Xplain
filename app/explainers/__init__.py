from app.explainers import shap_explainer, lime_explainer


def explain(model, input_data, explainer_type):
    if explainer_type == "shap":
        return shap_explainer.explain(model, input_data)
    elif explainer_type == "lime":
        return lime_explainer.explain(model, input_data)
    else:
        raise ValueError(f"Unknown explainer: {explainer_type}")
