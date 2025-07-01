import shap, numpy as np

def explain(model, input_data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(np.array([input_data]))
    return shap_values.tolist()
