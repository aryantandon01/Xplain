import shap, numpy as np
from app.model_loader import load_model

model = load_model()
explainer = shap.TreeExplainer(model)

def explain(input_data):
    input_array = np.array([input_data])
    shap_values = explainer.shap_values(input_array)
    # Return as list of floats
    return shap_values.tolist()
