from lime.lime_tabular import LimeTabularExplainer
import numpy as np

training_data = np.load("app/training_data.npy")

explainer = LimeTabularExplainer(
    training_data, mode="classification", feature_names=["f1", "f2", "f3", "f4"]
)


def lime_explain(model, input_data, model_type="sklearn"):
    input_array = np.array(input_data)

    # wrap model to provide predict_proba
    if model_type == "tensorflow":

        def predict_proba(X):
            return model.predict(X)

    elif model_type == "pytorch":
        import torch

        def predict_proba(X):
            with torch.no_grad():
                input_tensor = torch.tensor(X, dtype=torch.float32)
                outputs = model(input_tensor).numpy()
                return outputs

    else:  # sklearn
        predict_proba = model.predict_proba

    exp = explainer.explain_instance(input_array, predict_proba, num_features=4)
    return exp.as_list()
