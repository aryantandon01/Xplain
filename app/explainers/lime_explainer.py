from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# assume training_data is loaded at startup; simplify here
training_data = np.load("app/training_data.npy")

explainer = LimeTabularExplainer(
    training_data, mode="classification", feature_names=["f1", "f2", "f3", "f4"]
)


def explain(model, input_data):
    exp = explainer.explain_instance(
        np.array(input_data), model.predict_proba, num_features=4
    )
    # Return list of feature, importance pairs
    return exp.as_list()
