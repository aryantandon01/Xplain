# app/models/sklearn_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset and train at import time (only once)
X, y = load_iris(return_X_y=True)
sk_model = RandomForestClassifier()
sk_model.fit(X, y)

def load_model():
    return sk_model
