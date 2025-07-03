# app/models/sklearn_model.py
import joblib


def load_model():
    return joblib.load("models/model.pkl")
