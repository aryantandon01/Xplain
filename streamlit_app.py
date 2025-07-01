# streamlit_app.py
import streamlit as st
import requests
import shap
import numpy as np
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000"

st.title("Explainable AI as a Service")

# Input form
st.subheader("Enter feature values")
feature_1 = st.number_input("Feature 1", value=5.1)
feature_2 = st.number_input("Feature 2", value=3.5)
feature_3 = st.number_input("Feature 3", value=1.4)
feature_4 = st.number_input("Feature 4", value=0.2)

features = [feature_1, feature_2, feature_3, feature_4]

if st.button("Predict & Explain"):
    # Call /predict
    pred_resp = requests.post(f"{API_URL}/predict", json={"features": features})
    prediction = pred_resp.json()["prediction"]
    st.success(f"Predicted class: {prediction}")

    # Call /explain
    exp_resp = requests.post(f"{API_URL}/explain", json={"features": features})
    shap_values = exp_resp.json()["shap_values"]

    st.subheader("SHAP Explanation")

    # Pick first class for visualization (or do smarter logic)
    shap_values_array = np.array(shap_values)[0,:,0]  # shape (4,)

    # Make a simple bar plot
    fig, ax = plt.subplots()
    ax.bar(range(len(features)), shap_values_array)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(["F1", "F2", "F3", "F4"])
    ax.set_ylabel("SHAP value")
    st.pyplot(fig)
