from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from src.preprocess import load_and_clean_data

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Customer Churn Prediction")
st.write("Upload the Telco churn CSV and run prediction using the trained model.")

model_path = "model/churn_model.pkl"

try:
    model = joblib.load(model_path)
except Exception:
    st.warning("Trained model not found. Run training first: python src/train_model.py")
    st.stop()

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if "Churn" in data.columns:
        data_no_target = data.drop(columns=["Churn"])
    else:
        data_no_target = data.copy()

    if "customerID" in data_no_target.columns:
        data_no_target = data_no_target.drop(columns=["customerID"])

    if "TotalCharges" in data_no_target.columns:
        data_no_target["TotalCharges"] = pd.to_numeric(data_no_target["TotalCharges"], errors="coerce")

    pred = model.predict(data_no_target)
    result = pd.DataFrame({"Prediction": pred})
    result["Prediction"] = result["Prediction"].map({0: "Stay", 1: "Churn"})

    st.subheader("Predictions")
    st.dataframe(result)

    st.subheader("Churn Distribution")
    st.bar_chart(result["Prediction"].value_counts())

    csv_bytes = result.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", csv_bytes, "predictions.csv", "text/csv")
else:
    st.info("Use the Telco dataset CSV to generate predictions.")
