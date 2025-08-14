# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 11:59:10 2025

@author: renuka
"""

import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

st.title("Prognosis Prediction App")

# Load model scores
model_scores = joblib.load("model_scores.pkl")

# Model selection
model_choice = st.selectbox("Choose Model", ["SVM", "RandomForest", "LogisticRegression"])
st.info(f"Accuracy of {model_choice}: {model_scores[model_choice]*100:.2f}%")

# Load selected model and label encoder
model = joblib.load(f"{model_choice}_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Excel file path
history_file = "prediction_history.xlsx"

# Load existing history or create empty DataFrame
if os.path.exists(history_file):
    history_df = pd.read_excel(history_file)
else:
    history_df = pd.DataFrame(columns=[
        "DateTime", "Model", "Age", "Systolic_BP",
        "Diastolic_BP", "Cholesterol", "Predicted_Prognosis", "Confidence (%)"
    ])

# Input form
st.header("Enter Patient Details")
age = st.number_input("Age", min_value=0.0, step=0.1)
systolic_bp = st.number_input("Systolic BP", min_value=0.0, step=0.1)
diastolic_bp = st.number_input("Diastolic BP", min_value=0.0, step=0.1)
cholesterol = st.number_input("Cholesterol", min_value=0.0, step=0.1)

if st.button("Predict and Save to Report"):
    features = [[age, systolic_bp, diastolic_bp, cholesterol]]
    prediction = model.predict(features)
    prognosis = label_encoder.inverse_transform(prediction)[0]
    
    # Confidence score
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features).max() * 100
    else:
        proba = None
    
    st.success(f"Predicted Prognosis: {prognosis} ({proba:.2f}% confidence)" if proba else f"Predicted Prognosis: {prognosis}")

    # Create new entry
    new_entry = {
        "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model_choice,
        "Age": age,
        "Systolic_BP": systolic_bp,
        "Diastolic_BP": diastolic_bp,
        "Cholesterol": cholesterol,
        "Predicted_Prognosis": prognosis,
        "Confidence (%)": round(proba, 2) if proba else None
    }
    
    # Remove duplicates for the same Age, BP, and Cholesterol
    history_df = history_df[
        ~(
            (history_df["Age"] == age) &
            (history_df["Systolic_BP"] == systolic_bp) &
            (history_df["Diastolic_BP"] == diastolic_bp) &
            (history_df["Cholesterol"] == cholesterol)
        )
    ]
    
    # Add the updated entry
    history_df = pd.concat([history_df, pd.DataFrame([new_entry])], ignore_index=True)
    
    # Save updated report
    history_df.to_excel(history_file, index=False)

# Show full updated history
st.header("Prediction Report")
st.write(history_df)

# Download button
if os.path.exists(history_file):
    with open(history_file, "rb") as f:
        st.download_button(
            "Download Report",
            f,
            file_name="prediction_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
