import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("fraud_model.pkl")

THRESHOLD = 0.05



st.title("Credit Card Fraud Detection")

uploaded_file = st.file_uploader("Upload transaction CSV file")

if uploaded_file is not None:
    
    data = pd.read_csv(uploaded_file)
    
    X = data.drop(columns=["Class"], errors="ignore")
    
    proba = model.predict_proba(X)[:,1]
    
    prediction = (proba >= THRESHOLD).astype(int)
    
    data["fraud_probability"] = proba
    data["fraud_prediction"] = prediction
    
    st.write("Prediction Results")
    st.dataframe(data.head(50))
    
    fraud_count = prediction.sum()
    
    st.write(f"Total Transactions: {len(data)}")
    st.write(f"Fraud Detected: {fraud_count}")