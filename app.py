import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Load model
model = joblib.load("fraud_model.pkl")

# Page Config
st.set_page_config(
    page_title="UPI Fraud Detection",
    layout="centered"
)

# Title
st.title("🛡️ AI Based UPI Fraud Detection System")

st.markdown("---")

# Transaction Inputs
st.header("Input Transaction Details")

transaction_id = st.text_input(
    "Transaction ID",
    "TXN784512"
)

upi_id = st.text_input(
    "UPI ID",
    "mahesh@oksbi"
)

currency = st.selectbox(
    "Currency",
    ["INR"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["upi"]
)

upi_app = st.selectbox(
    "UPI App",
    [
        "Amazon Pay",
        "PhonePe",
        "Google Pay",
        "Paytm"
    ]
)

bank = st.selectbox(
    "Bank",
    [
        "Axis",
        "SBI",
        "HDFC",
        "ICICI"
    ]
)

status = st.selectbox(
    "Transaction Status",
    [
        "success",
        "failed"
    ]
)

amount_slab = st.selectbox(
    "Amount Slab",
    [
        "small",
        "medium",
        "large"
    ]
)

transaction_amount = st.number_input(
    "Transaction Amount",
    min_value=1.0,
    value=1000.0
)

date = st.date_input(
    "Transaction Date"
)

time = st.time_input(
    "Transaction Time"
)

year = st.number_input(
    "Year",
    min_value=2024,
    max_value=2100,
    value=2026
)

# Convert time to hour
hour = time.hour

# Manual encoding
currency_map = {
    "INR": 0
}

payment_method_map = {
    "upi": 0
}

upi_app_map = {
    "Amazon Pay": 0,
    "PhonePe": 1,
    "Google Pay": 2,
    "Paytm": 3
}

bank_map = {
    "Axis": 0,
    "SBI": 1,
    "HDFC": 2,
    "ICICI": 3
}

status_map = {
    "success": 0,
    "failed": 1
}

amount_map = {
    "small": 0,
    "medium": 1,
    "large": 2
}

# Detect Button
if st.button("Detect Fraud"):

    fraud_score = 0

    # Rule 1 : Very High Amount
    if transaction_amount >= 100000:
        fraud_score += 40

    # Rule 2 : Failed Transaction
    if status == "failed":
        fraud_score += 20

    # Rule 3 : Large Amount Slab
    if amount_slab == "large":
        fraud_score += 20

    # Rule 4 : Midnight Transactions
    if hour >= 0 and hour <= 4:
        fraud_score += 20

    # Machine Learning Prediction

    input_data = np.array([[
        currency_map[currency],
        payment_method_map[payment_method],
        upi_app_map[upi_app],
        bank_map[bank],
        status_map[status],
        amount_map[amount_slab],
        int(transaction_amount),
        year,
        hour
    ]])

    prediction = model.predict(input_data)

    # Final Decision

    st.markdown("---")

    st.header("Detection Result")

    if fraud_score >= 50:

        st.error("🚨 FRAUD TRANSACTION DETECTED")

        st.warning(f"Fraud Risk Score : {fraud_score}%")

    elif prediction[0] == 1:

        st.error("🚨 FRAUD TRANSACTION DETECTED")

        st.warning("Machine Learning Model Flagged This Transaction")

    else:

        st.success("✅ GENUINE TRANSACTION")

        st.info(f"Fraud Risk Score : {fraud_score}%")

    # Transaction Summary

    st.markdown("---")

    st.subheader("Transaction Summary")

    st.write("Transaction ID :", transaction_id)
    st.write("UPI ID :", upi_id)
    st.write("UPI App :", upi_app)
    st.write("Bank :", bank)
    st.write("Status :", status)
    st.write("Amount Slab :", amount_slab)
    st.write("Transaction Amount : ₹", transaction_amount)
    st.write("Date :", date)
    st.write("Time :", time)

# CSV Upload
st.markdown("---")

st.header("📂 Batch Fraud Detection")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("CSV Uploaded Successfully")

    st.dataframe(df)

# Footer
st.markdown("---")

st.caption(
    "AI Based UPI Fraud Detection System Using Random Forest Algorithm"
)
