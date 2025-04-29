import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained model and encoders
model = joblib.load("loan_model.pkl")
encoders = joblib.load("encoders.pkl")

# Streamlit App Title
st.title("Smart Loan Approval Predictor")

# Input fields for prediction
st.header("Enter loan applicant details:")

# Collect user inputs (make sure to match the order used in training)
applicant_details = {}

applicant_details["Gender"] = st.selectbox("Gender", ["Male", "Female"])
applicant_details["Married"] = st.selectbox("Marital Status", ["Married", "Single"])
applicant_details["Dependents"] = st.number_input("Dependents", min_value=0, max_value=10, step=1)
applicant_details["Education"] = st.selectbox("Education", ["Graduate", "Not Graduate"])
applicant_details["Self_Employed"] = st.selectbox("Self Employed", ["Yes", "No"])
applicant_details["ApplicantIncome"] = st.number_input("Applicant Income", min_value=1000, step=100)
applicant_details["CoapplicantIncome"] = st.number_input("Coapplicant Income", min_value=0, step=100)
applicant_details["LoanAmount"] = st.number_input("Loan Amount", min_value=100, step=100)
applicant_details["Loan_Amount_Term"] = st.selectbox("Loan Amount Term", [360, 180, 120])
applicant_details["Credit_History"] = st.selectbox("Credit History", [1, 0])
applicant_details["Property_Area"] = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Prepare the input data
input_data = pd.DataFrame([applicant_details])

# Apply encoding using the saved encoders
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
    input_data[col] = encoders[col].transform(input_data[col])

# Make prediction using the trained model
if st.button("Predict"):
    prediction = model.predict(input_data)
    
    if prediction == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Denied!")

