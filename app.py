
import streamlit as st
import pandas as pd
import joblib
import datetime

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

# Load model and encoders
model = joblib.load("loan_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("🏦 Smart Loan Approval Predictor")

st.markdown("Enter your information to see if your loan will likely be approved.")

# Input form
with st.form("loan_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.slider("Applicant Income", 0, 100000, 5000)
    loan_amount = st.slider("Loan Amount", 0, 1000, 100)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Check Loan Approval")

    if submitted:
        try:
            input_dict = {
                'Gender': encoders['Gender'].transform([gender])[0],
                'Married': encoders['Married'].transform([married])[0],
                'Education': encoders['Education'].transform([education])[0],
                'Self_Employed': encoders['Self_Employed'].transform([self_employed])[0],
                'ApplicantIncome': applicant_income,
                'LoanAmount': loan_amount,
                'Credit_History': credit_history,
                'Property_Area': encoders['Property_Area'].transform([property_area])[0],
            }

            input_df = pd.DataFrame([input_dict])

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]  # Confidence for approval

            result = "✅ Approved" if prediction == 1 else "❌ Not Approved"
            st.success(f"**Result:** {result}")
            st.info(f"**Confidence:** {probability * 100:.2f}%")

            # Logging
            log = input_dict.copy()
            log['Result'] = result
            log['Confidence'] = round(probability * 100, 2)
            log['Timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pd.DataFrame([log]).to_csv("logs.csv", mode='a', index=False, header=not pd.io.common.file_exists("logs.csv"))

        except Exception as e:
            st.error("Something went wrong. Check your inputs or the model.")
