
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("loan_model.pkl", "rb"))

st.title("🏦 Loan Approval Predictor")
st.write("Enter your details to see if your loan might be approved.")

income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_history = st.selectbox("Credit History (1: Good, 0: Bad)", [1, 0])

if st.button("Predict"):
    data = np.array([[income, loan_amount, credit_history]])
    prediction = model.predict(data)
    result = "✅ Approved" if prediction[0] == 1 else "❌ Not Approved"
    st.success(f"Loan Status: {result}")
