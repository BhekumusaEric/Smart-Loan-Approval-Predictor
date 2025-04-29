import streamlit as st
import joblib
import pandas as pd
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Custom CSS for styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
        background-color: #F8F9FA;
    }
    .main {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        max-width: 800px;
        margin: auto;
    }
    .stButton>button {
        background-color: #1A3C5A;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2A5C8A;
    }
    .stNumberInput input, .stSelectbox select {
        border-radius: 5px;
        border: 1px solid #CED4DA;
        padding: 8px;
    }
    .stSuccess {
        background-color: #28A745;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stError {
        background-color: #DC3545;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .section-header {
        color: #1A3C5A;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .input-label {
        color: #1A3C5A;
        font-size: 16px;
        font-weight: 400;
        margin-bottom: 5px;
    }
    .icon {
        margin-right: 10px;
        color: #1A3C5A;
    }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
""", unsafe_allow_html=True)

# Load the pre-trained model and encoders
try:
    model = joblib.load("loan_model.pkl")
    encoders = joblib.load("encoders.pkl")
except FileNotFoundError as e:
    st.error(f"Error: Model or encoders file not found. {str(e)}")
    st.stop()

# Streamlit App Title
st.markdown("""
    <div class="main">
        <h1 style="color: #1A3C5A; text-align: center;">
            <i class="fas fa-university icon"></i> Smart Loan Approval Predictor
        </h1>
        <p style="text-align: center; color: #6C757D;">
            Powered by Eric Bank System - Securely predict your loan approval status
        </p>
    </div>
""", unsafe_allow_html=True)

# Input form in an expander
with st.expander("Enter Loan Applicant Details", expanded=True):
    st.markdown('<p class="section-header">Personal Information</p>', unsafe_allow_html=True)
    
    applicant_details = {}
    
    # Gender
    st.markdown('<p class="input-label"><i class="fas fa-user icon"></i>Gender</p>', unsafe_allow_html=True)
    applicant_details["Gender"] = st.selectbox("Select Gender", ["Male", "Female"], key="gender", label_visibility="collapsed")
    
    # Married
    st.markdown('<p class="input-label"><i class="fas fa-ring icon"></i>Marital Status</p>', unsafe_allow_html=True)
    applicant_details["Married"] = st.selectbox("Select Marital Status", ["Yes", "No"], key="married", label_visibility="collapsed")
    
    # Dependents
    st.markdown('<p class="input-label"><i class="fas fa-users icon"></i>Dependents</p>', unsafe_allow_html=True)
    applicant_details["Dependents"] = st.number_input("Number of Dependents", min_value=0, max_value=3, step=1, key="dependents", label_visibility="collapsed")
    
    # Education
    st.markdown('<p class="input-label"><i class="fas fa-graduation-cap icon"></i>Education</p>', unsafe_allow_html=True)
    applicant_details["Education"] = st.selectbox("Select Education", ["Graduate", "Not Graduate"], key="education", label_visibility="collapsed")
    
    # Self Employed
    st.markdown('<p class="input-label"><i class="fas fa-briefcase icon"></i>Self Employed</p>', unsafe_allow_html=True)
    applicant_details["Self_Employed"] = st.selectbox("Select Self Employed Status", ["Yes", "No"], key="self_employed", label_visibility="collapsed")
    
    st.markdown('<p class="section-header">Financial Information</p>', unsafe_allow_html=True)
    
    # Applicant Income
    st.markdown('<p class="input-label"><i class="fas fa-money-bill-wave icon"></i>Applicant Income</p>', unsafe_allow_html=True)
    applicant_details["ApplicantIncome"] = st.number_input("Applicant Income", min_value=1000, step=100, key="applicant_income", label_visibility="collapsed")
    
    # Coapplicant Income
    st.markdown('<p class="input-label"><i class="fas fa-money-bill-wave icon"></i>Coapplicant Income</p>', unsafe_allow_html=True)
    applicant_details["CoapplicantIncome"] = st.number_input("Coapplicant Income", min_value=0, step=100, key="coapplicant_income", label_visibility="collapsed")
    
    # Loan Amount
    st.markdown('<p class="input-label"><i class="fas fa-wallet icon"></i>Loan Amount</p>', unsafe_allow_html=True)
    applicant_details["LoanAmount"] = st.number_input("Loan Amount", min_value=100, step=100, key="loan_amount", label_visibility="collapsed")
    
    # Loan Amount Term
    st.markdown('<p class="input-label"><i class="fas fa-clock icon"></i>Loan Amount Term (months)</p>', unsafe_allow_html=True)
    applicant_details["Loan_Amount_Term"] = st.selectbox("Select Loan Term", [360, 180, 120], key="loan_term", label_visibility="collapsed")
    
    # Credit History
    st.markdown('<p class="input-label"><i class="fas fa-credit-card icon"></i>Credit History</p>', unsafe_allow_html=True)
    applicant_details["Credit_History"] = st.selectbox("Select Credit History", ["1.0", "0.0"], key="credit_history", label_visibility="collapsed")
    
    # Property Area
    st.markdown('<p class="input-label"><i class="fas fa-home icon"></i>Property Area</p>', unsafe_allow_html=True)
    applicant_details["Property_Area"] = st.selectbox("Select Property Area", ["Urban", "Semiurban", "Rural"], key="property_area", label_visibility="collapsed")

# Prepare the input data
input_data = pd.DataFrame([applicant_details])

# Apply encoding using the saved encoders
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']:
    logger.debug(f"Encoding column {col}: Input value = {input_data[col].iloc[0]}, Encoder classes = {encoders[col].classes_}")
    try:
        input_value = str(input_data[col].iloc[0])  # Ensure input is string to match encoder
        if input_value not in encoders[col].classes_:
            st.warning(f"Warning: Value '{input_value}' in {col} not recognized. Using default: {encoders[col].classes_[0]}")
            input_data[col] = encoders[col].transform([encoders[col].classes_[0]])
        else:
            input_data[col] = encoders[col].transform([input_value])
    except Exception as e:
        st.error(f"Error encoding {col}: {str(e)}. Using default value.")
        input_data[col] = encoders[col].transform([encoders[col].classes_[0]])

# Make prediction using the trained model
if st.button("Predict Loan Status", key="predict_button"):
    try:
        prediction = model.predict(input_data)
        prediction_label = encoders['Loan_Status'].inverse_transform(prediction)[0]
        
        if prediction_label == 'Y':
            st.markdown("""
                <div class="stSuccess">
                    <i class="fas fa-check-circle icon"></i> Loan Approved! Your application has been approved by Eric Bank System.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="stError">
                    <i class="fas fa-times-circle icon"></i> Loan Denied! Unfortunately, your application did not meet the criteria.
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Footer
st.markdown("""
    <div style="text-align: center; color: #6C757D; margin-top: 20px;">
        <p>© 2025 Eric Bank System. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)