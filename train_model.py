# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("loan_data.csv")

# Drop rows with missing target or features
df = df.dropna()

# Encode categorical features
label_encoders = {}
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split features and target
X = df[['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome', 'LoanAmount', 'Credit_History', 'Property_Area']]
y = df['Loan_Status']

# Encode target
y = LabelEncoder().fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model & encoders
joblib.dump(model, "loan_model.pkl")
joblib.dump(label_encoders, "encoders.pkl")
