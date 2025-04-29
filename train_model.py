import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv("loan_data.csv")

# Preprocessing: Handle missing values, encode categorical features, etc.
df.fillna(0, inplace=True)

# Label encoding for categorical columns
label_encoders = {}
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save the encoder for future use

# Prepare the features and target variable
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "loan_model.pkl")

# Save the encoders
joblib.dump(label_encoders, "encoders.pkl")

print("Model and encoders saved successfully!")
