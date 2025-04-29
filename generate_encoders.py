import joblib
from sklearn.preprocessing import LabelEncoder

# Predefined possible values for each categorical column (must match training)
data = {
    "Gender": ["Male", "Female"],
    "Married": ["Yes", "No"],
    "Education": ["Graduate", "Not Graduate"],
    "Self_Employed": ["Yes", "No"],
    "Property_Area": ["Urban", "Semiurban", "Rural"]
}

encoders = {}

# Create and fit a LabelEncoder for each field
for column, values in data.items():
    le = LabelEncoder()
    le.fit(values)
    encoders[column] = le

# Save all encoders into one file
joblib.dump(encoders, "encoders.pkl")
print("✅ encoders.pkl created successfully.")
