import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv("loan_data.csv")
print("Dataset columns:", df.columns)

# Drop Loan_ID
df = df.drop(columns=['Loan_ID'], errors='ignore')
print("Columns after dropping Loan_ID:", df.columns)

# Handle missing values
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Label encoding for categorical columns
label_encoders = {}
categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Handle Dependents column (convert '3+' to 3)
df['Dependents'] = pd.to_numeric(df['Dependents'].replace('3+', 3), errors='coerce').fillna(0).astype(int)

# Encode target variable Loan_Status
le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])
label_encoders['Loan_Status'] = le

# Prepare the features and target variable
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

# Verify features
print("Features in X:", X.columns)
print("Data types in X:", X.dtypes)
print("First few rows of X:\n", X.head())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save the trained model and encoders
joblib.dump(model, "loan_model.pkl")
joblib.dump(label_encoders, "encoders.pkl")

print("Model and encoders saved successfully!")