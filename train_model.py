import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("loan_data.csv")
df = df.dropna()

X = df[['ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = df['Loan_Status'].map({'Y': 1, 'N': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)
