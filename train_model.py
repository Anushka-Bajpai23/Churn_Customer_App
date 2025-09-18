import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

if not os.path.exists('model'):
    os.makedirs('model')


# Load the dataset
try:
    df = pd.read_csv('data/telco_customer.csv')
except FileNotFoundError:
    print("Error: 'telco_customer.csv' not found. Make sure the file is in the 'data' directory.")
    exit()

# Data Preprocessing
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Preprocessing for numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include='object').columns

preprocessor = joblib.load('model/preprocessor.pkl') if 'preprocessor.pkl' in joblib.__dict__ else None
if preprocessor is None:
    # Create a preprocessor pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

# Train the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42))])

model.fit(X, y)

# Save the model and preprocessor
joblib.dump(model, 'model/churn_model.pkl')
print("Model trained and saved successfully as 'model/churn_model.pkl'.")