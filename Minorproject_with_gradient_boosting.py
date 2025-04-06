import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load CSV file
file_path = "data.csv"  # Ensure the file is in the working directory
df = pd.read_csv(file_path)

# Display basic information
print(df.info())  # Check column names and data types
print(df.head())  # Preview the first few rows

# Convert categorical columns to numeric using Label Encoding
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Drop unnecessary columns (like 'Unnamed: 0' and non-numeric identifiers)
df.drop(columns=['Unnamed: 0', 'idEye'], errors='ignore', inplace=True)

# Define features and target
X = df.drop(columns=['ESI.Posterior.'])  # Features
y = df['ESI.Posterior.']  # Target variable

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Gradient Boosting Classifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Gradient Boosting Classifier Accuracy: {accuracy:.4f}")
