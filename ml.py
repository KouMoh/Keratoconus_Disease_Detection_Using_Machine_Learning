import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv('data.csv')
    return dataset

# Data Cleaning and Preprocessing
def preprocess_data(dataset):
    # Handle missing values (if any)
    dataset = dataset.dropna()  # Remove rows with missing values

    # Encode categorical variables (if necessary)
    dataset['idEye'] = dataset['idEye'].astype('category').cat.codes

    # Define features and target
    features = ['Ks', 'Kf', 'AvgK', 'CYL', 'AA', 'Ecc.9.0mm.', 'ACCP']
    target = 'ESI.Posterior.'

    # Handle missing columns
    missing_cols = [col for col in features if col not in dataset.columns]
    if missing_cols:
        st.warning(f"Missing columns in dataset: {missing_cols}")
        return None, None, None, None

    X = dataset[features]
    y = dataset[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Train Model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Streamlit App
def main():
    st.title("Keratoconus Detection App")
    st.write("This app predicts Keratoconus using machine learning.")

    # Load data
    dataset = load_data()

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(dataset)
    
    if X_train is None or X_test is None or y_train is None or y_test is None:
        st.error("Data preprocessing failed. Check your dataset.")
        return

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    accuracy, report = evaluate_model(model, X_test, y_test)

    # Display model performance
    st.subheader("Model Performance:")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.text(report)

    # Data visualization
    st.subheader("Data Visualization:")
    
    # Feature distributions
    st.write("### Feature Distributions")
    for feature in ['Ks', 'Kf', 'AvgK']:
        fig, ax = plt.subplots()
        ax.hist(dataset[feature], bins=20)
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
