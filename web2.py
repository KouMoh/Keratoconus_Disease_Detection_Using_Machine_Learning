import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    if 'idEye' in dataset.columns:
        dataset['idEye'] = dataset['idEye'].astype('category').cat.codes
    else:
        st.warning("Column 'idEye' not found. Skipping encoding.")

    # Create the target variable 'Keratoconus-present' based on 'ESI.Posterior.'
    dataset['Keratoconus-present'] = dataset['ESI.Posterior.'].apply(lambda x: 'Yes' if x > 0 else 'No')

    # Define features and target
    features = ['Ks', 'Kf', 'AvgK', 'CYL', 'AA', 'Ecc.9.0mm.', 'ACCP']
    target = 'Keratoconus-present'

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataset[features], dataset[target], test_size=0.2, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, dataset  # Return the dataset

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
    return accuracy, report, y_pred

# Streamlit App
def main():
    st.title("Keratoconus Detection App")
    st.write("This app predicts Keratoconus using machine learning.")

    # Load data
    dataset = load_data()

    # Data Cleaning and Preprocessing
    X_train, X_test, y_train, y_test, processed_dataset = preprocess_data(dataset)

    # Display the results in a table
    st.write("### Prediction Results:")
    st.dataframe(processed_dataset[['idEye', 'Ks', 'Kf', 'AvgK', 'CYL', 'AA', 'Ecc.9.0mm.', 'ACCP', 'Keratoconus-present']])

    # Visualize the results using a bar graph
    st.write("### Distribution of Keratoconus Presence")
    keratoconus_counts = processed_dataset['Keratoconus-present'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=keratoconus_counts.index, y=keratoconus_counts.values, ax=ax)
    ax.set_title('Distribution of Keratoconus Presence')
    ax.set_xlabel('Keratoconus-present')
    ax.set_ylabel('Number of Samples')
    st.pyplot(fig)

    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        accuracy, report, y_pred = evaluate_model(model, X_test, y_test)

        # Display model performance
        st.subheader("Model Performance:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.text(report)
    else:
        st.error("Data preprocessing failed. Check your dataset.")

if __name__ == "__main__":
    main()
