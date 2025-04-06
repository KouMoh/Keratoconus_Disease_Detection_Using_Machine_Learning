import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Load the dataset
@st.cache
def load_data():
    dataset = pd.read_csv('data.csv')
    return dataset

# Data Cleaning and Processing
def preprocess_data(dataset):
    # Create a target variable 'Keratoconus-present' based on 'ESI.Posterior.'
    dataset['Keratoconus-present'] = dataset['ESI.Posterior.'].apply(lambda x: 'Yes' if x > 0 else 'No')
    return dataset

# Visualization Functions
def visualize_keratoconus_distribution(dataset):
    # Count the occurrences of 'Yes' and 'No' in 'Keratoconus-present'
    counts = dataset['Keratoconus-present'].value_counts()
    
    # Bar Chart
    st.subheader("Bar Chart: Keratoconus Presence")
    st.bar_chart(counts)

    # Pie Chart
    st.subheader("Pie Chart: Proportion of Keratoconus Cases")
    fig, ax = plt.subplots()
    counts.plot(kind='pie', autopct='%1.1f%%', colors=['blue', 'orange'], startangle=90, ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)

def visualize_stacked_bar(dataset):
    grouped_data = dataset.groupby(['idEye', 'Keratoconus-present']).size().unstack(fill_value=0)
    
    st.subheader("Stacked Bar Chart: Keratoconus Presence Across Eye Types")
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_data.plot(kind='bar', stacked=True, color=['blue', 'orange'], ax=ax)
    ax.set_title('Stacked Bar Chart: Keratoconus Presence Across Eye Types')
    ax.set_xlabel('Eye Type (idEye)')
    ax.set_ylabel('Number of Samples')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Streamlit App Layout
def main():
    st.title("Keratoconus Prediction and Visualization")
    
    # Load and preprocess data
    dataset = load_data()
    dataset = preprocess_data(dataset)
    
    # Display raw data
    if st.checkbox("Show Raw Data"):
        st.write(dataset.head())
    
    # Visualizations
    visualize_keratoconus_distribution(dataset)
    
    visualize_stacked_bar(dataset)

if __name__ == "__main__":
    main()
