import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
dataset = pd.read_csv('data.csv')

# Encode categorical variables (if necessary)
dataset['idEye'] = dataset['idEye'].astype('category').cat.codes

# Drop unnecessary columns and scale features
columns_to_drop = ['Unnamed: 0', 'Ks.Axis', 'Kf.Axis', 'AvgK', 'CYL', 'AA', 
                   'Ecc.9.0mm.', 'coma.5', 'coma.axis.5', 'SA.C40..5', 
                   'S35.coma.like..5', 'S46.sph..like..5', 'HOAs.S3456..5', 
                   'AA.5', 'En.Anterior.', 'ESI.Anterior.', 'ESI.Posterior.']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(dataset.drop(columns=columns_to_drop))

# Create a new DataFrame for scaled features and target variable
data_scaled = pd.DataFrame(scaled_features, columns=dataset.columns[1:-16])
data_scaled['target'] = dataset['ESI.Posterior.']

# Split the dataset into training and testing sets
X = data_scaled.drop(columns=['target'])
y = data_scaled['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Create a table with predictions
results_table = pd.DataFrame({
    'Sample Index': X_test.index,
    'Predicted': ['Yes' if pred == 1 else 'No' for pred in y_pred]
})

# Display the results table
print("Prediction Results:")
print(results_table)
