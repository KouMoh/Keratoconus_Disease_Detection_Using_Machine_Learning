import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate Synthetic Eye Data
np.random.seed(42)  # For reproducibility

# Features
corneal_thickness = np.random.normal(500, 50, 1000)  # Normal distribution (mean=500, std=50)
curvature = np.random.normal(44, 2, 1000)            # Normal distribution (mean=44, std=2)
age = np.random.randint(18, 80, 1000)                # Random integers between 18 and 80
intraocular_pressure = np.random.normal(15, 3, 1000) # Normal distribution (mean=15, std=3)

# Labels (80% Healthy (0), 20% Keratoconus (1))
labels = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])

# Combine into a DataFrame
data = pd.DataFrame({
    'Corneal_Thickness': corneal_thickness,
    'Curvature': curvature,
    'Age': age,
    'Intraocular_Pressure': intraocular_pressure,
    'Label': labels
})

# Save dataset to CSV (optional)
# data.to_csv("synthetic_eye_data.csv", index=False)

# 2. Visualize Corneal Thickness Distribution
plt.figure(figsize=(10, 6))
plt.hist(corneal_thickness, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Corneal Thickness', fontsize=14)
plt.xlabel('Corneal Thickness (micrometers)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True)
plt.show()

# 3. Train-Test Split
X = data[['Corneal_Thickness', 'Curvature', 'Age', 'Intraocular_Pressure']]
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on Test Set
y_pred = model.predict(X_test)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. Visualize Model Accuracy vs Error
healthy_accuracy = accuracy_score(y_test[y_test == 0], y_pred[y_test == 0]) * 100
keratoconus_error = (1 - accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])) * 100

categories = ['Healthy (0)', 'Keratoconus (1)']
values = [healthy_accuracy, keratoconus_error]

plt.figure(figsize=(8, 6))
plt.bar(categories, values, color=['green', 'red'], alpha=0.7)
plt.title('Model Accuracy vs Error', fontsize=14)
plt.ylabel('Percentage', fontsize=12)
plt.show()
