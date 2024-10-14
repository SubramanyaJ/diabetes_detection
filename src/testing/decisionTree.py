# Accuracy : 0.8699728596101653

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('../../datasets/data.csv')

# Separate the features and target variable
X = data.drop(columns=['Target'])
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Decision Tree Classifier
cart_model = DecisionTreeClassifier(random_state=42)

# Set up the hyperparameter grid
param_grid = {
    'criterion': ['entropy'],  # Split criteria
    'max_depth': [10, 12],  # Max depth of the tree
    'min_samples_split': [1, 2, 3],  # Minimum samples to split a node
    'min_samples_leaf': [3, 4, 5],  # Minimum samples required in a leaf
    'max_features': [None],  # Number of features to consider for the best split
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=cart_model, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1)

# Train the model with hyperparameter tuning
grid_search.fit(X_train_scaled, y_train)

# Get the best model and its parameters
best_cart_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions on the test data
y_pred = best_cart_model.predict(X_test_scaled)

# Calculate statistics
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print out the statistics
print(f"Best Parameters: {best_params}")
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

