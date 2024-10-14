# Accuracy : 0.8720684672668667

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data.csv')

# Separate the features and target variable
X = data.drop(columns=['Target'])
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create individual classifiers
naive_bayes = GaussianNB()
random_forest = RandomForestClassifier(random_state=40)

# Set up parameter grids for GridSearchCV
param_grid_nb = {
    'var_smoothing': [1e-8]  # Fixed value
}

param_grid_rf = {
    'n_estimators': [50],  # Reduced to one option
    'criterion': ['entropy'],
    'max_depth': [10],  # Only one option
    'min_samples_split': [2, 5],  # Fewer options
    'min_samples_leaf': [1],  # Fixed value
    'max_features': [None]
}

# Create the voting classifier
voting_clf = VotingClassifier(
    estimators=[('naive_bayes', naive_bayes), ('random_forest', random_forest)],
    voting='soft',  # Use 'soft' for probability-based voting
    weights=[1, 5]
)

# Set up the parameter grid for the voting classifier
param_grid_voting = {
    'naive_bayes__var_smoothing': param_grid_nb['var_smoothing'],
    'random_forest__n_estimators': param_grid_rf['n_estimators'],
    'random_forest__criterion': param_grid_rf['criterion'],
    'random_forest__max_depth': param_grid_rf['max_depth'],
    'random_forest__min_samples_split': param_grid_rf['min_samples_split'],
    'random_forest__min_samples_leaf': param_grid_rf['min_samples_leaf'],
    'random_forest__max_features': param_grid_rf['max_features']
}

# Set up GridSearchCV for the voting classifier with reduced cv
grid_search = GridSearchCV(estimator=voting_clf, param_grid=param_grid_voting, cv=5, scoring='accuracy')

# Train the model using GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred = best_model.predict(X_test_scaled)

# Calculate relevant statistics
accuracy = np.mean(y_pred == y_test)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print statistics
print(f"\nBest Hyperparameters: {grid_search.best_params_}")
print(f"\nAccuracy: {accuracy}")
print("\nClassification Report:")
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

