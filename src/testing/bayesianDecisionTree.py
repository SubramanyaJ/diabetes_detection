'''
Using git bash :
Execute the following commands in git bash:
    git clone https://github.com/SubramanyaJ/diabetes_detection.git
    cd /diabetes_detection/src/testing
    python3 bayesianDecisionTree.py
    (Libraries need to be installed)

Using Google Collab:
Copy this code into a python notebook
Upload data.csv OR main.csv which is in this directory to the runtime
(File icon in the left panel)
Run
'''

# Accuracy : 0.8729576525508503 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
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
decision_tree = DecisionTreeClassifier(random_state=40)

# Set up parameter grids for GridSearchCV
param_grid_nb = {
    'var_smoothing': [1e-8]  # Values from 1 to 1e-9
}

param_grid_dt = {
    'criterion': ['entropy'],
    'splitter': ['best'],
    'max_depth': [9],
    'min_samples_split': [20],
    'min_samples_leaf': [10],
    'max_features': [None, "best split"]
}

# Create the voting classifier
voting_clf = VotingClassifier(
    estimators=[('naive_bayes', naive_bayes), ('decision_tree', decision_tree)],
    voting='soft',  # Use 'soft' for probability-based voting
    weights=[1, 5]
)

# Set up the parameter grid for the voting classifier
param_grid_voting = {
    'naive_bayes__var_smoothing': param_grid_nb['var_smoothing'],
    'decision_tree__criterion': param_grid_dt['criterion'],
    'decision_tree__splitter': param_grid_dt['splitter'],
    'decision_tree__max_depth': param_grid_dt['max_depth'],
    'decision_tree__min_samples_split': param_grid_dt['min_samples_split'],
    'decision_tree__min_samples_leaf': param_grid_dt['min_samples_leaf'],
    'decision_tree__max_features': param_grid_dt['max_features']
}

# Set up GridSearchCV for the voting classifier
grid_search = GridSearchCV(estimator=voting_clf, param_grid=param_grid_voting, cv=10, scoring='accuracy')

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

