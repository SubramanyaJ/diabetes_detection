import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def perform_logistic_regression(input_file):
    df = pd.read_csv(input_file)

    # Separate features (X) and the target (y)
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create a Logistic Regression model
    model = LogisticRegression(max_iter=1000)

    # Define hyperparameters to tune
    param_grid = {
        'C': [200],  # Inverse of regularization strength
        'penalty': ['l1', 'l2'],        # Regularization type
        'solver': ['liblinear']         # Solver suitable for L1 and L2
    }
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='accuracy')
    
    # Train the model using GridSearchCV
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_

    # Predict on the test set using the best model
    y_pred = best_model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)

# Usage example
input_file = '../../datasets/data.csv'  # Replace with your file path
perform_logistic_regression(input_file)

