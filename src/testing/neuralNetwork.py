import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data.csv')

# Separate the features and target variable
X = data.drop(columns=['Target'])
y = data['Target']

# Fix: Subtract 1 from the target labels to map them to the range [0, 4]
y = y - 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build Neural Network Model
def build_nn_model(hidden_layers=[64, 32], learning_rate=0.0008, dropout_rate=0.0):
    model = Sequential()
    model.add(Dense(hidden_layers[0], input_shape=(X_train_scaled.shape[1],), activation='relu'))
    
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        if dropout_rate > 0:  # Add dropout if specified
            model.add(Dropout(dropout_rate))
    
    model.add(Dense(5, activation='softmax'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Define hyperparameter grid
hidden_layers_options = [[128, 64]]
learning_rates = [0.0008]
batch_sizes = [4, 8]
dropout_rates = [0.0]
epochs = [20, 22]

# Initialize variables to track the best model
best_model = None
best_accuracy = 0
best_params = {}

# Loop through hyperparameter combinations
for hidden_layers in hidden_layers_options:
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for dropout_rate in dropout_rates:
                for epoch in epochs:
                    # Print the current hyperparameter combination
                    print(f"\nTraining model with hidden_layers={hidden_layers}, learning_rate={learning_rate}, "
                          f"batch_size={batch_size}, dropout_rate={dropout_rate}, epochs={epoch}")
                    
                    # Build the neural network model with the current hyperparameters
                    model = build_nn_model(hidden_layers=hidden_layers, learning_rate=learning_rate, dropout_rate=dropout_rate)
                    
                    # Train the model and capture the training history
                    history = model.fit(X_train_scaled, y_train, validation_split=0.1, 
                                        epochs=epoch, batch_size=batch_size, verbose=0)
                    
                    # Evaluate the model on the test set
                    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
                    
                    # Print the test accuracy for the current model
                    print(f"Test Accuracy: {test_accuracy:.4f}")
                    
                    # Check if this model's accuracy is better than the best found so far
                    if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                        best_model = model
                        best_params = {
                            'hidden_layers': hidden_layers,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size,
                            'dropout_rate': dropout_rate,
                            'epochs': epoch
                        }
                        print(f"New best model found with accuracy: {best_accuracy:.4f}")

# Print the best parameters and accuracy
print(f"\nBest Parameters: {best_params}")
print(f"Best Test Accuracy: {best_accuracy:.4f}")

# Make predictions with the best model
y_pred = best_model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate relevant statistics
class_report = classification_report(y_test, y_pred_classes)
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Print classification report
print("\nClassification Report:")
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Plot training and validation accuracy over epochs for the best model
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot training and validation loss over epochs for the best model
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

