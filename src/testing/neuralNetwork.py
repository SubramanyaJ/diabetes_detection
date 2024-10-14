import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('../../datasets/data.csv')

# Separate the features and target variable
X = data.drop(columns=['Target'])
y = data['Target']

# **Fix**: Subtract 1 from the target labels to map them to the range [0, 4]
y = y - 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural Network Hyperparameters (accessible for tuning)
nn_params = {
    'input_shape': X_train_scaled.shape[1],  # Automatically inferred
    'hidden_layers': [64, 32],               # Tunable number of hidden layers and neurons
    'activation': 'relu',                    # Activation function for hidden layers
    'output_activation': 'softmax',          # Activation function for the output layer
    'optimizer': 'adam',                     # Optimizer type (e.g., 'adam', 'sgd')
    'learning_rate': 0.0015,                  # Learning rate for optimizer
    'epochs': 27,                            # Number of epochs to train the model
    'batch_size': 32                         # Batch size for training
}

# Build Neural Network Model
def build_nn_model(params):
    model = Sequential()
    
    # Input layer + first hidden layer
    model.add(Dense(params['hidden_layers'][0], input_shape=(params['input_shape'],), activation=params['activation']))
    
    # Additional hidden layers
    for units in params['hidden_layers'][1:]:
        model.add(Dense(units, activation=params['activation']))
    
    # Output layer (number of classes = 5 in this case, adjust if needed)
    model.add(Dense(5, activation=params['output_activation']))
    
    # Compile the model
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build and train the model
model = build_nn_model(nn_params)
history = model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=nn_params['epochs'], batch_size=nn_params['batch_size'], verbose=1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Accuracy: {test_accuracy}")

# Make predictions
y_pred = model.predict(X_test_scaled)
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

# Plot training and validation accuracy over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

