# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_excel('Laboratory.xlsx')

# Data Cleaning
# Drop rows with missing target column values
data = data.dropna(subset=['dissolution_av'])

# Drop columns that are not needed for training
columns_to_drop = ['size','start','batch']
data = data.drop(columns=columns_to_drop)

# Handle 'strength' column with variations like '5MG' or '5m'
def parse_strength(strength_value):
    try:
        return float(strength_value.replace('MG', '').replace('M', ''))
    except:
        return None

data['strength'] = data['strength'].apply(parse_strength)

# Convert remaining object columns to float
# object_columns = data.select_dtypes(include=['object']).columns
# data[object_columns] = data[object_columns].apply(pd.to_numeric, errors='coerce')

# Convert remaining object columns to float
object_columns = data.select_dtypes(include=['object']).columns
data[object_columns] = data[object_columns].apply(pd.to_numeric, errors='coerce')

data = data.fillna(data.median())

features = data.drop(['dissolution_av', 'dissolution_min', 'resodual_solvent', 'impurities_total', 'impurity_o', 'impurity_l'], axis=1)
target_columns = data[['dissolution_av', 'dissolution_min', 'resodual_solvent', 'impurities_total', 'impurity_o', 'impurity_l']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target_columns, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error

# Initialize the Neural Network model
model = keras.Sequential([
    keras.layers.Dense(units=46, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=6)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model using the training data
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Predict the target values using the trained model
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Print the Mean Squared Error
print(f"Mean Squared Error (Neural Network): {mse}")

import numpy as np

# Input feature values (adjust these according to your dataset)
input_features = np.array([[25, 5, 5, 2, 1, 2, 1, 1.53, 0.25, 0.13, 94.5, 1.27, 18.52, 109.999,
    0.05, 17, 50, 82, 4.251, 0.45, 0.33, 31.156, 112.141, 245.499,
    4.4, 3.012, 3.3, 3.4, 3.4, 3.4, 111, 116, 0.92, 0.72, 56.84, 68.6,
    46, 37, 56, 62.72, 6.1, 6.1, 1.412698315, 1.926183442, 95.785, 94.697]])

# Use the trained model to predict the target columns
predicted_values = model.predict(input_features)

# Print the predicted values for each target column
print("Predicted values:")
print(f"Dissolution_av: {predicted_values[0, 0]}")
print(f"Dissolution_min: {predicted_values[0, 1]}")
print(f"Residual_solvent: {predicted_values[0, 2]}")
print(f"Impurities_total: {predicted_values[0, 3]}")
print(f"Impurity_o: {predicted_values[0, 4]}")
print(f"Impurity_l: {predicted_values[0, 5]}")

import pickle
pickle.dump(model,open('model.pkl','wb'))