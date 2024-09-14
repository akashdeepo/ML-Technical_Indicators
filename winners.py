# winners.py

import pandas as pd
import numpy as np
from data_preprocessing import load_and_prepare_data, create_target_variable, z_score_volume
import technical_indicators as ti
from rfr import *
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure folders exist for saving outputs
os.makedirs('plots', exist_ok=True)
os.makedirs('metrics', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load and preprocess data
filepath = 'SPY_2024-08.csv'
data = load_and_prepare_data(filepath)
data = create_target_variable(data)
data = z_score_volume(data)

# Calculate the winning indicators
data = ti.calculate_cci(data)
data = ti.calculate_ema(data)
data = ti.calculate_sma(data)

# Drop NaN values introduced by indicator calculations
data.dropna(inplace=True)

# Define the base features and add the winning indicators
base_features = ['lr_open', 'lr_high', 'lr_low', 'lr_close', 'z_score_volume']
indicator_features = ['CCI', 'EMA', 'SMA']
features = base_features + indicator_features
target = 'lr_close_t+1'

print(f"Features used for the model: {features}")

# Split the data
X_train, X_test, y_train, y_test = split_data(data, features, target)
print(f"Training data: {X_train.index[0]} through {X_train.index[-1]}")
print(f"Testing data: {X_test.index[0]} through {X_test.index[-1]}")

# Tune hyperparameters
best_params = tune_hyperparameters(X_train, y_train)
print("Best hyperparameters found:", best_params)

# Train and evaluate the model
model_name = "Random_Forest_with_Winners"
model, y_train_pred, y_test_pred, metrics = train_and_evaluate_model(
    X_train, y_train, X_test, y_test, best_params, model_name
)

# Calculate and save metrics
metrics_df = calculate_and_save_metrics(
    y_train, y_train_pred, y_test, y_test_pred, model_name, metrics, features, data
)

# Plot and save results
plot_and_save_results(
    y_train, y_train_pred, y_test, y_test_pred, model, metrics, features, data, model_name
)

# Generate Correlation Matrix
corr_matrix = data[features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Features in the Hybrid Model')
plt.tight_layout()
plt.savefig(f"plots/{model_name}_correlation_matrix.png")
plt.close()

print("\nPerformance Metrics:")
print(metrics_df)
