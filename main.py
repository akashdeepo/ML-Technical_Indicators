import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from data_preprocessing import load_and_prepare_data, create_target_variable, z_score_volume
from rfr import calculate_and_save_metrics
from technical_indicators import *  # Import all technical indicators
from rfr import *
import os

# Ensure folders exist for saving outputs
os.makedirs('plots', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

# Initialize an empty DataFrame to store all metrics
all_metrics_df = pd.DataFrame()

# Load and preprocess data
filepath = 'SPY_2024-08.csv'
data = load_and_prepare_data(filepath)
data = create_target_variable(data)
data = z_score_volume(data)

# Define the base model features and target
base_features = ['lr_open', 'lr_high', 'lr_low', 'lr_close', 'z_score_volume']
target = 'lr_close_t+1'

# Split data for the base model
X_train, X_test, y_train, y_test = split_data(data, base_features, target)
print(f"Training data: {X_train.index[0]} through {X_train.index[-1]}")
print(f"Testing data: {X_test.index[0]} through {X_test.index[-1]}")

# Train and evaluate the base model
best_params = {
    'max_depth': 20, 
    'max_features': 'sqrt', 
    'min_samples_leaf': 2, 
    'min_samples_split': 2, 
    'n_estimators': 300
}
model_name = "Base_Random_Forest"
model, y_train_pred, y_test_pred, metrics = train_and_evaluate_model(X_train, y_train, X_test, y_test, best_params, model_name)

# Calculate and save metrics
base_metrics = calculate_and_save_metrics(y_train, y_train_pred, y_test, y_test_pred, model_name, metrics, base_features, data)
all_metrics_df = pd.concat([all_metrics_df, base_metrics], ignore_index=True)

# Plot results for the base model
plot_and_save_results(y_train, y_train_pred, y_test, y_test_pred, metrics, base_features, data, model_name)

# List of indicators to be calculated
indicators = {
    'SMA': calculate_sma,
    'EMA': calculate_ema,
    'MACD': calculate_macd,
    'RSI': calculate_rsi,
    'Bollinger Bands': calculate_bollinger_bands,
    'Stochastic Oscillator': calculate_stochastic_oscillator,
    'Fibonacci Retracement': calculate_fibonacci_retracement,
    'ADX': calculate_adx,
    'OBV': calculate_obv,
    'CCI': calculate_cci,
    'Ichimoku Cloud': calculate_ichimoku
}

# Calculate and evaluate models with each indicator added
for name, func in tqdm(indicators.items(), desc="Processing Indicators", unit="indicator"):
    # Process and evaluate with the indicator
    indicator_metrics = process_and_evaluate_with_indicator(name, func, data, base_features, target, best_params)
    all_metrics_df = pd.concat([all_metrics_df, indicator_metrics], ignore_index=True)

# Save the combined metrics DataFrame to a single CSV file
all_metrics_df.to_csv('metrics/all_model_performance_metrics.csv', index=False)

# Display the combined metrics as a table
print("\nCombined Metrics for All Models:")
print(all_metrics_df)
