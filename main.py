# main.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm  # Import tqdm for progress bar
from data_preprocessing import load_and_prepare_data, create_target_variable, z_score_volume
from technical_indicators import (
    calculate_sma, calculate_ema, calculate_macd, calculate_rsi,
    calculate_bollinger_bands, calculate_stochastic_oscillator,
    calculate_fibonacci_retracement, calculate_adx, calculate_obv,
    calculate_cci, calculate_ichimoku
)
from rfr import train_and_evaluate_model, split_data, plot_rfr_results, calculate_trend_accuracy, calculate_direction_accuracy
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
model, y_train_pred, y_test_pred, metrics = train_and_evaluate_model(X_train, y_train, X_test, y_test, best_params)

# Calculate additional metrics
train_dir_accuracy = calculate_direction_accuracy(y_train, y_train_pred)
test_dir_accuracy = calculate_direction_accuracy(y_test, y_test_pred)
trend_accuracy = calculate_trend_accuracy(y_test, y_test_pred)
train_corr_coef = np.corrcoef(y_train, y_train_pred)[0, 1]
test_corr_coef = np.corrcoef(y_test, y_test_pred)[0, 1]

# Save base model metrics to the DataFrame
base_metrics = pd.DataFrame({
    'Model': ['Base Random Forest'],
    'Train RMSE': [metrics['train_rmse']],
    'Test RMSE': [metrics['test_rmse']],
    'Train MAE': [metrics['train_mae']],
    'Test MAE': [metrics['test_mae']],
    'Train R²': [metrics['train_r2']],
    'Test R²': [metrics['test_r2']],
    'Trend Accuracy': [trend_accuracy],
    'Train Direction Accuracy': [train_dir_accuracy],
    'Test Direction Accuracy': [test_dir_accuracy],
    'Train Correlation Coefficient': [train_corr_coef],
    'Test Correlation Coefficient': [test_corr_coef]
})
all_metrics_df = pd.concat([all_metrics_df, base_metrics], ignore_index=True)

# Plot results for the base model
plot_rfr_results(
    y_train=y_train,
    y_train_pred=y_train_pred,
    y_test=y_test,
    y_test_pred=y_test_pred,
    feature_importances=metrics['feature_importances'],
    feature_names=base_features,
    data=data,
    model_name="Base Random Forest",
    last_n=70
)
plt.savefig('plots/base_model_results.png')

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
    # Copy data to avoid modifying the original DataFrame
    indicator_data = data.copy()
    
    # Apply the technical indicator calculation
    indicator_data = func(indicator_data)
    indicator_data.dropna(inplace=True)  # Drop any NaN values from indicator calculation
    
    # Define new feature set including the indicator
    indicator_features = base_features + [col for col in indicator_data.columns if col not in base_features and col not in [target]]
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(indicator_data, indicator_features, target)
    
    # Train and evaluate the model with the new indicator
    model, y_train_pred, y_test_pred, metrics = train_and_evaluate_model(X_train, y_train, X_test, y_test, best_params)
    
    # Calculate additional metrics
    train_dir_accuracy = calculate_direction_accuracy(y_train, y_train_pred)
    test_dir_accuracy = calculate_direction_accuracy(y_test, y_test_pred)
    trend_accuracy = calculate_trend_accuracy(y_test, y_test_pred)
    train_corr_coef = np.corrcoef(y_train, y_train_pred)[0, 1]
    test_corr_coef = np.corrcoef(y_test, y_test_pred)[0, 1]
    
    # Save metrics for the model with the indicator
    metrics_df = pd.DataFrame({
        'Model': [f'Random Forest with {name}'],
        'Train RMSE': [metrics['train_rmse']],
        'Test RMSE': [metrics['test_rmse']],
        'Train MAE': [metrics['train_mae']],
        'Test MAE': [metrics['test_mae']],
        'Train R²': [metrics['train_r2']],
        'Test R²': [metrics['test_r2']],
        'Trend Accuracy': [trend_accuracy],
        'Train Direction Accuracy': [train_dir_accuracy],
        'Test Direction Accuracy': [test_dir_accuracy],
        'Train Correlation Coefficient': [train_corr_coef],
        'Test Correlation Coefficient': [test_corr_coef]
    })
    all_metrics_df = pd.concat([all_metrics_df, metrics_df], ignore_index=True)

    # Plot results for the model with the indicator
    plot_rfr_results(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        feature_importances=metrics['feature_importances'],
        feature_names=indicator_features,
        data=indicator_data,
        model_name=f"Random Forest with {name}",
        last_n=70
    )
    plt.savefig(f'plots/{name.lower().replace(" ", "_")}_model_results.png')
    
    print(f"Completed evaluation for model with {name}")

# Save the combined metrics DataFrame to a single CSV file
all_metrics_df.to_csv('metrics/all_model_performance_metrics.csv', index=False)

# Display the combined metrics as a table
print("\nCombined Metrics for All Models:")
print(all_metrics_df)
