# rfr.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def split_data(data, features, target, test_size=0.2, shuffle=False):
    X = data[features]
    y = data[target]
    return train_test_split(X, y, test_size=test_size, shuffle=shuffle)

def train_and_evaluate_model(X_train, y_train, X_test, y_test, best_params, model_name):
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'feature_importances': model.feature_importances_
    }
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'y_test': y_test,
        'predictions': y_test_pred
    })
    predictions_df.to_csv(f'predictions_{model_name}.csv', index=False)
    
    return model, y_train_pred, y_test_pred, metrics

def calculate_trend_accuracy(y_true, y_pred):
    actual_trend = np.sign(np.diff(y_true))
    predicted_trend = np.sign(np.diff(y_pred))
    trend_accuracy = np.mean(actual_trend == predicted_trend)
    return trend_accuracy * 100

def calculate_direction_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred))

def plot_rfr_results(y_train, y_train_pred, y_test, y_test_pred, feature_importances, feature_names, data, model_name, last_n=70):
    if not os.path.exists('plots'):
        os.makedirs('plots')

    feature_importances_df = pd.DataFrame(feature_importances, index=feature_names, columns=['Importance']).sort_values('Importance', ascending=False)

    plt.figure(figsize=(25, 12))
    plot_scatter(y_train, y_train_pred, 'Training Data: Actual vs Predicted', 1)
    plot_residuals(y_train_pred, y_train, 'Training Data: Residuals', 2)
    plot_scatter(y_test, y_test_pred, 'Testing Data: Actual vs Predicted', 3)
    plot_residuals(y_test_pred, y_test, 'Testing Data: Residuals', 4)
    plot_feature_importance(feature_importances_df, 5)
    plot_time_series(data, y_train, y_train_pred, y_test, y_test_pred, last_n, 6)
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_results.png')
    plt.close()

    plot_last_n_time_series(data, y_test, y_test_pred, last_n, model_name)

def plot_scatter(y_true, y_pred, title, position):
    plt.subplot(2, 3, position)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)

def plot_residuals(y_pred, y_true, title, position):
    plt.subplot(2, 3, position)
    plt.scatter(y_pred, y_true - y_pred, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(title)

def plot_feature_importance(feature_importances_df, position):
    plt.subplot(2, 3, position)
    sns.barplot(x=feature_importances_df['Importance'], y=feature_importances_df.index)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Features')

def plot_time_series(data, y_train, y_train_pred, y_test, y_test_pred, last_n, position):
    plt.subplot(2, 3, position)
    plt.plot(data.index[-last_n:], y_train[-last_n:], label='Actual - Training', color='blue', marker='o', markersize=5, linestyle='-', linewidth=2, alpha=0.7)
    plt.plot(data.index[-last_n:], y_train_pred[-last_n:], label='Predicted - Training', color='orange', marker='x', markersize=5, linestyle='--', linewidth=2, alpha=0.7)
    plt.plot(data.index[-last_n:], y_test[-last_n:], label='Actual - Testing', color='green', marker='o', markersize=5, linestyle='-', linewidth=2, alpha=0.7)
    plt.plot(data.index[-last_n:], y_test_pred[-last_n:], label='Predicted - Testing', color='red', marker='x', markersize=5, linestyle='--', linewidth=2, alpha=0.7)
    plt.title('Actual vs Predicted Values Over Time (Training and Testing Data)')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)

def plot_last_n_time_series(data, y_test, y_test_pred, last_n, model_name):
    plt.figure(figsize=(16, 8))
    time_series = data.index[-last_n:]
    y_test_actual_last_n = y_test[-last_n:]
    y_test_pred_last_n = y_test_pred[-last_n:]

    plt.plot(time_series, y_test_actual_last_n, label='Actual', color='blue', marker='o')
    plt.plot(time_series, y_test_pred_last_n, label='Predicted', color='orange', marker='o')

    plt.title(f'Actual vs Predicted values over time (Last {last_n} Data Points)')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_last_n_time_series.png')
    plt.close()

def save_metrics_to_csv(metrics, model_name):
    results_df = pd.DataFrame(metrics)
    if not os.path.exists('metrics'):
        os.makedirs('metrics')
    results_df.to_csv(f'metrics/{model_name}_performance_metrics.csv', index=False)
    
def calculate_and_save_metrics(y_train, y_train_pred, y_test, y_test_pred, model_name, metrics, features, data):
    # Calculate additional metrics
    train_dir_accuracy = calculate_direction_accuracy(y_train, y_train_pred)
    test_dir_accuracy = calculate_direction_accuracy(y_test, y_test_pred)
    trend_accuracy = calculate_trend_accuracy(y_test, y_test_pred)
    train_corr_coef = np.corrcoef(y_train, y_train_pred)[0, 1]
    test_corr_coef = np.corrcoef(y_test, y_test_pred)[0, 1]

    metrics_df = pd.DataFrame({
        'Model': [model_name],
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
    save_metrics_to_csv(metrics_df, model_name)
    return metrics_df

def process_and_evaluate_with_indicator(name, func, data, base_features, target, best_params):
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
    model_name = f"Random_Forest_with_{name.replace(' ', '_')}"
    model, y_train_pred, y_test_pred, metrics = train_and_evaluate_model(X_train, y_train, X_test, y_test, best_params, model_name)
    
    # Calculate and save metrics
    metrics_df = calculate_and_save_metrics(y_train, y_train_pred, y_test, y_test_pred, model_name, metrics, indicator_features, indicator_data)
    
    # Plot results for the model with the indicator
    plot_and_save_results(y_train, y_train_pred, y_test, y_test_pred, metrics, indicator_features, indicator_data, model_name)
    
    print(f"Completed evaluation for model with {name}")
    return metrics_df

def plot_and_save_results(y_train, y_train_pred, y_test, y_test_pred, metrics, features, data, model_name):
    plot_rfr_results(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        feature_importances=metrics['feature_importances'],
        feature_names=features,
        data=data,
        model_name=model_name,
        last_n=70
    )
    plt.savefig(f'plots/{model_name.lower().replace(" ", "_")}_results.png')
