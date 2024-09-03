# RFR.py
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

def train_and_evaluate_model(X_train, y_train, X_test, y_test, best_params):
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
