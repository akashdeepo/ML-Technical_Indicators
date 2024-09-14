# rfr.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to split the data
def split_data(data, features, target, test_size=0.2):
    split_index = int(len(data) * (1 - test_size))
    X_train = data[features].iloc[:split_index]
    X_test = data[features].iloc[split_index:]
    y_train = data[target].iloc[:split_index]
    y_test = data[target].iloc[split_index:]
    return X_train, X_test, y_train, y_test

# Function to train and evaluate the Random Forest model
def train_and_evaluate_model(X_train, y_train, X_test, y_test, best_params, model_name):
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    
    y_train_pred = pd.Series(model.predict(X_train), index=y_train.index)
    y_test_pred = pd.Series(model.predict(X_test), index=y_test.index)
    
    metrics = calculate_metrics(y_train, y_train_pred, y_test, y_test_pred, model.feature_importances_)
    
    # Save predictions to CSV
    save_predictions_to_csv(y_test, y_test_pred, model_name)
    
    return model, y_train_pred, y_test_pred, metrics

# Function to calculate metrics
def calculate_metrics(y_train, y_train_pred, y_test, y_test_pred, feature_importances):
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'feature_importances': feature_importances
    }
    return metrics

# Function to save predictions to CSV
def save_predictions_to_csv(y_test, y_test_pred, model_name):
    predictions_df = pd.DataFrame({
        'y_test': y_test,
        'predictions': y_test_pred
    })
    predictions_df.to_csv(f'predictions_{model_name}.csv', index=False)

# Function to calculate trend accuracy
def calculate_trend_accuracy(y_true, y_pred):
    actual_trend = np.sign(np.diff(y_true))
    predicted_trend = np.sign(np.diff(y_pred))
    trend_accuracy = np.mean(actual_trend == predicted_trend)
    return trend_accuracy * 100

# Function to calculate direction accuracy
def calculate_direction_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

# Function to plot results and save them
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
    
    # Plot and save the last N time series
    plot_last_n_time_series(data, y_test, y_test_pred, last_n=70, model_name=model_name)

# Function to plot Random Forest Regression results
def plot_rfr_results(y_train, y_train_pred, y_test, y_test_pred,
                     feature_importances, feature_names, data, model_name, last_n=70):
    os.makedirs('plots', exist_ok=True)

    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(25, 12))

    plot_scatter(y_train, y_train_pred, 'Training Data: Actual vs Predicted', axs[0, 0])
    plot_residuals(y_train_pred, y_train, 'Training Data: Residuals', axs[0, 1])
    plot_scatter(y_test, y_test_pred, 'Testing Data: Actual vs Predicted', axs[0, 2])
    plot_residuals(y_test_pred, y_test, 'Testing Data: Residuals', axs[1, 0])
    plot_feature_importance(feature_importances, feature_names, axs[1, 1])
    plot_time_series(data, y_train, y_train_pred, y_test, y_test_pred, last_n, axs[1, 2])

    plt.tight_layout()

    # Save and close the figure here
    plt.savefig(f'plots/{model_name.lower().replace(" ", "_")}_results.png')
    plt.close()

# Function to plot the last N data points time series
def plot_last_n_time_series(data, y_test, y_test_pred, last_n, model_name):
    plt.figure(figsize=(16, 8))
    y_test_last_n = y_test[-last_n:]
    y_test_pred_last_n = y_test_pred[-last_n:]

    plt.plot(y_test_last_n.index, y_test_last_n, label='Actual', color='blue', marker='o')
    plt.plot(y_test_pred_last_n.index, y_test_pred_last_n, label='Predicted', color='orange', marker='o')

    plt.title(f'Actual vs Predicted values over time (Last {last_n} Data Points)')
    plt.xlabel('Time')
    plt.ylabel('Log Return of Close Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f'plots/{model_name.lower().replace(" ", "_")}_last_n_time_series.png')
    plt.close()

# Helper functions for plotting
def plot_scatter(y_true, y_pred, title, ax):
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)

def plot_residuals(y_pred, y_true, title, ax):
    ax.scatter(y_pred, y_true - y_pred, alpha=0.6)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residuals')
    ax.set_title(title)

def plot_feature_importance(feature_importances, feature_names, ax):
    sns.barplot(x=feature_importances, y=feature_names, ax=ax)
    ax.set_title('Feature Importances')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')

def plot_time_series(data, y_train, y_train_pred, y_test, y_test_pred, last_n, ax):
    y_actual = pd.concat([y_train, y_test])
    y_predicted = pd.concat([y_train_pred, y_test_pred])

    y_actual.sort_index(inplace=True)
    y_predicted.sort_index(inplace=True)

    y_actual_last_n = y_actual[-last_n:]
    y_predicted_last_n = y_predicted[-last_n:]

    ax.plot(y_actual_last_n.index, y_actual_last_n, label='Actual', color='blue', marker='o', markersize=5, linestyle='-', linewidth=2, alpha=0.7)
    ax.plot(y_predicted_last_n.index, y_predicted_last_n, label='Predicted', color='orange', marker='x', markersize=5, linestyle='--', linewidth=2, alpha=0.7)

    ax.set_title('Actual vs Predicted Values Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Log Return of Close Price')
    ax.legend()
    ax.grid(True)

# Function to calculate and save metrics to CSV
def calculate_and_save_metrics(y_train, y_train_pred, y_test, y_test_pred, model_name, metrics, features, data):
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
        'Trend Accuracy (%)': [trend_accuracy],
        'Train Direction Accuracy (%)': [train_dir_accuracy],
        'Test Direction Accuracy (%)': [test_dir_accuracy],
        'Train Correlation Coefficient': [train_corr_coef],
        'Test Correlation Coefficient': [test_corr_coef]
    })

    save_metrics_to_csv(metrics_df, model_name)
    return metrics_df

# Function to save metrics to CSV
def save_metrics_to_csv(metrics, model_name):
    os.makedirs('metrics', exist_ok=True)
    metrics.to_csv(f'metrics/{model_name}_performance_metrics.csv', index=False)

# Function to process and evaluate with an indicator
def process_and_evaluate_with_indicator(name, func, data, base_features, target, best_params):
    data_copy = data.copy()
    before_columns = set(data_copy.columns)
    
    # Apply the technical indicator calculation
    data_copy = func(data_copy)
    after_columns = set(data_copy.columns)
    new_columns = list(after_columns - before_columns)
    
    # Drop NaN values introduced by the indicator calculation
    data_copy.dropna(inplace=True)
    
    indicator_features = base_features + new_columns
    print(f"Features used for model with {name}: {indicator_features}")
    
    X_train, X_test, y_train, y_test = split_data(data_copy, indicator_features, target)
    
    model_name = f"Random_Forest_with_{name.replace(' ', '_')}"
    model, y_train_pred, y_test_pred, metrics = train_and_evaluate_model(X_train, y_train, X_test, y_test, best_params, model_name)
    
    metrics_df = calculate_and_save_metrics(y_train, y_train_pred, y_test, y_test_pred, model_name, metrics, indicator_features, data_copy)
    
    plot_and_save_results(y_train, y_train_pred, y_test, y_test_pred, metrics, indicator_features, data_copy, model_name)
    
    print(f"Completed evaluation for model with {name}")
    return metrics_df

# Function to tune hyperparameters
def tune_hyperparameters(X_train, y_train):
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np

    # Define the parameter grid to search
    param_distributions = {
        'n_estimators': [int(x) for x in np.linspace(start=100, stop=500, num=5)],
        'max_features': [1.0, 'sqrt', 'log2', None],  # Updated line
        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the model
    rf = RandomForestRegressor(random_state=42)

    # Set up RandomizedSearchCV
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )

    # Fit the random search model
    rf_random.fit(X_train, y_train)

    # Get the best parameters
    best_params = rf_random.best_params_

    # Optionally, you can print the best score
    print(f"Best Score: {rf_random.best_score_}")

    return best_params
