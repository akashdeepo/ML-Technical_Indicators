import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

# Function to calculate Rachev Ratio
def calculate_rachev_ratio(returns, beta=0.05, gamma=0.95):
    avar_beta = np.percentile(returns, 100 * beta)  # CVaR at lower quantile (losses)
    avar_gamma = np.percentile(returns, 100 * (1 - gamma))  # CVaR at upper quantile (gains)
    return avar_gamma / avar_beta

# Function to calculate Modified Rachev Ratio
def calculate_modified_rachev_ratio(returns, beta=0.05, gamma=0.95, delta=0.05, epsilon=0.95):
    avar_beta_gamma = np.percentile(returns, 100 * beta) / gamma
    avar_delta_epsilon = np.percentile(returns, 100 * delta) / epsilon
    return avar_delta_epsilon / avar_beta_gamma

# Distortion RRR
def calculate_distortion_rrr(returns, beta=0.05):
    distorted_returns_positive = np.percentile(returns, 100 * beta)
    distorted_returns_negative = np.percentile(returns, 100 * (1 - beta))
    return distorted_returns_positive / distorted_returns_negative

# Sortino-Satchell Ratio
def calculate_sortino_satchell(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    semi_deviation = np.std(downside_returns)
    return np.mean(excess_returns) / semi_deviation

# Gains-Loss Ratio
def calculate_gains_loss_ratio(returns):
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    return np.mean(positive_returns) / np.abs(np.mean(negative_returns))

# STAR Ratio
def calculate_star_ratio(returns, cvar_level=0.05):
    cvar = np.mean(np.sort(returns)[:int(len(returns) * cvar_level)])
    return np.mean(returns) / abs(cvar)

# MiniMax Ratio
def calculate_minimax_ratio(returns):
    drawdown = np.max(np.maximum.accumulate(returns) - returns)  # Maximum Drawdown
    return np.mean(returns) / drawdown

# Gini Ratio
def calculate_gini_ratio(returns):
    n = len(returns)
    sorted_returns = np.sort(returns)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_returns)) / (n * np.sum(sorted_returns)) - (n + 1) / n
    return gini

# Function to calculate all reward-risk ratios
def calculate_all_ratios(returns, risk_free_rate=0.0):
    ratios = {
        "Sharpe Ratio": calculate_sharpe_ratio(returns, risk_free_rate),
        "Rachev Ratio": calculate_rachev_ratio(returns),
        "Modified Rachev Ratio": calculate_modified_rachev_ratio(returns),
        "Distortion RRR": calculate_distortion_rrr(returns),
        "Sortino-Satchell Ratio": calculate_sortino_satchell(returns, risk_free_rate),
        "Gains-Loss Ratio": calculate_gains_loss_ratio(returns),
        "STAR Ratio": calculate_star_ratio(returns),
        "MiniMax Ratio": calculate_minimax_ratio(returns),
        "Gini Ratio": calculate_gini_ratio(returns)
    }
    return ratios

# Main function to load prediction files, calculate ratios, and save results to CSV
def main():
    # Directory where the prediction CSV files are located
    predictions_dir = './'

    # Initialize an empty DataFrame to store results
    all_results = []

    # Loop through each prediction file in the directory
    for filename in os.listdir(predictions_dir):
        if filename.startswith('predictions_') and filename.endswith('.csv'):
            # Get the model name from the filename
            model_name = filename.replace('predictions_', '').replace('.csv', '')

            # Load the CSV file
            filepath = os.path.join(predictions_dir, filename)
            df = pd.read_csv(filepath)

            # Extract y_test (actual returns) and predictions (predicted returns)
            y_test = df['y_test'].values
            predictions = df['predictions'].values

            # Calculate reward-risk ratios for the predictions
            ratios = calculate_all_ratios(predictions)

            # Append the model name and ratios to the results list
            results = {
                'Model': model_name,
                **ratios
            }
            all_results.append(results)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save the results to a CSV file
    results_df.to_csv('reward_risk_ratios_results.csv', index=False)

    print("Reward-risk ratios calculated and saved to 'reward_risk_ratios_results.csv'")

    # Generate a heatmap and radar chart
    generate_heatmap(results_df)
    generate_radar_chart(results_df)

# Function to generate and save a heatmap for the calculated ratios
def generate_heatmap(df):
    # Select the columns for metrics
    metrics_df = df.set_index('Model')[['Sharpe Ratio', 'Sortino-Satchell Ratio', 'Gains-Loss Ratio', 'STAR Ratio', 'MiniMax Ratio', 'Gini Ratio']]

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(metrics_df, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Reward-Risk Ratios Heatmap')

    # Save the heatmap
    plt.tight_layout()
    plt.savefig('reward_risk_ratios_heatmap.png')
    plt.show()

# Function to generate and save a radar chart (spider chart) for the models
def generate_radar_chart(df):
    # Set up the radar chart variables
    metrics = ['Sharpe Ratio', 'Rachev Ratio', 'Modified Rachev Ratio', 'Distortion RRR', 'Sortino-Satchell Ratio', 'Gains-Loss Ratio', 'STAR Ratio', 'MiniMax Ratio', 'Gini Ratio']
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create the figure and polar axis
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot each model
    for i, row in df.iterrows():
        values = row[metrics].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, label=row['Model'], linewidth=2)
        ax.fill(angles, values, alpha=0.25)

    # Fix the labels around the circle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    # Add legend and title
    plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), fontsize='small', ncol=1)
    plt.title('Radar Chart Comparison for All Models', size=20, color='blue', y=1.1)

    # Save the radar chart
    plt.tight_layout()
    plt.savefig('reward_risk_ratios_radar_chart.png')
    plt.show()

# Entry point for script execution
if __name__ == "__main__":
    main()
