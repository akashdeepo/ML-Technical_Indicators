import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def trading_strategy(predictions, prices):
    upper_third = np.percentile(predictions, 66)
    lower_third = np.percentile(predictions, 33)

    cash = 10000  # Starting with $10,000
    shares = 0
    num_trades = 0
    portfolio_values = []

    for i in range(len(prices)):
        if predictions[i] > upper_third:
            # Buy signal
            if shares <= 0:
                shares += 1
                cash -= prices[i]
                num_trades += 1
        elif predictions[i] < lower_third:
            # Sell signal
            if shares > 0:
                shares -= 1
                cash += prices[i]
                num_trades += 1
        # Hold signal does nothing

        portfolio_value = cash + shares * prices[i]
        portfolio_values.append(portfolio_value)

    final_value = cash + shares * prices[-1]
    return final_value, num_trades, portfolio_values

def plot_and_save_individual_pl_chart(portfolio_values, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label='Cumulative P/L')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value')
    plt.title(f'P/L Chart for {model_name}')
    plt.legend()
    plt.grid(True)

    output_path = f'plots/pl_chart_{model_name}.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"P/L chart saved for {model_name} at {output_path}.")

def plot_and_save_combined_pl_chart(all_portfolio_values, model_names):
    plt.figure(figsize=(12, 8))
    for model_name, portfolio_values in all_portfolio_values.items():
        plt.plot(portfolio_values, label=model_name)
    
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value')
    plt.title('Combined P/L Chart for All Models')
    plt.legend()
    plt.grid(True)

    output_path = 'plots/combined_pl_chart.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Combined P/L chart saved at {output_path}.")

def simulate_trading(predictions_file, prices_file):
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    if not os.path.exists(prices_file):
        raise FileNotFoundError(f"Prices file not found: {prices_file}")

    predictions = pd.read_csv(predictions_file)['predictions'].values
    prices_df = pd.read_csv(prices_file, parse_dates=['timestamp'])
    prices_df = prices_df.sort_values('timestamp')
    prices = prices_df['close'].values

    min_length = min(len(predictions), len(prices))
    predictions = predictions[-min_length:]
    prices = prices[-min_length:]

    final_value, num_trades, portfolio_values = trading_strategy(predictions, prices)
    return final_value, num_trades, portfolio_values

def run_experiment(prices_file, prediction_files):
    results = {}
    all_portfolio_values = {}

    for prediction_file in prediction_files:
        try:
            final_portfolio_value, num_trades, portfolio_values = simulate_trading(prediction_file, prices_file)
            model_name = prediction_file.split('predictions_')[1].replace('.csv', '')
            results[model_name] = (final_portfolio_value, num_trades)
            all_portfolio_values[model_name] = portfolio_values
            print(f"Final portfolio value for {model_name}: {final_portfolio_value}, Number of trades: {num_trades}")

            plot_and_save_individual_pl_chart(portfolio_values, model_name)

        except FileNotFoundError as e:
            print(e)
    
    plot_and_save_combined_pl_chart(all_portfolio_values, list(results.keys()))

    results_df = pd.DataFrame(list(results.items()), columns=['Model+Indicator', 'Results'])
    results_df[['Final Portfolio Value', 'Number of Trades']] = pd.DataFrame(results_df['Results'].tolist(), index=results_df.index)
    results_df.drop(columns=['Results'], inplace=True)
    results_df.sort_values(by='Final Portfolio Value', ascending=False, inplace=True)
    results_df.to_csv('metrics/trading_experiment_results.csv', index=False)

    print("\nExperiment completed. Results saved to 'metrics/trading_experiment_results.csv'.")
    print("\nTop Performing Models:")
    print(results_df.head())

if __name__ == "__main__":
    prices_file = 'SPY_2024-08.csv'
    prediction_files = [f for f in os.listdir() if f.startswith('predictions_') and f.endswith('.csv')]
    run_experiment(prices_file, prediction_files)