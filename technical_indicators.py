# technical_indicators.py
import pandas as pd
import numpy as np

def calculate_sma(data, window=10):
    """Calculate Simple Moving Average (SMA) using log returns."""
    data['SMA'] = data['lr_close'].rolling(window=window).mean()
    return data

def calculate_ema(data, span=10):
    """Calculate Exponential Moving Average (EMA) using log returns."""
    data['EMA'] = data['lr_close'].ewm(span=span, adjust=False).mean()
    return data

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """Calculate Moving Average Convergence Divergence (MACD) using log returns."""
    data['MACD_Line'] = data['lr_close'].ewm(span=short_window, adjust=False).mean() - data['lr_close'].ewm(span=long_window, adjust=False).mean()
    data['MACD_Signal'] = data['MACD_Line'].ewm(span=signal_window, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD_Line'] - data['MACD_Signal']
    return data

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI) using log returns."""
    delta = data['lr_close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands using log returns."""
    rolling_mean = data['lr_close'].rolling(window).mean()
    rolling_std = data['lr_close'].rolling(window).std()
    data['Bollinger_Upper'] = rolling_mean + (rolling_std * num_std)
    data['Bollinger_Lower'] = rolling_mean - (rolling_std * num_std)
    return data

def calculate_stochastic_oscillator(data, window=14):
    """Calculate Stochastic Oscillator using log returns."""
    data['Lowest_Low'] = data['lr_low'].rolling(window).min()
    data['Highest_High'] = data['lr_high'].rolling(window).max()
    data['Stochastic_%K'] = 100 * ((data['lr_close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low']))
    data['Stochastic_%D'] = data['Stochastic_%K'].rolling(window=3).mean()
    return data

def calculate_fibonacci_retracement(data):
    """Calculate Fibonacci Retracement Levels using log returns."""
    max_price = data['lr_close'].max()
    min_price = data['lr_close'].min()
    levels = [0.236, 0.382, 0.5, 0.618, 0.764]
    for level in levels:
        data[f'Fib_{level}'] = min_price + (max_price - min_price) * level
    return data

def calculate_adx(data, window=14):
    """Calculate Average Directional Index (ADX) using high, low, and close prices with log returns."""
    data['High-Low'] = data['lr_high'] - data['lr_low']
    data['High-Close'] = np.abs(data['lr_high'] - data['lr_close'].shift(1))
    data['Low-Close'] = np.abs(data['lr_low'] - data['lr_close'].shift(1))
    data['TR'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    
    high_diff = data['lr_high'].diff()
    low_diff = -data['lr_low'].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    plus_di = 100 * (plus_dm.rolling(window).mean() / data['TR'].rolling(window).mean())
    minus_di = 100 * (minus_dm.rolling(window).mean() / data['TR'].rolling(window).mean())
    data['ADX'] = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di)).rolling(window).mean()
    return data

def calculate_obv(data):
    """Calculate On-Balance Volume (OBV) using log returns and normalized volume."""
    obv = [0]
    for i in range(1, len(data)):
        if data['lr_close'].iloc[i] > 0:
            obv.append(obv[-1] + data['z_score_volume'].iloc[i])
        elif data['lr_close'].iloc[i] < 0:
            obv.append(obv[-1] - data['z_score_volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv
    return data


def calculate_cci(data, window=20):
    tp = (data['high'] + data['low'] + data['close']) / 3
    sma = tp.rolling(window=window).mean()
    mean_deviation = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma) / (0.015 * mean_deviation)
    data['CCI'] = cci
    return data

def calculate_ichimoku(data):
    """Calculate Ichimoku Cloud using log returns."""
    high_9 = data['lr_high'].rolling(window=9).max()
    low_9 = data['lr_low'].rolling(window=9).min()
    high_26 = data['lr_high'].rolling(window=26).max()
    low_26 = data['lr_low'].rolling(window=26).min()
    high_52 = data['lr_high'].rolling(window=52).max()
    low_52 = data['lr_low'].rolling(window=52).min()
    
    data['Tenkan_sen'] = (high_9 + low_9) / 2  # Conversion Line
    data['Kijun_sen'] = (high_26 + low_26) / 2  # Base Line
    data['Senkou_Span_A'] = ((data['Tenkan_sen'] + data['Kijun_sen']) / 2).shift(26)  # Leading Span A
    data['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)  # Leading Span B
    data['Chikou_Span'] = data['lr_close'].shift(-26)  # Lagging Span
    return data
