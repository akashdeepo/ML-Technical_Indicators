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
    """Calculate MACD using EMAs of log returns."""
    exp1 = data['lr_close'].ewm(span=short_window, adjust=False).mean()
    exp2 = data['lr_close'].ewm(span=long_window, adjust=False).mean()
    data['MACD_Line'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD_Line'].ewm(span=signal_window, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD_Line'] - data['MACD_Signal']
    return data

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI) using log returns."""
    delta = data['lr_close']
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
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
    max_return = data['lr_close'].max()
    min_return = data['lr_close'].min()
    diff = max_return - min_return
    levels = [0.236, 0.382, 0.5, 0.618, 0.764]
    for level in levels:
        data[f'Fib_{level}'] = max_return - diff * level
    return data

def calculate_adx(data, window=14):
    """Calculate Average Directional Index (ADX) using log returns."""
    data['TR'] = data[['lr_high', 'lr_low', 'lr_close']].max(axis=1) - data[['lr_high', 'lr_low', 'lr_close']].min(axis=1)
    
    up_move = data['lr_high'].diff()
    down_move = -data['lr_low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_rolling = data['TR'].rolling(window=window).sum()
    plus_dm_rolling = pd.Series(plus_dm, index=data.index).rolling(window=window).sum()
    minus_dm_rolling = pd.Series(minus_dm, index=data.index).rolling(window=window).sum()
    
    plus_di = 100 * (plus_dm_rolling / tr_rolling)
    minus_di = 100 * (minus_dm_rolling / tr_rolling)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    data['ADX'] = dx.rolling(window=window).mean()
    return data

def calculate_obv(data):
    """Calculate On-Balance Volume (OBV) using log returns and volume."""
    obv = [0]
    for i in range(1, len(data)):
        if data['lr_close'].iloc[i] > 0:
            obv.append(obv[-1] + data['volume'].iloc[i])
        elif data['lr_close'].iloc[i] < 0:
            obv.append(obv[-1] - data['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv
    return data

def calculate_cci(data, window=20):
    """Calculate Commodity Channel Index (CCI) using log returns."""
    tp = (data['lr_high'] + data['lr_low'] + data['lr_close']) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    data['CCI'] = (tp - sma) / (0.015 * mad)
    return data

def calculate_ichimoku(data):
    """Calculate Ichimoku Cloud using log returns."""
    high_9 = data['lr_high'].rolling(window=9).max()
    low_9 = data['lr_low'].rolling(window=9).min()
    high_26 = data['lr_high'].rolling(window=26).max()
    low_26 = data['lr_low'].rolling(window=26).min()
    high_52 = data['lr_high'].rolling(window=52).max()
    low_52 = data['lr_low'].rolling(window=52).min()
    
    data['Tenkan_sen'] = (high_9 + low_9) / 2
    data['Kijun_sen'] = (high_26 + low_26) / 2
    data['Senkou_Span_A'] = ((data['Tenkan_sen'] + data['Kijun_sen']) / 2).shift(26)
    data['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
    data['Chikou_Span'] = data['lr_close'].shift(-26)
    return data
