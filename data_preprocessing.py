# data_preprocessing.py

import pandas as pd
import numpy as np

def load_and_prepare_data(filepath, sort_by='timestamp', ascending=True):
    data = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
    data.sort_values(by=sort_by, ascending=ascending, inplace=True)
    for col in ['open', 'high', 'low', 'close']:
        data[f'lr_{col}'] = np.log(data[col] / data[col].shift(1))
    data.dropna(inplace=True)
    return data

def create_target_variable(data, shift_by=-1):
    data['lr_close_t+1'] = data['lr_close'].shift(shift_by)
    data.dropna(inplace=True)
    return data

def z_score_volume(data, window=60):
    rolling_mean = data['volume'].rolling(window).mean()
    rolling_std = data['volume'].rolling(window).std()
    data['z_score_volume'] = (data['volume'] - rolling_mean) / rolling_std
    data.dropna(inplace=True)
    return data
