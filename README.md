# Financial Market Forecasting with Random Forest and Technical Indicators

This repository contains code for predicting financial market movements using a Random Forest Regressor (RFR) model with various technical indicators. Run main.py once all libraries are installed.

## Overview

The project is divided into four main Python scripts:

1. **`data_preprocessing.py`**: Data loading, preparation, target variable creation, and Z-score normalization.
2. **`rfr.py`**: Functions for training, evaluating the model, and plotting results.
3. **`technical_indicators.py`**: Functions for calculating various technical indicators (SMA, EMA, MACD, RSI, etc.).
4. **`main.py`**: Orchestrates data preparation, model training, and evaluation.
5. **`trading_simulation.py`**: Contains the simulation logic for a simple trading strategy. This script simulates a trading strategy using the model's outputs to generate buy, hold, or sell signals, ignoring transaction costs, and allowing fractional shares.

## Prerequisites

- Python 3.7 or higher
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`

