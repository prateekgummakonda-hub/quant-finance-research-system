# Quant Finance Research System

A quantitative finance research project built using Python and Google Colab.

## Overview

This project explores:

- Multi-asset portfolio backtesting
- Strategy comparison (Buy & Hold, Momentum, Alternate-Day)
- Risk and performance metrics
- Machine learning-based market direction prediction using Random Forest

## Data

- Source: Yahoo Finance (via yfinance)
- Assets:
  - AAPL
  - TSLA
  - NVDA
  - SPY

## Methodology

### Portfolio Construction
- Equal-weight allocation across assets
- Daily returns aggregation

### Performance Metrics
- Total Return
- Sharpe Ratio
- Maximum Drawdown

### Strategies Implemented
- Buy & Hold
- Alternate-Day Trading
- Momentum Strategy

### Machine Learning Model
- Model: Random Forest Classifier
- Features:
  - Lag returns (1–5 days)
  - Momentum indicators
  - Moving average ratios
  - Volatility
  - RSI-style indicator

- Target:
  - Next-day market direction (Up/Down)

## Results

- Compared portfolio performance against SPY benchmark
- Evaluated multiple strategies
- Tested predictive power of engineered features

## Key Insight

While machine learning can detect patterns in market data, achieving consistent outperformance remains challenging due to market noise and randomness.

## Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- yfinance

## Future Improvements

- Hyperparameter tuning
- Feature engineering enhancements
- Deep learning models (LSTM)
- Transaction cost modeling
