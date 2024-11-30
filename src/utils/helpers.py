import pandas as pd
import numpy as np
from typing import Union, List, Dict
import logging
from datetime import datetime, timedelta

def validate_date_range(start_date: str, end_date: str) -> bool:
    """Validate if the date range is correct."""
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        return start < end
    except ValueError as e:
        logging.error(f"Date validation error: {str(e)}")
        return False

def calculate_returns(prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Calculate percentage returns from price series."""
    return np.diff(prices) / prices[:-1] * 100

def moving_average_crossover(short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
    """Generate trading signals based on MA crossover."""
    return pd.Series(np.where(short_ma > long_ma, 1, 0), index=short_ma.index)

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.01) -> float:
    """Calculate the Sharpe ratio of returns."""
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

def format_currency(value: float) -> str:
    """Format number as currency string."""
    return f"${value:,.2f}"

def calculate_drawdown(prices: pd.Series) -> Dict[str, float]:
    """Calculate maximum drawdown and its duration."""
    rolling_max = prices.expanding().max()
    drawdown = (prices - rolling_max) / rolling_max
    
    max_drawdown = drawdown.min()
    max_drawdown_duration = (drawdown != 0).sum()
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_drawdown_duration
    }

def remove_outliers(data: pd.Series, n_std: float = 3) -> pd.Series:
    """Remove outliers based on standard deviation."""
    mean = data.mean()
    std = data.std()
    return data[(data > mean - n_std * std) & (data < mean + n_std * std)]
