import pytest
import pandas as pd
import numpy as np
from src.features.technical_indicators import TechnicalIndicators

@pytest.fixture
def indicators():
    return TechnicalIndicators()

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Close': np.random.randn(100).cumsum(),
        'High': np.random.randn(100).cumsum() + 1,
        'Low': np.random.randn(100).cumsum() - 1,
        'Volume': np.random.randint(1000, 10000, 100)
    })

def test_calculate_sma(indicators, sample_data):
    """Test Simple Moving Average calculation."""
    sma = indicators.calculate_sma(sample_data['Close'], 20)
    assert len(sma) == len(sample_data)
    assert isinstance(sma, pd.Series)
    assert not sma[:19].isnull().any()  # First 19 values should be NaN

def test_calculate_rsi(indicators, sample_data):
    """Test RSI calculation."""
    rsi = indicators.calculate_rsi(sample_data['Close'])
    assert len(rsi) == len(sample_data)
    assert all((0 <= x <= 100) for x in rsi.dropna())

def test_calculate_macd(indicators, sample_data):
    """Test MACD calculation."""
    macd_data = indicators.calculate_macd(sample_data['Close'])
    assert isinstance(macd_data, pd.DataFrame)
    assert all(col in macd_data.columns for col in ['MACD', 'Signal', 'MACD_Hist'])

def test_calculate_bollinger_bands(indicators, sample_data):
    """Test Bollinger Bands calculation."""
    bb_data = indicators.calculate_bollinger_bands(sample_data)
    assert all(col in bb_data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower'])
    assert all(bb_data['BB_Lower'] <= bb_data['BB_Middle'])
    assert all(bb_data['BB_Middle'] <= bb_data['BB_Upper'])
