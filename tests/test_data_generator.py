import pytest
import pandas as pd
import numpy as np
from src.data.stock_data import StockDataLoader

@pytest.fixture
def data_loader():
    return StockDataLoader()

def test_fetch_stock_data(data_loader):
    """Test if data fetching works correctly."""
    df = data_loader.fetch_stock_data('AAPL')
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

def test_prepare_data(data_loader):
    """Test data preparation functionality."""
    # Create sample data
    sample_data = pd.DataFrame({
        'Close': np.random.randn(100),
        'Volume': np.random.randint(1000, 10000, 100),
        'Dividends': np.zeros(100),
        'Stock Splits': np.zeros(100)
    })
    
    X, y = data_loader.prepare_data(sample_data)
    
    assert len(X) == len(y)
    assert 'Target' not in X.columns
    assert not X.isnull().any().any()

def test_train_test_split(data_loader):
    """Test train-test split functionality."""
    X = pd.DataFrame(np.random.randn(100, 5))
    y = pd.Series(np.random.randn(100))
    
    X_train, X_test, y_train, y_test = data_loader.generate_train_test_split(X, y)
    
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
