import yfinance as yf
import pandas as pd
import logging
from typing import Optional, Tuple
import yaml

class StockDataLoader:
    """Class for loading and preprocessing stock data."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the data loader with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=self.config['logging']['level'],
            format=self.config['logging']['format'],
            filename=self.config['logging']['file']
        )
    
    def fetch_stock_data(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol. If None, uses config default.
        
        Returns:
            DataFrame containing stock data.
        """
        ticker = ticker or self.config['data']['ticker']
        try:
            self.logger.info(f"Fetching data for {ticker}")
            stock = yf.Ticker(ticker)
            data = stock.history(
                start=self.config['data']['start_date'],
                end=self.config['data']['end_date']
            )
            self.logger.info(f"Successfully fetched {len(data)} records")
            return data
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training and testing.
        
        Args:
            data: Raw stock data DataFrame
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            # Create feature matrix X and target variable y
            target_col = self.config['model']['target_column']
            features = data.drop(columns=['Dividends', 'Stock Splits'])
            
            # Create target variable (next day's closing price)
            features['Target'] = features[target_col].shift(-1)
            
            # Remove last row where target is NaN
            features = features.dropna()
            
            # Separate features and target
            X = features.drop('Target', axis=1)
            y = features['Target']
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def generate_train_test_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """
        Generate train-test split preserving time series order.
        
        Args:
            X: Feature DataFrame
            y: Target Series
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * (1 - self.config['data']['train_test_split']))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test