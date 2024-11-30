import pandas as pd
import numpy as np
from typing import List
import logging
import yaml

class TechnicalIndicators:
    """Class for calculating technical indicators."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = df.copy()
        try:
            # Calculate SMA for different periods
            for period in self.config['features']['sma_periods']:
                df[f'SMA_{period}'] = self.calculate_sma(df['Close'], period)
            
            # Calculate RSI
            df['RSI'] = self.calculate_rsi(df['Close'])
            
            # Calculate MACD
            macd_data = self.calculate_macd(df['Close'])
            df = pd.concat([df, macd_data], axis=1)
            
            # Calculate Bollinger Bands
            df = self.calculate_bollinger_bands(df)
            
            # Calculate momentum indicators
            df['ROC'] = self.calculate_roc(df['Close'])
            df['ATR'] = self.calculate_atr(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {str(e)}")
            raise
    
    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period).mean()
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate Relative Strength Index."""
        period = self.config['features']['rsi_period']
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        conf = self.config['features']['macd']
        
        exp1 = prices.ewm(span=conf['short_period']).mean()
        exp2 = prices.ewm(span=conf['long_period']).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=conf['signal_period']).mean()
        
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal,
            'MACD_Hist': macd - signal
        })
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        df = df.copy()
        df['BB_Middle'] = df['Close'].rolling(window=period).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=period).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=period).std()
        return df
    
    @staticmethod
    def calculate_roc(prices: pd.Series, period: int = 12) -> pd.Series:
        """Calculate Rate of Change."""
        return prices.pct_change(periods=period) * 100
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()