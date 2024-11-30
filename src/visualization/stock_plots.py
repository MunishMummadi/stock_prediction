import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict
import yaml

class StockVisualizer:
    """Class for creating stock analysis visualizations."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize visualizer with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        plt.style.use(self.config['visualization']['style'])
    
    def plot_price_prediction(self, y_true: pd.Series, y_pred: pd.Series,
                            title: str = "Stock Price Predictions vs Actual") -> None:
        """Plot actual vs predicted prices."""
        plt.figure(figsize=self.config['visualization']['figure_size'])
        
        plt.plot(y_true.index, y_true.values, label='Actual', alpha=0.8)
        plt.plot(y_true.index, y_pred, label='Predicted', alpha=0.8)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f"{self.config['visualization']['save_path']}price_predictions.png",
                   dpi=self.config['visualization']['dpi'])
        plt.close()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                              title: str = "Feature Importance") -> None:
        """Plot feature importance."""
        plt.figure(figsize=self.config['visualization']['figure_size'])
        
        sns.barplot(data=importance_df.head(10),
                   x='importance', y='feature')
        
        plt.title(title)
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        plt.savefig(f"{self.config['visualization']['save_path']}feature_importance.png",
                   dpi=self.config['visualization']['dpi'])
        plt.close()
    
    def plot_technical_indicators(self, data: pd.DataFrame) -> None:
        """Plot technical indicators."""
        # Create subplot figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
# Plot price and SMAs
        ax1.plot(data.index, data['Close'], label='Price', alpha=0.8)
        for period in self.config['features']['sma_periods']:
            ax1.plot(data.index, data[f'SMA_{period}'], 
                    label=f'SMA {period}', alpha=0.6)
        ax1.set_title('Price and Moving Averages')
        ax1.set_ylabel('Price')
        ax1.legend()
        
        # Plot RSI
        ax2.plot(data.index, data['RSI'], color='purple', alpha=0.8)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.set_title('RSI Indicator')
        ax2.set_ylabel('RSI')
        
        # Plot MACD
        ax3.plot(data.index, data['MACD'], label='MACD', alpha=0.8)
        ax3.plot(data.index, data['Signal'], label='Signal', alpha=0.8)
        ax3.bar(data.index, data['MACD_Hist'], label='MACD Histogram', alpha=0.5)
        ax3.set_title('MACD')
        ax3.set_ylabel('MACD')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.config['visualization']['save_path']}technical_indicators.png",
                   dpi=self.config['visualization']['dpi'])
        plt.close()
    
    def plot_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Plot model performance metrics."""
        plt.figure(figsize=(10, 6))
        
        # Create bar plot of metrics
        metrics_df = pd.DataFrame(list(metrics.items()), 
                                columns=['Metric', 'Value'])
        sns.barplot(data=metrics_df, x='Metric', y='Value')
        
        plt.title('Model Performance Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(f"{self.config['visualization']['save_path']}performance_metrics.png",
                   dpi=self.config['visualization']['dpi'])
        plt.close()