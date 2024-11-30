from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import joblib
import logging
import yaml
from typing import Dict, Tuple

class StockPredictor:
    """Class for training and making stock price predictions."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the predictor with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the prediction model.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        try:
            self.logger.info("Starting model training")
            self.model = RandomForestRegressor(
                n_estimators=self.config['model']['n_estimators'],
                random_state=self.config['model']['random_state']
            )
            self.model.fit(X_train, y_train)
            self.logger.info("Model training completed")
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
        
        Returns:
            Dictionary of evaluation metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)
    
    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}")