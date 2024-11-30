from src.data.stock_data import StockDataLoader
from src.features.technical_indicators import TechnicalIndicators
from src.models.predictor import StockPredictor
from src.visualization.stock_plots import StockVisualizer
import logging
import yaml

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Set up logging
    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format'],
        filename=config['logging']['file']
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        data_loader = StockDataLoader()
        technical_indicators = TechnicalIndicators()
        predictor = StockPredictor()
        visualizer = StockVisualizer()
        
        # Load and prepare data
        logger.info("Loading stock data...")
        raw_data = data_loader.fetch_stock_data()
        
        # Calculate technical indicators
        logger.info("Calculating technical indicators...")
        data_with_features = technical_indicators.calculate_all_features(raw_data)
        
        # Prepare features and target
        X, y = data_loader.prepare_data(data_with_features)
        X_train, X_test, y_train, y_test = data_loader.generate_train_test_split(X, y)
        
        # Train model
        logger.info("Training model...")
        predictor.train(X_train, y_train)
        
        # Make predictions
        y_pred = predictor.predict(X_test)
        
        # Evaluate model
        metrics = predictor.evaluate(y_test, y_pred)
        feature_importance = predictor.get_feature_importance(X.columns)
        
        # Create visualizations
        logger.info("Generating visualizations...")
        visualizer.plot_price_prediction(y_test, y_pred)
        visualizer.plot_feature_importance(feature_importance)
        visualizer.plot_technical_indicators(data_with_features)
        visualizer.plot_performance_metrics(metrics)
        
        # Save model
        predictor.save_model('models/stock_predictor.joblib')
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()