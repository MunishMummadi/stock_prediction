# Stock Price Prediction Model

## Overview
This project implements a machine learning-based stock price prediction system using technical indicators and Random Forest algorithm. The system fetches historical stock data, calculates various technical indicators, trains a prediction model, and generates comprehensive visualizations and analysis reports.

## Features
- Real-time stock data fetching via Yahoo Finance API
- Advanced technical indicator calculations:
  - Simple Moving Averages (SMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Average True Range (ATR)
- Machine learning model using Random Forest Regression
- Interactive visualizations and performance metrics
- Comprehensive logging and error handling
- Configurable parameters via YAML

## Project Structure
```
stock_prediction/
│
├── config/
│   └── config.yaml          # Configuration parameters
├── src/
│   ├── data/               # Data handling modules
│   ├── features/           # Feature engineering
│   ├── models/             # Prediction models
│   ├── visualization/      # Plotting utilities
│   └── utils/              # Helper functions
├── tests/                  # Unit tests
├── notebooks/             # Jupyter notebooks
├── reports/               # Generated reports
│   └── figures/           # Generated visualizations
└── logs/                  # Application logs
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MunishMummadi/stock_prediction.git
cd stock_prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure parameters in `config/config.yaml`

2. Run the main script:
```bash
python main.py
```

3. Access results in the `reports/` directory

## Configuration
Key parameters in `config.yaml`:
```yaml
model:
  algorithm: 'random_forest'
  n_estimators: 100
  test_size: 0.2

features:
  sma_periods: [20, 50, 200]
  rsi_period: 14

data:
  ticker: 'AAPL'
  start_date: '2020-01-01'
```

## Output
The system generates:
- Price prediction visualizations
- Technical indicator plots
- Feature importance analysis
- Performance metrics
- Detailed logs
- Exportable model files

## Dependencies
- pandas
- numpy
- scikit-learn
- yfinance
- matplotlib
- seaborn
- pyyaml

## Testing
Run unit tests:
```bash
python -m pytest tests/
```

## Future Improvements
- [ ] Add support for multiple stock symbols
- [ ] Implement deep learning models
- [ ] Add backtesting capabilities
- [ ] Create web interface
- [ ] Add portfolio optimization
- [ ] Implement real-time predictions
- [ ] Add sentiment analysis

## License
This project is licensed under the MIT License - see the LICENSE file for details.

