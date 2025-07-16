# Stock Market Prediction System

A machine learning-based system that predicts whether individual stocks will outperform or underperform the S&P 500 index using fundamental financial data.

## Overview

This project uses Support Vector Machines (SVM) to classify stocks as likely to outperform or underperform the market based on 35 key financial metrics including profitability ratios, growth metrics, valuation ratios, and market sentiment indicators.

## Features

- **Data Collection**: Fetch stock price data via Quandl API
- **Data Processing**: Parse HTML files containing stock fundamental data
- **Machine Learning**: SVM-based binary classification for performance prediction
- **Performance Analysis**: Backtesting with simulated investment scenarios
- **Command-Line Interface**: Easy-to-use CLI for different operations

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stockmlproject
   ```

2. **Set up virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

#### Run Full Analysis
```bash
python3 main.py analyze
```

#### Train a Model
```bash
python3 main.py train --data data/key_stats_acc_perf_WITH_NA.csv --test-size 2900
```

#### Fetch Stock Data (requires Quandl API key)
```bash
python3 main.py fetch --api-key YOUR_API_KEY --tickers AAPL MSFT GOOGL
```

#### Process Raw Data
```bash
python3 main.py process --input data/raw_data.csv --output data/processed_data.csv
```

## Project Structure

```
stockmlproject/
├── main.py                    # CLI entry point
├── ml.py                     # Legacy HTML data processor
├── requirements.txt          # Python dependencies
├── CLAUDE.md                # Project documentation for Claude Code
├── models/
│   └── stock_predictor.py   # Main ML pipeline
├── utils/
│   ├── data_fetcher.py      # Quandl API integration
│   └── data_processor.py    # Data preprocessing utilities
├── data/                    # CSV files and datasets
├── forward_json/           # Individual stock JSON files
├── backup_old_files_*/     # Archived legacy files
└── venv/                   # Virtual environment
```

## Data Sources

### Financial Metrics Used
The system analyzes 35 key financial features:

**Valuation Ratios**:
- P/E Ratio (Trailing & Forward)
- Price/Sales, Price/Book
- PEG Ratio
- Enterprise Value/Revenue, Enterprise Value/EBITDA

**Profitability Metrics**:
- Profit Margin, Operating Margin
- Return on Assets (ROA), Return on Equity (ROE)
- Gross Profit, EBITDA, Net Income

**Growth Metrics**:
- Earnings Growth, Revenue Growth
- Revenue Per Share

**Financial Health**:
- Debt/Equity Ratio, Current Ratio
- Total Cash, Total Debt
- Cash Flow, Book Value Per Share

**Market Sentiment**:
- Beta, Market Cap
- Insider Holdings, Institutional Holdings
- Short Interest, Short Ratio

### Data Format
- **Input**: CSV files with stock fundamental data
- **Target**: Binary classification (outperformed vs underperformed S&P 500)
- **Features**: 35 numerical financial metrics
- **Missing Values**: Handled via mean imputation

## Machine Learning Pipeline

### Data Processing
1. **Data Loading**: CSV files with stock fundamentals
2. **Data Cleaning**: Handle missing values and outliers
3. **Feature Scaling**: StandardScaler for normalization
4. **Target Creation**: Binary labels based on S&P 500 comparison

### Model Training
- **Algorithm**: Support Vector Machine (SVM) with linear kernel
- **Features**: 35 financial metrics
- **Cross-validation**: Train/test split
- **Performance Metrics**: Accuracy, investment returns

### Prediction Strategy
The system predicts whether to invest in stocks that are likely to outperform the market, simulating a strategy where:
- $10,000 is invested in each "outperform" prediction
- Returns are compared against equivalent S&P 500 investments
- Performance is measured by total strategy returns vs market returns

## Performance Analysis

The system evaluates model performance using:
- **Accuracy**: Percentage of correct predictions
- **Investment Simulation**: Compares strategy returns vs market returns
- **Return Metrics**: Average return per investment

Example output:
```
Total samples: 15000
Accuracy: 65.23%
Market return: $450.32
Strategy return: $687.15
Difference: $236.83
```

## Configuration

### Environment Setup
- Create a `.env` file for API keys and configuration
- Update file paths in `ml.py` for your local setup
- Adjust data file paths in `main.py` as needed

### MCP Integration
This project includes Model Context Protocol (MCP) server configuration for enhanced capabilities:
- Enhanced file system operations
- Persistent memory across sessions
- Sequential thinking for complex problem-solving
- Web scraping capabilities for financial data

Run `./setup_mcp.sh` to install MCP servers.

## Development

### Running Individual Modules
```bash
# Run stock predictor directly
python3 models/stock_predictor.py

# Run legacy data processor
python3 ml.py
```

### Testing
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .
```

## Dependencies

### Core Requirements
- pandas >= 1.5.0
- numpy >= 1.21.0  
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0

### Data Sources
- quandl >= 3.7.0
- yfinance >= 0.2.0

See `requirements.txt` for complete dependency list.

## Known Limitations

1. **Historical Data Dependency**: Relies on historical fundamental data
2. **Market Condition Sensitivity**: Performance may vary across different market cycles
3. **Data Quality**: Dependent on accuracy of input financial data
4. **Path Configuration**: Some hardcoded paths need manual adjustment
5. **Feature Selection**: Current feature set may not capture all relevant factors

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with data provider terms of service when using financial data APIs.

## Disclaimer

This software is for educational purposes only. It should not be used as the sole basis for investment decisions. Past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.

## Support

For issues and questions:
1. Check the documentation in `CLAUDE.md`
2. Review existing GitHub issues
3. Create a new issue with detailed information

---

*Built with Python, scikit-learn, and financial market data*