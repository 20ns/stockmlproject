"""
Main entry point for stock market prediction application.

This script provides command-line functionality to run different aspects
of the stock market prediction system.
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from utils.data_fetcher import fetch_stock_prices
from utils.data_processor import process_raw_data, prepare_model_data
from models.stock_predictor import run_analysis, build_data_set, train_model, save_model, load_model, analyze_performance


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Stock Market Prediction Tool')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Fetch data command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch stock price data')
    fetch_parser.add_argument('--api-key', required=True, help='Quandl API key')
    fetch_parser.add_argument('--tickers', nargs='+', help='Stock tickers to fetch')
    fetch_parser.add_argument('--start-date', default='2000-12-12', help='Start date (YYYY-MM-DD)')
    fetch_parser.add_argument('--end-date', default=None, help='End date (YYYY-MM-DD)')
    fetch_parser.add_argument('--output', default='data/stock_prices.csv', help='Output file path')
    
    # Process data command
    process_parser = subparsers.add_parser('process', help='Process raw data')
    process_parser.add_argument('--input', required=True, help='Input CSV file')
    process_parser.add_argument('--output', default=None, help='Output CSV file')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train prediction model')
    train_parser.add_argument('--data', default='data/key_stats_acc_perf_WITH_NA.csv', help='Data file path')
    train_parser.add_argument('--test-size', type=int, default=2900, help='Number of samples for testing')
    train_parser.add_argument('--model-output', default='models/stock_model.pkl', help='Output model path')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with trained model')
    predict_parser.add_argument('--model', required=True, help='Path to trained model')
    predict_parser.add_argument('--data', required=True, help='Data file for prediction')
    
    # Run full analysis
    analysis_parser = subparsers.add_parser('analyze', help='Run full analysis pipeline')
    analysis_parser.add_argument('--test-size', type=int, default=2900, help='Number of samples for testing')
    analysis_parser.add_argument('--save-model', action='store_true', help='Save the trained model')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process commands
    if args.command == 'fetch':
        print(f"Fetching stock data for {args.tickers or 'all available tickers'}...")
        fetch_stock_prices(
            api_key=args.api_key,
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.output
        )
    
    elif args.command == 'process':
        print(f"Processing data from {args.input}...")
        process_raw_data(
            input_file=args.input,
            output_file=args.output
        )
    
    elif args.command == 'train':
        print(f"Training model on {args.data}...")
        # Build the dataset
        X, y, Z = build_data_set(args.data)
        print(f"Total samples: {len(X)}")
        
        # Split data into training and testing sets
        test_size = args.test_size
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        # Train the model
        print("Training model...")
        clf = train_model(X_train, y_train)
        
        # Save the model
        save_model(clf, args.model_output)
    
    elif args.command == 'predict':
        # This would need to be implemented based on your prediction needs
        print("Prediction functionality not yet implemented")
    
    elif args.command == 'analyze':
        print("Running full analysis...")
        run_analysis(test_size=args.test_size, save=args.save_model)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
