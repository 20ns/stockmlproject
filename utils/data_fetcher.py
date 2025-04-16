"""
Data fetching utilities for stock market analysis.

This module provides functions for fetching stock market data from Quandl API.
"""

import pandas as pd
import os
import quandl
from datetime import datetime

def fetch_stock_prices(api_key, tickers=None, start_date="2000-12-12", end_date=None, output_path="../data/stock_prices.csv"):
    """
    Fetch stock prices from Quandl API and save to CSV.
    
    Args:
        api_key (str): Quandl API key.
        tickers (list): List of stock tickers to fetch. If None, will detect from file structure.
        start_date (str): Start date for data fetching in YYYY-MM-DD format.
        end_date (str): End date for data fetching in YYYY-MM-DD format. Defaults to today.
        output_path (str): Path to save the CSV file.
        
    Returns:
        pandas.DataFrame: DataFrame containing stock prices.
    """
    # Configure API key
    quandl.ApiConfig.api_key = api_key
    
    # Set default end date to today if not provided
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Initialize empty DataFrame
    df = pd.DataFrame()
    
    # If tickers not provided, attempt to get from directory structure
    if tickers is None:
        # You will need to adjust this path to where your ticker data is stored
        stats_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stocks")
        if os.path.exists(stats_path):
            stock_dirs = [x for x in os.listdir(stats_path) if os.path.isdir(os.path.join(stats_path, x))]
            tickers = stock_dirs
        else:
            raise ValueError("No tickers provided and default stock directory not found.")
    
    # Fetch data for each ticker
    for ticker in tickers:
        try:
            print(f"Processing ticker: {ticker}")
            name = "WIKI/" + ticker.upper()
            data = quandl.get(name, start_date=start_date, end_date=end_date)
            print(f"Data retrieved for {ticker}: {data.shape}")
            data[ticker.upper()] = data["Adj. Close"]
            df = pd.concat([df, data[ticker.upper()]], axis=1)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    # Save to CSV if data was fetched successfully
    if not df.empty:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path)
        print(f"Stock prices saved to {output_path}")
    
    return df


def fetch_stock_fundamentals(api_key, tickers=None, output_path="../data/stock_fundamentals.csv"):
    """
    Fetch stock fundamental data from Quandl API and save to CSV.
    
    Args:
        api_key (str): Quandl API key.
        tickers (list): List of stock tickers to fetch.
        output_path (str): Path to save the CSV file.
        
    Returns:
        pandas.DataFrame: DataFrame containing fundamental data.
    """
    # This function would need to be implemented based on how your project
    # fetches fundamental data, which wasn't clear from the provided files.
    # This is a placeholder for now.
    pass
