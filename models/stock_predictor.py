"""
Stock Market Performance Predictor

This module provides functionality to predict stock market performance
using machine learning techniques, specifically Support Vector Machines (SVM).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from matplotlib import style
import os
import pickle
from datetime import datetime

# Set plotting style
style.use("ggplot")

# Features used for prediction
FEATURES = [
    'DE Ratio', 'Trailing P/E', 'Price/Sales', 'Price/Book',
    'Profit Margin', 'Operating Margin', 'Return on Assets',
    'Return on Equity', 'Revenue Per Share', 'Market Cap',
    'Enterprise Value', 'Forward P/E', 'PEG Ratio',
    'Enterprise Value/Revenue', 'Enterprise Value/EBITDA',
    'Revenue', 'Gross Profit', 'EBITDA', 'Net Income Avl to Common ',
    'Diluted EPS', 'Earnings Growth', 'Revenue Growth',
    'Total Cash', 'Total Cash Per Share', 'Total Debt',
    'Current Ratio', 'Book Value Per Share', 'Cash Flow',
    'Beta', 'Held by Insiders', 'Held by Institutions',
    'Shares Short (as of', 'Short Ratio', 'Short % of Float',
    'Shares Short (prior '
]


def build_data_set(csv_file="../data/key_stats_acc_perf_WITH_NA.csv"):
    """
    Build and preprocess the dataset for model training and testing.
    
    Args:
        csv_file (str): Path to the CSV file containing stock data.
        
    Returns:
        tuple: Processed features (X), labels (y), and additional stock/market data (Z).
    """
    # Load the data
    data_df = pd.read_csv(csv_file)
    
    # Shuffle the data
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    
    # Replace string NaNs with np.nan
    data_df = data_df.replace(["NaN", "N/A"], np.nan)
    
    # Impute NaN values with column mean
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(data_df[FEATURES])
    
    # Assign labels: 0 for underperformed, 1 for outperformed
    y = np.where(data_df['Status'] == "underperformed", 0, 1)
    
    # Scale the features to have zero mean and unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Get the stock and S&P 500 percentage changes
    Z = data_df[["stock_p_change", "sp500_p_change"]].values
    
    return X, y, Z


def train_model(X, y, kernel="linear", C=1.0):
    """
    Train an SVM model on the provided data.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Labels.
        kernel (str): Kernel type for SVM.
        C (float): Regularization parameter.
        
    Returns:
        sklearn.svm.SVC: Trained SVM model.
    """
    clf = svm.SVC(kernel=kernel, C=C)
    clf.fit(X, y)
    return clf


def save_model(model, filename="../models/stock_model.pkl"):
    """
    Save the trained model to a file.
    
    Args:
        model: The trained model to save.
        filename (str): Path where the model will be saved.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")


def load_model(filename="../models/stock_model.pkl"):
    """
    Load a trained model from a file.
    
    Args:
        filename (str): Path to the model file.
        
    Returns:
        The loaded model.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


def analyze_performance(model, X_test, y_test, Z_test, invest_amount=10000):
    """
    Analyze the performance of the trained model.
    
    Args:
        model: The trained model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        Z_test (numpy.ndarray): Stock and market percentage changes.
        invest_amount (int): Initial investment amount.
        
    Returns:
        tuple: Accuracy, market return, and strategy return.
    """
    test_size = len(X_test)
    correct_count = 0
    total_invests = 0
    if_market = 0
    if_strat = 0
    
    for x in range(test_size):
        prediction = model.predict(X_test[x].reshape(1, -1))[0]
        
        # Check if prediction matches the actual label
        if prediction == y_test[x]:
            correct_count += 1
        
        # If the prediction is 1 (outperformed), calculate returns
        if prediction == 1:
            total_invests += 1
            stock_change = Z_test[x][0] / 100  # Convert percentage to decimal
            sp500_change = Z_test[x][1] / 100  # Convert percentage to decimal
            
            if_market += invest_amount * sp500_change
            if_strat += invest_amount * stock_change
    
    # Calculate accuracy
    accuracy = (correct_count / test_size) * 100
    
    # Calculate average returns
    market_return = if_market / total_invests if total_invests > 0 else 0
    strategy_return = if_strat / total_invests if total_invests > 0 else 0
    
    return accuracy, market_return, strategy_return


def run_analysis(test_size=2900, save=True):
    """
    Run the full analysis pipeline: load data, train model, and evaluate performance.
    
    Args:
        test_size (int): Number of samples to use for testing.
        save (bool): Whether to save the trained model.
        
    Returns:
        tuple: Accuracy, market return, strategy return, and the trained model.
    """
    # Build the dataset
    X, y, Z = build_data_set()
    print(f"Total samples: {len(X)}")
    
    # Split data into training and testing sets
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    Z_test = Z[-test_size:]
    
    # Train the model
    clf = train_model(X_train, y_train)
    
    # Analyze performance
    accuracy, market_return, strategy_return = analyze_performance(clf, X_test, y_test, Z_test)
    
    # Print results
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Market return: ${market_return:.2f}")
    print(f"Strategy return: ${strategy_return:.2f}")
    print(f"Difference: ${strategy_return - market_return:.2f}")
    
    # Save the model if requested
    if save:
        save_model(clf)
    
    return accuracy, market_return, strategy_return, clf


if __name__ == "__main__":
    run_analysis()
