"""
Data processing utilities for stock market analysis.

This module provides functions for processing and preparing stock market data.
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def process_raw_data(input_file, output_file=None, features=None):
    """
    Process raw stock data and prepare it for machine learning.
    
    Args:
        input_file (str): Path to input CSV file.
        output_file (str): Path to output CSV file. If None, will not save to file.
        features (list): List of features to include. If None, will use all features.
        
    Returns:
        pandas.DataFrame: Processed DataFrame.
    """
    # Read data
    data = pd.read_csv(input_file)
    
    # Handle missing values
    data = data.replace(["NaN", "N/A"], np.nan)
    
    # If specific features are requested, select only those
    if features is not None:
        # Make sure all requested features exist in the data
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            print(f"Warning: The following features were not found in the data: {missing_features}")
        
        # Select only features that exist in the data
        available_features = [f for f in features if f in data.columns]
        data = data[available_features]
    
    # Fill missing values with mean
    for col in data.select_dtypes(include=[np.number]).columns:
        data[col] = data[col].fillna(data[col].mean())
    
    # Save processed data if output file specified
    if output_file:
        data.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
    
    return data


def load_json_data(json_dir, output_file=None):
    """
    Load data from multiple JSON files and combine into a DataFrame.
    
    Args:
        json_dir (str): Directory containing JSON files.
        output_file (str): Path to output CSV file. If None, will not save to file.
        
    Returns:
        pandas.DataFrame: Combined DataFrame.
    """
    data = []
    
    # Check if directory exists
    if not os.path.exists(json_dir):
        raise ValueError(f"Directory not found: {json_dir}")
    
    # Get list of JSON files
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    if not json_files:
        raise ValueError(f"No JSON files found in {json_dir}")
    
    # Load each JSON file
    for file in json_files:
        try:
            with open(os.path.join(json_dir, file), 'r') as f:
                file_data = json.load(f)
            
            # Extract ticker from filename
            ticker = os.path.splitext(file)[0].upper()
            
            # Add ticker to data
            if isinstance(file_data, dict):
                file_data['ticker'] = ticker
                data.append(file_data)
            else:
                print(f"Warning: Unexpected format in {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV if output file specified
    if output_file and not df.empty:
        df.to_csv(output_file, index=False)
        print(f"JSON data saved to {output_file}")
    
    return df


def prepare_model_data(data, features, target='Status', scale=True):
    """
    Prepare data for machine learning model.
    
    Args:
        data (pandas.DataFrame): Input DataFrame.
        features (list): List of features to use.
        target (str): Target column name.
        scale (bool): Whether to scale the features.
        
    Returns:
        tuple: X (features), y (target), and feature_names.
    """
    # Select features that exist in the data
    available_features = [f for f in features if f in data.columns]
    
    # Extract features and target
    X = data[available_features].values
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # Scale features if requested
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Convert target to binary
    if target in data.columns:
        if data[target].dtype == object:
            # Assuming 'underperformed' is 0 and anything else is 1
            y = np.where(data[target] == "underperformed", 0, 1)
        else:
            y = data[target].values
    else:
        y = None
        print(f"Warning: Target column '{target}' not found in data.")
    
    return X, y, available_features
