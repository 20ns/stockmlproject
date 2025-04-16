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
# Added for hyperparameter tuning, evaluation metrics, cross-validation and visualization
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

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


def tune_and_train_model(X_train, y_train):
    """Hyperparameter tuning using GridSearchCV"""
    param_grid = {
        'kernel': ['linear','rbf','poly'],
        'C': [0.1,1,10],
        'gamma': ['scale','auto']
    }
    svc = svm.SVC(probability=True)
    grid = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}, CV score: {grid.best_score_:.4f}")
    return grid.best_estimator_


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


def evaluate_model(model, X_test, y_test, Z_test, invest_amount=10000):
    """Extended evaluation: classification report, confusion matrix, ROC curve, returns."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('ROC Curve'); plt.legend(loc='lower right')
    plt.show()
    return analyze_performance(model, X_test, y_test, Z_test, invest_amount)


def run_analysis(test_size=2900, save=True, tune=False, cv=5):
    """Run full pipeline with optional tuning and cross-validation."""
    X, y, Z = build_data_set()
    print(f"Total samples: {len(X)}")
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    Z_test = Z[-test_size:]
    # Baseline cross-validation
    baseline = svm.SVC(kernel='linear')
    cv_scores = cross_val_score(baseline, X_train, y_train, cv=cv)
    print(f"Baseline {cv}-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    # Train or tune
    clf = tune_and_train_model(X_train, y_train) if tune else train_model(X_train, y_train)
    # Evaluate
    accuracy, market_return, strategy_return = evaluate_model(clf, X_test, y_test, Z_test)
    print(f"Accuracy: {accuracy:.2f}% | Market: ${market_return:.2f} | Strategy: ${strategy_return:.2f}")
    if save: save_model(clf)
    return accuracy, market_return, strategy_return, clf


# CLI support for flexibility
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stock Predictor CLI")
    parser.add_argument("--test-size", type=int, default=2900)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()
    run_analysis(test_size=args.test_size, save=not args.no_save, tune=args.tune, cv=args.cv)
