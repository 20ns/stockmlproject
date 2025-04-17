"""
Fast version of stock prediction analysis optimized for machines without GPU.
Should complete in under 5 minutes.
"""

import os
import sys
import traceback
import time

# Make sure we can import from the models directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.stock_predictor import build_data_set, train_model, analyze_performance
except Exception as e:
    print(f"Error importing from stock_predictor: {e}")
    traceback.print_exc()
    sys.exit(1)

def main():
    print("Running QUICK stock market prediction analysis...")
    start_time = time.time()
    
    # Use the correct file path
    data_file = "data/key_stats_acc_perf_WITH_NA.csv"
    print(f"Using data file: {data_file}")
    
    # Build dataset with basic preprocessing only
    print("Building dataset with basic preprocessing...")
    X, y, Z = build_data_set(csv_file=data_file, use_knn_imputer=False)
    print(f"Total samples: {len(X)}")
    
    # Split data - use smaller test sample for faster results
    test_size = 1500  # Using smaller test size for speed
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    Z_test = Z[-test_size:]
    
    # Train a Random Forest model (faster than SVM and often better)
    from sklearn.ensemble import RandomForestClassifier
    print("Training Random Forest model (faster than SVM)...")
    clf = RandomForestClassifier(
        n_estimators=50,  # Fewer trees for speed
        max_depth=10,     # Limit depth for speed
        min_samples_split=10,
        class_weight='balanced',
        n_jobs=-1,        # Use all CPU cores
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Analyze performance
    print("Analyzing performance...")
    accuracy, market_return, strategy_return = analyze_performance(clf, X_test, y_test, Z_test)
    
    # Show results
    previous_accuracy = 55.24
    print(f"\n===== COMPARING RESULTS =====")
    print(f"Previous accuracy: {previous_accuracy:.2f}%")
    print(f"New accuracy: {accuracy:.2f}%")
    print(f"Improvement: {accuracy - previous_accuracy:.2f} percentage points")
    
    # Print run time
    run_time = time.time() - start_time
    print(f"\nTotal runtime: {run_time:.2f} seconds")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
