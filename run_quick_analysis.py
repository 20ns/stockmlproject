"""
Script to run a faster version of the stock market prediction analysis.
"""

import os
import sys
import traceback

# Make sure we can import from the models directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.stock_predictor import build_data_set, train_model, analyze_performance, tune_and_train_model, run_analysis
except Exception as e:
    print(f"Error importing from stock_predictor: {e}")
    traceback.print_exc()
    sys.exit(1)

def main():
    print("Running quick stock market prediction analysis...")
    
    # Use the correct file path
    data_file = "data/key_stats_acc_perf_WITH_NA.csv"
    print(f"Using data file: {data_file}")
    
    # Use the improved run_analysis function with fewer time-intensive enhancements
    print("Using faster model configuration...")
    accuracy, market_return, strategy_return, clf = run_analysis(
        test_size=2900, 
        tune=False,  # Skip time-consuming hyperparameter tuning
        use_ensemble=True,  # Still use ensemble model for better accuracy
        use_knn_imputer=False,  # Use faster mean imputation
        use_feature_selection=True,  # Keep feature selection for better performance
        feature_count=15,  # Use fewer features
        save=True
    )
    
    # Show previous results for comparison
    previous_accuracy = 55.24
    print(f"\n===== COMPARING RESULTS =====")
    print(f"Previous accuracy: {previous_accuracy:.2f}%")
    print(f"New accuracy: {accuracy:.2f}%")
    print(f"Improvement: {accuracy - previous_accuracy:.2f} percentage points")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
