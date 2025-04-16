import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from matplotlib import style

style.use("ggplot")

FEATURES = ['DE Ratio',
            'Trailing P/E',
            'Price/Sales',
            'Price/Book',
            'Profit Margin',
            'Operating Margin',
            'Return on Assets',
            'Return on Equity',
            'Revenue Per Share',
            'Market Cap',
            'Enterprise Value',
            'Forward P/E',
            'PEG Ratio',
            'Enterprise Value/Revenue',
            'Enterprise Value/EBITDA',
            'Revenue',
            'Gross Profit',
            'EBITDA',
            'Net Income Avl to Common ',
            'Diluted EPS',
            'Earnings Growth',
            'Revenue Growth',
            'Total Cash',
            'Total Cash Per Share',
            'Total Debt',
            'Current Ratio',
            'Book Value Per Share',
            'Cash Flow',
            'Beta',
            'Held by Insiders',
            'Held by Institutions',
            'Shares Short (as of',
            'Short Ratio',
            'Short % of Float',
            'Shares Short (prior ']

def build_data_set():
    # Load the data
    data_df = pd.read_csv("key_stats_acc_perf_WITH_NA.csv")
    
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

def analysis():
    test_size = 2900  # Number of samples to use for testing
    
    invest_amount = 10000  # Amount to invest
    total_invests = 0  # Counter for total investments
    if_market = 0  # Total return if following market
    if_strat = 0  # Total return if following strategy
    
    X, y, Z = build_data_set()
    print(f"Total samples: {len(X)}")

    # Initialize and train the SVM
    clf = svm.SVC(kernel="linear", C=1.0)
    clf.fit(X[:-test_size], y[:-test_size])
    
    correct_count = 0
    
    for x in range(1, test_size + 1):
        prediction = clf.predict(X[-x].reshape(1, -1))[0]
        
        # Check if prediction matches the actual label
        if prediction == y[-x]:
            correct_count += 1
        
        # If the prediction is 1 (outperformed), calculate returns
        if prediction == 1:
            stock_change = Z[-x][0]
            sp500_change = Z[-x][1]
            
            if not np.isnan(stock_change) and not np.isnan(sp500_change):
                invest_return = invest_amount + (invest_amount * (stock_change / 100))
                market_return = invest_amount + (invest_amount * (sp500_change / 100))
                
                total_invests += 1
                if_market += market_return
                if_strat += invest_return    
    # Calculate accuracy
    accuracy = (correct_count / test_size) * 100.00
    print(f"Accuracy: {accuracy}%")
    
    # Print investment results
    print(f"Total Trades: {total_invests}")
    print(f"Ending with Strategy: {if_strat}")
    print(f"Ending with Market: {if_market}")
    do_nothing = total_invests*invest_amount

    avg_market = ((if_market - do_nothing) / do_nothing) * 100    
    avg_strat = ((if_strat - do_nothing) / do_nothing) * 100    

    # Calculate and print the comparison with the market
    if if_market > 0:
        compared = ((if_strat - if_market) / if_market) * 100
        print(f"Compared to market, we earn: {compared}% more")
        print("Average investment return: ", str(avg_strat) + "%" )
        print("Average market return: ", str(avg_market) + "%" )

    else:
        print("Market return is zero or negative, comparison not possible")

# Run the analysis
analysis()
