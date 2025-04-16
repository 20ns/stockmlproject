import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from matplotlib import style

style.use("ggplot")

FEATURES=['DE Ratio',
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
    data_df = pd.read_csv("key_stats.csv")
    #data_df = data_df[:100]

    data_df = data_df.reindex(np.random.permutation(data_df.index))

    X = np.array(data_df[FEATURES].values)
    y = np.where(data_df['Status'] == "underperformed", 0, 1)  # Corrected label assignment

    X = preprocessing.scale(X)
    return X, y

def analysis():

    test_size = 2900

    X, y = build_data_set()
    print(len(X))

    clf = svm.SVC(kernel="linear", C=1.0)
    clf.fit(X[:-test_size], y[:-test_size])

    correct_count = 0

    for x in range(1,test_size+1):
        if clf.predict(X[-x].reshape(1, -1))[0] == y[-x]:
            correct_count +=1

    print("Accuracy: ", (correct_count/test_size) *100.00)


# def Randomising():
#     df = pd.DataFrame({'D1':range(5), 'D2':range(5)})
#     print(df)
#     df2 = df.reindex(np.random.permutation(df.index))
#     print(df2)




analysis()
