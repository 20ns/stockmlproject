import pandas as pd
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import re

style.use("dark_background")

path = "C:/Users/xboxo/Documents/MachineLearningProjectStock/intraQuarter"

def Key_Stats(gather=["Total Debt/Equity",
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
                      'Shares Short (prior ']):
    
    statspath = os.path.join(path, '_KeyStats')
    stock_list = [x[0] for x in os.walk(statspath)]
    data = []
    ticker_list = []

    sp500_df = pd.read_csv("YAHOO-INDEX_GSPC.csv", index_col=0, parse_dates=True)
    stock_df = pd.read_csv("stock_prices.csv", index_col=0, parse_dates=True)

    for each_dir in stock_list[1:]:
        each_file = os.listdir(each_dir)
        ticker = os.path.basename(each_dir)
        ticker_list.append(ticker)

        if len(each_file) > 0:
            for file in each_file:
                try:
                    date_stamp = datetime.strptime(file, '%Y%m%d%H%M%S.html')
                    unix_time = time.mktime(date_stamp.timetuple())
                    full_file_path = os.path.join(each_dir, file)
                    with open(full_file_path, 'r') as source_file:
                        source = source_file.read()

                    try:
                        value_list = []

                        for each_data in gather:
                            try:
                                regex = re.escape(each_data) + r'.*?(\d{1,8}\.\d{1,8}M?B?|N/A)%?</td>'
                                value = re.search(regex, source)
                                if value:
                                    value = value.group(1)
                                    if 'B' in value:
                                        value = float(value.replace("B", "")) * 1000000000
                                    elif 'M' in value:
                                        value = float(value.replace("M", "")) * 1000000
                                else:
                                    value = 'N/A'
                                value_list.append(value)
                            except Exception as e:
                                value_list.append('N/A')

                        try:
                            sp500_date = datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d')
                            sp500_value = float(sp500_df.loc[sp500_date]["Adj Close"])
                        except KeyError:
                            sp500_date = datetime.fromtimestamp(unix_time - 259200).strftime('%Y-%m-%d')
                            sp500_value = float(sp500_df.loc[sp500_date]["Adj Close"])

                        one_year_later = int(unix_time + 31536000)

                        try:
                            sp500_1y = datetime.fromtimestamp(one_year_later).strftime('%Y-%m-%d')
                            sp500_1y_value = float(sp500_df.loc[sp500_1y]["Adj Close"])
                        except KeyError:
                            sp500_1y = datetime.fromtimestamp(one_year_later - 259200).strftime('%Y-%m-%d')
                            sp500_1y_value = float(sp500_df.loc[sp500_1y]["Adj Close"])

                        try:
                            stock_1y_price = datetime.fromtimestamp(one_year_later).strftime('%Y-%m-%d')
                            stock_1y_value = float(stock_df.loc[stock_1y_price][ticker.upper()])
                        except KeyError:
                            stock_1y_price = datetime.fromtimestamp(one_year_later - 259200).strftime('%Y-%m-%d')
                            stock_1y_value = float(stock_df.loc[stock_1y_price][ticker.upper()])

                        try:
                            stock_price = datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d')
                            stock_price_value = float(stock_df.loc[stock_price][ticker.upper()])
                        except KeyError:
                            stock_price = datetime.fromtimestamp(unix_time - 259200).strftime('%Y-%m-%d')
                            stock_price_value = float(stock_df.loc[stock_price][ticker.upper()])

                        stock_p_change = round((((stock_1y_value - stock_price_value) / stock_price_value) * 100), 2)
                        sp500_p_change = round((((sp500_1y_value - sp500_value) / sp500_value) * 100), 2)

                        difference = stock_p_change - sp500_p_change
                        if difference > 0:
                            status = "outperformed"
                        else:
                            status = "underperformed"

                        if value_list.count("N/A") > 15:
                            continue

                        data.append({'Date': date_stamp,
                                     'Unix': unix_time,
                                     'Ticker': ticker,
                                     'Price': stock_price_value,
                                     'stock_p_change': stock_p_change,
                                     'SP500': sp500_value,
                                     'sp500_p_change': sp500_p_change,
                                     'Difference': difference,
                                     'DE Ratio': value_list[0],
                                     'Trailing P/E': value_list[1],
                                     'Price/Sales': value_list[2],
                                     'Price/Book': value_list[3],
                                     'Profit Margin': value_list[4],
                                     'Operating Margin': value_list[5],
                                     'Return on Assets': value_list[6],
                                     'Return on Equity': value_list[7],
                                     'Revenue Per Share': value_list[8],
                                     'Market Cap': value_list[9],
                                     'Enterprise Value': value_list[10],
                                     'Forward P/E': value_list[11],
                                     'PEG Ratio': value_list[12],
                                     'Enterprise Value/Revenue': value_list[13],
                                     'Enterprise Value/EBITDA': value_list[14],
                                     'Revenue': value_list[15],
                                     'Gross Profit': value_list[16],
                                     'EBITDA': value_list[17],
                                     'Net Income Avl to Common ': value_list[18],
                                     'Diluted EPS': value_list[19],
                                     'Earnings Growth': value_list[20],
                                     'Revenue Growth': value_list[21],
                                     'Total Cash': value_list[22],
                                     'Total Cash Per Share': value_list[23],
                                     'Total Debt': value_list[24],
                                     'Current Ratio': value_list[25],
                                     'Book Value Per Share': value_list[26],
                                     'Cash Flow': value_list[27],
                                     'Beta': value_list[28],
                                     'Held by Insiders': value_list[29],
                                     'Held by Institutions': value_list[30],
                                     'Shares Short (as of': value_list[31],
                                     'Short Ratio': value_list[32],
                                     'Short % of Float': value_list[33],
                                     'Shares Short (prior ': value_list[34],
                                     'Status': status})

                    except Exception as e:
                        print(f"Error parsing value or S&P 500 data: {e}")

                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    if data:
        df = pd.DataFrame(data)
        df.to_csv("key_stats_acc_perf_WITH_NA.csv", index=False)
    else:
        print("No data to save.")

Key_Stats()
