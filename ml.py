"""
Legacy data processing module for stock market prediction.

This module parses HTML files containing stock financial data to extract
key statistics and performance metrics for machine learning analysis.
"""

import pandas as pd
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import re

style.use("dark_background")

# Note: This path needs to be updated to match your local setup
# TODO: Make this configurable or relative to project root
path = "data/intraQuarter"  # Updated from hardcoded Windows path

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
    """
    Extract key financial statistics from HTML files for stock analysis.
    
    Args:
        gather (list): List of financial metrics to extract from HTML files.
        
    Returns:
        None: Saves processed data to 'key_stats.csv' file.
    """
    statspath = os.path.join(path, '_KeyStats')
    stock_list = [x[0] for x in os.walk(statspath)]
    data = []
    ticker_list = []

    sp500_df = pd.read_csv("data/YAHOO-INDEX_GSPC.csv", index_col=0, parse_dates=True)

    for each_dir in stock_list[1:]:
        each_file = os.listdir(each_dir)
        ticker = os.path.basename(each_dir)
        ticker_list.append(ticker)

        starting_stock_value = None
        starting_sp500_value = None

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
                                regex = re.escape(each_data)+r'.*?(\d{1,8}\.\d{1,8}M?B?|N/A)%?</td>'
                                value = re.search(regex,source)
                                value = (value.group(1))

                                if 'B' in value:
                                    value = float(value.replace("B","")) * 1000000000
                                elif 'M' in value:
                                    value = float(value.replace("M", "")) * 1000000
                                value_list.append(value)


                            except Exception as e:
                                value = 'N/A'
                                value_list.append(value)
                        try:
                            sp500_date = datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d')
                            sp500_value = float(sp500_df.loc[sp500_date]["Adj Close"])
                        except KeyError:
                            sp500_date = datetime.fromtimestamp(unix_time - 259200).strftime('%Y-%m-%d')
                            sp500_value = float(sp500_df.loc[sp500_date]["Adj Close"])

                        try:
                            stock_price = float(source.split('</small><big><b>')[1].split('</b></big>')[0])
                        except Exception as e:
                            try:
                                stock_price_str = source.split('</small><big><b>')[1].split('</b></big>')[0]
                                stock_price = re.search(r'(\d{1,8}\.\d{1,8})', stock_price_str)
                                stock_price = float(stock_price.group(1))
                            except Exception as e:
                                stock_price_str = source.split('<span class="time_rtq_ticker">')[1].split('</span>')[0]
                                stock_price = re.search(r'(\d{1,8}\.\d{1,8})', stock_price_str)
                                stock_price = float(stock_price.group(1))
                        if starting_stock_value is None:
                            starting_stock_value = stock_price
                        if starting_sp500_value is None:
                            starting_sp500_value = sp500_value

                        stock_p_change = ((stock_price - starting_stock_value) / starting_stock_value) * 100
                        sp500_p_change = ((sp500_value - starting_sp500_value) / starting_sp500_value) * 100

                        difference = stock_p_change - sp500_p_change
                        if difference > 0:
                            status = "outperformed"
                        else:
                            status = "underperformed"
                        
                        if value_list.count("N/A") > 0:
                            pass
                        else:

                            data.append({'Date':date_stamp,
                                                'Unix':unix_time,
                                                'Ticker':ticker,
                                                
                                                'Price':stock_price,
                                                'stock_p_change':stock_p_change,
                                                'SP500':sp500_value,
                                                'sp500_p_change':sp500_p_change,
                                                'Difference':difference,
                                                'DE Ratio':value_list[0],
                                                #'Market Cap':value_list[1],
                                                'Trailing P/E':value_list[1],
                                                'Price/Sales':value_list[2],
                                                'Price/Book':value_list[3],
                                                'Profit Margin':value_list[4],
                                                'Operating Margin':value_list[5],
                                                'Return on Assets':value_list[6],
                                                'Return on Equity':value_list[7],
                                                'Revenue Per Share':value_list[8],
                                                'Market Cap':value_list[9],
                                                'Enterprise Value':value_list[10],
                                                'Forward P/E':value_list[11],
                                                'PEG Ratio':value_list[12],
                                                'Enterprise Value/Revenue':value_list[13],
                                                'Enterprise Value/EBITDA':value_list[14],
                                                'Revenue':value_list[15],
                                                'Gross Profit':value_list[16],
                                                'EBITDA':value_list[17],
                                                'Net Income Avl to Common ':value_list[18],
                                                'Diluted EPS':value_list[19],
                                                'Earnings Growth':value_list[20],
                                                'Revenue Growth':value_list[21],
                                                'Total Cash':value_list[22],
                                                'Total Cash Per Share':value_list[23],
                                                'Total Debt':value_list[24],
                                                'Current Ratio':value_list[25],
                                                'Book Value Per Share':value_list[26],
                                                'Cash Flow':value_list[27],
                                                'Beta':value_list[28],
                                                'Held by Insiders':value_list[29],
                                                'Held by Institutions':value_list[30],
                                                'Shares Short (as of':value_list[31],
                                                'Short Ratio':value_list[32],
                                                'Short % of Float':value_list[33],
                                                'Shares Short (prior ':value_list[34],
                                                'Status':status})

                    except Exception as e:
                        print(f"Error parsing value or S&P 500 data: {e}")

                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    if data:
        df = pd.DataFrame(data)
        #save = gather.replace(' ', '').replace(')', '').replace('(', '').replace('/', '') + '.csv'
        #df.to_csv(save, index=False)

    #     for each_ticker in ticker_list:
    #         try:
    #             plot_df = df[df['Ticker'] == each_ticker]
    #             plot_df = plot_df.set_index(['Date'])

    #             if plot_df['Status'][-1] == "underperformed":
    #                 color = 'r'
    #             else:
    #                 color = 'g'

    #             plot_df['Difference'].plot(label=each_ticker, color = color)

    #             plt.legend()
    #         except Exception as e:
    #             print(f"Error plotting data for ticker {each_ticker}: {e}")
    #     plt.show()
    # else:
    #     print("No data to save or plot.")

    df.to_csv("data/key_stats.csv")


if __name__ == "__main__":
    Key_Stats()
