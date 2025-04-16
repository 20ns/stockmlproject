import pandas as pd
import os
import json

def Forward(gather=['debtToEquity',
                    'trailingPE',
                    'priceToSalesTrailing12Months',
                    'priceToBook',
                    'profitMargins',
                    'operatingMargins',
                    'returnOnAssets',
                    'returnOnEquity',
                    'revenuePerShare',
                    'marketCap',
                    'enterpriseValue',
                    'forwardPE',
                    'pegRatio',
                    'enterpriseToRevenue',
                    'enterpriseToEbitda',
                    'totalRevenue',
                    'grossProfit',
                    'ebitda',
                    'netIncomeToCommon',
                    'trailingEps',
                    'earningsGrowth',
                    'revenueGrowth',
                    'totalCash',
                    'totalCashPerShare',
                    'totalDebt',
                    'currentRatio',
                    'bookValue',
                    'operatingCashflow',
                    'beta',
                    'heldPercentInsiders',
                    'heldPercentInstitutions',
                    'sharesShort',
                    'shortRatio',
                    'shortPercentOfFloat',
                    'sharesShortPriorMonth',
                    'currentPrice',
                    'sharesOutstanding']):

    df = pd.DataFrame(columns = ['Date',
                                'Unix',
                                'Ticker',
                                'Price',
                                'stock_p_change',
                                'SP500',
                                'sp500_p_change',
                                'Difference',
                                'DE Ratio',
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
                                'Net Income Avl to Common',
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
                                'Shares Short (prior',
                                'Current Price',
                                'Shares Outstanding',
                                'Status'])

    file_list = os.listdir("forward_json")
    rows = []

    for each_file in file_list:
        ticker = each_file.split(".json")[0]
        full_file_path = "forward_json/" + each_file

        with open(full_file_path, "r") as file:
            source = json.load(file)

        try:
            value_list = []
            statistics = source.get("statistics", {})

            for each_data in gather:
                value = statistics.get(each_data, "N/A")

                if isinstance(value, str):
                    if "B" in value:
                        value = float(value.replace("B", '')) * 1e9
                    elif "M" in value:
                        value = float(value.replace("M", '')) * 1e6
                    elif "K" in value:
                        value = float(value.replace("K", '')) * 1e3
                    elif value == "N/A":
                        pass  # Keep value as "N/A"
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            value = "N/A"
                elif isinstance(value, (int, float)):
                    value = float(value)
                else:
                    value = "N/A"

                value_list.append(value)

            if value_list.count("N/A") > 15:
                continue

            current_price = statistics.get("currentPrice", "N/A")
            shares_outstanding = statistics.get("sharesOutstanding", "N/A")

            try:
                trailing_pe = float(statistics.get("trailingEps", "N/A"))
                trailing_pe = float(current_price) / trailing_pe if trailing_pe != "N/A" else "N/A"
            except (TypeError, ValueError):
                trailing_pe = "N/A"

            try:
                market_cap = float(current_price) * float(shares_outstanding) if current_price != "N/A" and shares_outstanding != "N/A" else "N/A"
            except (TypeError, ValueError):
                market_cap = "N/A"

            row = {
                'Date': "N/A",
                'Unix': "N/A",
                'Ticker': ticker,
                'Price': "N/A",
                'stock_p_change': "N/A",
                'SP500': "N/A",
                'sp500_p_change': "N/A",
                'Difference': "N/A",
                'DE Ratio': value_list[0],
                'Trailing P/E': trailing_pe,
                'Price/Sales': value_list[2],
                'Price/Book': value_list[3],
                'Profit Margin': value_list[4],
                'Operating Margin': value_list[5],
                'Return on Assets': value_list[6],
                'Return on Equity': value_list[7],
                'Revenue Per Share': value_list[8],
                'Market Cap': market_cap,
                'Enterprise Value': value_list[10],
                'Forward P/E': value_list[11],
                'PEG Ratio': value_list[12],
                'Enterprise Value/Revenue': value_list[13],
                'Enterprise Value/EBITDA': value_list[14],
                'Revenue': value_list[15],
                'Gross Profit': value_list[16],
                'EBITDA': value_list[17],
                'Net Income Avl to Common': value_list[18],
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
                'Shares Short (prior': value_list[34],
                'Current Price': current_price,
                'Shares Outstanding': shares_outstanding,
                'Status': "N/A"
            }

            # Add only non-empty and non-NA rows
            if any(v != "N/A" for v in row.values()):
                rows.append(row)

        except Exception as e:
            print(f"Error processing file {each_file}: {e}")
            continue

    # Only concatenate non-empty dataframes
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv("forward_sample_WITH_NA.csv", index=False)

Forward()
