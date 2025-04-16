import pandas as pd
import os
import quandl

quandl.ApiConfig.api_key = 'hTwzUzc2rTtHePbRbz2U'
path = "C:/Users/xboxo/Documents/MachineLearningProjectStock/intraQuarter"

def stockprice():
    df = pd.DataFrame()
    statspath = path+"//_KeyStats"
    stock_list = [x[0] for x in os.walk(statspath)]

    for each_dir in stock_list[1:]:
        try:
            ticker = each_dir.split("\\")[1]
            print(f"Processing ticker: {ticker}")
            name = "WIKI/" + ticker.upper()
            data = quandl.get(name, start_date="2000-12-12", end_date="2024-05-30")
            print(f"Data retrieved for {ticker}: {data.shape}")
            data[ticker.upper()] = data["Adj. Close"]
            df = pd.concat([df, data[ticker.upper()]], axis=1)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    print("Final dataframe shape: ", df.shape)
    df.to_csv("stock_prices.csv")

stockprice()
