import yfinance as yf
import os
import json

path = "C:/Users/xboxo/Documents/MachineLearningProjectStock/intraQuarter"

def Check_Yahoo():
    statspath = os.path.join(path, '_KeyStats')
    stock_list = [x[0] for x in os.walk(statspath)]

    counter = 0
    for e in stock_list[1:]:
        try:
            ticker = e.split(os.sep)[-1]
            stock = yf.Ticker(ticker.upper())

            # Fetch the required data
            financial_data = {
                "statistics": stock.info
            }

            # Create the save directory if it doesn't exist
            save_dir = "forward_json"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{ticker}.json")

            # Save the JSON data
            with open(save_path, "w", encoding='utf-8') as store:
                json.dump(financial_data, store, indent=4)

            # Print status messages
            counter += 1
            print(f"Stored {ticker}.json")
            print(f"We now have {counter} JSON files in the directory.")
        except Exception as ex:
            print(f"Error fetching data for {ticker}: {ex}")

Check_Yahoo()
