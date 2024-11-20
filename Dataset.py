import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define the stock symbols you want to fetch data for
stock_symbols = ['RELIANCE.NS', 'TATAMOTORS.NS', 'HDFCBANK.NS']  # Add more symbols as needed

# Define the date range for the stock data
start_date = '2010-01-01'
end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

# Folder to save CSV files
output_folder = 'smlProject'

# Fetch data for each stock symbol and save it to a CSV file
for symbol in stock_symbols:
    # Fetch the data from Yahoo Finance
    stock_df = yf.download(symbol, start=start_date, end=end_date)

    # Check if data was successfully fetched
    if not stock_df.empty:
        # Define the filename based on the stock symbol
        csv_filename = f"{output_folder}{symbol}_data.csv"

        # Save the DataFrame to a CSV file
        stock_df.to_csv(csv_filename)
        print(f"Data for {symbol} saved to {csv_filename}")
    else:
        print(f"No data found for {symbol}")

