import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Set start and end dates for fetching stock data
start = '2010-01-01'
end = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

# Streamlit title
st.title('StockSage\n By-Vikram Madhad(E23CSEU1717) \n\nStock Trend Prediction')

# Stock symbol input
user_input = st.text_input('Enter Stock Symbol', 'RELIANCE.NS')

# Button to show data and prediction
if st.button('Show Data and Prediction'):
    
    # Fetch stock data
    df = yf.download(user_input, start=start, end=end)

    # Describing data
    st.subheader('Data from 1-Jan-2010 to Present Day')
    st.write(df.describe())

    # Visualizations
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.plot(df.Close, 'b')
    st.pyplot(fig)

    # Splitting data into training and testing sets
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Load the pre-trained model
    model = load_model('keras_model.keras')

    # Testing part - using past 100 days of training data + testing data
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

    # Transform the final dataframe (without refitting the scaler)
    input_data = scaler.fit_transform(final_df)

    # Splitting testing data into x_test and y_test
    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # Predicting the prices using the model
    y_predicted = model.predict(x_test)

    # Use the scaler.scale_ attribute from the correct scaler instance (don't overwrite scaler)
    scaler = scaler.scale_
    scale_factor = 1 / scaler[0]

    # Rescale the predicted and test data back to original scale
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Plotting predictions vs original
    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)


















import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go

# Streamlit app title
st.title('Candlestick Charts for Stocks')

# Input field for stock tickers
stock_input = st.text_input('Enter stock symbols separated by commas (e.g., RELIANCE.NS, TATAMOTORS.NS, HDFCBANK.NS )', 'RELIANCE.NS, TATAMOTORS.NS, HDFCBANK.NS')

# Split the input by commas and strip any extra spaces
stock_symbols = [symbol.strip() for symbol in stock_input.split(',')]

# Date range for the last 50 days
days_to_fetch = 50
end_date = pd.Timestamp.today().date()
start_date = end_date - pd.Timedelta(days=days_to_fetch * 1.5)  # Roughly fetching 75 days to ensure 50 trading days

# Fetch stock data from Yahoo Finance
if st.button('Show Candlestick Charts'):
    if len(stock_symbols) > 0:
        for symbol in stock_symbols:
            # Fetching stock data
            stock_df = yf.download(symbol, start=start_date, end=end_date)
            
            # If there are less than 50 trading days, adjust accordingly
            if len(stock_df) > days_to_fetch:
                stock_df = stock_df.tail(days_to_fetch)

            if not stock_df.empty:
                # Create candlestick chart
                fig = go.Figure(data=[go.Candlestick(x=stock_df.index,
                                                     open=stock_df['Open'],
                                                     high=stock_df['High'],
                                                     low=stock_df['Low'],
                                                     close=stock_df['Close'])])

                # Set title for the chart
                fig.update_layout(title=f'{symbol} - Latest 50 Daily Candlesticks',
                                  xaxis_title='Date',
                                  yaxis_title='Price',
                                  xaxis_rangeslider_visible=False)

                # Display the chart
                st.plotly_chart(fig)
            else:
                st.write(f"No data available for {symbol}.")
    else:
        st.write("Please enter valid stock symbols.")












# RSI Calculation Function
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Streamlit app title
st.title('Portfolio RSI and Stock Signal Visualization')

# Input field for stock tickers
portfolio_input = st.text_input('Enter your portfolio stocks separated by commas (e.g., RELIANCE.NS, TATAMOTORS.NS, HDFCBANK.NS )', 'RELIANCE.NS, TATAMOTORS.NS, HDFCBANK.NS')

# Split the input by commas and strip any extra spaces
portfolio = [ticker.strip() for ticker in portfolio_input.split(',')]

# Dates
start = '2020-01-01'  # Reduced date range for RSI calculation
end = '2024-10-13'

# Set the RSI threshold
rsi_threshold_overbought = 70
rsi_threshold_oversold = 30

# Fetch stock data from Yahoo Finance
if st.button('Show Portfolio Signals'):
    if len(portfolio) > 0:
        portfolio_data = {}
        signal_data = {}
        rsi_values = []

        # Fetch data for each stock
        for ticker in portfolio:
            stock_df = yf.download(ticker, start=start, end=end)
            stock_df['RSI'] = calculate_rsi(stock_df)
            portfolio_data[ticker] = stock_df

            # Get the latest RSI value
            latest_rsi = stock_df['RSI'].iloc[-1]
            rsi_values.append(latest_rsi)

            # Determine the signal (Overbought, Oversold, Hold)
            if latest_rsi > rsi_threshold_overbought:
                signal_data[ticker] = 'Overbought'
            elif latest_rsi < rsi_threshold_oversold:
                signal_data[ticker] = 'Oversold'
            else:
                signal_data[ticker] = 'Hold'

        # Data for plotting
        stocks = list(signal_data.keys())
        signals = list(signal_data.values())
        
        # Define the color scheme based on the signal
        colors = ['red' if signal == 'Overbought' else 'green' if signal == 'Oversold' else 'gray' for signal in signals]

        # Create scatter plot for the stock signals
        st.subheader('Stock Signal Chart (Based on RSI)')
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each stock as a point (using numerical indices on x-axis)
        ax.scatter(range(len(stocks)), rsi_values, color=colors, s=200)  # s is the size of the points

        # Add horizontal lines for RSI thresholds
        ax.axhline(rsi_threshold_overbought, color='red', linestyle='--', label='Overbought Threshold (70)')
        ax.axhline(rsi_threshold_oversold, color='green', linestyle='--', label='Oversold Threshold (30)')
        
        # Remove x-axis labels (since stock names are on the pointers)
        ax.set_xticks([])

        # Set plot labels and title
        ax.set_xlabel('Stock Symbols (labeled on pointers)')
        ax.set_ylabel('RSI Value')
        ax.set_title('Stock Signals: Overbought, Oversold, or Hold')
        ax.legend()

        # Annotate each stock with its ticker and RSI signal (Overbought, Oversold, Hold)
        for i, stock in enumerate(stocks):
            signal_text = f"{stock} ({signal_data[stock]})"
            ax.text(i, rsi_values[i] + 2, signal_text, ha='center', fontsize=10)

        # Show the plot
        st.pyplot(fig)

        # Display the signals below the plot
        st.subheader('RSI Signal Summary')
        for ticker, signal in signal_data.items():
            rsi_value = portfolio_data[ticker]['RSI'].iloc[-1]
            st.write(f"**{ticker}: {signal} (RSI: {rsi_value:.2f})**")

    else:
        st.write("Please enter valid stock symbols.")
