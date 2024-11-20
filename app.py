import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Enhanced CSS for Times New Roman font and improved visibility
# Enhanced CSS for Times New Roman font and improved visibility
# Enhanced CSS for Times New Roman font and improved visibility
# Enhanced CSS for Times New Roman font and improved visibility
# Enhanced CSS for Times New Roman font and improved visibility
# Enhanced CSS for Times New Roman font and improved visibility
st.markdown("""
    <style>
    /* Apply Times New Roman font throughout */
    .main * { font-family: 'Times New Roman', Times, serif; }

    /* Title styling with increased padding */
    .title-section {
        color: #FFD700; /* Bright gold for "StockFlux" */
        font-weight: 700;
        font-size: 2.5em;
        text-align: center;
        padding-top: 1em;
        padding-bottom: 1em;
    }

    /* Author names styled with a balanced red color and additional spacing */
    .authors {
        color: #C41E3A; /* Slightly muted bright red */
        font-weight: 700;
        font-size: 1.25em; /* Smaller than title */
        text-align: center;
        margin-bottom: 1.5em; /* Increased margin for more free space */
    }

    /* Section header styling for Stock Trend Prediction and other headers */
    .section-header {
        color: #1E90FF; /* Bright blue for section headers */
        font-weight: 600;
        font-size: 1.8em; /* Matches the size of Stock Trend Prediction title */
        display: flex;
        align-items: center;
        gap: 0.5em; /* Space between icon and text */
    }

    /* Subheader styling */
    .subheader {
        color: #FFD700;
        font-size: 1.2em;
    }

    /* Input Boxes */
    .stTextInput, .stDateInput {
        background-color: #2b2b2b;
        border: 1px solid #5C5C5C;
        color: #FFFFFF;
        font-size: 1em;
        padding: 0.5em;
        border-radius: 8px;
    }

    /* Buttons */
    .stButton button {
        background-color: #FF4500; /* Bright orange-red */
        color: #FFFFFF;
        border: none;
        font-size: 1em;
        font-weight: bold;
        padding: 0.5em 1em;
        border-radius: 5px;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #1E90FF; /* Bright blue on hover */
    }

    /* Divider styling */
    .divider {
        border-top: 2px solid #FFD700;
        margin: 20px 0;
    }

    /* Plot background styling */
    .stPlotlyChart, .stPyplot {
        border-radius: 8px;
        background-color: #1E1E1E;
        padding: 1em;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
    }

    /* Instruction box */
    .instruction-box {
        background-color: #333333;
        padding: 1em;
        border-left: 4px solid #1E90FF; /* Bright blue */
        border-radius: 6px;
        color: #FFD700;
    }
    </style>
""", unsafe_allow_html=True)

# App Header and Authors with spacing
st.markdown('<div class="title-section">📈 StockFlux - Stock Trend Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="authors">By Riddhi Chaplot (E23CSEU0425) & Akinchan Jain (E23CSEU0423)</div>', unsafe_allow_html=True)

# Spacer for free space before the next section
st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# Main Section - "Stock Trend Prediction" header with icon
st.markdown('<div class="section-header">📊 Stock Trend Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Analyze stock trends with a data-driven approach</div>', unsafe_allow_html=True)

# Stock symbol input field
user_input = st.text_input('Enter Stock Symbol (e.g., RELIANCE.NS)', 'RELIANCE.NS')

# Divider
# st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# # Candlestick Chart Section with matching header size and icon
# st.markdown('<div class="section-header">📉 Candlestick Charts for Stocks</div>', unsafe_allow_html=True)
# st.write("Visualize stock price patterns over time with interactive candlestick charts")

# # Portfolio RSI and Signal Visualization Section with matching header size and icon
# st.markdown('<div class="section-header">📊 Portfolio RSI and Stock Signal Visualization</div>', unsafe_allow_html=True)
# portfolio_input = st.text_input(
#     'Enter your portfolio stocks separated by commas',
#     'RELIANCE.NS, TATAMOTORS.NS, HDFCBANK.NS'
# )




# Divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# # Candlestick Chart Section
# st.header("📊 Candlestick Charts for Stocks")
# st.write("Visualize stock price patterns over time with interactive candlestick charts")

# # Candlestick chart stock input field
# stock_input = st.text_input('Enter stock symbols separated by commas', 'RELIANCE.NS, TATAMOTORS.NS, HDFCBANK.NS')

# # Divider
# st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Portfolio RSI and Signal Visualization Section
# st.header("📉 Portfolio RSI and Stock Signal Visualization")
# portfolio_input = st.text_input(
#     'Enter your portfolio stocks separated by commas',
#     'RELIANCE.NS, TATAMOTORS.NS, HDFCBANK.NS'
# )

# Instruction box for RSI guidance
# st.markdown("""
#     <div class="instruction-box">
#     <p>**RSI Guidance**: A value **above 70** indicates an <span style="color:#FF4500;">Overbought</span> condition, while **below 30** indicates an <span style="color:#FFD700;">Oversold</span> condition.</p>
#     </div>
# """, unsafe_allow_html=True)

# Dates for fetching stock data
start = '2010-01-01'
end = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

# Button for showing data and prediction
# Button for showing data and prediction
# Button for showing data and prediction
if st.button('Show Data and Prediction'):
    
    # Fetch stock data
    df = yf.download(user_input, start=start, end=end)

    # Data description and summary
    st.subheader('Data Summary')
    st.write(df.describe())

    # Visualization: Closing Price vs Time chart
    st.subheader('Closing Price vs Time')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, color='darkblue', label="Closing Price")  # Dark blue line for closing price
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig)

    # Visualization: Closing Price vs Time with 100MA
    st.subheader('Closing Price vs Time with 100-Day Moving Average')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, label="100-Day MA", color='black')  # Black for 100-Day MA
    plt.plot(df.Close, label="Closing Price", color='darkblue')  # Dark blue for closing price
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig)

    # Visualization: Closing Price vs Time with 100MA & 200MA
    st.subheader('Closing Price vs Time with 100-Day & 200-Day Moving Averages')
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, label="100-Day MA", color='black')  # Black for 100-Day MA
    plt.plot(ma200, label="200-Day MA", color='red')    # Red for 200-Day MA
    plt.plot(df.Close, label="Closing Price", color='darkblue')  # Dark blue for closing price
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig)

    # Data split and model loading for prediction
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Load the pre-trained model
    model = load_model('keras_model.keras')

    # Prepare input data for testing and scaling
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    # Create x_test and y_test arrays for prediction
    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # Perform prediction
    y_predicted = model.predict(x_test)

    # Rescale predictions to original scale
    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Plot predictions vs original data
    st.subheader('Predicted vs Actual Prices')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Price', color='darkblue')  # Dark blue for actual prices
    plt.plot(y_predicted, label='Predicted Price', color='red')  # Red for predicted prices
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
st.markdown("""
    <div class="instruction-box">
    <p>**RSI Guidance**: A value **above 70** indicates an <span style="color:#FF4500;">Overbought</span> condition, while **below 30** indicates an <span style="color:#FF4500;">Oversold</span> condition.</p>
    </div>
""", unsafe_allow_html=True)

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
