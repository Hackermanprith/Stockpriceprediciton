# backtesting_platform.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
from binance.client import Client

from alpha import AlphaEngine
from fragility_module import calculate_fragility_score
from advanced_chart import render_all_charts  # Import charts module

# --- Binance Setup ---
api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"
client = Client(api_key, api_secret)

# --- Data Loader ---
def load_binance_data(symbol: str, interval: str, days: int = 365*3):
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=days)
    klines = client.get_historical_klines(symbol, interval, start_time.strftime("%d %b %Y %H:%M:%S"))
    df = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                       'Close Time', 'Quote Asset Volume', 'Number of Trades',
                                       'Taker buy base volume', 'Taker buy quote volume', 'Ignore'])
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df = df[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df.set_index('Open Time', inplace=True)
    return df

# --- Optional File Loader ---
def load_uploaded_file(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df['Open Time'] = pd.to_datetime(df['Open Time'])
    df.set_index('Open Time', inplace=True)
    return df

# --- Metrics Calculator ---
def compute_metrics(balance_history):
    returns = np.diff(balance_history) / balance_history[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
    return {
        "Final Balance": balance_history[-1],
        "Total Return %": (balance_history[-1] - balance_history[0]) / balance_history[0] * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown %": (1 - min(balance_history / np.maximum.accumulate(balance_history))) * 100
    }

# --- Backtesting Engine ---
def run_backtest(df, initial_balance=1000):
    df['alpha_signal'] = alpha_signal(df)
    df['hedge_signal'] = hedge_signal(df)
    df['fragility'] = calculate_fragility_score(df)

    position = 0
    entry_price = 0.0
    balance = initial_balance
    balances = [balance]
    trades = []

    for i in range(1, len(df)):
        price = df.iloc[i]['Close']
        signal = df.iloc[i]['alpha_signal']
        hedge = df.iloc[i]['hedge_signal']

        if position == 0:
            if signal == 'buy' and hedge != 'hedge':
                position = 1
                entry_price = price
                trades.append((df.index[i], 'BUY', price))
            elif signal == 'sell' and hedge != 'hedge':
                position = -1
                entry_price = price
                trades.append((df.index[i], 'SELL', price))

        elif position == 1:
            if signal == 'sell' or hedge == 'hedge':
                ret = (price - entry_price) / entry_price
                balance *= (1 + ret)
                balances.append(balance)
                trades.append((df.index[i], 'EXIT LONG', price))
                position = 0

        elif position == -1:
            if signal == 'buy' or hedge == 'hedge':
                ret = (entry_price - price) / entry_price
                balance *= (1 + ret)
                balances.append(balance)
                trades.append((df.index[i], 'EXIT SHORT', price))
                position = 0

        balances.append(balance)

    return np.array(balances), trades, df

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📊 Crypto Strategy Backtesting Platform")

with st.sidebar:
    symbol = st.text_input("Symbol", value="BTCUSDT")
    interval = st.selectbox("Interval", ["1d", "1h", "15m"])
    days = st.slider("Days of Data", 30, 1095, 365)
    initial_balance = st.number_input("Initial Capital (USD)", value=1000.0)
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx'])
    run_infinitely = st.checkbox("Run Backtest Infinitely Until Stopped")
    run_backtest_button = st.button("🚀 Run Backtest")
    stop_signal = st.empty()

if run_backtest_button:
    stop_button = stop_signal.button("🛑 Stop")
    with st.spinner("Fetching data and running strategy..."):
        if uploaded_file:
            df = load_uploaded_file(uploaded_file)
        else:
            df = load_binance_data(symbol, interval, days)

        placeholder = st.empty()

        while True:
            balances, trade_log, df_with_signals = run_backtest(df.copy(), initial_balance)
            metrics = compute_metrics(balances)

            with placeholder.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Final Balance", f"${metrics['Final Balance']:.2f}")
                    st.metric("Total Return", f"{metrics['Total Return %']:.2f}%")
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                    st.metric("Max Drawdown", f"{metrics['Max Drawdown %']:.2f}%")

                st.subheader("📈 Balance Over Time")
                st.line_chart(balances)

                st.subheader("🧾 Trade Log")
                trade_df = pd.DataFrame(trade_log, columns=["Date", "Action", "Price"])
                st.dataframe(trade_df)

                st.subheader("📊 Advanced Analytics")
                render_all_charts(df_with_signals)

            time.sleep(10)  # wait between iterations
            if stop_button:
                st.success("Backtest stopped.")
                break
