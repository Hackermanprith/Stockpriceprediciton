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
    raw_df = None
    try:
        raw_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None, [], {} # df, raw_columns, auto_detected_mapping

    raw_columns = list(raw_df.columns)
    df = raw_df.copy() # Work on a copy for processing

    auto_detected_mapping = {
        'Timestamp': None, 'Open': None, 'High': None, 'Low': None, 'Close': None, 'Volume': None
    }
    processed_df = None
    timestamp_col_used = None

    # 1. Detect Timestamp column
    if 'timestamp' in df.columns:
        auto_detected_mapping['Timestamp'] = 'timestamp'
    elif 'Timestamp' in df.columns: # Exact match
        auto_detected_mapping['Timestamp'] = 'Timestamp'
    elif 'Open Time' in df.columns:
        auto_detected_mapping['Timestamp'] = 'Open Time'

    if auto_detected_mapping['Timestamp']:
        timestamp_col_used = auto_detected_mapping['Timestamp']
        try:
            df[timestamp_col_used] = pd.to_datetime(df[timestamp_col_used])
            df.set_index(timestamp_col_used, inplace=True)
        except Exception as e:
            st.warning(f"Could not automatically process timestamp column '{timestamp_col_used}': {e}. Please map manually.")
            # Don't return None yet, allow manual mapping
            timestamp_col_used = None # Reset as it failed
    else:
        st.warning("No obvious timestamp column ('timestamp' or 'Open Time') found. Please map manually.")

    # 2. Detect OHLCV columns (case-insensitive) and prepare for renaming
    essential_fields_pascal = ['Open', 'High', 'Low', 'Close', 'Volume']
    rename_map = {}
    found_all_essentials_auto = True

    for field in essential_fields_pascal: # e.g. field = 'Open'
        found_for_field = None
        for col in df.columns: # Iterate actual columns from file
            if col.lower() == field.lower():
                found_for_field = col
                auto_detected_mapping[field] = col
                if col != field: # if 'open' was found for 'Open'
                    rename_map[col] = field
                break
        if not found_for_field:
            found_all_essentials_auto = False
            st.warning(f"Could not automatically find a column for '{field}'. Please map manually.")

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # 3. Attempt to build processed_df if timestamp and all essentials were auto-detected
    if timestamp_col_used and found_all_essentials_auto:
        try:
            # Ensure all PascalCase columns are present after renaming
            final_cols_to_check = essential_fields_pascal[:] # Copy
            missing_renamed = [col for col in final_cols_to_check if col not in df.columns]
            if missing_renamed:
                 st.warning(f"Some columns seemed to be found but are missing after rename: {missing_renamed}. This is unexpected.")
                 processed_df = None # Mark as failed
            else:
                df[essential_fields_pascal] = df[essential_fields_pascal].astype(float)
                processed_df = df[essential_fields_pascal] # Select only these columns
                st.success("Successfully auto-processed uploaded file.")
        except ValueError as e:
            st.warning(f"Error converting auto-detected columns to numeric types: {e}. Please map manually or check data.")
            processed_df = None # Mark as failed
        except Exception as e:
            st.warning(f"An unexpected error occurred during auto-processing: {e}")
            processed_df = None
    else:
        st.info("Initial auto-detection incomplete. Please review and map columns manually below if needed.")
        processed_df = None # Indicate auto-processing was not fully successful

    return processed_df, raw_columns, auto_detected_mapping


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
def run_backtest(df, initial_balance=1000, symbol: str = "BTCUSDT"): # Added symbol parameter
    engine = AlphaEngine() # Initialize AlphaEngine
    df['fragility'] = calculate_fragility_score(df) # Kept fragility for now

    signals = [] # New list to store signals
    forecast_details_log = [] # Initialize log for forecast details

    # Loop to generate signals using AlphaEngine
    for i in range(len(df)):
        if i == 0: # Cannot generate signal for the very first row with no prior data
            signals.append(None)
            # Add a placeholder for forecast_details_log for alignment, or skip
            forecast_details_log.append({
                'Timestamp': df.index[i],
                'Current Price': df.iloc[i]['Close'],
                'Signal': 'N/A (Initial Row)',
            })
            continue

        current_df_slice = df.iloc[:i+1]
        forecast_obj = None # Initialize forecast_obj to ensure it's defined
        current_signal_val = None # Initialize current_signal_val

        try:
            forecast_obj = engine.run_forecast(symbol=symbol, df_override=current_df_slice)
            current_signal_val = forecast_obj.get('signal')
            signals.append(current_signal_val)

            # Log forecast details
            log_entry = {
                'Timestamp': df.index[i],
                'Current Price': df.iloc[i]['Close'],
                'Forecast Price': forecast_obj.get('forecast_price'),
                'Signal': current_signal_val,
                'Confidence': forecast_obj.get('confidence'),
                'Predicted Price Change %': forecast_obj.get('price_change_pct'),
                'Volatility': forecast_obj.get('volatility'),
                'Uncertainty': forecast_obj.get('uncertainty'),
                'Fragility Score': forecast_obj.get('fragility_score'),
                'Fragility Interpretation': forecast_obj.get('fragility_interpretation'),
                'Breakout Direction': forecast_obj.get('breakout_direction'),
                'Breakout Confidence': forecast_obj.get('breakout_confidence'),
                'Lower Bound': forecast_obj.get('lower_bound'),
                'Upper Bound': forecast_obj.get('upper_bound'),
            }
            forecast_details_log.append(log_entry)

        except Exception as e:
            print(f"Error generating signal at step {i} for {symbol}: {e}") # Print to console
            signals.append(None)
            forecast_details_log.append({ # Log error or minimal info
                'Timestamp': df.index[i],
                'Current Price': df.iloc[i]['Close'],
                'Signal': 'Error',
                'Error Details': str(e)
            })

    df['signal'] = signals # Add signals to DataFrame

    position = 0
    entry_price = 0.0
    balance = initial_balance
    balances = [balance]
    trades = []

    for i in range(1, len(df)): # Start from 1 because signal for row 0 might be None
        price = df.iloc[i]['Close']
        # Use the new 'signal' column. Hedge is removed.
        current_signal = df.iloc[i]['signal']

        if pd.isna(current_signal): # Skip if signal is None/NaN
            balances.append(balance)
            continue

        if position == 0:
            if current_signal == 'buy': # Removed hedge condition
                position = 1
                entry_price = price
                trades.append((df.index[i], 'BUY', price))
            elif current_signal == 'sell': # Removed hedge condition
                position = -1
                entry_price = price
                trades.append((df.index[i], 'SELL', price))

        elif position == 1:
            if current_signal == 'sell': # Removed hedge condition
                ret = (price - entry_price) / entry_price
                balance *= (1 + ret)
                # balances.append(balance) # Balance is appended at the end of the loop
                trades.append((df.index[i], 'EXIT LONG', price))
                position = 0

        elif position == -1:
            if current_signal == 'buy': # Removed hedge condition
                ret = (entry_price - price) / entry_price
                balance *= (1 + ret)
                # balances.append(balance) # Balance is appended at the end of the loop
                trades.append((df.index[i], 'EXIT SHORT', price))
                position = 0

        balances.append(balance) # Append balance at each step regardless of trade

    return np.array(balances), trades, df, forecast_details_log

# --- Paper Trading Core Logic ---
def execute_paper_trade_iteration(active_symbol, asset_type):
    st.write(f"Executing paper trade iteration for {active_symbol} ({asset_type})...")

    user_api_key = st.session_state.get('binance_api_key_input')
    user_api_secret = st.session_state.get('binance_api_secret_input')
    engine_api_key = user_api_key if user_api_key else None
    engine_api_secret = user_api_secret if user_api_secret else None

    engine = AlphaEngine(api_key=engine_api_key, api_secret=engine_api_secret)

    forecast_obj = None
    current_price_for_trade = None
    log_entry_forecast = {'Timestamp': datetime.datetime.now(), 'Symbol': active_symbol}

    try:
        # For paper trading, AlphaEngine fetches its own live data.
        forecast_obj = engine.run_forecast(symbol=active_symbol)
        log_entry_forecast.update(forecast_obj if forecast_obj else {})
        current_price_for_trade = forecast_obj.get('current_price') # Price used for forecast
        if current_price_for_trade is None: # Fallback if not in forecast_obj (should be)
            st.warning("Could not get current price from forecast object. Attempting direct fetch for trade execution price.")
            # This is a simplified fallback; ideally, AlphaEngine always returns current_price
            # For crypto:
            if asset_type == "Cryptocurrency":
                kline = client.get_klines(symbol=active_symbol, interval="1m", limit=1)
                if kline: current_price_for_trade = float(kline[0][4])
            # For stocks, yfinance would be needed here if not using AlphaEngine's price
            # else: current_price_for_trade = ... (e.g. yf.Ticker(active_symbol).history(period="1d")['Close'].iloc[-1])

        if current_price_for_trade is None:
            st.error(f"Paper Trading: Could not determine current price for {active_symbol} to execute trade.")
            log_entry_forecast['Error'] = "Failed to get current price for trade."
            st.session_state.paper_forecast_log.append(log_entry_forecast)
            return

    except Exception as e:
        st.error(f"Error getting forecast for {active_symbol}: {e}")
        log_entry_forecast['Error'] = str(e)
        st.session_state.paper_forecast_log.append(log_entry_forecast)
        return # Do not proceed with trade logic if forecast fails

    st.session_state.paper_forecast_log.append(log_entry_forecast)

    if forecast_obj and 'signal' in forecast_obj and current_price_for_trade is not None:
        signal = forecast_obj['signal']

        cash = st.session_state.paper_portfolio['cash']
        held_asset_symbol = st.session_state.current_asset_held
        quantity = st.session_state.current_asset_quantity

        # Trade Size Logic
        trade_size = 0
        if asset_type == "Stock":
            trade_size = 10  # Example: 10 shares
        elif asset_type == "Cryptocurrency":
            if current_price_for_trade > 1000:  # e.g., BTC
                trade_size = 0.01
            elif current_price_for_trade > 0: # Avoid division by zero for very low priced assets
                trade_size = 100 / current_price_for_trade # Spend $100 USD
            else:
                trade_size = 0 # Cannot determine trade size

        if trade_size == 0 and signal != 'HOLD': # Do not proceed if trade size is zero unless HOLD
            st.warning(f"Paper Trading: Trade size is zero for {active_symbol} at price {current_price_for_trade}. Signal ignored.")
            return

        trade_action = None
        trade_details = {}

        if signal == 'BUY':
            if held_asset_symbol is not None and held_asset_symbol != active_symbol:
                # Simulate selling the old asset (using its last known forecast price or refetch)
                # This part is simplified; a real system might need to fetch current price of old asset
                st.warning(f"Paper Trading: Portfolio already holds {held_asset_symbol}. Simplified: Assuming it's sold before buying {active_symbol}.")
                # For simplicity, let's assume it was sold at the price it was acquired or last known.
                # A more complex version would fetch current price of held_asset_symbol.
                # Here, we'll just clear it to allow buying the new one.
                st.session_state.paper_portfolio['cash'] += 0 # Placeholder for proceeds of hypothetical old asset sale
                st.session_state.paper_trade_log.append({
                    'Timestamp': datetime.datetime.now(),
                    'Symbol': held_asset_symbol,
                    'Action': 'SELL (Pre-Switch)',
                    'Price': 'N/A (Simplified)',
                    'Quantity': quantity,
                    'Reason': f'Switching to {active_symbol}'
                })
                st.session_state.current_asset_held = None
                st.session_state.current_asset_quantity = 0.0
                held_asset_symbol = None # Update for current logic

            if held_asset_symbol is None: # Ready to buy new asset
                cost = current_price_for_trade * trade_size
                if cash >= cost:
                    st.session_state.paper_portfolio['cash'] -= cost
                    st.session_state.current_asset_held = active_symbol
                    st.session_state.current_asset_quantity = trade_size
                    trade_action = 'BUY'
                    trade_details = {'Price': current_price_for_trade, 'Quantity': trade_size, 'Cost': cost}
                    st.success(f"Paper TRADED: BOUGHT {trade_size:.4f} {active_symbol} @ ${current_price_for_trade:.2f}")
                else:
                    st.warning(f"Paper Trading: Insufficient cash to buy {active_symbol}. Need ${cost:.2f}, have ${cash:.2f}")
            elif held_asset_symbol == active_symbol:
                 st.info(f"Paper Trading: Already holding {active_symbol}. No BUY action taken.")


        elif signal == 'SELL':
            if held_asset_symbol == active_symbol and quantity > 0:
                proceeds = current_price_for_trade * quantity
                st.session_state.paper_portfolio['cash'] += proceeds
                trade_action = 'SELL'
                trade_details = {'Price': current_price_for_trade, 'Quantity': quantity, 'Proceeds': proceeds}
                st.session_state.current_asset_held = None
                st.session_state.current_asset_quantity = 0.0
                st.success(f"Paper TRADED: SOLD {quantity:.4f} {active_symbol} @ ${current_price_for_trade:.2f}")
            elif held_asset_symbol != active_symbol and held_asset_symbol is not None:
                 st.info(f"Paper Trading: SELL signal for {active_symbol}, but holding {held_asset_symbol}. No action taken on {active_symbol}.")
            else:
                 st.info(f"Paper Trading: SELL signal for {active_symbol}, but no holdings. No action taken.")

        if trade_action:
            log_trade = {
                'Timestamp': datetime.datetime.now(),
                'Symbol': active_symbol,
                'Action': trade_action,
                **trade_details, # Add price, quantity, cost/proceeds
                'Signal Source': forecast_obj.get('signal'),
                'Signal Confidence': forecast_obj.get('confidence')
            }
            st.session_state.paper_trade_log.append(log_trade)

    # Update total portfolio value
    asset_value = 0
    if st.session_state.current_asset_held and current_price_for_trade is not None: # Use current_price_for_trade for consistency
        asset_value = st.session_state.current_asset_quantity * current_price_for_trade

    st.session_state.paper_portfolio['total_value'] = st.session_state.paper_portfolio['cash'] + asset_value
    st.info(f"Paper portfolio updated. Cash: ${st.session_state.paper_portfolio['cash']:.2f}, Asset Value: ${asset_value:.2f}, Total: ${st.session_state.paper_portfolio['total_value']:.2f}")


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📊 Crypto Strategy Backtesting Platform")

with st.sidebar:
    st.header("⚙️ Configuration")

    # Asset Type Selector
    asset_type = st.selectbox("Select Asset Type", ["Cryptocurrency", "Stock"], key="asset_type_selector", index=0)

    # Initialize symbol variable
    symbol = None

    if asset_type == "Cryptocurrency":
        crypto_symbol = st.text_input("Crypto Symbol (e.g., BTCUSDT, ETHUSDT)", value="BTCUSDT", key="crypto_symbol_input")
        symbol = crypto_symbol
    elif asset_type == "Stock":
        stock_symbol = st.text_input("Stock Ticker (e.g., AAPL, MSFT)", value="AAPL", key="stock_symbol_input")
        st.caption("Note: Stock data fetched from Yahoo Finance.")
        symbol = stock_symbol

    # Ensure symbol is set for other parts of the app if needed immediately,
    # though it's primarily used when "Run Backtest" is clicked.
    # If using session state extensively, values are directly accessible via st.session_state.key_name

    interval = st.selectbox("Interval", ["1d", "1h", "15m"], help="Data interval. Note: Stock data typically uses '1d' for longer histories from Yahoo Finance.")
    days = st.slider("Days of Data", 30, 1095*2, 365, help="Number of days of historical data to fetch.") # Increased max days

    st.header("🔑 API Configuration (Optional)")
    st.text_input("Binance API Key", type="password", key="binance_api_key_input", help="Only required for live paper trading with Binance or direct data fetching via your account.")
    st.text_input("Binance API Secret", type="password", key="binance_api_secret_input", help="Only required for live paper trading with Binance or direct data fetching via your account.")
    st.caption("API keys are stored in session state and used for Binance data/paper trading.")

    st.header("💰 Backtest Settings")
    initial_balance = st.number_input("Initial Capital (USD)", value=1000.0)
    uploaded_file = st.file_uploader("Upload Custom Data (Excel)", type=['xlsx'], help="Upload your own OHLCV data. Ensure columns are clearly named (e.g., Timestamp, Open, High, Low, Close, Volume).")

    st.header("▶️ Execution Controls")
    run_infinitely_disabled = uploaded_file is not None
    run_infinitely = st.checkbox("Run Backtest Infinitely Until Stopped (Legacy for non-uploaded data backtest)",
                                 value=False if run_infinitely_disabled else True, # Default to True if not disabled
                                 disabled=run_infinitely_disabled,
                                 help="For continuous backtesting simulation on fetched data. Disabled if a file is uploaded. Paper trading provides live simulation.")
    if run_infinitely_disabled and uploaded_file:
        st.caption("Infinite run is disabled for uploaded files. Use Paper Trading for continuous simulation.")

    run_backtest_button = st.button("🚀 Run Backtest")
    stop_signal = st.empty() # Used by the infinite backtest loop if enabled

# Initialize session state variables
# Asset type and symbol inputs are already managed by Streamlit's widget state using their keys.
# We can access them directly using st.session_state.asset_type_selector,
# st.session_state.crypto_symbol_input, or st.session_state.stock_symbol_input when needed.

if 'raw_columns' not in st.session_state:
    st.session_state.raw_columns = []
if 'auto_detected_mapping' not in st.session_state:
    st.session_state.auto_detected_mapping = {}
if 'user_column_mappings' not in st.session_state:
    st.session_state.user_column_mappings = {}
if 'processed_df_from_upload' not in st.session_state:
    st.session_state.processed_df_from_upload = None
if 'uploaded_file_name' not in st.session_state: # To detect new file uploads
    st.session_state.uploaded_file_name = None

# Initialize session state for paper trading
if 'paper_trading_active' not in st.session_state:
    st.session_state.paper_trading_active = False
if 'paper_portfolio' not in st.session_state:
    st.session_state.paper_portfolio = {'cash': 0.0, 'total_value': 0.0} # Added total_value
if 'paper_trade_log' not in st.session_state:
    st.session_state.paper_trade_log = []
if 'paper_forecast_log' not in st.session_state:
    st.session_state.paper_forecast_log = []
if 'current_asset_held' not in st.session_state: # Stores symbol of asset held
    st.session_state.current_asset_held = None
if 'current_asset_quantity' not in st.session_state:
    st.session_state.current_asset_quantity = 0.0
if 'initial_paper_balance_set' not in st.session_state: # To ensure balance is set only once per "Start"
    st.session_state.initial_paper_balance_set = False
if 'last_paper_trade_time' not in st.session_state: # For timed execution
    st.session_state.last_paper_trade_time = None


# Main UI logic for data source selection and processing (for Backtesting)
data_source_df = None # This df is for the backtesting section

if uploaded_file is not None:
    # Check if it's a new file or the same one
    if st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.processed_df_from_upload, st.session_state.raw_columns, st.session_state.auto_detected_mapping = load_uploaded_file(uploaded_file)
        # Initialize user_column_mappings based on auto_detected_mapping for the new file
        st.session_state.user_column_mappings = st.session_state.auto_detected_mapping.copy()
        # If auto-processing was successful, this df can be used directly
        if st.session_state.processed_df_from_upload is not None:
             data_source_df = st.session_state.processed_df_from_upload
        else: # Auto-processing failed or incomplete, force manual mapping by clearing data_source_df
             data_source_df = None


    with st.expander("Map Excel Columns", expanded=data_source_df is None): # Expand if auto-processing failed
        required_mapping_fields = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']

        # Use a form for the mapping section
        with st.form(key='column_mapping_form'):
            for field in required_mapping_fields:
                # Try to get the auto-detected column for default, otherwise None
                default_selection = st.session_state.auto_detected_mapping.get(field, None)
                # Ensure default_selection is in raw_columns, else set index to 0 or a placeholder
                try:
                    default_index = st.session_state.raw_columns.index(default_selection) if default_selection in st.session_state.raw_columns else 0
                except ValueError: # Should not happen if default_selection is from raw_columns
                    default_index = 0

                # If raw_columns is empty, selectbox will error, add a placeholder if needed
                options_list = st.session_state.raw_columns if st.session_state.raw_columns else ["<No columns found>"]
                if not st.session_state.raw_columns and default_selection is None: # Handle case of no columns and no default
                     default_index = 0 # selectbox needs a valid index

                selected_col = st.selectbox(
                    f"Select column for '{field}':",
                    options=options_list,
                    index=default_index,
                    key=f"map_{field.lower()}" # Unique key for each selectbox
                )
                st.session_state.user_column_mappings[field] = selected_col if selected_col != "<No columns found>" else None

            submit_mapping_button = st.form_submit_button("Process Mapped Excel File")

        if submit_mapping_button:
            try:
                # Re-read the original uploaded file
                raw_df_on_submit = pd.read_excel(uploaded_file)

                # Validate all fields are mapped
                all_fields_mapped = True
                for field in required_mapping_fields:
                    if not st.session_state.user_column_mappings.get(field):
                        st.error(f"Please map all required fields. '{field}' is missing.")
                        all_fields_mapped = False
                        break
                if not all_fields_mapped:
                    st.session_state.processed_df_from_upload = None # Signal failure
                    data_source_df = None
                else:
                    # Construct the DataFrame based on user mappings
                    mapped_data = {}
                    # Timestamp processing
                    ts_col_name = st.session_state.user_column_mappings['Timestamp']
                    mapped_data[ts_col_name] = pd.to_datetime(raw_df_on_submit[ts_col_name])

                    current_df = pd.DataFrame({ts_col_name: mapped_data[ts_col_name]})
                    current_df.set_index(ts_col_name, inplace=True)

                    # OHLCV processing
                    temp_rename_map = {}
                    for field in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        user_selected_col = st.session_state.user_column_mappings[field]
                        # Check for duplicate selections for different fields
                        # (e.g., user maps 'excel_col_A' to 'Open' and also to 'Close')
                        # This check is tricky here, better to check if all selected user_selected_col for OHLCV are unique.
                        current_df[field] = raw_df_on_submit[user_selected_col].astype(float)
                        # No rename needed here as we are assigning directly to 'Open', 'High' etc.

                    # Final check for uniqueness of mapped source columns (excluding timestamp)
                    ohlcv_source_cols = [st.session_state.user_column_mappings[f] for f in ['Open', 'High', 'Low', 'Close', 'Volume']]
                    if len(ohlcv_source_cols) != len(set(ohlcv_source_cols)):
                        st.error("Duplicate columns selected for Open, High, Low, Close, or Volume fields. Please use unique columns.")
                        st.session_state.processed_df_from_upload = None
                        data_source_df = None
                    else:
                        st.session_state.processed_df_from_upload = current_df[['Open', 'High', 'Low', 'Close', 'Volume']]
                        data_source_df = st.session_state.processed_df_from_upload
                        st.success("Successfully processed mapped Excel file.")

            except Exception as e:
                st.error(f"Error processing mapped Excel data: {e}")
                st.session_state.processed_df_from_upload = None
                data_source_df = None

    # If after all upload logic, data_source_df is still None (e.g. new file, auto-fail, no manual map yet)
    # but we have a previously processed df in session state, use that.
    # This handles re-runs where the file is already processed.
    if data_source_df is None and st.session_state.processed_df_from_upload is not None:
        data_source_df = st.session_state.processed_df_from_upload

elif not uploaded_file: # No file uploaded, clear session state related to uploads
    st.session_state.raw_columns = []
    st.session_state.auto_detected_mapping = {}
    st.session_state.user_column_mappings = {}
    st.session_state.processed_df_from_upload = None
    st.session_state.uploaded_file_name = None
    # Fallback to Binance data if no file uploaded
    # data_source_df = load_binance_data(symbol, interval, days) # This will be decided later

if run_backtest_button:
    # Determine final df for backtest
    final_df_for_backtest = None
    if data_source_df is not None: # This implies an uploaded file was processed successfully
        final_df_for_backtest = data_source_df
        st.info("Using data from uploaded Excel file for backtest.")
    elif not uploaded_file: # No upload attempted, use chosen data source
        # Determine the correct symbol to use based on asset_type
        # This logic should ideally be at the top where `symbol` is derived,
        # but for the backtest button, we re-evaluate here to ensure it's current.
        current_asset_type = st.session_state.asset_type_selector
        if current_asset_type == "Cryptocurrency":
            active_symbol = st.session_state.crypto_symbol_input
            data_source_name = "Binance"
        else: # Stock
            active_symbol = st.session_state.stock_symbol_input
            data_source_name = "Yahoo Finance" # AlphaEngine's fetch_data handles Yahoo

        with st.spinner(f"Fetching {data_source_name} data for {active_symbol}..."):
            # The load_binance_data function is specific to Binance.
            # AlphaEngine's fetch_data method is more general and handles Yahoo Finance.
            # For simplicity here, we'll assume load_binance_data is used for crypto
            # and if we were to fully integrate stock backtesting here (without paper trading setup yet),
            # we'd need a more general data loader or use AlphaEngine's fetch directly.
            # For now, this part of backtesting will primarily work for Crypto via load_binance_data.
            # If asset_type is Stock, load_binance_data will likely fail or fetch nothing if symbol is like 'AAPL'.
            # This will be more relevant when AlphaEngine is used for data fetching in Option 2.
            if current_asset_type == "Cryptocurrency":
                 final_df_for_backtest = load_binance_data(active_symbol, interval, days)
                 st.info(f"Using data from {data_source_name} for {active_symbol} for backtest.")
            else:
                 # Placeholder: Actual stock data fetching for backtesting via this path would need
                 # to use yfinance directly here or adapt AlphaEngine's fetch_data for direct use.
                 # For now, we'll simulate an error or empty df for stock backtesting via this old path.
                 st.warning(f"Direct backtesting for Stocks via this UI path is currently using a placeholder. "
                            f"AlphaEngine will fetch stock data if used directly. For now, an empty DataFrame will be used for {active_symbol}.")
                 final_df_for_backtest = pd.DataFrame() # Empty DataFrame for stock type in this specific path

    else: # Uploaded file exists, but data_source_df is None (error or not processed)
        st.error("Uploaded file is present but not yet processed or failed processing. Please map columns and click 'Process Mapped Excel File' or check errors.")
        final_df_for_backtest = None


    if final_df_for_backtest is not None:
        stop_button = stop_signal.button("🛑 Stop")
        with st.spinner("Running strategy..."):
            placeholder = st.empty()
            # Ensure df is not empty
            if final_df_for_backtest.empty:
                st.error("The DataFrame is empty. Cannot run backtest.")
            else:
                while True: # This loop is for the "run infinitely" feature
                    # Determine the active symbol again for the run_backtest call
                    current_asset_type_runtime = st.session_state.asset_type_selector
                    if current_asset_type_runtime == "Cryptocurrency":
                        active_symbol_runtime = st.session_state.crypto_symbol_input
                    else: # Stock
                        active_symbol_runtime = st.session_state.stock_symbol_input

                    balances, trade_log, df_with_signals, forecast_log = run_backtest(final_df_for_backtest.copy(), initial_balance, symbol=active_symbol_runtime) # Use the determined df and active_symbol
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

                # Display AlphaModel Decision Insights
                if forecast_log: # Check if the log is not empty
                    st.subheader("🔍 AlphaModel Decision Insights")
                    st.caption("This table shows the detailed outputs from the AlphaModel at each step of the backtest.")
                    insights_df = pd.DataFrame(forecast_log)
                    # Format timestamp and floats for insights_df
                    if 'Timestamp' in insights_df.columns:
                        try:
                            insights_df['Timestamp'] = pd.to_datetime(insights_df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                        except: pass # Keep original if formatting fails

                    float_cols_insights = insights_df.select_dtypes(include='float').columns
                    format_dict_insights = {col: '{:.2f}' for col in float_cols_insights if 'Price' in col or 'Bound' in col or 'Score' in col or 'Cash' in col or '%' in col}
                    format_dict_insights.update({col: '{:.4f}' for col in float_cols_insights if 'Volatility' in col or 'Uncertainty' in col or 'Confidence' in col})
                    format_dict_insights.update({col: '{:.6f}' for col in float_cols_insights if 'Quantity' in col}) # For crypto quantities

                    st.dataframe(insights_df.style.format(format_dict_insights, na_rep="-"))


            time.sleep(10)  # wait between iterations (only if run_infinitely is true and not disabled)
            if not (run_infinitely and not run_infinitely_disabled): # If not running infinitely, or if it's disabled
                if stop_button: st.success("Backtest stopped.")
                else: st.success("Backtest run complete.")
                break
            if stop_button: # This implies run_infinitely was true and enabled
                st.success("Infinite backtest stopped.")
                break

# --- Paper Trading Zone ---
st.divider() # Visual separator
st.header("📜 Paper Trading Zone")
st.caption("Simulate live trading based on AlphaModel forecasts. Ensure API keys are set in the sidebar if using non-public data sources within AlphaEngine, though paper trading itself does not execute real orders via Binance API with these keys.")

col_paper_1, col_paper_2 = st.columns(2)
with col_paper_1:
    st.number_input("Initial Paper Trading Balance (USD)", min_value=100.0, value=10000.0, step=100.0, key="initial_paper_balance", help="Set your starting cash balance for paper trading.")

with col_paper_2:
    st.selectbox("Paper Trading Interval",
                 options=["Manual Trigger", "1 minute", "5 minutes", "15 minutes", "1 hour"],
                 key="paper_trading_interval",
                 help="Select how often paper trades should be attempted. 'Manual Trigger' requires clicking 'Execute Manual Paper Trade'. Timed intervals require the app to remain open in the browser.")

    if st.session_state.get("paper_trading_interval") == "Manual Trigger" and st.session_state.paper_trading_active:
        st.button("📈 Execute Manual Paper Trade", key="execute_manual_paper_trade_button", help="Click to attempt a paper trade based on the latest forecast.")


start_paper_button = st.button("🚀 Start Paper Trading", key="start_paper_trading_button", use_container_width=True, help="Begin the paper trading simulation.")
stop_paper_button = st.button("🛑 Stop Paper Trading", key="stop_paper_trading_button", use_container_width=True, help="Halt the paper trading simulation.")

if start_paper_button:
    active_paper_symbol = None
    selected_asset_type = st.session_state.get('asset_type_selector', 'Cryptocurrency')
    if selected_asset_type == "Cryptocurrency":
        active_paper_symbol = st.session_state.get('crypto_symbol_input')
    else: # Stock
        active_paper_symbol = st.session_state.get('stock_symbol_input')

    if not active_paper_symbol:
        st.error("⚠️ Please select an asset type and enter a symbol in the sidebar configuration before starting paper trading.")
    else:
        st.session_state.paper_trading_active = True
        if not st.session_state.initial_paper_balance_set:
            st.session_state.paper_portfolio['cash'] = st.session_state.initial_paper_balance
            st.session_state.paper_portfolio['total_value'] = st.session_state.initial_paper_balance
            st.session_state.current_asset_held = None
            st.session_state.current_asset_quantity = 0.0
            st.session_state.paper_trade_log = []
            st.session_state.paper_forecast_log = []
            st.session_state.initial_paper_balance_set = True
        st.session_state.last_paper_trade_time = datetime.datetime.now()
        st.success(f"Paper trading started for {active_paper_symbol} with ${st.session_state.initial_paper_balance:.2f}. Interval: {st.session_state.paper_trading_interval}")

if stop_paper_button:
    st.session_state.paper_trading_active = False
    st.session_state.initial_paper_balance_set = False
    st.warning("Paper trading stopped by user.")

st.subheader("📈 Portfolio Status")
portfolio_display = st.empty()
st.subheader("📋 Paper Trade Log")
paper_trade_log_display = st.empty()
st.subheader("🔮 AlphaModel Forecasts (Paper Trading)")
paper_forecast_log_display = st.empty()

# Update display areas
# Improved Portfolio Display
if st.session_state.paper_trading_active or st.session_state.initial_paper_balance_set : # Show if active or was active
    cash_display = f"**Cash:** ${st.session_state.paper_portfolio.get('cash', 0.0):.2f}"
    asset_display_list = [cash_display]
    asset_value_num = 0.0 # For total calculation

    if st.session_state.current_asset_held:
        current_price_info = "N/A (Waiting for next forecast)"
        # Try to get current price from the last forecast for display
        if st.session_state.paper_forecast_log and isinstance(st.session_state.paper_forecast_log[-1], dict):
            last_forecast = st.session_state.paper_forecast_log[-1]
            if last_forecast.get('Symbol') == st.session_state.current_asset_held and last_forecast.get('current_price') is not None:
                current_price_info = f"${last_forecast['current_price']:.2f}"
                asset_value_num = st.session_state.current_asset_quantity * last_forecast['current_price']
                asset_display_list.append(f"**{st.session_state.current_asset_held}:** {st.session_state.current_asset_quantity:.6f} units (Est. Value: ${asset_value_num:.2f} at {current_price_info} last seen)")
            else: # Asset held, but price not in last forecast for this asset
                 asset_display_list.append(f"**{st.session_state.current_asset_held}:** {st.session_state.current_asset_quantity:.6f} units (Current value pending next forecast)")
        else: # Asset held, but no forecast log yet or not a dict
            asset_display_list.append(f"**{st.session_state.current_asset_held}:** {st.session_state.current_asset_quantity:.6f} units (Current value pending first forecast)")

    # Update total portfolio value in session state before displaying
    st.session_state.paper_portfolio['total_value'] = st.session_state.paper_portfolio.get('cash', 0.0) + asset_value_num
    total_value_display = f"**Total Portfolio Value:** ${st.session_state.paper_portfolio.get('total_value', 0.0):.2f}"
    asset_display_list.append(f"\n{total_value_display}") # Add some space
    portfolio_display.markdown("\n\n".join(asset_display_list))
else:
    portfolio_display.info("Paper trading not started. Click 'Start Paper Trading' to begin.")


# Helper for DataFrame styling
def style_dataframe(df):
    if df.empty:
        return df

    # Timestamp formatting
    if 'Timestamp' in df.columns:
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception: pass # Keep original if formatting fails

    # Float formatting
    float_cols = df.select_dtypes(include='float').columns
    format_dict = {}
    for col in float_cols:
        if any(kw in col for kw in ['Price', 'Cost', 'Proceeds', 'Bound', 'Balance', 'Value', 'Cash']):
            format_dict[col] = '${:,.2f}'
        elif any(kw in col for kw in ['%', 'Ratio', 'Change']):
            format_dict[col] = '{:,.2f}%'
        elif 'Quantity' in col:
             # Check if it's crypto (many decimals) or stock (few decimals)
            is_crypto_trade = asset_type == "Cryptocurrency" if 'asset_type' in locals() else False # Assuming asset_type is available
            if is_crypto_trade and df[col].abs().max() < 100: # Heuristic for crypto quantities
                 format_dict[col] = '{:,.6f}'
            else: # Stocks or large crypto amounts
                 format_dict[col] = '{:,.2f}'
        elif any(kw in col for kw in ['Confidence', 'Score', 'Uncertainty', 'Volatility']):
            format_dict[col] = '{:,.4f}'
        else:
            format_dict[col] = '{:,.2f}' # Default for other floats

    return df.style.format(format_dict, na_rep="-")


if st.session_state.paper_trade_log:
    paper_trade_log_df = pd.DataFrame(st.session_state.paper_trade_log).sort_values(by="Timestamp", ascending=False)
    paper_trade_log_display.dataframe(style_dataframe(paper_trade_log_df))
else:
    paper_trade_log_display.info("No paper trades made yet.")

if st.session_state.paper_forecast_log:
    paper_forecast_log_df = pd.DataFrame(st.session_state.paper_forecast_log).sort_values(by="Timestamp", ascending=False)
    paper_forecast_log_display.dataframe(style_dataframe(paper_forecast_log_df))
else:
    paper_forecast_log_display.info("No AlphaModel forecasts logged for paper trading yet.")


# --- Timed / Manual Paper Trading Execution Logic ---
current_asset_type_for_paper = st.session_state.get('asset_type_selector', "Cryptocurrency")
active_symbol_for_paper = None
if current_asset_type_for_paper == "Cryptocurrency":
    active_symbol_for_paper = st.session_state.get('crypto_symbol_input')
else: # Stock
    active_symbol_for_paper = st.session_state.get('stock_symbol_input')

if st.session_state.get('execute_manual_paper_trade_button'): # Check if button was pressed
    if st.session_state.paper_trading_active:
        if active_symbol_for_paper:
            execute_paper_trade_iteration(active_symbol_for_paper, current_asset_type_for_paper)
            st.session_state.last_paper_trade_time = datetime.datetime.now() # Reset timer on manual execution
        else:
            st.warning("No symbol selected for paper trading. Please select one in the sidebar.")
    else:
        st.warning("Paper trading is not active. Click 'Start Paper Trading'.")

if st.session_state.paper_trading_active and st.session_state.paper_trading_interval != "Manual Trigger":
    interval_str = st.session_state.paper_trading_interval
    interval_seconds = 0
    if interval_str == "1 minute": interval_seconds = 60
    elif interval_str == "5 minutes": interval_seconds = 5 * 60
    elif interval_str == "15 minutes": interval_seconds = 15 * 60
    elif interval_str == "1 hour": interval_seconds = 60 * 60

    if interval_seconds > 0:
        last_trade_time = st.session_state.get('last_paper_trade_time', datetime.datetime.min)
        if datetime.datetime.now() - last_trade_time >= datetime.timedelta(seconds=interval_seconds):
            if active_symbol_for_paper:
                st.info(f"Auto-executing paper trade for {active_symbol_for_paper} based on {interval_str} interval.")
                execute_paper_trade_iteration(active_symbol_for_paper, current_asset_type_for_paper)
                st.session_state.last_paper_trade_time = datetime.datetime.now()
                st.rerun() # Rerun to update UI immediately after timed execution
            else:
                st.warning("No symbol selected for paper trading (timed execution). Please select one in the sidebar.")
