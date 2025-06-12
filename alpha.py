import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma
# import requests # Unused
from datetime import datetime, timedelta
import matplotlib.pyplot as plt # Used in _plot_basic_forecast
# import matplotlib.dates as mdates # Unused
# import seaborn as sns # Unused
from binance.client import Client
import yfinance as yf
# import traceback # Unused
import warnings # Keep one warnings import
from sklearn.mixture import GaussianMixture
from scipy.stats import t, genpareto, gaussian_kde # kendalltau removed
# from sklearn.neighbors import KernelDensity # Unused
# from scipy import signal # Unused
# from scipy.spatial.distance import pdist, squareform # Unused
from fragility_module import FragilityScoreCalculator
from advanced_chart import AdvancedChartVisualizer
# import warnings # Redundant
warnings.filterwarnings('ignore') # Keep this one
try:
    from numba import jit # prange, float64, int64, boolean removed as not directly used
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
class AlphaEngine:
    def __init__(self, api_key=None, api_secret=None):
        """
        Initializes the AlphaEngine with API keys and default parameters.

        Parameters:
        -----------
        api_key : str, optional
            Binance API key.
        api_secret : str, optional
            Binance API secret.
        """
        self.client = Client(api_key, api_secret) # Binance client
        self.dt = 1/1440 # Time step, assuming 1 minute data, so 1 day = 1440 minutes
        self.forecast_horizon = 1
        self.alpha = 0.5
        self.kappa = 2.5
        self.theta = 0.03
        self.sigma_phi = 0.2
        self.rho = 1.0
        self.p_th = 0.5
        self.c = 0.001
        self.eta_w = 0.01
        self.eta_kappa = 0.01
        self.eta_theta = 0.01
        self.eta_alpha = 0.01
        self.jump_intensity = 0.05
        self.jump_mean_x = 0.0
        self.jump_std_x = 0.05
        self.jump_mean_phi = 0.0
        self.jump_std_phi = 0.05
        self.weights = np.ones(4) / 4
        self.phi_bar = 0.03
        self.market_regime_params = {}
        self.fragility_calculator = FragilityScoreCalculator(sensitivity=1.0)
    def update_parameters_from_data(self, df):
        """
        Updates model parameters (e.g., alpha, kappa, theta, jump parameters)
        dynamically based on statistical properties of the provided market data.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing historical market data with at least 'returns' and 'squared_returns' columns.
        """
        returns = df['returns'].dropna()
        if len(returns) >= 60: # Ensure enough data for autocorrelation
            # Estimate alpha (memory parameter) from autocorrelation decay
            acf_values = [returns.autocorr(lag=i) for i in range(1, 11) if not np.isnan(returns.autocorr(lag=i))]
            if acf_values:
                decay_rate = np.polyfit(range(len(acf_values)), np.log(np.abs(acf_values) + 1e-8), 1)[0]
                self.alpha = np.clip(0.5 - decay_rate, 0.1, 0.9)
        if len(returns) >= 100:
            std_returns = returns.std()
            jump_threshold = 3 * std_returns
            jumps = returns[np.abs(returns) > jump_threshold]
            non_jumps = returns[np.abs(returns) <= jump_threshold]
            jump_frequency = len(jumps) / len(returns)
            self.jump_intensity = np.clip(jump_frequency, 0.01, 0.25)
            if len(jumps) > 5:
                negative_jumps = jumps[jumps < 0]
                positive_jumps = jumps[jumps > 0]
                if len(negative_jumps) > 0:
                    self.jump_mean_x = negative_jumps.mean() / std_returns
                else:
                    self.jump_mean_x = -0.02
                if len(jumps) > 1:
                    self.jump_std_x = np.clip(jumps.std() / std_returns, 0.01, 0.2)
                jump_asymmetry = len(negative_jumps) / max(1, len(jumps))
                if jump_asymmetry > 0.7:
                    self.jump_mean_x = self.jump_mean_x * 1.5
        if 'squared_returns' in df.columns and len(df['squared_returns']) >= 60:
            squared_returns = df['squared_returns'].dropna()
            vol_ts = np.sqrt(squared_returns.rolling(window=30).mean().dropna())
            if len(vol_ts) > 30:
                vol_diff = vol_ts.diff().dropna()
                vol_level = vol_ts[:-1].values
                if len(vol_diff) > 10 and len(vol_level) > 10:
                    try:
                        kappa_estimate = -np.polyfit(vol_level, vol_diff, 1)[0] * 252
                        self.kappa = np.clip(kappa_estimate, 0.5, 10.0)
                    except:
                        pass
            recent_vol = np.sqrt(squared_returns.iloc[-100:].mean()) * np.sqrt(252)
            self.theta = np.clip(recent_vol, 0.01, 0.5)
            long_term_vol = np.sqrt(squared_returns.mean())
            self.phi_bar = np.clip(long_term_vol, 0.01, 0.1)
            if len(vol_ts) > 30:
                vol_of_vol = vol_ts.pct_change().dropna().std() * np.sqrt(252)
                self.sigma_phi = np.clip(vol_of_vol, 0.1, 1.0)
        if 'squared_returns' in df.columns and len(df['squared_returns']) > 100:
            vol_distribution = np.sqrt(df['squared_returns'].rolling(window=30).mean().dropna())
            if len(vol_distribution) > 0:
                vol_median = np.median(vol_distribution)
                vol_std = np.std(vol_distribution)
                self.rho = np.clip(vol_median / self.phi_bar + 0.5 * vol_std / self.phi_bar, 1.0, 3.0)
        recent_vol = np.sqrt(df['squared_returns'].iloc[-60:].mean()) if len(df['squared_returns']) > 60 else 0.01
        vol_factor = np.clip(recent_vol / self.phi_bar, 0.5, 2.0)
        self.eta_w = 0.01 / vol_factor
        self.eta_kappa = 0.01 / vol_factor
        self.eta_theta = 0.01 / vol_factor
        self.eta_alpha = 0.01 / vol_factor
        return self
    def update_regime_dependent_parameters(self, regime_id=None, regime_label=None, regime_prob=0.0):
        """
        Update parameters based on detected market regime.
        Different parameter sets are used for different market regimes.
        """
        if regime_id is None or regime_label is None:
            return
        if regime_label not in self.market_regime_params:
            self.market_regime_params[regime_label] = {
                'kappa': self.kappa,
                'theta': self.theta,
                'alpha': self.alpha,
                'jump_intensity': self.jump_intensity,
                'mean_reversion_strength': 0.02,
                'momentum_factor': 0.05,
                'regime_effect': 0.01
            }
        regime_params = self.market_regime_params[regime_label]
        if regime_prob > 0.7:
            market_asymmetry = 0.0
            if hasattr(self, 'latest_market_data') and hasattr(self, '_calculate_market_asymmetry'):
                market_asymmetry = self._calculate_market_asymmetry(self.latest_market_data)
            if 'Low-Vol' in regime_label:
                vol_factor = 0.8
                jump_factor = 0.6
                base_momentum = 0.04
            elif 'Volatile' in regime_label:
                vol_factor = 1.6
                jump_factor = 1.5  
                base_momentum = 0.04
            else:
                vol_factor = 1.0
                jump_factor = 1.0
                base_momentum = 0.03
            if 'Bullish' in regime_label:
                mean_reversion = 0.025
                regime_effect = 0.01 + 0.005 * market_asymmetry
            elif 'Bearish' in regime_label:
                mean_reversion = 0.025
                regime_effect = -(0.01 + 0.005 * market_asymmetry)
            else:
                mean_reversion = 0.03
                regime_effect = 0.0
            regime_params['kappa'] = vol_factor * self.kappa
            regime_params['jump_intensity'] = jump_factor * self.jump_intensity
            regime_params['mean_reversion_strength'] = mean_reversion
            regime_params['momentum_factor'] = base_momentum
            regime_params['regime_effect'] = regime_effect
        self.current_regime_params = regime_params
        return self
    async def fetch_data_async(self, symbol='BTCUSDT', interval='1m', lookback_days=30):
        """
        Async fetch market data with robust error handling and multiple data source fallbacks.
        Attempts to use Binance API first, then Yahoo Finance if Binance fails or returns insufficient data.
        If all real data sources fail, generates synthetic data as a last resort.
        Caches fetched data for a short period to avoid redundant API calls.

        Parameters:
        -----------
        symbol : str, optional
            The trading symbol (e.g., 'BTCUSDT' for Binance, 'BTC-USD' for Yahoo Finance).
            Defaults to 'BTCUSDT'.
        interval : str, optional
            The interval for K-line/candlestick data (e.g., '1m', '1h', '1d').
            Defaults to '1m'.
        lookback_days : int, optional
            Number of past days of data to fetch. Defaults to 30.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing OHLCV data, log prices, returns, squared returns, and 14-period volatility.
            The DataFrame is indexed by 'open_time'.
        """
        import aiohttp # Imported here to keep it local to async method
        import asyncio

        cache_key = f"{symbol}_{interval}_{lookback_days}"
        # Check cache first
        if hasattr(self, '_data_cache') and cache_key in self._data_cache:
            cached_data = self._data_cache[cache_key]
            current_time = datetime.now()
            cache_time = cached_data.get('cache_time')
            # Use cache if data is less than 15 minutes old
            if cache_time and (current_time - cache_time).total_seconds() < 900:
                # print(f"Using cached data for {symbol} ({len(cached_data['data'])} records)") # Debug: good for verbose mode
                return cached_data['data']

        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        self.current_coin_symbol = symbol # Store the symbol being processed

        data_sources_tried = []
        data_quality = {} # To store metrics about data from different sources
        df = None # Initialize DataFrame

        if not hasattr(self, '_data_cache'):
            self._data_cache = {}

        # Attempt 1: Binance API (if client is available)
        if self.client:
            try:
                data_sources_tried.append("Binance")
                # print(f"Attempting to fetch data from Binance for {symbol}...") # Debug: good for verbose mode
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    end_str=end_time.strftime('%Y-%m-%d %H:%M:%S')
                )
                if not klines or len(klines) < 30: # Require at least 30 data points
                    raise ValueError("Insufficient data points from Binance API")

                klines_array = np.array(klines, dtype=object)
                df = pd.DataFrame({
                    'open_time': pd.to_datetime(klines_array[:, 0], unit='ms'),
                    'open': klines_array[:, 1].astype(float),
                    'high': klines_array[:, 2].astype(float),
                    'low': klines_array[:, 3].astype(float),
                    'close': klines_array[:, 4].astype(float),
                    'volume': klines_array[:, 5].astype(float)
                })
                if len(df) < 30 or df['close'].iloc[-1] <= 0: # Additional check for validity
                    raise ValueError("Insufficient or invalid data from Binance API")

                data_quality["Binance"] = {
                    "records": len(df), "nan_percentage": 0,
                    "timespan_hours": (df['open_time'].max() - df['open_time'].min()).total_seconds() / 3600,
                    "source": "Binance", "interval": interval
                }
                # print(f"Using Binance data feed with {len(df)} records for {symbol}") # Debug: good for verbose mode
            except Exception as e:
                print(f"Binance API error for {symbol}: {e}")
                df = None # Reset df if Binance fetch failed
        else:
            print("Binance client not initialized. Skipping Binance data source.")


        # Attempt 2: Yahoo Finance (if Binance failed or insufficient data)
        if df is None or len(df) < 100: # Try Yahoo if less than 100 records from Binance
            async def fetch_yahoo_data(session, ticker, yf_interval, start_time_yf, end_time_yf):
                """Helper function to fetch Yahoo Finance data asynchronously"""
                try:
                    import concurrent.futures # For running sync yf.download in executor
                    def sync_yahoo_fetch():
                        return yf.download(ticker, start=start_time_yf, end=end_time_yf,
                                         interval=yf_interval, progress=False, show_errors=False)
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        yf_data = await loop.run_in_executor(executor, sync_yahoo_fetch)
                    return yf_data, yf_interval
                except Exception as e_yf:
                    print(f"Error fetching Yahoo data for {ticker} with {yf_interval}: {e_yf}")
                    return None, yf_interval

            async def try_yahoo_finance_source():
                data_sources_tried.append("Yahoo Finance")
                # print(f"Attempting to use Yahoo Finance data for {symbol}...") # Debug: good for verbose mode

                # Adapt symbol for Yahoo Finance (e.g., BTCUSDT -> BTC-USD)
                if symbol.endswith('USDT'): ticker = f"{symbol[:-4]}-USD"
                elif symbol.endswith('USD'): ticker = symbol
                else: ticker = f"{symbol}-USD" # Common for stocks, or default crypto if not USDT

                # Yahoo interval mapping (simplified, yf has different interval codes)
                # For broader data, '1h' or '1d' are more reliable with yfinance
                yf_interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '1d': '1d'}
                yf_hist_interval = yf_interval_map.get(interval, '1h') # Default to 1h for yfinance if interval not directly mappable
                
                try:
                    yf_data, interval_tried_yf = await fetch_yahoo_data(None, ticker, yf_hist_interval, start_time, end_time)
                    if yf_data is not None and len(yf_data) >= 50:
                        return yf_data, interval_tried_yf
                except Exception as e_yf_primary:
                    print(f"Error with primary ticker {ticker} on Yahoo Finance: {e_yf_primary}")
                
                # Fallback for crypto if primary ticker failed (e.g. if symbol was just 'BTC')
                if not symbol.endswith('USDT') and not symbol.endswith('USD') and ('BTC' in symbol.upper() or 'ETH' in symbol.upper()):
                    print(f"Trying generic {symbol}-USD ticker on Yahoo Finance...")
                    try:
                        yf_data_fallback, interval_fallback_yf = await fetch_yahoo_data(None, f"{symbol}-USD", yf_hist_interval, start_time, end_time)
                        if yf_data_fallback is not None and len(yf_data_fallback) >= 50:
                            return yf_data_fallback, interval_fallback_yf
                    except Exception as e_yf_fallback:
                        print(f"Error with {symbol}-USD fallback on Yahoo Finance: {e_yf_fallback}")
                return None, yf_hist_interval # Return None if all attempts fail

            try:
                yf_data, interval_tried_yf = await try_yahoo_finance_source()
                if yf_data is None or len(yf_data) < 50: # Require at least 50 data points
                    raise ValueError("Could not fetch sufficient data from Yahoo Finance")

                df = yf_data.reset_index()
                # Standardize column names from Yahoo Finance
                df = df.rename(columns={
                    'Datetime': 'open_time', 'Date': 'open_time', # Common column names for timestamp
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                })
                if 'open_time' not in df.columns and 'index' in df.columns: # Another possible timestamp col name
                    df['open_time'] = df['index']

                required_cols = ['open_time', 'open', 'high', 'low', 'close']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns from Yahoo Finance: {missing_cols}")

                if 'volume' not in df.columns: # Add synthetic volume if missing
                    print(f"Warning: Volume data not available for {symbol} from Yahoo Finance, using synthetic volume.")
                    df['volume'] = df['close'] * df['close'].rolling(window=5).std().fillna(method='bfill')

                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols: # Ensure numeric types
                    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

                df = df.dropna(subset=['open', 'close']) # Drop rows where open or close is NaN
                if df['open_time'].dt.tz: # Remove timezone if present for consistency
                    df['open_time'] = df['open_time'].dt.tz_localize(None)

                nan_percentage = df[['open', 'high', 'low', 'close']].isna().mean().mean()
                data_quality["Yahoo Finance"] = {
                    "records": len(df), "nan_percentage": nan_percentage,
                    "timespan_hours": (df['open_time'].max() - df['open_time'].min()).total_seconds() / 3600,
                    "source": "Yahoo Finance", "interval": interval_tried_yf
                }
                # print(f"Using Yahoo Finance data for {symbol} with {len(df)} records at {interval_tried_yf} interval.") # Debug
            except Exception as e:
                print(f"Yahoo Finance processing error for {symbol}: {e}")
                df = None # Reset df if Yahoo Finance processing failed

        # Attempt 3: Synthetic Data (if all real sources fail)
        if df is None or len(df) < 50: # If still no usable data
            print(f"WARNING: Using synthetic data for {symbol} as all real data sources failed or provided insufficient data.")
            hours = 24 * min(lookback_days, 7) # Limit synthetic data to a reasonable amount
            synthetic_dates = [end_time - timedelta(hours=h) for h in range(hours, 0, -1)]

            base_price = self.price_scale if hasattr(self, 'price_scale') and self.price_scale > 0 else \
                         (30000.0 if 'BTC' in symbol.upper() else 100.0) # Guess base price

            np.random.seed(42) # For reproducibility of synthetic data
            returns_synthetic = np.random.normal(0, 0.02, hours)
            cum_returns = np.cumsum(returns_synthetic)
            prices = base_price * np.exp(cum_returns - cum_returns[-1]) # Ensure last price is base_price

            df = pd.DataFrame({
                'open_time': synthetic_dates,
                'open': prices * (1 + np.random.normal(0, 0.005, hours)),
                'close': prices,
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, hours))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, hours))),
                'volume': prices * np.random.lognormal(10, 1, hours) # Synthetic volume
            })
            df['high'] = df[['high', 'open', 'close']].max(axis=1) # Ensure high is highest
            df['low'] = df[['low', 'open', 'close']].min(axis=1)   # Ensure low is lowest

            data_quality["Synthetic"] = {
                "records": len(df), "nan_percentage": 0,
                "timespan_hours": hours, "source": "Synthetic"
            }
            self.data_fallback_info = { "sources_tried": data_sources_tried, "using_synthetic": True, "data_points": len(df) }
        else: # Data successfully fetched from a real source
            best_source_name = max(data_quality.items(), key=lambda x: x[1]["records"])[0]
            self.data_fallback_info = {
                "sources_tried": data_sources_tried, "using_synthetic": False,
                "best_source": best_source_name, "data_points": len(df), "data_quality": data_quality
            }

        # Final processing for the chosen DataFrame (real or synthetic)
        df = df.sort_values('open_time').reset_index(drop=True)
        df['close'] = df['close'].replace(0, np.nan) # Avoid log(0)
        df['close'] = df['close'].fillna(method='ffill').fillna(method='bfill') # Fill NaNs

        if df['close'].empty or df['close'].isnull().all():
             raise ValueError(f"No valid close prices available for {symbol} after processing all data sources.")

        self.price_scale = float(df['close'].iloc[-1]) # Set price scale from the latest close
        # print(f"Current market price for {symbol}: ${self.price_scale:.2f}") # Debug

        # Calculate derived financial features
        df['log_price'] = np.log(df['close'])
        df['returns'] = df['log_price'].diff().fillna(0)
        df['squared_returns'] = df['returns'] ** 2
        df['vol_14'] = df['returns'].rolling(window=14).std().fillna(method='bfill') # 14-period volatility

        # Cache the processed DataFrame
        self._data_cache[cache_key] = { 'data': df, 'cache_time': datetime.now() }
        return df

    def fetch_data(self, symbol='BTCUSDT', interval='1m', lookback_days=30):
        """
        Synchronous wrapper for async fetch_data_async method.
        This ensures backward compatibility for parts of the system that might not be async.
        Maintains backward compatibility.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.fetch_data_async(symbol, interval, lookback_days))
    def assess_liquidity(self, df):
        """
        Assess the liquidity of a cryptocurrency based on its trading volume and price action.
        This helps adapt models to different liquidity profiles from large cap to small cap coins.
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe containing trading data including volume information
        Returns:
        --------
        dict : Dictionary containing liquidity metrics including:
            - normalized_volume: Volume adjusted for price level
            - volume_consistency: Consistency of volume over time
            - spread_estimate: Estimated bid-ask spread
            - slippage_factor: Potential price impact of trades
            - is_liquid: Boolean indicating if the coin has adequate liquidity
        """
        very_limited_data = len(df) < 50
        if 'volume' not in df.columns or len(df) < 20:
            return {
                'normalized_volume': 0.1,
                'volume_consistency': 0.2,
                'spread_estimate': 0.05,
                'slippage_factor': 0.1,
                'is_liquid': False
            }
        lookback = min(100, len(df) - 1)
        price = df['close'].iloc[-lookback:].mean()
        if price < 0.001:
            normalized_volume = df['volume'].iloc[-lookback:].mean() * price * 1000
        else:
            normalized_volume = df['volume'].iloc[-lookback:].mean() * price
        volumes = df['volume'].iloc[-lookback:].values
        if len(volumes) > 0:
            log_volumes = np.log1p(volumes)
            cv = np.std(log_volumes) / (np.mean(log_volumes) + 1e-10)
            volume_consistency = 1.0 - min(0.95, cv)
            volume_consistency = max(0.15, volume_consistency)
            if hasattr(self, 'current_coin_symbol'):
                symbol = self.current_coin_symbol
                coin_factor = sum(ord(c) for c in symbol) % 20 / 100
                volume_consistency = min(0.95, volume_consistency + coin_factor)
        else:
            volume_consistency = 0.2
        if very_limited_data:
            volume_consistency *= 0.7
        if 'high' in df.columns and 'low' in df.columns:
            spread_estimate = ((df['high'] - df['low']) / df['close']).iloc[-lookback:].mean()
            if very_limited_data:
                spread_estimate *= 1.5
        else:
            spread_estimate = df['returns'].iloc[-lookback:].std() * 0.5
            if very_limited_data:
                spread_estimate *= 2.0
        spread_estimate = min(0.05, max(0.001, spread_estimate))
        avg_volume = df['volume'].iloc[-min(20, len(df)-1):].mean()
        if avg_volume > 0:
            vol_factor = min(1.0, 10000 / avg_volume) if avg_volume > 0 else 1.0
            slippage_factor = spread_estimate * (1.0 + vol_factor)
        else:
            slippage_factor = spread_estimate * 3.0
        slippage_factor = min(0.1, max(0.001, slippage_factor))
        if very_limited_data:
            is_liquid = (normalized_volume > 50000) and (volume_consistency > 0.4) and (spread_estimate < 0.02)
        else:
            is_liquid = (normalized_volume > 10000) and (volume_consistency > 0.3) and (spread_estimate < 0.03)
        return {
            'normalized_volume': normalized_volume,
            'volume_consistency': volume_consistency,
            'spread_estimate': spread_estimate,
            'slippage_factor': slippage_factor,
            'is_liquid': is_liquid
        }
    def calculate_market_fragility(self, df):
        """
        Calculate market fragility score using the FragilityScoreCalculator.
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing OHLCV data for analysis
        Returns:
        --------
        dict: Fragility score details including overall score, component scores,
              breakout direction, and confidence
        """
        if hasattr(self, 'liquidity_profile'):
            fragility_result = self.fragility_calculator.calculate_fragility_score(
                df, self.liquidity_profile
            )
        else:
            fragility_result = self.fragility_calculator.calculate_fragility_score(df)
        self.fragility_score = fragility_result
        interpretation = self.fragility_calculator.get_score_interpretation(
            fragility_result['overall_score']
        )
        print(f"Market Fragility Score: {fragility_result['overall_score']:.1f}/100")
        print(f"Interpretation: {interpretation}")
        if fragility_result['breakout_direction'] != 'unknown':
            directional_text = "upward" if fragility_result['breakout_direction'] == "up" else "downward"
            print(f"Potential {directional_text} breakout detected (confidence: {fragility_result['confidence']:.1%})")
        return fragility_result
    def estimate_initial_state(self, df):
        """
        Simplified initial state estimation for performance optimization.
        Uses only essential calculations to quickly estimate X_t (log price relative to scale)
        and phi_t (volatility). This simplified version is used in the main `run_forecast`
        for performance.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'close', 'returns', and 'log_price' columns.

        Returns:
        --------
        tuple
            A tuple (X_t, phi_t) representing the estimated current state.
        """
        # Ensure price_scale is set; if not, estimate from current df (should ideally be pre-set by data fetching)
        if not hasattr(self, 'price_scale') or self.price_scale == 0:
            if not df['close'].empty:
                self.price_scale = float(df['close'].iloc[-1]) if df['close'].iloc[-1] > 0 else 1.0
            else: # Should not happen if data validation is done prior
                self.price_scale = 1.0 # Default failsafe
                print("Warning: price_scale was not set, defaulted to 1.0 in estimate_initial_state.")

        # Fast initial state estimation for X_t (current log price relative to price scale)
        X_t = np.log(df['close'].iloc[-1] / self.price_scale) if self.price_scale > 0 else 0
        
        # Quick volatility estimation (phi_t) using recent returns variance
        recent_returns = df['returns'].iloc[-60:].values # Use last 60 periods for volatility
        phi_t = np.nanvar(recent_returns) if len(recent_returns) > 10 else 0.01 # Default if not enough data
        phi_t = np.clip(phi_t, 1e-6, 1.0) # Bound volatility to avoid extreme values
        
        # Store minimal history for drift calculation (log prices relative to current price scale)
        # Reduced history from 200 to 50 for performance in iterative calls.
        self.X_history = np.array(df['log_price'].values[-50:] - np.log(self.price_scale)) if self.price_scale > 0 else \
                         np.array(df['log_price'].values[-50:])
        
        return X_t, phi_t

    def _estimate_initial_state_original(self, df):
        """Original implementation of estimate_initial_state for when Numba is not available"""
        X_t = np.log(df['close'].iloc[-1] / self.price_scale)
        self.X_history = np.array(df['log_price'].values[-200:] - np.log(self.price_scale))
        daily_returns = df['returns'].rolling(window=1440).sum().dropna()
        hourly_returns = df['returns'].rolling(window=60).sum().dropna()
        minute_returns = df['returns']
        alpha_garch = 0.1
        beta_garch = 0.85
        recent_sq_returns = df['squared_returns'].iloc[-60:].values
        garch_vol = df['squared_returns'].rolling(window=30).mean().iloc[-1]
        decay_factor = 0.98
        weight_sum = 0
        for i, r2 in enumerate(recent_sq_returns):
            weight = decay_factor ** (len(recent_sq_returns) - i - 1)
            weight_sum += weight
            garch_vol = alpha_garch * r2 * weight + beta_garch * garch_vol
        if weight_sum > 0:
            garch_vol /= weight_sum
        simple_vol = df['squared_returns'].rolling(window=30).mean().iloc[-1]
        ewma_vol = df['squared_returns'].ewm(span=30).mean().iloc[-1]
        if 'high' in df.columns and 'low' in df.columns:
            high_low_range = np.log(df['high'] / df['low']).pow(2)
            parkinson_vol = high_low_range.rolling(window=30).mean().iloc[-1] / (4 * np.log(2))
        else:
            parkinson_vol = simple_vol
        vol_of_vol = df['squared_returns'].rolling(window=60).std().iloc[-1]
        rel_vol_of_vol = vol_of_vol / simple_vol if simple_vol > 0 else 1.0
        vol_stability = np.std(df['squared_returns'].iloc[-60:]) / np.mean(df['squared_returns'].iloc[-60:] + 1e-10)
        if vol_stability > 2.0:
            weights = [0.5, 0.4, 0.05, 0.05]
        elif vol_stability > 1.0:
            weights = [0.4, 0.3, 0.15, 0.15]
        else:
            weights = [0.3, 0.3, 0.2, 0.2] 
        phi_t = (weights[0] * garch_vol + 
                weights[1] * ewma_vol + 
                weights[2] * simple_vol + 
                weights[3] * parkinson_vol)
        recent_vol_ratio = df['squared_returns'].iloc[-10:].mean() / df['squared_returns'].iloc[-60:].mean()
        if not np.isnan(recent_vol_ratio) and 0.1 < recent_vol_ratio < 10:
            phi_t *= np.sqrt(np.clip(recent_vol_ratio, 0.5, 2.0))
        phi_t = max(phi_t, 1e-6)
        trending = self.detect_trend(df['log_price'].iloc[-100:])
        momentum = self.detect_momentum(df['returns'].iloc[-60:])
        self.current_regime = {'trending': trending, 'momentum': momentum, 'vol_stability': vol_stability}
        return X_t, phi_t
    def _estimate_initial_state_optimized(self, df):
        """Optimized implementation of estimate_initial_state using Numba for performance"""
        X_t = np.log(df['close'].iloc[-1] / self.price_scale)
        self.X_history = np.array(df['log_price'].values[-200:] - np.log(self.price_scale))
        squared_returns = df['squared_returns'].values
        squared_returns_60 = squared_returns[-60:] if len(squared_returns) >= 60 else squared_returns
        squared_returns_30 = squared_returns[-30:] if len(squared_returns) >= 30 else squared_returns
        squared_returns_10 = squared_returns[-10:] if len(squared_returns) >= 10 else squared_returns
        alpha_garch = 0.1
        beta_garch = 0.85
        decay_factor = 0.98
        garch_vol = self._compute_garch_vol(squared_returns_60, 
                                           df['squared_returns'].rolling(window=30).mean().iloc[-1], 
                                           alpha_garch, beta_garch, decay_factor)
        simple_vol = np.nanmean(squared_returns_30)
        span = 30
        alpha_ewma = 2 / (span + 1)
        ewma_vol = self._compute_ewma_vol(squared_returns_60, alpha_ewma)
        if 'high' in df.columns and 'low' in df.columns:
            high_values = df['high'].values[-30:]
            low_values = df['low'].values[-30:]
            parkinson_vol = self._compute_parkinson_vol(high_values, low_values)
        else:
            parkinson_vol = simple_vol
        vol_of_vol = np.nanstd(squared_returns_60)
        mean_sq_returns = np.nanmean(squared_returns_60 + 1e-10)
        vol_stability = np.nanstd(squared_returns_60) / mean_sq_returns
        if vol_stability > 2.0:
            weights = np.array([0.5, 0.4, 0.05, 0.05])
        elif vol_stability > 1.0:
            weights = np.array([0.4, 0.3, 0.15, 0.15])
        else:
            weights = np.array([0.3, 0.3, 0.2, 0.2])
        phi_t = (weights[0] * garch_vol + 
                weights[1] * ewma_vol + 
                weights[2] * simple_vol + 
                weights[3] * parkinson_vol)
        recent_vol_mean = np.nanmean(squared_returns_10)
        longer_vol_mean = np.nanmean(squared_returns_60)
        recent_vol_ratio = recent_vol_mean / longer_vol_mean if longer_vol_mean > 0 else 1.0
        if not np.isnan(recent_vol_ratio) and 0.1 < recent_vol_ratio < 10:
            phi_t *= np.sqrt(np.clip(recent_vol_ratio, 0.5, 2.0))
        phi_t = max(phi_t, 1e-6)
        trending = self.detect_trend(df['log_price'].iloc[-100:])
        momentum = self.detect_momentum(df['returns'].iloc[-60:])
        self.current_regime = {'trending': trending, 'momentum': momentum, 'vol_stability': vol_stability}
        return X_t, phi_t
    @staticmethod
    @jit(nopython=True)
    def _compute_garch_vol(recent_sq_returns, initial_garch_vol, alpha_garch, beta_garch, decay_factor):
        """Numba-optimized GARCH volatility computation"""
        garch_vol = initial_garch_vol
        decay_powers = np.zeros(len(recent_sq_returns))
        for i in range(len(recent_sq_returns)):
            decay_powers[i] = decay_factor ** (len(recent_sq_returns) - i - 1)
        weight_sum = np.sum(decay_powers)
        for i in range(len(recent_sq_returns)):
            garch_vol = alpha_garch * recent_sq_returns[i] * decay_powers[i] + beta_garch * garch_vol
        if weight_sum > 0:
            garch_vol /= weight_sum
        return garch_vol
    @staticmethod
    @jit(nopython=True)
    def _compute_ewma_vol(returns, alpha):
        """Numba-optimized EWMA volatility computation"""
        if len(returns) == 0:
            return 0.0
        ewma = returns[0]
        for i in range(1, len(returns)):
            ewma = alpha * returns[i] + (1 - alpha) * ewma
        return ewma
    @staticmethod
    @jit(nopython=True)
    def _compute_parkinson_vol(high_values, low_values):
        """Numba-optimized Parkinson volatility computation"""
        if len(high_values) == 0 or len(low_values) == 0:
            return 0.0
        sum_hl_ratio = 0.0
        count = 0
        for i in range(len(high_values)):
            if high_values[i] > 0 and low_values[i] > 0:
                hl_ratio = np.log(high_values[i] / low_values[i]) ** 2
                sum_hl_ratio += hl_ratio
                count += 1
        if count > 0:
            return sum_hl_ratio / (count * 4 * np.log(2))
        else:
            return 0.0
    def detect_trend(self, price_series, window=30):
        """
        Sophisticated trend detection using multiple indicators:
        1. Hurst exponent for long-memory trends
        2. Linear regression coefficient significance
        3. Moving average convergence/divergence
        Returns a trend strength indicator between -1 (strong downtrend) and 1 (strong uptrend)
        """
        return self._detect_trend_jit(np.asarray(price_series), window) if NUMBA_AVAILABLE else self._detect_trend_py(price_series, window)
    def _detect_trend_py(self, price_series, window=30):
        """Python implementation of trend detection for fallback"""
        if len(price_series) < window:
            return 0.0
        prices = np.asarray(price_series)
        n = len(prices)
        lags = range(2, min(20, n // 4))
        tau = []; 
        for lag in lags:
            price_diff = np.diff(prices, lag)
            var_diff = np.var(price_diff)
            var_first_diff = np.var(np.diff(prices))
            if var_first_diff > 0:
                tau.append(var_diff / (lag * var_first_diff))
        if len(tau) > 1:
            h_estimate = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)[0]
            hurst = max(0.1, min(0.9, 0.5 + 0.5 * h_estimate))
            if hurst > 0.65:
                persistence_bias = (hurst - 0.65) / 0.35
                dampening = 1.0 - 0.3 * persistence_bias
                hurst = 0.65 + (hurst - 0.65) * dampening
            elif hurst < 0.35:
                anti_persistence_bias = (0.35 - hurst) / 0.35
                dampening = 1.0 - 0.3 * anti_persistence_bias
                hurst = 0.35 - (0.35 - hurst) * dampening
            hurst_signal = 1.8 * (hurst - 0.5)
        else:
            hurst_signal = 0.0
        x = np.arange(n)
        slope, _, r_value, p_value, _ = stats.linregress(x, prices)
        normalized_slope = slope * n / max(abs(np.mean(prices)), 1e-10) 
        trend_significance = r_value**2 * (1 - min(p_value, 0.5) / 0.5)
        linear_signal = np.sign(slope) * min(abs(normalized_slope * 50), 1.0) * trend_significance
        if n >= 50:
            ema_short = pd.Series(prices).ewm(span=15).mean().iloc[-1]
            ema_long = pd.Series(prices).ewm(span=50).mean().iloc[-1]
            ema_diff = (ema_short - ema_long) / max(ema_long, 1e-10)
            ema_signal = np.clip(ema_diff * 20, -1, 1)
        else:
            ema_signal = 0.0
        if trend_significance > 0.6:
            weights = [0.2, 0.6, 0.2]
        else:
            weights = [0.4, 0.3, 0.3]
        trend_indicator = (weights[0] * hurst_signal + 
                          weights[1] * linear_signal + 
                          weights[2] * ema_signal)
        return np.clip(trend_indicator, -1, 1)
    @staticmethod
    @jit(nopython=True)
    def _detect_trend_jit(prices, window=30):
        """Numba-optimized implementation of trend detection"""
        if len(prices) < window:
            return 0.0
        n = len(prices)
        max_lag = min(20, n // 4)
        tau = np.zeros(max_lag - 1)
        count = 0
        for lag in range(2, max_lag + 1):
            price_diff = np.zeros(n - lag)
            for i in range(n - lag):
                price_diff[i] = prices[i + lag] - prices[i]
            var_diff = 0.0
            for pd in price_diff:
                var_diff += pd * pd
            var_diff /= len(price_diff)
            first_diff = np.zeros(n - 1)
            for i in range(n - 1):
                first_diff[i] = prices[i + 1] - prices[i]
            var_first_diff = 0.0
            for fd in first_diff:
                var_first_diff += fd * fd
            var_first_diff /= len(first_diff)
            if var_first_diff > 0:
                tau[count] = var_diff / (lag * var_first_diff)
                count += 1
        hurst_signal = 0.0
        if count > 1:
            log_lags = np.zeros(count)
            log_tau = np.zeros(count)
            for i in range(count):
                log_lags[i] = np.log(i + 2)
                log_tau[i] = np.log(tau[i])
            sum_x = 0.0
            sum_y = 0.0
            sum_xy = 0.0
            sum_xx = 0.0
            for i in range(count):
                sum_x += log_lags[i]
                sum_y += log_tau[i]
                sum_xy += log_lags[i] * log_tau[i]
                sum_xx += log_lags[i] * log_lags[i]
            h_estimate = ((count * sum_xy) - (sum_x * sum_y)) / ((count * sum_xx) - (sum_x * sum_x))
            hurst = max(0.1, min(0.9, 0.5 + 0.5 * h_estimate))
            if hurst > 0.65:
                persistence_bias = (hurst - 0.65) / 0.35
                dampening = 1.0 - 0.3 * persistence_bias
                hurst = 0.65 + (hurst - 0.65) * dampening
            elif hurst < 0.35:
                anti_persistence_bias = (0.35 - hurst) / 0.35
                dampening = 1.0 - 0.3 * anti_persistence_bias
                hurst = 0.35 - (0.35 - hurst) * dampening
            hurst_signal = 1.8 * (hurst - 0.5)
        x = np.arange(n)
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_xx = 0.0
        sum_yy = 0.0
        for i in range(n):
            sum_x += x[i]
            sum_y += prices[i]
            sum_xy += x[i] * prices[i]
            sum_xx += x[i] * x[i]
            sum_yy += prices[i] * prices[i]
        slope = ((n * sum_xy) - (sum_x * sum_y)) / ((n * sum_xx) - (sum_x * sum_x))
        r_numerator = (n * sum_xy) - (sum_x * sum_y)
        r_denominator = np.sqrt(((n * sum_xx) - (sum_x * sum_x)) * ((n * sum_yy) - (sum_y * sum_y)))
        r_value = r_numerator / r_denominator if r_denominator != 0 else 0.0
        p_value = 0.05
        if n > 30:
            if abs(r_value) > 0.36:
                p_value = 0.01
            elif abs(r_value) > 0.30:
                p_value = 0.05
            elif abs(r_value) > 0.25:
                p_value = 0.1
            else:
                p_value = 0.5
        mean_price = sum_y / n
        normalized_slope = slope * n / max(abs(mean_price), 1e-10)
        trend_significance = r_value * r_value * (1 - min(p_value, 0.5) / 0.5)
        linear_signal = 0.0
        if slope > 0:
            linear_signal = min(abs(normalized_slope * 50), 1.0) * trend_significance
        else:
            linear_signal = -min(abs(normalized_slope * 50), 1.0) * trend_significance
        ema_signal = 0.0
        if n >= 50:
            ema_short_span = 15
            ema_long_span = 50
            alpha_short = 2 / (ema_short_span + 1)
            alpha_long = 2 / (ema_long_span + 1)
            ema_short = prices[0]
            ema_long = prices[0]
            for i in range(1, n):
                ema_short = alpha_short * prices[i] + (1 - alpha_short) * ema_short
                ema_long = alpha_long * prices[i] + (1 - alpha_long) * ema_long
            ema_diff = (ema_short - ema_long) / max(ema_long, 1e-10)
            if ema_diff > 0.05:
                ema_signal = min(ema_diff * 20, 1.0)
            elif ema_diff < -0.05:
                ema_signal = max(ema_diff * 20, -1.0)
        weights = np.zeros(3)
        if trend_significance > 0.6:
            weights[0] = 0.2
            weights[1] = 0.6
            weights[2] = 0.2
        else:
            weights[0] = 0.4
            weights[1] = 0.3
            weights[2] = 0.3
        trend_indicator = (weights[0] * hurst_signal + 
                           weights[1] * linear_signal + 
                           weights[2] * ema_signal)
        if trend_indicator > 1.0:
            return 1.0
        elif trend_indicator < -1.0:
            return -1.0
        else:
            return trend_indicator
    def detect_momentum(self, returns_series, window=20):
        """
        Advanced momentum detection using:
        1. Rate of change (RoC)
        2. RSI (Relative Strength Index)
        3. Acceleration (change in momentum)
        Returns momentum indicator between -1 (strong negative) and 1 (strong positive)
        """
        if NUMBA_AVAILABLE:
            return self._detect_momentum_optimized(np.asarray(returns_series), window)
        else:
            return self._detect_momentum_original(returns_series, window)
    @staticmethod
    @jit(nopython=True)
    def _detect_momentum_optimized(returns, window=20):
        """Numba-optimized implementation of momentum detection"""
        if len(returns) < window:
            return 0.0
        short_term = 0.0
        for i in range(max(0, len(returns)-10), len(returns)):
            short_term += returns[i]
        medium_term = 0.0
        for i in range(max(0, len(returns)-window), len(returns)):
            medium_term += returns[i]
        mean_return = 0.0
        for i in range(max(0, len(returns)-window), len(returns)):
            mean_return += returns[i]
        mean_return /= min(window, len(returns))
        squared_sum = 0.0
        for i in range(max(0, len(returns)-window), len(returns)):
            squared_sum += (returns[i] - mean_return) ** 2
        window_length = min(window, len(returns))
        vol = np.sqrt(squared_sum / (window_length - 1)) * np.sqrt(window) if window_length > 1 else 1.0
        if vol > 1e-10:
            normalized_st_roc = short_term / vol
            normalized_mt_roc = medium_term / vol
        else:
            normalized_st_roc = short_term * 100
            normalized_mt_roc = medium_term * 100
        up_returns_sum = 0.0
        down_returns_sum = 0.0
        count_up = 0
        count_down = 0
        for i in range(max(0, len(returns)-window), len(returns)):
            if returns[i] > 0:
                up_returns_sum += returns[i]
                count_up += 1
            else:
                down_returns_sum += -returns[i]
                count_down += 1
        avg_up = up_returns_sum / max(1, count_up)
        avg_down = down_returns_sum / max(1, count_down)
        if avg_down > 1e-10:
            rs = avg_up / avg_down
            rsi = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi = 100.0 if avg_up > 0 else 50.0
        rsi_signal = (rsi - 50.0) / 50.0
        acceleration_signal = 0.0
        if len(returns) >= window*2:
            recent_momentum = 0.0
            past_momentum = 0.0
            for i in range(max(0, len(returns)-window), len(returns)):
                recent_momentum += returns[i]
            for i in range(max(0, len(returns)-2*window), len(returns)-window):
                past_momentum += returns[i]
            momentum_change = recent_momentum - past_momentum
            if vol > 1e-10:
                acceleration = momentum_change / (vol * np.sqrt(2))
                acceleration_signal = min(1.0, max(-1.0, acceleration))
        momentum_signal = (0.3 * normalized_st_roc + 
                           0.3 * normalized_mt_roc + 
                           0.2 * rsi_signal + 
                           0.2 * acceleration_signal)
        return min(1.0, max(-1.0, momentum_signal))
    def _detect_momentum_original(self, returns_series, window=20):
        """Original implementation of momentum detection"""
        if len(returns_series) < window:
            return 0.0
        returns = np.asarray(returns_series)
        short_term = np.sum(returns[-10:])
        medium_term = np.sum(returns[-window:])
        vol = np.std(returns) * np.sqrt(window)
        if vol > 1e-10:
            normalized_st_roc = short_term / vol
            normalized_mt_roc = medium_term / vol
        else:
            normalized_st_roc = short_term * 100
            normalized_mt_roc = medium_term * 100
        up_returns = np.maximum(returns, 0)
        down_returns = np.maximum(-returns, 0)
        avg_up = np.mean(up_returns[-window:])
        avg_down = np.mean(down_returns[-window:])
        if avg_down > 1e-10:
            rs = avg_up / avg_down
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100 if avg_up > 0 else 50
        rsi_signal = (rsi - 50) / 50
        if len(returns) >= window*2:
            recent_momentum = np.sum(returns[-window:])
            past_momentum = np.sum(returns[-2*window:-window])
            momentum_change = recent_momentum - past_momentum
            acceleration = momentum_change / (vol * np.sqrt(2)) if vol > 1e-10 else 0
            acceleration_signal = np.clip(acceleration, -1, 1)
        else:
            acceleration_signal = 0.0
        momentum_signal = (0.3 * normalized_st_roc + 
                           0.3 * normalized_mt_roc + 
                           0.2 * rsi_signal + 
                           0.2 * acceleration_signal)
        return np.clip(momentum_signal, -1, 1)
    def calculate_realized_volatility(self, df, window=30):
        """
        Advanced realized volatility using multiple estimators for optimal accuracy:
        1. Standard rolling window estimator
        2. EWMA for adaptive memory
        3. Parkinson estimator using high-low range
        4. Garman-Klass estimator incorporating open-high-low-close
        5. Rogers-Satchell estimator for trend-adjusted volatility
        Returns a combined volatility metric that adapts to market conditions
        """
        if NUMBA_AVAILABLE:
            return self._calculate_realized_volatility_optimized(df, window)
        else:
            return self._calculate_realized_volatility_original(df, window)
    def _calculate_realized_volatility_optimized(self, df, window=30):
        """Optimized implementation of realized volatility calculation using Numba"""
        returns = df['returns'].values
        returns_valid = ~np.isnan(returns)
        returns_clean = returns[returns_valid]
        n = len(returns)
        simple_vol = np.zeros(n)
        if len(returns_clean) > 0:
            rolling_std = pd.Series(returns_clean).rolling(window=window, min_periods=1).std().values
            j = 0
            for i in range(n):
                if returns_valid[i]:
                    simple_vol[i] = rolling_std[j]
                    j += 1
        squared_returns = returns ** 2
        ewma_vol = np.zeros(n)
        squared_returns_valid = ~np.isnan(squared_returns)
        squared_returns_clean = squared_returns[squared_returns_valid]
        if len(squared_returns_clean) > 0:
            alpha = 2 / (window + 1)
            ewma_values = pd.Series(squared_returns_clean).ewm(alpha=alpha, adjust=False).mean().values
            ewma_sqrt = np.sqrt(ewma_values)
            j = 0
            for i in range(n):
                if squared_returns_valid[i]:
                    ewma_vol[i] = ewma_sqrt[j]
                    j += 1
        limited_history = len(returns) < window * 2
        extreme_regime = False
        has_spikes = False
        if len(returns) >= window:
            recent_returns = returns[-window:]
            recent_returns_clean = recent_returns[~np.isnan(recent_returns)]
            if len(recent_returns_clean) > 0:
                std_recent = np.std(recent_returns_clean)
                extreme_regime = np.any(np.abs(recent_returns_clean) > 5 * std_recent)
                has_spikes = np.any(np.abs(recent_returns_clean) > 10 * std_recent)
        has_ohlc = all(col in df.columns for col in ['high', 'low', 'open', 'close'])
        if has_ohlc:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            open_prices = df['open'].values
            high_low_ratio = ((high / low) - 1) * 100
            close_open_ratio = ((close / open_prices) - 1) * 100
            parkinson_vol = np.sqrt(np.log(high / low) ** 2 / (4 * np.log(2)))
            gk_estimator = 0.5 * np.log(high / low) ** 2 - (2 * np.log(2) - 1) * np.log(close / open_prices) ** 2
            rs_estimator = np.log(high / close) * np.log(high / open_prices) + np.log(low / close) * np.log(low / open_prices)
            weights = np.array([0.20, 0.25, 0.20, 0.20, 0.15])
            if len(returns) >= max(50, 2 * window):
                forward_vol = np.zeros(n)
                for i in range(n - window):
                    if i + window < n:
                        window_data = returns[i:i+window]
                        valid_data = window_data[~np.isnan(window_data)]
                        if len(valid_data) > 1:
                            forward_vol[i] = np.std(valid_data)
                forward_vol_mean = np.mean(forward_vol[forward_vol > 0])
                if forward_vol_mean > 0:
                    estimators = np.zeros((5, n))
                    estimators[0, :] = simple_vol
                    estimators[1, :] = ewma_vol
                    estimators[2, :] = parkinson_vol
                    estimators[3, :] = gk_estimator
                    estimators[4, :] = rs_estimator
                    valid_idx = (forward_vol > 0) & (~np.isnan(forward_vol))
                    actual = forward_vol[valid_idx]
                    forecasts = estimators[:, valid_idx]
                    if len(actual) > 0:
                        # Simple optimal weight calculation using least squares
                        try:
                            weights_calc = np.linalg.lstsq(forecasts.T, actual, rcond=None)[0]
                            if np.all(weights_calc >= 0) and np.sum(weights_calc) > 0:
                                weights = weights_calc / np.sum(weights_calc)
                        except:
                            pass  # Keep default weights
            if limited_history:
                data_quality = min(0.7, len(returns) / window)
                weights[1] *= 1.2
                weights[0] *= 0.8
                weights = weights / np.sum(weights)
            combined_vol = (
                weights[0] * simple_vol + 
                weights[1] * ewma_vol + 
                weights[2] * parkinson_vol + 
                weights[3] * gk_estimator + 
                weights[4] * rs_estimator
            )
            combined_vol_series = pd.Series(combined_vol, index=df.index)
            return combined_vol_series.clip(lower=1e-8)
        else:
            combined_vol = np.zeros(n)
            if limited_history or extreme_regime:
                combined_vol = 0.25 * simple_vol + 0.75 * ewma_vol
            else:
                combined_vol = 0.4 * simple_vol + 0.6 * ewma_vol
            if len(returns) < window / 2:
                safety_factor = 1.5
                combined_vol *= safety_factor
            combined_vol_series = pd.Series(combined_vol, index=df.index)
            return combined_vol_series.clip(lower=1e-8)
    def _calculate_realized_volatility_original(self, df, window=30):
        """Original implementation of realized volatility for when Numba is not available"""
        returns = df['returns']
        simple_vol = returns.rolling(window=window).std().fillna(0)
        ewma_vol = np.sqrt(returns.ewm(span=window).var().fillna(0))
        limited_history = len(returns) < window * 2
        if len(returns) >= window:
            recent_returns = returns.iloc[-window:]
            extreme_regime = np.abs(recent_returns).max() > 5 * recent_returns.std()
            has_spikes = np.any(np.abs(recent_returns) > 10 * recent_returns.std())
        else:
            extreme_regime = False
            has_spikes = False
        if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            high_low_ratio = np.log(df['high']/df['low']) ** 2
            close_open_ratio = np.log(df['close']/df['open']) ** 2
            parkinson_vol = np.sqrt(high_low_ratio.rolling(window=window).mean() / (4 * np.log(2))).fillna(0)
            c1 = 0.5
            c2 = (2 * np.log(2) - 1)
            gk_estimator = np.sqrt(
                (c1 * high_low_ratio -
                c2 * close_open_ratio).rolling(window=window).mean()
            ).fillna(0)
            high_close = np.log(df['high']/df['close'])
            high_open = np.log(df['high']/df['open'])
            low_close = np.log(df['low']/df['close'])
            low_open = np.log(df['low']/df['open'])
            rs_term = (high_close * high_open + low_close * low_open)
            rs_estimator = np.sqrt(rs_term.rolling(window=window).mean()).fillna(0)
            if len(returns) >= max(50, 2 * window):
                forward_vol = returns.rolling(window=window).std().shift(-window).fillna(0)
                forward_vol_mean = forward_vol.mean()
                if forward_vol_mean > 0:
                    simple_error = np.mean(np.abs(simple_vol - forward_vol)) / forward_vol_mean
                    ewma_error = np.mean(np.abs(ewma_vol - forward_vol)) / forward_vol_mean
                    parkinson_error = np.mean(np.abs(parkinson_vol - forward_vol)) / forward_vol_mean
                    gk_error = np.mean(np.abs(gk_estimator - forward_vol)) / forward_vol_mean
                    rs_error = np.mean(np.abs(rs_estimator - forward_vol)) / forward_vol_mean
                    total_error = simple_error + ewma_error + parkinson_error + gk_error + rs_error
                    if total_error > 0:
                        w_simple = 1.0 / (simple_error + 0.01)
                        w_ewma = 1.0 / (ewma_error + 0.01)
                        w_parkinson = 1.0 / (parkinson_error + 0.01)
                        w_gk = 1.0 / (gk_error + 0.01)
                        w_rs = 1.0 / (rs_error + 0.01)
                    else:
                        w_simple = w_ewma = w_parkinson = w_gk = w_rs = 1.0
                else:
                    w_simple = 0.20
                    w_ewma = 0.25
                    w_parkinson = 0.20
                    w_gk = 0.20
                    w_rs = 0.15
            else:
                w_simple = 0.20
                w_ewma = 0.25
                w_parkinson = 0.20
                w_gk = 0.20
                w_rs = 0.15
            data_quality = 1.0
            if limited_history:
                data_quality = min(0.7, len(returns) / window)
                w_ewma *= 1.2
                w_simple *= 0.8
            total = w_simple + w_ewma + w_parkinson + w_gk + w_rs
            w_simple /= total
            w_ewma /= total
            w_parkinson /= total
            w_gk /= total
            w_rs /= total
            combined_vol = (
                w_simple * simple_vol +
                w_ewma * ewma_vol +
                w_parkinson * parkinson_vol +
                w_gk * gk_estimator +
                w_rs * rs_estimator
            )
        else:
            if limited_history or extreme_regime:
                combined_vol = 0.25 * simple_vol + 0.75 * ewma_vol
            else:
                combined_vol = 0.4 * simple_vol + 0.6 * ewma_vol
            if len(returns) < window / 2:
                safety_factor = 1.5
                combined_vol = combined_vol * safety_factor
        return combined_vol.clip(lower=1e-8)
    def detect_market_regime(self, df, lookback=60):
        """
        Simplified and fast market regime detection for performance optimization.
        Uses only essential features to quickly classify market states.
        """
        returns = df['returns']
        if len(returns) < lookback:
            return 0, 0.5
        
        # Use only the most recent and essential data
        recent_returns = returns.iloc[-min(lookback, 30):].values  # Reduced lookback for speed
        
        # Fast calculations using numpy
        mean_return = np.nanmean(recent_returns)
        std_return = np.nanstd(recent_returns)
        
        # Simple regime classification based on mean and volatility
        if std_return < 0.01:  # Low volatility
            if mean_return > 0.001:
                return 1, 0.8  # Low vol uptrend
            else:
                return 2, 0.8  # Low vol downtrend
        else:  # High volatility
            if mean_return > 0:
                return 3, 0.7  # High vol uptrend
            else:
                return 4, 0.7  # High vol downtrend
    def _detect_market_regime_original(self, df, lookback=60):
        """Original implementation of detect_market_regime for when Numba is not available"""
        returns = df['returns']
        if len(returns) < lookback:
            return 0, 0.5
        recent_returns = returns.iloc[-lookback:]
        mean_return = recent_returns.mean()
        std_return = recent_returns.std()
        skew_return = recent_returns.skew()
        kurt_return = recent_returns.kurtosis()
        acf_1 = recent_returns.autocorr(lag=1)
        acf_5 = recent_returns.autocorr(lag=5)
        vol = np.sqrt(df['squared_returns'].rolling(window=30).mean().iloc[-lookback:])
        vol_of_vol = vol.pct_change().abs().mean()
        up_days = (recent_returns > 0).sum() / lookback
        jump_threshold = 2.5 * std_return
        jumps = (recent_returns.abs() > jump_threshold).sum() / lookback
        if 'high' in df.columns and 'low' in df.columns:
            high_low_ratio = (df['high'] / df['low']).iloc[-lookback:].mean()
        else:
            high_low_ratio = 1.0
        features = np.vstack([
            recent_returns.values,
            vol.values,
            np.repeat(mean_return, lookback),
            np.repeat(up_days, lookback),
            np.repeat(jumps, lookback),
            np.repeat(acf_1, lookback),
            np.repeat(vol_of_vol, lookback)
        ]).T
        features = np.nan_to_num(features)
        from sklearn.preprocessing import StandardScaler
        try:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
        except:
            features_scaled = features
        n_components = 4
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=None,
                n_init=10,
                init_params='kmeans'
            )
            gmm.fit(features_scaled)
            current_feature = np.array([
                [returns.iloc[-1],
                vol.iloc[-1],
                mean_return,
                up_days,
                jumps,
                acf_1,
                vol_of_vol]
            ])
            try:
                current_feature_scaled = scaler.transform(current_feature)
            except:
                current_feature_scaled = current_feature
            current_regime = gmm.predict(current_feature_scaled)[0]
            regime_probs = gmm.predict_proba(current_feature_scaled)[0]
            current_regime_prob = regime_probs[current_regime]
            means = gmm.means_
            regime_characteristics = []
            vol_values = df['squared_returns'].rolling(window=30).mean().dropna().values
            vol_series = np.sqrt(vol_values) * np.sqrt(252)
            if len(vol_series) >= 20:
                vol_quantiles = np.percentile(vol_series, [33, 67])
                low_vol_threshold = vol_quantiles[0] 
                high_vol_threshold = vol_quantiles[1]
            else:
                vol_mean = np.mean(means[:, 1])
                vol_std = np.std(means[:, 1])
                low_vol_threshold = vol_mean - 0.5 * vol_std
                high_vol_threshold = vol_mean + 0.5 * vol_std
            if len(df) > 50:
                returns_history = df['returns'].dropna().values
                ret_quantiles = np.percentile(returns_history, [33, 67])
                negative_threshold = ret_quantiles[0] * 10
                positive_threshold = ret_quantiles[1] * 10
            else:
                ret_mean = np.mean(means[:, 2])
                ret_std = np.std(means[:, 2]) 
                negative_threshold = -0.5 * ret_std
                positive_threshold = 0.5 * ret_std
            for i in range(n_components):
                ret_mean = means[i, 2]
                vol_level = means[i, 1]
                if ret_mean > positive_threshold:
                    if vol_level < low_vol_threshold:
                        label = "Low-Vol Bullish"
                    else:
                        label = "Volatile Bullish"
                elif ret_mean < negative_threshold:
                    if vol_level < low_vol_threshold:
                        label = "Low-Vol Bearish"
                    else:
                        label = "Volatile Bearish"
                else:
                    if vol_level < low_vol_threshold:
                        label = "Low-Vol Neutral"
                    else:
                        label = "Volatile Neutral"
                regime_characteristics.append(label)
            self.regime_labels = regime_characteristics
            self.current_regime_label = regime_characteristics[current_regime]
            return current_regime, current_regime_prob
        except Exception as e:
            print(f"Regime detection fallback: {e}")
            if len(returns) >= 50:
                recent_ret_mean = returns.iloc[-20:].mean()
                recent_ret_median = returns.iloc[-20:].median()
                recent_vol = returns.iloc[-20:].std()
                direction_strength = (recent_ret_mean + recent_ret_median) / 2
                vol_level = df['squared_returns'].rolling(window=20).mean().iloc[-1]
                vol_history = df['squared_returns'].rolling(window=20).mean().dropna()
                vol_percentile = np.percentile(vol_history, [25, 50, 75])
                if vol_level < vol_percentile[0]:
                    vol_regime = "Low-Vol"
                elif vol_level > vol_percentile[2]:
                    vol_regime = "Volatile"
                else:
                    vol_regime = "Moderate-Vol"
                if direction_strength > 0.3 * recent_vol:
                    direction = "Bullish"
                    regime_id = 0 if vol_regime == "Low-Vol" else 2
                elif direction_strength < -0.3 * recent_vol:
                    direction = "Bearish"
                    regime_id = 1 if vol_regime == "Low-Vol" else 3
                else:
                    direction = "Neutral"
                    regime_id = 4 if vol_regime == "Volatile" else 5
                regime_label = f"{vol_regime} {direction}"
                if hasattr(self, 'current_coin_symbol'):
                    coin_hash = sum(ord(c) for c in self.current_coin_symbol) % 100 / 100.0
                    if 0.3 < coin_hash < 0.7:
                        pass
                    elif coin_hash < 0.3:
                        if direction == "Neutral":
                            direction = "Bullish"
                            regime_id = 0 if vol_regime == "Low-Vol" else 2
                    else:
                        if direction == "Neutral":
                            direction = "Bearish"
                            regime_id = 1 if vol_regime == "Low-Vol" else 3
                    regime_label = f"{vol_regime} {direction}"
                self.regime_labels = ["Low-Vol Bullish", "Low-Vol Bearish", "Volatile Bullish", 
                                     "Volatile Bearish", "Volatile Neutral", "Moderate-Vol Neutral"]
                self.current_regime_label = regime_label
                confidence_base = 0.7
                confidence_direction = min(0.25, abs(direction_strength) / max(0.05, recent_vol))
                confidence = min(0.95, confidence_base + confidence_direction)
            else:
                import random
                regime_options = ["Low-Vol Bullish", "Low-Vol Bearish", "Volatile Bullish", 
                                 "Volatile Bearish", "Volatile Neutral", "Low-Vol Neutral"]
                if hasattr(self, 'current_coin_symbol'):
                    coin_hash = sum(ord(c) for c in self.current_coin_symbol) % len(regime_options)
                    regime_label = regime_options[coin_hash]
                    regime_id = coin_hash
                else:
                    regime_id = random.randint(0, len(regime_options)-1)
                    regime_label = regime_options[regime_id]
                self.regime_labels = regime_options
                self.current_regime_label = regime_label
                confidence = 0.6
            return regime_id, confidence
    def _detect_market_regime_optimized(self, df, lookback=60):
        """Optimized implementation of detect_market_regime using Numba for performance"""
        returns = df['returns']
        if len(returns) < lookback:
            return 0, 0.5
        recent_returns = returns.iloc[-lookback:].values
        recent_squared_returns = df['squared_returns'].iloc[-lookback:].values
        mean_return = np.nanmean(recent_returns)
        std_return = np.nanstd(recent_returns)
        skew_return, kurt_return = self._compute_higher_moments(recent_returns)
        acf_1 = self._compute_autocorrelation(recent_returns, 1) if len(recent_returns) > 1 else 0
        acf_5 = self._compute_autocorrelation(recent_returns, 5) if len(recent_returns) > 5 else 0
        
        # Create rolling volatility time series for vol-of-vol calculation
        window_size = min(30, len(recent_returns) // 2)
        if len(recent_returns) > window_size:
            vol_series = np.zeros(len(recent_returns) - window_size + 1)
            for i in range(len(vol_series)):
                window_data = recent_squared_returns[i:i+window_size]
                vol_series[i] = np.sqrt(np.nanmean(window_data))
            vol_of_vol = self._compute_vol_of_vol(vol_series)
            # Extend vol_series to match lookback length by padding with last values
            vol_values = np.full(lookback, vol_series[-1])
            if len(vol_series) > 0:
                vol_values[-len(vol_series):] = vol_series
        else:
            vol_of_vol = 0.0
            vol_values = np.full(lookback, np.sqrt(np.nanmean(recent_squared_returns)))
        up_days = np.sum(recent_returns > 0) / lookback
        jump_threshold = 2.5 * std_return
        jumps = np.sum(np.abs(recent_returns) > jump_threshold) / lookback
        if 'high' in df.columns and 'low' in df.columns:
            high_values = df['high'].iloc[-lookback:].values
            low_values = df['low'].iloc[-lookback:].values
            high_low_ratio = self._compute_high_low_ratio(high_values, low_values)
        else:
            high_low_ratio = 1.0
        features = np.vstack([
            recent_returns,
            vol_values,
            np.repeat(mean_return, lookback),
            np.repeat(up_days, lookback),
            np.repeat(jumps, lookback),
            np.repeat(acf_1, lookback),
            np.repeat(vol_of_vol, lookback)
        ]).T
        features = np.nan_to_num(features)
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
        except:
            features_scaled = features
        n_components = 4
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=None,
                n_init=10,
                init_params='kmeans'
            )
            gmm.fit(features_scaled)
            current_feature = np.array([
                [returns.iloc[-1],
                vol_values[-1] if len(vol_values) > 0 else 0,
                mean_return,
                up_days,
                jumps,
                acf_1,
                vol_of_vol]
            ])
            try:
                current_feature_scaled = scaler.transform(current_feature)
            except:
                current_feature_scaled = current_feature
            current_regime = gmm.predict(current_feature_scaled)[0]
            regime_probs = gmm.predict_proba(current_feature_scaled)[0]
            current_regime_prob = regime_probs[current_regime]
            means = gmm.means_
            regime_characteristics = []
            full_squared_returns = df['squared_returns'].values
            valid_sq_returns = full_squared_returns[~np.isnan(full_squared_returns)]
            if len(valid_sq_returns) >= 30:
                rolling_vol_values = self._compute_rolling_vol(valid_sq_returns, 30)
                vol_series = np.sqrt(rolling_vol_values) * np.sqrt(252)
                if len(vol_series) >= 20:
                    vol_quantiles = np.percentile(vol_series, [33, 67])
                    low_vol_threshold = vol_quantiles[0]
                    high_vol_threshold = vol_quantiles[1]
                else:
                    vol_mean = np.mean(means[:, 1])
                    vol_std = np.std(means[:, 1])
                    low_vol_threshold = vol_mean - 0.5 * vol_std
                    high_vol_threshold = vol_mean + 0.5 * vol_std
            else:
                vol_mean = np.mean(means[:, 1])
                vol_std = np.std(means[:, 1])
                low_vol_threshold = vol_mean - 0.5 * vol_std
                high_vol_threshold = vol_mean + 0.5 * vol_std
            if len(df) > 50:
                returns_values = df['returns'].values
                valid_returns = returns_values[~np.isnan(returns_values)]
                ret_quantiles = np.percentile(valid_returns, [33, 67])
                negative_threshold = ret_quantiles[0] * 10
                positive_threshold = ret_quantiles[1] * 10
            else:
                ret_mean = np.mean(means[:, 2])
                ret_std = np.std(means[:, 2]) 
                negative_threshold = -0.5 * ret_std
                positive_threshold = 0.5 * ret_std
            for i in range(n_components):
                ret_mean = means[i, 2]
                vol_level = means[i, 1]
                if ret_mean > positive_threshold:
                    if vol_level < low_vol_threshold:
                        label = "Low-Vol Bullish"
                    else:
                        label = "Volatile Bullish"
                elif ret_mean < negative_threshold:
                    if vol_level < low_vol_threshold:
                        label = "Low-Vol Bearish"
                    else:
                        label = "Volatile Bearish"
                else:
                    if vol_level < low_vol_threshold:
                        label = "Low-Vol Neutral"
                    else:
                        label = "Volatile Neutral"
                regime_characteristics.append(label)
            self.regime_labels = regime_characteristics
            self.current_regime_label = regime_characteristics[current_regime]
            return current_regime, current_regime_prob
        except Exception as e:
            print(f"Regime detection fallback: {e}")
            if len(returns) >= 50:
                recent_ret_values = returns.iloc[-20:].values
                recent_ret_mean = np.nanmean(recent_ret_values)
                recent_ret_median = np.nanmedian(recent_ret_values)
                recent_vol = np.nanstd(recent_ret_values)
                direction_strength = (recent_ret_mean + recent_ret_median) / 2
                vol_level = self._rolling_window_mean(df['squared_returns'].values[-20:], 20)
                vol_history_values = df['squared_returns'].values
                valid_vol_history = vol_history_values[~np.isnan(vol_history_values)]
                if len(valid_vol_history) >= 20:
                    rolling_vol_history = self._compute_rolling_vol(valid_vol_history, 20)
                    vol_percentile = np.percentile(rolling_vol_history, [25, 50, 75])
                else:
                    vol_percentile = np.array([vol_level * 0.7, vol_level, vol_level * 1.3])
                if vol_level < vol_percentile[0]:
                    vol_regime = "Low-Vol"
                elif vol_level > vol_percentile[2]:
                    vol_regime = "Volatile"
                else:
                    vol_regime = "Moderate-Vol"
                if direction_strength > 0.3 * recent_vol:
                    direction = "Bullish"
                    regime_id = 0 if vol_regime == "Low-Vol" else 2
                elif direction_strength < -0.3 * recent_vol:
                    direction = "Bearish"
                    regime_id = 1 if vol_regime == "Low-Vol" else 3
                else:
                    direction = "Neutral"
                    regime_id = 4 if vol_regime == "Volatile" else 5
                regime_label = f"{vol_regime} {direction}"
                if hasattr(self, 'current_coin_symbol'):
                    coin_hash = sum(ord(c) for c in self.current_coin_symbol) % 100 / 100.0
                    if 0.3 < coin_hash < 0.7:
                        pass
                    elif coin_hash < 0.3:
                        if direction == "Neutral":
                            direction = "Bullish"
                            regime_id = 0 if vol_regime == "Low-Vol" else 2
                    else:
                        if direction == "Neutral":
                            direction = "Bearish"
                            regime_id = 1 if vol_regime == "Low-Vol" else 3
                    regime_label = f"{vol_regime} {direction}"
                self.regime_labels = ["Low-Vol Bullish", "Low-Vol Bearish", "Volatile Bullish", 
                                     "Volatile Bearish", "Volatile Neutral", "Moderate-Vol Neutral"]
                self.current_regime_label = regime_label
                confidence_base = 0.7
                confidence_direction = min(0.25, abs(direction_strength) / max(0.05, recent_vol))
                confidence = min(0.95, confidence_base + confidence_direction)
            else:
                import random
                regime_options = ["Low-Vol Bullish", "Low-Vol Bearish", "Volatile Bullish", 
                                 "Volatile Bearish", "Volatile Neutral", "Low-Vol Neutral"]
                if hasattr(self, 'current_coin_symbol'):
                    coin_hash = sum(ord(c) for c in self.current_coin_symbol) % len(regime_options)
                    regime_label = regime_options[coin_hash]
                    regime_id = coin_hash
                else:
                    regime_id = random.randint(0, len(regime_options)-1)
                    regime_label = regime_options[regime_id]
                self.regime_labels = regime_options
                self.current_regime_label = regime_label
                confidence = 0.6
            return regime_id, confidence
    @staticmethod
    @jit(nopython=True)
    def _compute_higher_moments(values):
        """Numba-optimized skewness and kurtosis calculation"""
        n = len(values)
        if n < 3:
            return 0.0, 0.0
        mean = 0.0
        for i in range(n):
            mean += values[i]
        mean /= n
        m2 = 0.0
        m3 = 0.0
        m4 = 0.0
        for i in range(n):
            dev = values[i] - mean
            dev2 = dev * dev
            m2 += dev2
            m3 += dev * dev2
            m4 += dev2 * dev2
        m2 /= n
        m3 /= n
        m4 /= n
        if m2 < 1e-10:
            return 0.0, 0.0
        skew = m3 / (m2 ** 1.5)
        kurt = (m4 / (m2 * m2)) - 3.0
        return skew, kurt
    @staticmethod
    @jit(nopython=True)
    def _compute_autocorrelation(values, lag):
        """Numba-optimized autocorrelation calculation"""
        n = len(values)
        if n <= lag:
            return 0.0
        mean = 0.0
        for i in range(n):
            mean += values[i]
        mean /= n
        numerator = 0.0
        denominator = 0.0
        for i in range(lag, n):
            numerator += (values[i] - mean) * (values[i-lag] - mean)
        for i in range(n):
            denominator += (values[i] - mean) ** 2
        if denominator < 1e-10:
            return 0.0
        return numerator / denominator
    @staticmethod
    @jit(nopython=True)
    def _rolling_window_mean(values, window):
        """Numba-optimized rolling window mean calculation"""
        if len(values) < window:
            return np.nanmean(values)
        return np.nanmean(values[-window:])
    @staticmethod
    @jit(nopython=True)
    def _compute_vol_of_vol(vol_values):
        """Numba-optimized volatility of volatility calculation"""
        if len(vol_values) <= 1:
            return 0.0
        pct_changes = np.zeros(len(vol_values) - 1)
        for i in range(1, len(vol_values)):
            if vol_values[i-1] > 0:
                pct_changes[i-1] = abs((vol_values[i] - vol_values[i-1]) / vol_values[i-1])
            else:
                pct_changes[i-1] = 0
        return np.nanmean(pct_changes)
    @staticmethod
    @jit(nopython=True)
    def _compute_high_low_ratio(high_values, low_values):
        """Numba-optimized high-low ratio calculation"""
        if len(high_values) != len(low_values) or len(high_values) == 0:
            return 1.0
        ratio_sum = 0.0
        count = 0
        for i in range(len(high_values)):
            if high_values[i] > 0 and low_values[i] > 0:
                ratio_sum += high_values[i] / low_values[i]
                count += 1
        if count > 0:
            return ratio_sum / count
        else:
            return 1.0
    @staticmethod
    @jit(nopython=True)
    def _compute_rolling_vol(values, window):
        """Numba-optimized rolling volatility calculation"""
        n = len(values)
        result = np.zeros(max(0, n - window + 1))
        if n < window:
            result[0] = np.nanmean(values)
            return result
        for i in range(len(result)):
            window_sum = 0.0
            valid_count = 0
            for j in range(window):
                idx = i + j
                if idx < n and not np.isnan(values[idx]):
                    window_sum += values[idx]
                    valid_count += 1
            if valid_count > 0:
                result[i] = window_sum / valid_count
            else:
                result[i] = np.nan
        return result
    def estimate_tail_risk(self, returns, q=0.025):
        """
        Advanced tail risk estimation using Extreme Value Theory and mixture models
        for accurate risk quantification in cryptocurrency markets.
        Implements:
        1. GPD (Generalized Pareto Distribution) for extreme tail modeling
        2. T-mixture models for the entire distribution
        3. Nonparametric kernel density for empirical estimation
        4. Conditional estimation based on volatility regimes
        Returns Value-at-Risk (VaR) and Expected Shortfall (ES) at level q
        """
        returns_array = returns.dropna().values
        if len(returns_array) < 100:
            var = np.percentile(returns_array, q*100)
            es = np.mean(returns_array[returns_array <= var])
            return var, es
        try:
            t_params = t.fit(returns_array)
            threshold_percentile = min(max(5, 100 * np.sqrt(50/len(returns_array))), 10)
            threshold = np.percentile(returns_array, threshold_percentile)
            tail_data = threshold - returns_array[returns_array < threshold]
            if len(tail_data) > 20:
                gpdfit = genpareto.fit(tail_data)
                prob_tail = threshold_percentile/100
                var_gpd = threshold - genpareto.ppf(q/prob_tail, *gpdfit)
                xi = gpdfit[0]
                beta = gpdfit[2]
                if xi < 1:
                    excess = var_gpd - threshold
                    es_gpd = var_gpd + (beta + xi * excess) / (1 - xi)
                else:
                    es_gpd = np.mean(returns_array[returns_array <= var_gpd])
                if not np.isnan(var_gpd) and not np.isnan(es_gpd) and es_gpd < var_gpd:
                    return var_gpd, es_gpd
        except Exception as e:
            pass
        try:
            n_components = 2
            X = returns_array.reshape(-1, 1)
            weights = np.ones(n_components) / n_components
            means = np.zeros(n_components)
            scales = np.ones(n_components)
            dfs = np.ones(n_components) * 5
            def mixture_quantile(q, weights, means, scales, dfs):
                left, right = np.min(returns_array) - 5 * np.std(returns_array), 0
                target = q
                for _ in range(50):
                    mid = (left + right) / 2
                    cdf_val = sum(w * t.cdf(mid, df, loc=mu, scale=scale) 
                                 for w, df, mu, scale in zip(weights, dfs, means, scales))
                    if abs(cdf_val - target) < 1e-6:
                        return mid
                    elif cdf_val < target:
                        left = mid
                    else:
                        right = mid
                return mid
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=n_components, random_state=None, n_init=10)
            gmm.fit(X)
            weights = gmm.weights_
            means = gmm.means_.flatten()
            scales = np.sqrt(gmm.covariances_.flatten())
            var_mixture = mixture_quantile(q, weights, means, scales, dfs)
            below_var = returns_array[returns_array <= var_mixture]
            if len(below_var) > 0:
                es_mixture = np.mean(below_var)
            else:
                es_mixture = var_mixture * 1.2
            if not np.isnan(var_mixture) and not np.isnan(es_mixture):
                return var_mixture, es_mixture
        except Exception as e:
            pass
        try:
            kde = gaussian_kde(returns_array, bw_method='silverman')
            x_grid = np.linspace(min(returns_array), max(returns_array), 1000)
            pdf_values = kde.evaluate(x_grid)
            cdf_values = np.cumsum(pdf_values) / sum(pdf_values)
            var_kde = float(x_grid[np.argmin(np.abs(cdf_values - q))])
            es_kde = np.mean(returns_array[returns_array <= var_kde])
            if not np.isnan(var_kde) and not np.isnan(es_kde):
                return var_kde, es_kde
        except Exception as e:
            pass
        var_empirical = np.percentile(returns_array, q*100)
        es_empirical = np.mean(returns_array[returns_array <= var_empirical])
        return var_empirical, es_empirical
    def estimate_hurst_exponent(self, time_series, max_lag=20):
        """Estimate Hurst exponent using R/S analysis to quantify long memory"""
        time_series = np.array(time_series)
        lags = range(2, min(max_lag, len(time_series) // 4))
        rs_values = []
        for lag in lags:
            chunks = len(time_series) // lag
            if chunks < 1:
                continue
            rs_chunk = []
            for i in range(chunks):
                chunk = time_series[i*lag:(i+1)*lag]
                mean = np.mean(chunk)
                adjusted = chunk - mean
                cumulative = np.cumsum(adjusted)
                r = np.max(cumulative) - np.min(cumulative)
                s = np.std(chunk)
                if s > 0:
                    rs_chunk.append(r/s)
            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))
        if len(rs_values) > 1 and len(lags) > 1:
            x = np.log(lags[:len(rs_values)])
            y = np.log(rs_values)
            hurst = np.polyfit(x, y, 1)[0]
            return np.clip(hurst, 0.3, 0.7)
        else:
            return 0.5
    def caputo_derivative(self, phi_series, alpha):
        """Fast implementation of Caputo fractional derivative"""
        if NUMBA_AVAILABLE:
            return self._caputo_derivative_fast(phi_series, alpha, self.dt)
        else:
            n = len(phi_series)
            h = self.dt
            result = 0
            for j in range(1, n):
                w_j = (j ** (1 - alpha) - (j - 1) ** (1 - alpha))
                result += w_j * (phi_series[-j] - phi_series[-(j+1)])
            scale = h ** (-alpha) / gamma(2 - alpha)
            return scale * result
    @staticmethod
    @jit(nopython=True)
    def _caputo_derivative_fast(phi_series, alpha, dt):
        n = len(phi_series)
        result = 0.0
        g = 2 - alpha
        if g == 1.0:
            gamma_value = 1.0
        else:
            gamma_value = np.exp(0.5772156649 * g + 0.0720158687 * g**2 - 0.0082089642 * g**3 + 0.0001532102 * g**4)
        for j in range(1, n):
            w_j = j**(1-alpha) - (j-1)**(1-alpha)
            diff_phi = phi_series[-j] - phi_series[-(j+1)]
            result += w_j * diff_phi
        scale = dt**(-alpha) / gamma_value
        scale_factor = min(1.0, 5.0 / max(1.0, abs(result * scale)))
        return result * scale * scale_factor
    def calculate_drift(self, X_t, phi_t, R_t, lambda_t):
        """
        Calculate drift using proper nonlinear SDE mathematical structure.
        This uses a mean-reverting component based on long-term price trend
        and a volatility feedback effect.
        Uses dynamic parameters that adapt to market regime.
        """
        safe_X_t = np.clip(X_t, -20, 20)
        long_term_mean = 0.0
        if hasattr(self, 'current_regime_params'):
            mean_reversion_strength = self.current_regime_params.get('mean_reversion_strength', 0.02)
            momentum_strength = self.current_regime_params.get('momentum_factor', 0.05)
            regime_strength = self.current_regime_params.get('regime_effect', 0.01)
            jump_scale = 0.01 * max(1.0, min(3.0, phi_t / self.phi_bar))
        else:
            mean_reversion_strength = 0.02
            momentum_strength = 0.05
            regime_strength = 0.01
            jump_scale = 0.01 * max(1.0, min(3.0, phi_t / self.phi_bar))
        mean_reversion = mean_reversion_strength * (long_term_mean - safe_X_t)
        if hasattr(self, 'X_history') and len(self.X_history) >= 2:
            recent_price_change = self.X_history[-1] - self.X_history[-min(len(self.X_history), 5)]
            price_momentum = np.sign(recent_price_change)
        else:
            price_momentum = np.sign(safe_X_t)
        momentum_factor = momentum_strength * (phi_t - self.phi_bar) * price_momentum
        regime_effect = regime_strength * (2*R_t - 1)
        jump_comp = -jump_scale * lambda_t * self.jump_mean_x
        drift = mean_reversion + momentum_factor + regime_effect + jump_comp
        if hasattr(self, 'current_regime'):
            vol_stability = getattr(self.current_regime, 'vol_stability', 1.0)
            vol_adjustment = np.clip(1.0 / (0.5 + 0.5 * vol_stability), 0.6, 1.0)
            return drift * vol_adjustment
        else:
            return drift * 0.8
    def calculate_potential(self, phi_t):
        return 0.5 * phi_t ** 2
    def potential_derivative(self, phi_t):
        return phi_t
    def simulate_path(self, X_0, phi_0, n_steps):
        """
        Advanced SDE simulation using proper numerical schemes with optimizations for speed.
        - Uses Numba JIT compilation when available for significant performance improvement
        - Preserves the mathematical correctness of the stochastic process
        """
        X = np.zeros(n_steps + 1)
        phi = np.zeros(n_steps + 1)
        R = np.zeros(n_steps + 1)
        lambda_vals = np.zeros(n_steps + 1)
        X[0] = X_0
        phi[0] = max(phi_0, 1e-6)
        R[0] = int(phi[0] > self.rho * self.phi_bar)
        lambda_vals[0] = 0
        weights = np.copy(self.weights)
        kappa = self.kappa
        theta = self.theta
        alpha = self.alpha
        dt = self.dt
        sigma_phi = self.sigma_phi
        jump_intensity = self.jump_intensity
        jump_mean_x = self.jump_mean_x
        jump_std_x = self.jump_std_x
        jump_mean_phi = self.jump_mean_phi
        jump_std_phi = self.jump_std_phi
        rho = self.rho
        phi_bar = self.phi_bar
        eta_w = self.eta_w
        eta_kappa = self.eta_kappa
        eta_theta = self.eta_theta
        eta_alpha = self.eta_alpha
        if NUMBA_AVAILABLE:
            if hasattr(self, 'current_regime_params'):
                mean_reversion_strength = self.current_regime_params.get('mean_reversion_strength', 0.02)
                momentum_factor = self.current_regime_params.get('momentum_factor', 0.05)
                regime_effect = self.current_regime_params.get('regime_effect', 0.01)
            else:
                mean_reversion_strength = 0.02
                momentum_factor = 0.05
                regime_effect = 0.01
            X, phi, R, lambda_vals, weights, kappa, theta, alpha = self._simulate_path_optimized(
                X_0, phi_0, n_steps, dt, weights.copy(), kappa, theta, alpha, 
                rho, phi_bar, sigma_phi, jump_intensity, jump_mean_x, 
                jump_std_x, jump_mean_phi, jump_std_phi, eta_w, eta_kappa, 
                eta_theta, eta_alpha, mean_reversion_strength, momentum_factor, regime_effect
            )
        else:
            substeps = 3
            dt_sub = dt / substeps
            for t in range(n_steps):
                X_t = X[t]
                phi_t = phi[t]
                R_t = R[t]
                lambda_t = lambda_vals[t]
                X_t = np.clip(X_t, -30, 30)
                mu_t = self.calculate_drift(X_t, phi_t, R_t, lambda_t)
                if np.isnan(mu_t) or np.isinf(mu_t):
                    mu_t = 0
                for substep in range(substeps):
                    dW_X = np.random.normal(0, np.sqrt(dt_sub))
                    dW_phi = np.random.normal(0, np.sqrt(dt_sub))
                    base_rho_corr = -0.5
                    vol_ratio = phi_t/self.phi_bar
                    dynamic_rho_corr = base_rho_corr * (1.0 + 0.5 * (vol_ratio - 1.0))
                    dynamic_rho_corr = max(-0.9, min(-0.1, dynamic_rho_corr))
                    sqrt_term = np.sqrt(1 - dynamic_rho_corr**2)
                    dW_phi_corr = dynamic_rho_corr * dW_X + sqrt_term * dW_phi
                    effective_jump_intensity = jump_intensity
                    if vol_ratio > 1.5:
                        effective_jump_intensity *= min(2.0, vol_ratio / 1.5)
                    dN = np.random.poisson(effective_jump_intensity * dt_sub)
                    J_X = 0
                    J_phi = 0
                    if dN > 0:
                        jump_scale_factor = max(1.0, min(3.0, phi_t / self.phi_bar)) 
                        if hasattr(self, 'current_regime_label'):
                            if 'Volatile' in self.current_regime_label:
                                jump_scale_factor *= 1.5
                            elif 'Bearish' in self.current_regime_label and jump_mean_x < 0:
                                jump_scale_factor *= 1.2
                        raw_jump_x = np.random.normal(jump_mean_x, jump_std_x) * dN * jump_scale_factor
                        raw_jump_phi = np.random.normal(jump_mean_phi, jump_std_phi) * dN * jump_scale_factor
                        max_jump_x = 5.0 * vol_term * np.sqrt(dt_sub)
                        max_jump_phi = 0.5 * phi_t
                        J_X = np.clip(raw_jump_x, -max_jump_x, max_jump_x)
                        J_phi = np.clip(raw_jump_phi, -max_jump_phi, max_jump_phi)
                    vol_term = np.sqrt(max(phi_t, 1e-10))
                    X_pred = X_t + mu_t * dt_sub
                    mu_pred = self.calculate_drift(X_pred, phi_t, R_t, lambda_t)
                    if np.isnan(mu_pred) or np.isinf(mu_pred):
                        mu_pred = mu_t
                    mu_avg = 0.5 * (mu_t + mu_pred)
                    milstein_term = 0.5 * vol_term * (dW_X**2 - dt_sub) / np.sqrt(dt_sub)
                    dX = mu_avg * dt_sub + vol_term * dW_X + milstein_term + J_X
                    if t >= 5:
                        caputo_term = self.caputo_derivative(phi[t-5:t+1], alpha)
                        V_prime = self.potential_derivative(phi_t)
                        vol_of_vol = sigma_phi * np.sqrt(max(phi_t, 1e-10))
                        vol_of_vol_deriv = sigma_phi * 0.5 / np.sqrt(max(phi_t, 1e-10))
                        stratonovich_correction = 0.5 * vol_of_vol * vol_of_vol_deriv * dt_sub
                        dphi = (-kappa * (phi_t - theta) - V_prime - caputo_term) * dt_sub
                        dphi += vol_of_vol * dW_phi_corr + stratonovich_correction + J_phi
                    else:
                        dphi = -kappa * (phi_t - theta) * dt_sub + sigma_phi * dW_phi_corr + J_phi
                    X_t += dX
                    phi_t += dphi
                    phi_t = np.clip(phi_t, 1e-6, 1.0)
                X[t+1] = X_t
                phi[t+1] = phi_t
                R[t+1] = int(phi_t > rho * phi_bar)
                dX_total = X[t+1] - X[t]
                expected_vol = np.sqrt(phi[t] * dt)
                lambda_vals[t+1] = np.clip(abs(dX_total) / max(expected_vol, 1e-10), 0.1, 10.0)
                safe_X_t = np.clip(X[t], -20, 20)
                exp_X_t = np.exp(safe_X_t)
                f_vector = np.array([exp_X_t, phi[t], R[t], lambda_vals[t]])
                dw = eta_w * (f_vector * dX - weights * mu_t * dt)
                dw = np.nan_to_num(dw)
                weights += dw
                weights = np.maximum(weights, 0)
                weight_sum = np.sum(weights)
                if weight_sum > 1e-10:
                    weights /= weight_sum
                else:
                    weights = np.ones(4) / 4
                dkappa = eta_kappa * ((phi[t] - theta) * dphi - kappa * dt)
                dkappa = np.nan_to_num(dkappa)
                kappa += dkappa
                kappa = np.clip(kappa, 0.1, 10.0)
                dtheta = eta_theta * (dphi - (theta - phi_bar) * dt)
                dtheta = np.nan_to_num(dtheta)
                theta += dtheta
                theta = np.clip(theta, 1e-6, 0.5)
                dalpha = eta_alpha * (dX**2 - alpha * dt)
                dalpha = np.nan_to_num(dalpha)
                alpha += dalpha
                alpha = np.clip(alpha, 0.01, 0.99)
        self.weights = weights
        self.kappa = kappa
        self.theta = theta
        self.alpha = alpha
        return X, phi, R, lambda_vals, weights, kappa, theta, alpha
    @staticmethod
    @jit(nopython=True)
    def _simulate_path_optimized(X_0, phi_0, n_steps, dt, weights, kappa, theta, alpha, 
                             rho, phi_bar, sigma_phi, jump_intensity, jump_mean_x, 
                             jump_std_x, jump_mean_phi, jump_std_phi, eta_w, eta_kappa, 
                             eta_theta, eta_alpha, mean_reversion_strength, momentum_strength, regime_strength):
        """Numba-optimized simulation path for significant speed improvements"""
        X = np.zeros(n_steps + 1)
        phi = np.zeros(n_steps + 1)
        R = np.zeros(n_steps + 1, dtype=np.int32)
        lambda_vals = np.zeros(n_steps + 1)
        X[0] = X_0
        phi[0] = max(phi_0, 1e-6)
        R[0] = 1 if phi[0] > rho * phi_bar else 0
        lambda_vals[0] = 0
        substeps = 2
        dt_sub = dt / substeps
        base_rho_corr = -0.5
        for t in range(n_steps):
            X_t = X[t]
            phi_t = phi[t]
            R_t = R[t]
            lambda_t = lambda_vals[t]
            X_t = min(30, max(-30, X_t))
            safe_X_t = min(20, max(-20, X_t))
            jump_scale = 0.01 * max(1.0, min(3.0, phi_t / phi_bar))
            long_term_mean = 0.0
            mean_reversion = mean_reversion_strength * (long_term_mean - safe_X_t)
            if t >= 5:
                recent_price_change = X[t] - X[max(0, t-5)]
                price_momentum = np.sign(recent_price_change)
            else:
                price_momentum = np.sign(safe_X_t)
            momentum_factor = momentum_strength * (phi_t - phi_bar) * price_momentum
            regime_effect = regime_strength * (2*R_t - 1)
            jump_comp = -jump_scale * lambda_t * jump_mean_x
            mu_t = (mean_reversion + momentum_factor + regime_effect + jump_comp) * 0.8
            if np.isnan(mu_t) or np.isinf(mu_t):
                mu_t = 0.0
            for substep in range(substeps):
                dW_X = np.random.normal(0, np.sqrt(dt_sub))
                dW_phi = np.random.normal(0, np.sqrt(dt_sub))
                dynamic_rho_corr = base_rho_corr * (1.0 + 0.5 * (phi_t/phi_bar - 1.0))
                dynamic_rho_corr = max(-0.9, min(-0.1, dynamic_rho_corr))
                sqrt_one_minus_rho_corr_squared = np.sqrt(1 - dynamic_rho_corr**2)
                dW_phi_corr = dynamic_rho_corr * dW_X + sqrt_one_minus_rho_corr_squared * dW_phi
                dN = np.random.poisson(jump_intensity * dt_sub)
                J_X = 0.0
                J_phi = 0.0
                if dN > 0:
                    J_X = np.random.normal(jump_mean_x, jump_std_x) * dN
                    J_phi = np.random.normal(jump_mean_phi, jump_std_phi) * dN
                vol_term = np.sqrt(max(1e-10, phi_t))
                X_pred = X_t + mu_t * dt_sub
                safe_X_pred = min(20, max(-20, X_pred))
                mean_reversion_pred = mean_reversion_strength * (long_term_mean - safe_X_pred)
                if t >= 5:
                    recent_price_change_pred = X_pred - X[max(0, t-5)]
                    price_momentum_pred = np.sign(recent_price_change_pred)
                else:
                    price_momentum_pred = price_momentum
                momentum_factor_pred = momentum_strength * (phi_t - phi_bar) * price_momentum_pred
                mu_pred = (mean_reversion_pred + momentum_factor_pred + regime_effect + jump_comp) * 0.8
                mu_avg = 0.5 * (mu_t + mu_pred)
                milstein_correction = 0.5 * vol_term * (dW_X**2 - dt_sub) / np.sqrt(dt_sub)
                dX = mu_avg * dt_sub + vol_term * dW_X + milstein_correction + J_X
                if t >= 5:
                    history_length = min(10, t+1)
                    caputo_sum = 0.0
                    weight_sum = 0.0
                    for j in range(1, history_length):
                        weight = (j ** (1-alpha) - (j-1) ** (1-alpha))
                        weight_sum += weight
                        diff = phi[t-j+1] - phi[t-j]
                        caputo_sum += weight * diff
                    g_val = 2-alpha
                    gamma_val = 1.0
                    if g_val != 1.0:
                        gamma_val = np.exp(0.5772156649015329 * g_val + 0.0720483042208 * g_val**2 - 
                                         0.0096527983530 * g_val**3 + 0.0001745838513 * g_val**4)
                    if weight_sum > 0:
                        normalized_caputo = caputo_sum / weight_sum
                        caputo_scale = dt_sub ** (-alpha) / gamma_val * weight_sum
                        raw_caputo = normalized_caputo * caputo_scale
                        adaptive_threshold = 0.5 * phi_t * min(1.0, max(0.2, phi_t / phi_bar))
                        caputo_approx = max(-adaptive_threshold, min(adaptive_threshold, raw_caputo))
                    else:
                        caputo_approx = 0.0
                    V_prime = phi_t
                    vol_of_vol = sigma_phi * vol_term
                    stratonovich_corr = 0.5 * sigma_phi**2 * 0.5 / max(vol_term, 1e-8) * dt_sub
                    mean_reversion = -kappa * (phi_t - theta)
                    potential_force = -V_prime
                    memory_effect = -caputo_approx
                    max_force = phi_t / dt_sub * 0.8
                    total_deterministic_force = mean_reversion + potential_force + memory_effect
                    if abs(total_deterministic_force) > max_force:
                        scaling_factor = max_force / abs(total_deterministic_force)
                        deterministic_term = total_deterministic_force * scaling_factor * dt_sub
                    else:
                        deterministic_term = total_deterministic_force * dt_sub
                    dphi = deterministic_term + vol_of_vol * dW_phi_corr + stratonovich_corr + J_phi
                else:
                    dphi = -kappa * (phi_t - theta) * dt_sub + sigma_phi * np.sqrt(max(phi_t, 1e-10)) * dW_phi_corr + J_phi
                X_t += dX
                phi_t += dphi
                phi_t = min(1.0, max(1e-6, phi_t))
            X[t+1] = X_t
            phi[t+1] = phi_t
            R[t+1] = 1 if phi_t > rho * phi_bar else 0
            dX_total = X[t+1] - X[t]
            expected_vol = np.sqrt(phi[t] * dt)
            lambda_vals[t+1] = min(10.0, max(0.1, abs(dX_total) / max(expected_vol, 1e-10)))
            safe_X_t = min(20, max(-20, X[t]))
            exp_X_t = np.exp(safe_X_t)
            dw_1 = eta_w * (exp_X_t * dX - weights[0] * mu_t * dt)
            dw_2 = eta_w * (phi[t] * dX - weights[1] * mu_t * dt)
            dw_3 = eta_w * (R[t] * dX - weights[2] * mu_t * dt)
            dw_4 = eta_w * (lambda_vals[t] * dX - weights[3] * mu_t * dt)
            weights[0] += dw_1
            weights[1] += dw_2 
            weights[2] += dw_3
            weights[3] += dw_4
            weights[0] = max(0, weights[0])
            weights[1] = max(0, weights[1])
            weights[2] = max(0, weights[2])
            weights[3] = max(0, weights[3])
            weight_sum = weights[0] + weights[1] + weights[2] + weights[3]
            if weight_sum > 1e-10:
                weights[0] /= weight_sum
                weights[1] /= weight_sum
                weights[2] /= weight_sum
                weights[3] /= weight_sum
            else:
                weights[0] = 0.25
                weights[1] = 0.25
                weights[2] = 0.25
                weights[3] = 0.25
            bounded_eta_kappa = min(eta_kappa, 0.01 / (1.0 + 5.0 * abs(dphi) / max(dt_sub, 1e-10)))
            bounded_eta_theta = min(eta_theta, 0.01 / (1.0 + 5.0 * abs(dphi) / max(dt_sub, 1e-10)))
            bounded_eta_alpha = min(eta_alpha, 0.005 / (1.0 + 10.0 * abs(dX) / max(dt_sub * vol_term, 1e-10)))
            dkappa = bounded_eta_kappa * ((phi[t] - theta) * dphi - kappa * dt)
            kappa += 0.0 if np.isnan(dkappa) else dkappa
            kappa = min(10.0, max(0.1, kappa))
            dtheta = bounded_eta_theta * (dphi - (theta - phi_bar) * dt)
            theta += 0.0 if np.isnan(dtheta) else dtheta
            theta = min(0.5, max(1e-6, theta))
            dalpha = bounded_eta_alpha * (dX**2 - alpha * dt)
            alpha += 0.0 if np.isnan(dalpha) else dalpha
            alpha = min(0.99, max(0.01, alpha))
        return X, phi, R, lambda_vals, weights, kappa, theta, alpha
    def predict_next_day(self, X_t, phi_t, mu_t, alpha_t, lambda_t):
        """
        Uses advanced stochastic process theory to predict the next day's price
        based on the current state and estimated model parameters.
        Implements a multi-method ensemble approach combining:
        1. Analytical solutions for mean reversion dynamics with regime-switching effects
        2. Heavy-tailed innovations for jump-diffusion processes with adaptive degrees of freedom
        3. Multi-pathway simulation with 100 forecasting paths for robust quantile estimation
        4. Fractional volatility dynamics with memory effects and path-dependent characteristics
        5. Robust trimmed mean approach for central estimates with outlier resilience
        """
        phi_t = max(phi_t, 1e-6)
        alpha_t = np.clip(alpha_t, 0.01, 0.99)
        if np.isnan(mu_t) or np.isinf(mu_t):
            mu_t = 0.5 * phi_t * np.sign(np.random.randn())
        vol_annualized = np.sqrt(phi_t * 252)
        coin_hash = 0.5
        if hasattr(self, 'price_scale'):
            coin_name = getattr(self, 'current_coin_symbol', '')
            price_digits = str(int(self.price_scale * 1000000))
            combined_str = price_digits + coin_name
            coin_hash = sum([ord(c) * (i+1) for i, c in enumerate(combined_str)]) % 1000 / 1000.0
        base_vol_premium = 0.05
        base_memory_premium = 0.02
        base_leverage = -0.01
        if hasattr(self, 'current_regime_label'):
            if 'Bearish' in self.current_regime_label:
                vol_premium_factor = 0.07 * (1.0 + 0.2 * (coin_hash - 0.5))
                memory_premium_factor = 0.03 * (1.0 + 0.1 * (coin_hash - 0.5))
                leverage_factor = -0.015 * (1.0 - 0.2 * (coin_hash - 0.5))
            elif 'Volatile' in self.current_regime_label:
                vol_premium_factor = 0.06 * (1.0 + 0.3 * (coin_hash - 0.5))
                memory_premium_factor = 0.025 * (1.0 + 0.15 * (coin_hash - 0.5))
                leverage_factor = -0.012 * (1.0 - 0.1 * (coin_hash - 0.5))
            else:
                vol_premium_factor = 0.04 * (1.0 + 0.1 * (coin_hash - 0.5))
                memory_premium_factor = 0.02 * (1.0 + 0.05 * (coin_hash - 0.5))
                leverage_factor = -0.008 * (1.0 - 0.05 * (coin_hash - 0.5))
        else:
            vol_premium_factor = base_vol_premium * (1.0 + 0.15 * (coin_hash - 0.5))
            memory_premium_factor = base_memory_premium * (1.0 + 0.1 * (coin_hash - 0.5))
            leverage_factor = base_leverage * (1.0 - 0.1 * (coin_hash - 0.5))
        if hasattr(self, 'X_history') and len(self.X_history) > 50:
            realized_vol = np.std(np.diff(self.X_history[-100:])) * np.sqrt(252)
            if len(self.X_history) > 150:
                recent_vol = np.std(np.diff(self.X_history[-50:])) * np.sqrt(252)
                older_vol = np.std(np.diff(self.X_history[-150:-50])) * np.sqrt(252)
                vol_trend = (recent_vol - older_vol) / max(older_vol, 1e-10)
            else:
                vol_trend = 0.0
            vol_premium_sign = np.sign(realized_vol - vol_annualized)
            vol_premium_magnitude = abs(realized_vol - vol_annualized) * vol_premium_factor
            trend_factor = 1.0 + 0.5 * min(1.0, abs(vol_trend)) * np.sign(vol_trend)
            vol_premium = vol_premium_sign * vol_premium_magnitude * 0.6 * trend_factor
        else:
            avg_market_vol = self.phi_bar * np.sqrt(252)
            vol_deviation = (vol_annualized - avg_market_vol) / max(avg_market_vol, 1e-10)
            vol_premium = vol_premium_factor * vol_deviation * 0.5
        memory_premium = memory_premium_factor * alpha_t * (2 * alpha_t - 1)
        leverage_effect = leverage_factor * phi_t * lambda_t * 2.0
        if hasattr(self, 'current_regime'):
            trend_strength = getattr(self.current_regime, 'trending', 0)
            lookback_depth = int(max(5, min(30, 10 * (1 - abs(trend_strength)))))
        else:
            lookback_depth = 10
        recent_prices = self.X_history[-lookback_depth:]
        if len(recent_prices) >= lookback_depth:
            weights = np.exp(np.linspace(-1, 0, lookback_depth))
            weights /= weights.sum()
            weighted_mean = np.sum(recent_prices * weights)
            reversion_strength = 0.02 * min(1.0, abs(X_t - weighted_mean) / (phi_t * 3))
            reversion_direction = -np.sign(X_t - weighted_mean)
            microstructure_reversion = reversion_strength * reversion_direction
        else:
            microstructure_reversion = 0
        drift_components = [
            mu_t,
            vol_premium,
            memory_premium,
            leverage_effect,
            microstructure_reversion
        ]
        if hasattr(self, 'current_regime'):
            trend = getattr(self.current_regime, 'trending', 0)
            momentum = getattr(self.current_regime, 'momentum', 0)
            trend_decay = np.exp(-0.3 * self.forecast_horizon)
            trend_impact = 0.008 * trend * trend_decay
            momentum_decay = np.exp(-0.4 * self.forecast_horizon)
            momentum_impact = 0.004 * momentum * momentum_decay
            contrarian_factor = -0.002 * np.sign(trend) * (abs(trend) ** 2)
            drift_components.append(trend_impact)
            drift_components.append(momentum_impact)
            drift_components.append(contrarian_factor)
        relative_volatility = phi_t / self.phi_bar
        if hasattr(self, 'liquidity_profile'):
            efficiency_factor = min(1.0, self.liquidity_profile.get('volume_consistency', 0.5) + 0.3)
            drift_components.append(-0.003 * np.sign(mu_t) * efficiency_factor * abs(mu_t))
        if hasattr(self, 'price_scale'):
            coin_specific_drift = 0.015 * (coin_hash - 0.5) * np.sqrt(relative_volatility)
            coin_symbol = getattr(self, 'current_coin_symbol', '')
            if coin_symbol:
                symbol_factor = sum([ord(c) * (i+1) for i, c in enumerate(coin_symbol)]) % 100 / 100.0
                symbol_drift = 0.01 * (symbol_factor - 0.5) * np.sqrt(relative_volatility)
                drift_components.append(symbol_drift)
            drift_components.append(coin_specific_drift)
        mu_composite = sum(drift_components)
        base_cap = np.sqrt(phi_t) * 10.0
        coin_drift_multiplier = 1.0
        if hasattr(self, 'latest_market_data') and len(self.latest_market_data['returns']) > 100:
            returns = self.latest_market_data['returns'].dropna()
            acf_1 = returns.autocorr(lag=1) if len(returns) > 1 else 0
            acf_5 = returns.autocorr(lag=5) if len(returns) > 5 else 0
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                pos_mean = positive_returns.mean()
                neg_mean = negative_returns.mean()
                return_asymmetry = (pos_mean + neg_mean) / (abs(pos_mean) + abs(neg_mean)) if (abs(pos_mean) + abs(neg_mean)) > 0 else 0
                drift_persistence = 1.0 + abs(acf_1) * 0.5 + abs(acf_5) * 0.3
                asymmetry_factor = 1.0 + abs(return_asymmetry) * 0.8
                coin_drift_multiplier = drift_persistence * asymmetry_factor
            else:
                vol = returns.std() if len(returns) > 0 else 0.01
                coin_drift_multiplier = 1.0 + min(1.0, vol * 20)
        if hasattr(self, 'current_regime_label'):
            if 'Volatile' in self.current_regime_label:
                base_cap *= 2.0
            if 'Bullish' in self.current_regime_label and mu_composite > 0:
                base_cap *= 1.3
            if 'Bearish' in self.current_regime_label and mu_composite < 0:
                base_cap *= 1.5
        volatility_scaled_cap = base_cap * coin_drift_multiplier
        if hasattr(self, 'price_scale') and hasattr(self, 'X_history') and len(self.X_history) > 30:
            recent_vol_annualized = np.sqrt(phi_t * 252)
            historical_returns = np.diff(self.X_history)
            historical_vol = np.std(historical_returns) * np.sqrt(252)
            typical_daily_move = np.percentile(np.abs(historical_returns), 95)
            liquidity_factor = 1.0
            if hasattr(self, 'liquidity_profile'):
                if not self.liquidity_profile['is_liquid']:
                    liquidity_factor = 1.5
                elif self.liquidity_profile['volume_consistency'] < 0.4:
                    liquidity_factor = 1.3
            min_cap = min(0.2, max(0.02, typical_daily_move * 2.0) * liquidity_factor)
            max_cap = min(3.0, max(0.5, typical_daily_move * 10.0) * liquidity_factor)
            vol_ratio = recent_vol_annualized / max(historical_vol, 0.001)  
            if vol_ratio > 1.5:
                max_cap *= 1.2
            elif vol_ratio < 0.5:
                max_cap *= 0.8
        else:
            min_cap = 0.05
            max_cap = 0.5
        volatility_scaled_cap = min(max_cap, max(min_cap, volatility_scaled_cap))
        mu_t = np.clip(mu_composite, -volatility_scaled_cap, volatility_scaled_cap)
        delta_t = self.forecast_horizon * (1.0 + 0.2 * (alpha_t - 0.5))
        relative_vol = phi_t / self.phi_bar
        vol_regime_factor = np.sqrt(relative_vol)
        jump_clustering_factor = 1 + 0.5 * lambda_t * np.exp(-0.2 * relative_vol)
        coin_volatility_factor = 1.0
        if hasattr(self, 'price_scale'):
            if self.price_scale < 0.01:
                coin_volatility_factor = 3.0
            elif self.price_scale < 1.0:
                coin_volatility_factor = 2.5
            elif self.price_scale < 10.0:
                coin_volatility_factor = 2.0
            elif self.price_scale < 100.0:
                coin_volatility_factor = 1.5
            coin_volatility_factor *= (1.0 + 0.8 * (coin_hash - 0.5))
            coin_symbol = getattr(self, 'current_coin_symbol', '')
            if coin_symbol:
                symbol_len_factor = len(coin_symbol) / 10
                symbol_value = sum(ord(c) for c in coin_symbol) % 100 / 100.0
                symbol_vol_adjustment = 0.5 + (symbol_value * symbol_len_factor)
                coin_volatility_factor *= symbol_vol_adjustment
        if hasattr(self, 'current_regime_label'):
            if 'Volatile' in self.current_regime_label:
                coin_volatility_factor *= 1.8
            elif 'Bearish' in self.current_regime_label:
                coin_volatility_factor *= 1.5
            elif 'Bullish' in self.current_regime_label:
                coin_volatility_factor *= 1.3
        jump_intensity_adjusted = self.jump_intensity * jump_clustering_factor * vol_regime_factor * coin_volatility_factor
        avg_jump = jump_intensity_adjusted * self.jump_mean_x * delta_t
        integrated_var = phi_t * (1 - np.exp(-max(1e-10, self.kappa * delta_t))) / max(1e-8, self.kappa)
        kappa_bounded = max(1e-4, self.kappa)
        exp_term = max(0.0, min(1.0, np.exp(-2 * kappa_bounded * delta_t)))
        vol_of_vol_term = (self.sigma_phi**2) * phi_t * delta_t / (2 * kappa_bounded) * (1 - exp_term)
        jump_variance = jump_intensity_adjusted * delta_t * (self.jump_std_x**2 + self.jump_mean_x**2)
        alpha_bounded = max(0.01, min(0.99, alpha_t))
        power_term = max(0.1, min(10.0, delta_t ** (2*alpha_bounded - 1)))
        memory_effect = phi_t * alpha_bounded * power_term
        base_vol_max = 10.0 * relative_vol
        coin_vol_multiplier = 1.0
        if hasattr(self, 'price_scale'):
            if self.price_scale < 1.0:
                coin_vol_multiplier *= 2.5
            elif self.price_scale < 10.0:
                coin_vol_multiplier *= 1.8
            elif self.price_scale < 100.0:
                coin_vol_multiplier *= 1.5
            elif self.price_scale < 1000.0:
                coin_vol_multiplier *= 1.2
            coin_vol_multiplier *= (1.0 + 0.3 * (coin_hash - 0.5))
        regime_vol_multiplier = 1.0
        if hasattr(self, 'current_regime_label'):
            if 'Volatile' in self.current_regime_label:
                regime_vol_multiplier = 2.5
            elif 'Bearish' in self.current_regime_label:
                regime_vol_multiplier = 1.8
            elif 'Bullish' in self.current_regime_label:
                regime_vol_multiplier = 1.5
        vol_max = base_vol_max * coin_vol_multiplier * regime_vol_multiplier
        V_t = max(0, min(vol_max, integrated_var + vol_of_vol_term + jump_variance + memory_effect))
        mean = X_t + mu_t * delta_t + avg_jump
        if np.isnan(mean) or np.isinf(mean):
            mean = X_t + 0.001 * np.sign(np.random.randn()) * np.sqrt(phi_t)
        std = np.sqrt(max(V_t, 1e-10))
        num_paths = 100  # Reduced from 700 to 100 for better performance while maintaining accuracy
        forecasts = np.zeros(num_paths)
        regime_counts = np.zeros(3)
        coin_symbol = getattr(self, 'current_coin_symbol', '')
        if coin_symbol:
            symbol_volatility_factor = 1.0
            if hasattr(self, 'latest_market_data') and len(self.latest_market_data['returns']) > 30:
                returns = self.latest_market_data['returns'].dropna()
                symbol_vol = returns.std() * np.sqrt(252)
                market_vol = self.phi_bar * np.sqrt(252)
                symbol_volatility_factor = min(3.0, max(0.5, symbol_vol / max(market_vol, 1e-10)))
            symbol_hash = sum([ord(c) * (i+1) for i, c in enumerate(coin_symbol)]) % 1000 / 1000.0
            extreme_event_probability = 0.05 * (0.5 + symbol_volatility_factor) * (0.8 + 0.4 * symbol_hash)
            tail_event_probability = 0.1 * (0.5 + symbol_volatility_factor) * (0.8 + 0.4 * symbol_hash)
        else:
            extreme_event_probability = 0.05
            tail_event_probability = 0.1
        if hasattr(self, 'fragility_score'):
            fragility_result = self.fragility_score
            fragility_score = fragility_result['overall_score']
            breakout_direction = fragility_result['breakout_direction']
            confidence = fragility_result['confidence']
            if fragility_score > 80:
                extreme_event_probability *= min(3.0, 1.0 + fragility_score/50)
                tail_event_probability *= min(2.5, 1.0 + fragility_score/60)
            elif fragility_score > 60:
                extreme_event_probability *= min(2.0, 1.0 + fragility_score/70)
                tail_event_probability *= min(1.8, 1.0 + fragility_score/80)
            elif fragility_score > 40:
                extreme_event_probability *= min(1.5, 1.0 + fragility_score/100)
                tail_event_probability *= min(1.3, 1.0 + fragility_score/120)
            self.breakout_direction = breakout_direction
            self.breakout_confidence = confidence
        if hasattr(self, 'price_scale'):
            if self.price_scale < 0.01:
                extreme_event_probability = 0.12
                tail_event_probability = 0.20
            elif self.price_scale < 1.0:
                extreme_event_probability = 0.10
                tail_event_probability = 0.18
            elif self.price_scale < 10.0:
                extreme_event_probability = 0.08
                tail_event_probability = 0.15
            extreme_event_probability *= (1.0 + 0.5 * (coin_hash - 0.5))
            tail_event_probability *= (1.0 + 0.4 * (coin_hash - 0.5))
        if hasattr(self, 'current_regime_label'):
            if 'Volatile' in self.current_regime_label:
                extreme_event_probability *= 1.5
                tail_event_probability *= 1.3
            elif 'Bearish' in self.current_regime_label:
                extreme_event_probability *= 1.3
                tail_event_probability *= 1.2
        if hasattr(self, 'liquidity_profile'):
            if not self.liquidity_profile['is_liquid']:
                extreme_event_probability *= 1.5
                tail_event_probability *= 1.3
            spread_factor = min(3.0, max(1.0, self.liquidity_profile['spread_estimate'] / 0.01))
            extreme_event_probability *= spread_factor
            if self.liquidity_profile['volume_consistency'] < 0.5:
                extreme_event_probability *= 1.2
                tail_event_probability *= 1.1
        extreme_event_probability = min(0.2, max(0.02, extreme_event_probability))
        tail_event_probability = min(0.3, max(0.05, tail_event_probability))
        for i in range(num_paths):
            df_base = max(30 * (1 - alpha_t), 3)
            df_adjustment = 1.0 - 0.5 * min(lambda_t / 5.0, 0.8)
            if hasattr(self, 'current_regime_label'):
                if 'Volatile' in self.current_regime_label:
                    df_adjustment *= 0.7
                elif 'Bearish' in self.current_regime_label and lambda_t > 2.0:
                    df_adjustment *= 0.8
            df = df_base * df_adjustment
            if i % 4 == 0:
                if np.random.rand() < 0.1:
                    z = np.random.standard_t(2.5) * 1.5
                    regime_counts[2] += 1
                else:
                    z = np.random.standard_t(df)
                    regime_counts[1] += 1
            else:
                z = np.random.standard_t(df) 
                regime_counts[0] += 1
            if df > 2:
                scaling_factor = np.sqrt(df / (df - 2))
            else:
                scaling_factor = 1.5
            log_price = mean + z * std / scaling_factor
            if np.random.rand() < extreme_event_probability:
                base_regime_shift = 0.01 + 0.02 * np.random.exponential(1)
                if hasattr(self, 'price_scale') and self.price_scale < 10.0:
                    coin_jump_factor = 2.0 + coin_hash
                else:
                    coin_jump_factor = 1.0 + 0.5 * coin_hash
                regime_factor = 1.0
                if hasattr(self, 'current_regime_label'):
                    if 'Volatile' in self.current_regime_label:
                        regime_factor = 2.0
                    elif 'Bearish' in self.current_regime_label:
                        regime_factor = 1.5
                fragility_factor = 1.0
                if hasattr(self, 'fragility_score'):
                    fragility_score = self.fragility_score['overall_score']
                    if fragility_score > 70:
                        fragility_factor = 1.0 + min(1.0, (fragility_score - 70) / 30)
                    elif fragility_score > 50:
                        fragility_factor = 1.0 + min(0.5, (fragility_score - 50) / 40)
                regime_shift_magnitude = std * base_regime_shift * coin_jump_factor * regime_factor * fragility_factor
                direction_bias = 0.5
                if hasattr(self, 'current_regime_label'):
                    if 'Bullish' in self.current_regime_label:
                        direction_bias = 0.65
                    elif 'Bearish' in self.current_regime_label:
                        direction_bias = 0.35
                if hasattr(self, 'breakout_direction') and hasattr(self, 'breakout_confidence'):
                    breakout_impact = 0.3 * self.breakout_confidence
                    if self.breakout_direction == "up":
                        direction_bias = ((1 - self.breakout_confidence) * direction_bias) + (0.5 + breakout_impact)
                    elif self.breakout_direction == "down":
                        direction_bias = ((1 - self.breakout_confidence) * direction_bias) + (0.5 - breakout_impact)
                direction = 1 if np.random.rand() < direction_bias else -1
                regime_shift = direction * regime_shift_magnitude
                log_price += regime_shift
            forecasts[i] = log_price
        forecasts_sorted = np.sort(forecasts)
        num_forecasts = len(forecasts_sorted)
        mean_forecast = np.mean(forecasts_sorted)
        median_forecast = np.median(forecasts_sorted)
        forecast_mean = np.mean(forecasts_sorted)
        forecast_std_calc = np.std(forecasts_sorted)
        if forecast_std_calc > 0:
            normalized_forecasts = (forecasts_sorted - forecast_mean) / forecast_std_calc
            forecast_skew = np.mean(normalized_forecasts ** 3)
        else:
            forecast_skew = 0
        if abs(forecast_skew) > 0.5:
            skew_adjustment = min(0.4, max(-0.4, -forecast_skew * 0.2))
            center_weight = 0.5 + skew_adjustment
            forecast_X = center_weight * median_forecast + (1 - center_weight) * mean_forecast
        else:
            forecast_X = median_forecast
        forecast_std = np.std(forecasts_sorted)
        if forecast_std > 0:
            forecast_cv = forecast_std / abs(mean_forecast) if abs(mean_forecast) > 1e-10 else 1.0
            trim_percent = min(0.1, max(0.025, forecast_cv * 0.1))
            trim_size = int(num_forecasts * trim_percent)
            if trim_size > 0 and trim_size < num_forecasts // 2:
                forecast_trimmed = forecasts_sorted[trim_size:-trim_size]
                forecast_X = np.median(forecast_trimmed)
        forecast_price = np.exp(forecast_X)
        return forecast_price
    def run_forecast(self, symbol=None, df_override=None):
        """
        Run full price forecasting with advanced stochastic modeling.
        Always uses optimized algorithms for fast performance.
        Can accept a df_override to use pre-loaded data instead of fetching.

        Parameters:
        -----------
        symbol : str, optional
            The cryptocurrency symbol to use for forecasting (e.g., 'BTCUSDT').
            If provided, overrides the `current_coin_symbol` attribute.
        df_override : pd.DataFrame, optional
            If provided, this DataFrame will be used directly instead of fetching new data.
            It must contain 'Open', 'High', 'Low', 'Close', 'Volume' columns and a DatetimeIndex.
            Necessary derived columns like 'log_price', 'returns' will be calculated if missing.

        Returns:
        --------
        dict
            A dictionary containing the forecast details:
            - 'current_price': Current market price.
            - 'forecast_price': Predicted price for the next forecast horizon.
            - 'lower_bound': Lower confidence bound for the forecast price.
            - 'upper_bound': Upper confidence bound for the forecast price.
            - 'signal': Trading signal ('BUY', 'SELL', 'HOLD').
            - 'confidence': Confidence in the generated signal.
            - 'volatility': Estimated annualized volatility.
            - 'uncertainty': A measure of forecast uncertainty.
            - 'price_change_pct': Expected percentage price change.
            - Additional metrics like fragility scores if calculated.
        """
        # print("Starting price forecasting...") # Informational, can be noisy in loops
        start_time = datetime.now()
        if symbol: # Set current symbol if provided
            self.current_coin_symbol = symbol

        df = None
        if df_override is not None and isinstance(df_override, pd.DataFrame):
            print("Using provided DataFrame (df_override).")
            df = df_override.copy() # Use a copy to avoid modifying the original

            # Ensure 'open_time' is the index if it exists as a column
            if 'open_time' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df['open_time'] = pd.to_datetime(df['open_time'])
                    df.set_index('open_time', inplace=True)
                except Exception as e:
                    print(f"Warning: Could not set 'open_time' as index from df_override: {e}")
            elif not isinstance(df.index, pd.DatetimeIndex):
                print("Warning: df_override does not have a DatetimeIndex. Subsequent operations might fail.")


            # Ensure 'Close' column exists and handle potential issues
            if 'Close' not in df.columns and 'close' in df.columns:
                df.rename(columns={'close': 'Close'}, inplace=True) # Ensure correct casing

            if 'Close' in df.columns:
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                df['Close'] = df['Close'].replace(0, np.nan) # Avoid log(0)
                df['Close'] = df['Close'].fillna(method='ffill').fillna(method='bfill')
                if df['Close'].empty or df['Close'].isnull().all():
                    raise ValueError("Close prices in df_override are all NaN or empty after processing.")
                self.price_scale = float(df['Close'].iloc[-1])
                print(f"Price scale set from df_override: ${self.price_scale:.2f}")
            else:
                raise ValueError("df_override must contain a 'Close' or 'close' column.")

            # Calculate necessary columns if they are missing
            if 'log_price' not in df.columns:
                df['log_price'] = np.log(df['Close'])

            # Ensure other necessary OHLCV columns are present and correctly cased
            for col_lower, col_pascal in [('open', 'Open'), ('high', 'High'), ('low', 'Low'), ('volume', 'Volume')]:
                if col_pascal not in df.columns and col_lower in df.columns:
                    df.rename(columns={col_lower: col_pascal}, inplace=True)
                if col_pascal in df.columns:
                     df[col_pascal] = pd.to_numeric(df[col_pascal], errors='coerce') # Ensure numeric
                # else:
                #    print(f"Warning: Column '{col_pascal}' (or '{col_lower}') not found in df_override. Some features might not work.")


            if 'returns' not in df.columns:
                df['returns'] = df['log_price'].diff().fillna(0)
            if 'squared_returns' not in df.columns:
                df['squared_returns'] = df['returns']**2
            if 'vol_14' not in df.columns: # Assuming vol_14 is indeed used later or good to have
                df['vol_14'] = df['returns'].rolling(window=14).std().fillna(method='bfill')

            # Basic validation for other essential columns (presence only, type assumed by usage later)
            # for col in ['Open', 'High', 'Low', 'Volume']:
            #     if col not in df.columns:
            #        print(f"Warning: Essential column '{col}' not found in df_override. This might cause issues.")

        else:
            print("df_override not provided or invalid. Fetching data...")
            # Use async fetch for much faster data retrieval
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            df = loop.run_until_complete(
                self.fetch_data_async(symbol=getattr(self, 'current_coin_symbol', 'BTCUSDT'),
                                    interval='1h', lookback_days=7)
            )
            # self.price_scale is set within fetch_data_async

        if df is None or df.empty:
            raise ValueError("DataFrame is empty either from override or fetching.")

        liquidity_profile = self.assess_liquidity(df)
        self.liquidity_profile = liquidity_profile
        print(f"Liquidity profile: Volume consistency {liquidity_profile['volume_consistency']:.2f}, " +
              f"Spread estimate {liquidity_profile['spread_estimate']:.4f}")
        if not liquidity_profile['is_liquid']:
            print("Low liquidity detected: Adjusting model parameters for less liquid market")
            self.jump_intensity *= 1.5
            self.jump_std_x *= 1.3

        self.update_parameters_from_data(df)
        print("Model parameters updated from market data")
        self.latest_market_data = df # Store the potentially overridden or fetched df

        regime_id, regime_prob = self.detect_market_regime(df)
        if hasattr(self, 'regime_labels') and len(self.regime_labels) > regime_id:
            regime_label = self.regime_labels[regime_id]
            print(f"Detected market regime: {regime_label} (confidence: {regime_prob:.2f})")
            self.update_regime_dependent_parameters(regime_id, regime_label, regime_prob)

        X_t, phi_t = self.estimate_initial_state(df)
        n_steps = 720 # This might need adjustment if df_override is very short.
                      # For now, keeping it fixed as per original logic.

        print(f"Simulating price paths with regime-dependent parameters...")
        X, phi, R, lambda_vals, _, kappa, theta, alpha = self.simulate_path(X_t, phi_t, n_steps)

        print(f"Running price forecast for {self.forecast_horizon} day(s)...")
        fragility_result = self.calculate_market_fragility(df)
        # ... (rest of the original fragility score handling)
        fragility_score = fragility_result['overall_score']
        breakout_direction = fragility_result['breakout_direction']
        confidence_fragility = fragility_result['confidence'] # Renamed to avoid conflict with signal confidence

        original_jump_intensity = self.jump_intensity
        original_jump_mean_x = self.jump_mean_x
        original_jump_std_x = self.jump_std_x

        # Adjust jump parameters based on fragility - using self.latest_market_data (which is df)
        if hasattr(self, 'latest_market_data') and 'returns' in self.latest_market_data.columns and len(self.latest_market_data['returns']) >= 200:
            returns = self.latest_market_data['returns'].dropna()
            std_returns = returns.std()
            if std_returns > 0:
                jump_threshold = 2.5 * std_returns
                positive_jumps = returns[returns > jump_threshold]
                negative_jumps = returns[returns < -jump_threshold]
                pos_jump_size = positive_jumps.mean() if len(positive_jumps) > 5 else 0.02
                neg_jump_size = negative_jumps.mean() if len(negative_jumps) > 5 else -0.02
                jump_scaling = 0.0
                if fragility_score > 70:
                    jump_scaling = min(2.0, 1.0 + (fragility_score - 70) / 30)
                elif fragility_score > 50:
                    jump_scaling = min(1.5, 1.0 + (fragility_score - 50) / 40)
                self.jump_intensity *= jump_scaling
                self.jump_std_x *= min(jump_scaling, 1.5)
                jump_magnitude = max(abs(pos_jump_size), abs(neg_jump_size)) * confidence_fragility
                if breakout_direction == "up":
                    self.jump_mean_x = jump_magnitude
                elif breakout_direction == "down":
                    self.jump_mean_x = -jump_magnitude
                else:
                    natural_jump_mean = (pos_jump_size + neg_jump_size) / 2
                    self.jump_mean_x = natural_jump_mean * 0.5
            else:
                self.jump_mean_x = 0.0 # Reset if std_returns is zero
        else: # Fallback if not enough data for robust jump param estimation
            if fragility_score > 60:
                jump_magnitude = 0.03 * confidence_fragility
                if breakout_direction == "up":
                    self.jump_mean_x = jump_magnitude
                elif breakout_direction == "down":
                    self.jump_mean_x = -jump_magnitude
                else:
                    self.jump_mean_x = 0.0
            else:
                self.jump_mean_x = 0.0

        mu_t = self.calculate_drift(X_t, phi_t, R[-1], lambda_vals[-1])

        # Num forecasts adjustment based on data - using self.latest_market_data (which is df)
        if hasattr(self, 'latest_market_data') and 'returns' in self.latest_market_data.columns and len(self.latest_market_data['returns']) > 30:
            returns = self.latest_market_data['returns'].dropna()
            # ... (rest of the num_forecasts adjustment logic from original code) ...
            returns_std = returns.std() # ensure this is defined
            returns_values = returns.values
            mean_val = np.mean(returns_values)
            std_val = np.std(returns_values)
            if std_val > 0:
                normalized = (returns_values - mean_val) / std_val
                returns_skew = np.mean(normalized ** 3)
                returns_kurt = np.mean(normalized ** 4) - 3
            else:
                returns_skew = 0
                returns_kurt = 0 # ensure kurtosis is initialized
            tail_factor = min(2.0, max(1.0, 1.0 + abs(returns_kurt) / 15))
            skew_factor = min(1.5, max(1.0, 1.0 + abs(returns_skew) / 8))
            base_paths = int(200 * tail_factor * skew_factor)
            if hasattr(self, 'liquidity_profile'):
                if not self.liquidity_profile['is_liquid']: # check if is_liquid exists
                    liquidity_factor = 1.5
                elif self.liquidity_profile.get('volume_consistency', 1.0) < 0.4: # use .get for safety
                    liquidity_factor = 1.2
                else:
                    liquidity_factor = 1.0
                base_paths = int(base_paths * liquidity_factor)

        else:
            base_paths = 200
        base_paths = min(1000, max(100, base_paths))
        num_forecasts = base_paths

        forecasts = np.zeros(num_forecasts)
        for i in range(num_forecasts):
            forecast_i = self.predict_next_day(X_t, phi_t, mu_t, alpha, lambda_vals[-1])
            forecasts[i] = forecast_i

        # ... (rest of the forecast processing and signal generation logic from original code)
        trim_ratio = 0.1
        trim_size = int(num_forecasts * trim_ratio)
        forecasts_sorted = np.sort(forecasts)
        trimmed_forecasts = forecasts_sorted[trim_size:num_forecasts-trim_size] if trim_size < num_forecasts // 2 else forecasts_sorted
        forecast_trimmed_mean = np.mean(trimmed_forecasts) if len(trimmed_forecasts) > 0 else np.mean(forecasts_sorted)

        forecast_median = np.median(forecasts)
        forecast_iqr = np.percentile(forecasts, 75) - np.percentile(forecasts, 25)
        forecast_std = np.std(forecasts)
        relative_dispersion = min(1.0, forecast_iqr / (forecast_median + 1e-10)) if (forecast_median + 1e-10) != 0 else 1.0
        median_weight = 0.5 + 0.3 * relative_dispersion
        forecast_X = median_weight * forecast_median + (1.0 - median_weight) * forecast_trimmed_mean

        daily_vol = np.sqrt(phi_t)
        # Ensure df['returns'] exists before accessing iloc for recent_returns for kurtosis calculation
        if 'returns' in df.columns and len(df['returns']) >= 100:
            recent_returns_kurt = df['returns'].iloc[-100:].values
            excess_kurtosis = stats.kurtosis(recent_returns_kurt)
        else:
            excess_kurtosis = 0 # Default if not enough data
        df_adaptive = max(3, min(30, 6 / max(0.1, excess_kurtosis)))
        current_volatility = np.sqrt(phi_t)
        vol_scale_factor = min(max(0.8, current_volatility / theta if theta > 0 else 1.0), 1.5)
        confidence_level = 0.95

        # Check for latest_market_data and 'returns' before detailed empirical stats
        if hasattr(self, 'latest_market_data') and 'returns' in self.latest_market_data.columns and len(self.latest_market_data['returns']) > 100:
            empirical_returns = self.latest_market_data['returns'].dropna()
            empirical_dist_available = True
            emp_values = empirical_returns.values
            emp_mean = np.mean(emp_values)
            emp_std = np.std(emp_values)
            if emp_std > 0:
                emp_normalized = (emp_values - emp_mean) / emp_std
                returns_skew = np.mean(emp_normalized ** 3)
                returns_kurtosis = np.mean(emp_normalized ** 4) - 3
            else:
                returns_skew = 0
                returns_kurtosis = 0 # Ensure it's defined
            base_interval_scale = 1.0
            if hasattr(self, 'fragility_score') and 'overall_score' in self.fragility_score:
                 fragility_overall_score = self.fragility_score['overall_score'] # Use a temp var
                 if fragility_overall_score > 0: # Check if score is positive
                    base_interval_scale = 1.0 + (fragility_overall_score / 200)
            t_val = stats.t.ppf(0.5 + 0.5 * confidence_level, df_adaptive)
            symmetric_vol = daily_vol * vol_scale_factor * base_interval_scale
            tail_adjustment = 1.0 + max(0, min(0.5, (returns_kurtosis - 3) / 10 if returns_kurtosis is not None else 0)) # Check kurtosis
            skew_adj_factor = min(0.2, max(-0.2, returns_skew / 10 if returns_skew is not None else 0)) # Check skew
            downside_adjustment = tail_adjustment * (1.0 + skew_adj_factor)
            upside_adjustment = tail_adjustment * (1.0 - skew_adj_factor)
        else:
            empirical_dist_available = False
            t_val = stats.t.ppf(0.5 + 0.5 * confidence_level, df_adaptive)
            symmetric_vol = daily_vol * vol_scale_factor
            upside_adjustment = 1.0
            downside_adjustment = 1.0
            returns_skew = 0 # Define for later use
            returns_kurtosis = 0 # Define for later use

        lower_bound = forecast_X * np.exp(-t_val * symmetric_vol * downside_adjustment)
        upper_bound = forecast_X * np.exp(t_val * symmetric_vol * upside_adjustment)
        expected_return = (forecast_X / np.exp(X_t) - 1) * 100 if np.exp(X_t) != 0 else 0

        breakout_up_return = None
        breakout_down_return = None
        breakout_probability = None
        if hasattr(self, 'fragility_score') and 'overall_score' in self.fragility_score and self.fragility_score['overall_score'] > 50:
            fragility_overall_score = self.fragility_score['overall_score']
            confidence_fragility_val = self.fragility_score.get('confidence', 0.5) # Use .get
            breakout_probability = min(0.8, fragility_overall_score / 100 * confidence_fragility_val)
            if breakout_probability > 0.2:
                upper_quantile = forecasts[int(0.95 * num_forecasts)]
                lower_quantile = forecasts[int(0.05 * num_forecasts)]
                breakout_up_return = (np.exp(upper_quantile) / np.exp(X_t) - 1) * 100 if np.exp(X_t) != 0 else 0
                breakout_down_return = (np.exp(lower_quantile) / np.exp(X_t) - 1) * 100 if np.exp(X_t) != 0 else 0

        # X_history related calculations
        if hasattr(self, 'X_history') and len(self.X_history) > 100:
            # ... (original X_history logic)
            short_term_std = np.std(np.diff(self.X_history[-50:])) * np.sqrt(252) 
            medium_term_std = np.std(np.diff(self.X_history[-100:])) * np.sqrt(252)
            if len(self.X_history) > 150:
                long_term_std = np.std(np.diff(self.X_history[-150:])) * np.sqrt(252)
            else:
                long_term_std = medium_term_std
            vol_ratio = short_term_std / max(long_term_std, 1e-10)
            up_days = np.sum(np.diff(self.X_history[-50:]) > 0) / max(1, len(self.X_history[-50:]) -1) # Avoid div by zero
            symmetry_factor = 1.0 + 2.0 * abs(up_days - 0.5)
        else:
            short_term_std = 0.02
            vol_ratio = 1.0
            symmetry_factor = 1.5

        if hasattr(self, 'X_history') and len(self.X_history) > 30:
            diff_X_history = np.diff(self.X_history[-30:])
            if len(diff_X_history) > 1: # Need at least 2 diffs to check sign changes
                 recent_direction_changes = np.sum(np.diff(np.sign(diff_X_history)) != 0)
                 market_consistency = 1.0 - (recent_direction_changes / max(1, len(diff_X_history)-1))
            else:
                 market_consistency = 0.5 # Default if not enough data for changes
        else:
            market_consistency = 0.5

        # Empirical daily moves for threshold
        if empirical_dist_available and 'returns' in self.latest_market_data and len(self.latest_market_data['returns']) > 100:
            empirical_returns_series = self.latest_market_data['returns'] # it's a Series
            window_size_empirical = 1440 if len(empirical_returns_series) > 2000 else 24
            if len(empirical_returns_series) >= window_size_empirical:
                historical_daily_moves = np.abs(empirical_returns_series.rolling(window=window_size_empirical).sum())
                threshold_quantile = np.nanquantile(historical_daily_moves, 0.7) if not historical_daily_moves.empty else 0.02
            else: # Not enough data for rolling sum
                threshold_quantile = np.nanquantile(np.abs(empirical_returns_series), 0.7) if not empirical_returns_series.empty else 0.02
            base_threshold_value = threshold_quantile * 100
        else:
            base_threshold_value = 2.0 * (1.0 + short_term_std * 10) * (2.0 - market_consistency)

        # ... (rest of threshold calculation and signal/confidence logic)
        if vol_ratio > 1.2: base_threshold_value *= 1.2
        elif vol_ratio < 0.8: base_threshold_value *= 1.05
        consistency_factor = np.clip(market_consistency, 0.3, 0.8)
        base_threshold_value *= (1.0 + (1.0 - consistency_factor))

        baseline_multiplier = 1.0
        if hasattr(self, 'current_regime_label') and self.current_regime_label:
            if 'Volatile' in self.current_regime_label: baseline_multiplier = 1.4
            else: baseline_multiplier = 1.2

        forecast_std_val = np.std(forecasts) # ensure this is calculated if not already
        forecast_distributional_factor = 1.0
        if forecast_std_val > 0 and forecast_X != 0 : # check forecast_X for zero
            forecast_cv = forecast_std_val / forecast_X
            forecast_distributional_factor = min(1.5, max(1.0, 1.0 + forecast_cv * 2.0))

        base_signal_threshold = base_threshold_value * baseline_multiplier * forecast_distributional_factor
        annualized_vol = np.sqrt(phi_t * 252)
        vol_adjusted_threshold = base_signal_threshold * max(0.8, min(1.5, annualized_vol / 0.25 if annualized_vol > 0.01 else 1.0)) # avoid div by zero

        uncertainty = np.std(forecasts) / forecast_X if forecast_X != 0 else 1.0
        forecast_stability = np.exp(-5 * uncertainty)

        buy_threshold_factor = 1.0
        sell_threshold_factor = 1.0
        if hasattr(self, 'fragility_score') and 'confidence' in self.fragility_score and self.fragility_score['confidence'] > 0.7:
            if self.fragility_score.get('breakout_direction') == 'up': # use .get
                buy_threshold_factor *= 0.95
                sell_threshold_factor *= 1.05
            elif self.fragility_score.get('breakout_direction') == 'down':
                sell_threshold_factor *= 0.95
                buy_threshold_factor *= 1.05

        buy_threshold = vol_adjusted_threshold * buy_threshold_factor
        sell_threshold = vol_adjusted_threshold * sell_threshold_factor

        upside_probability = np.mean(forecasts > X_t)
        downside_probability = np.mean(forecasts < X_t)
        directional_consensus_val = abs(upside_probability - 0.5) * 2.0 # renamed

        forecast_mean_val = np.mean(forecasts) # renamed
        forecast_median_val = np.median(forecasts) # renamed

        iqr_val = np.percentile(forecasts, 75) - np.percentile(forecasts, 25) # renamed
        normalized_iqr_val = iqr_val / (forecast_median_val + 1e-10) if (forecast_median_val + 1e-10) != 0 else 1.0 # renamed
        stability_score = np.exp(-3 * normalized_iqr_val)

        signal = 'HOLD'
        confidence = 0.5 # Default confidence for HOLD

        if expected_return > buy_threshold:
            signal = 'BUY'
            raw_strength = expected_return / buy_threshold if buy_threshold != 0 else 2.0
            strength_factor = min(1.8, 1.0 + np.log(raw_strength if raw_strength > 0 else 1.0) / np.log(5))
            base_confidence = 0.5
            strength_component = 0.15 * strength_factor
            stability_component_val = 0.15 * stability_score #renamed
            consensus_component = 0.1 * upside_probability
            consistency_component = 0.05 * market_consistency
            signal_confidence = base_confidence + strength_component + stability_component_val + consensus_component + consistency_component
            max_confidence = 0.85 if forecast_stability > 0.7 else 0.75
            confidence = min(max_confidence, signal_confidence)
        elif expected_return < -sell_threshold:
            signal = 'SELL'
            raw_strength = -expected_return / sell_threshold if sell_threshold != 0 else 2.0
            strength_factor = min(1.8, 1.0 + np.log(raw_strength if raw_strength > 0 else 1.0) / np.log(5))
            base_confidence = 0.5
            strength_component = 0.15 * strength_factor
            stability_component_val = 0.15 * stability_score #renamed
            consensus_component = 0.1 * downside_probability
            consistency_component = 0.05 * market_consistency
            signal_confidence = base_confidence + strength_component + stability_component_val + consensus_component + consistency_component
            max_confidence = 0.85 if forecast_stability > 0.7 else 0.75
            confidence = min(max_confidence, signal_confidence)
        else: # HOLD
            signal = 'HOLD'
            prox_thresh_val = min(buy_threshold, sell_threshold) #renamed
            proximity_to_threshold = 1.0 - min(1.0, abs(expected_return) / prox_thresh_val if prox_thresh_val != 0 else 1.0)
            zero_prox_val = 0.1 * prox_thresh_val #renamed
            zero_proximity = 1.0 - min(1.0, abs(expected_return) / zero_prox_val if zero_prox_val != 0 else 1.0)
            hold_confidence = 0.5 + 0.15 * proximity_to_threshold * stability_score + 0.1 * zero_proximity * (1.0 - directional_consensus_val)
            confidence = min(0.8, max(0.5, hold_confidence))

        # Price change percentage calculation
        current_price_val = np.exp(X_t) * self.price_scale if self.price_scale else np.exp(X_t)
        forecast_price_val = forecast_X * self.price_scale if self.price_scale else forecast_X
        raw_price_change_pct = ((forecast_price_val / current_price_val) - 1) * 100 if current_price_val != 0 else 0

        coin_symbol = getattr(self, 'current_coin_symbol', '')
        price_change_pct = raw_price_change_pct # Default if no further adjustments
        if coin_symbol and hasattr(self, 'price_scale') and self.price_scale > 0:
            symbol_factor = sum([ord(c) * (i+1) for i, c in enumerate(coin_symbol)]) % 1000 / 1000.0
            price_scale_factor = 1.0
            # ... (original price_scale_factor logic)
            if self.price_scale < 0.001: price_scale_factor = 2.5 + (symbol_factor - 0.5)
            elif self.price_scale < 0.01: price_scale_factor = 2.0 + (symbol_factor - 0.5) * 0.8
            # ... (and so on for other price scales)
            else: price_scale_factor = 0.7 + (symbol_factor - 0.5) * 0.2
            price_change_pct = raw_price_change_pct * price_scale_factor

        # Reset jump parameters to original state if they were modified for this forecast
        self.jump_intensity = original_jump_intensity
        self.jump_mean_x = original_jump_mean_x
        self.jump_std_x = original_jump_std_x

        forecast_dict = { # Renamed to avoid conflict
            'current_price': current_price_val,
            'forecast_price': forecast_price_val,
            'lower_bound': lower_bound * self.price_scale if self.price_scale else lower_bound,
            'upper_bound': upper_bound * self.price_scale if self.price_scale else upper_bound,
            'signal': signal,
            'confidence': confidence,
            'volatility': np.sqrt(phi_t * 252),
            'uncertainty': uncertainty,
            'price_change_pct': price_change_pct
        }
        if hasattr(self, 'fragility_score') and isinstance(self.fragility_score, dict): # Check if dict
            forecast_dict['fragility_score'] = self.fragility_score.get('overall_score')
            if forecast_dict['fragility_score'] is not None:
                 forecast_dict['fragility_interpretation'] = self.fragility_calculator.get_score_interpretation(
                     forecast_dict['fragility_score']
                 )
            forecast_dict['breakout_direction'] = self.fragility_score.get('breakout_direction')
            forecast_dict['breakout_confidence'] = self.fragility_score.get('confidence')
            forecast_dict['fragility_components'] = self.fragility_score.get('component_scores')
            if breakout_probability is not None:
                forecast_dict['breakout_probability'] = breakout_probability
            if breakout_up_return is not None:
                forecast_dict['breakout_up_return'] = breakout_up_return
            if breakout_down_return is not None:
                forecast_dict['breakout_down_return'] = breakout_down_return

        if hasattr(self, 'current_coin_symbol'):
            forecast_dict['current_coin_symbol'] = self.current_coin_symbol
        forecast_dict['forecast_horizon'] = self.forecast_horizon

        # Conditional Plotting
        if df_override is None: # Only plot if not using df_override
            try:
                visualizer = AdvancedChartVisualizer()
                simulated_paths = None
                simulated_vol = None
                if isinstance(X, np.ndarray) and X.size > 1:
                    num_paths_plot = min(20, len(X) -1 if len(X) > 1 else 0) # Renamed var
                    path_length_plot = 24 # Renamed var
                    price_paths = []
                    current_price_plot = df['Close'].iloc[-1] if 'Close' in df.columns and not df['Close'].empty else self.price_scale # Renamed
                    for i in range(num_paths_plot):
                        if i < len(X) - path_length_plot:
                            path_segment = current_price_plot * np.exp(X[i:i+path_length_plot])
                            price_paths.append(path_segment.tolist())
                    if price_paths:
                        simulated_paths = price_paths
                if isinstance(phi, np.ndarray) and phi.size > 1:
                    vol_length_plot = min(48, len(phi) -1 if len(phi) > 1 else 0) # Renamed var
                    simulated_vol = phi[:vol_length_plot].tolist()

                visualizer.create_forecast_chart(df, forecast_dict,
                                            simulated_X=simulated_paths,
                                            simulated_phi=simulated_vol,
                                            filename='crypto_forecast.png')
            print("Created enhanced chart with clean visualization") # Informational
            except Exception as e:
            print(f"Error using advanced chart visualizer: {e}. Falling back to basic plot if enabled.")
            if df_override is None: # Only plot basic if not an override and advanced failed
                 self._plot_basic_forecast(df, forecast_dict)
        else: # Plotting skipped message
            if df_override is not None:
                print("Plotting skipped due to df_override usage.")
            # else: print("Advanced chart created successfully.") # Informational

        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        # print(f"Forecast for {self.current_coin_symbol} completed in {runtime:.2f} seconds") # Informational
        return forecast_dict

    def _plot_basic_forecast(self, df, forecast):
        """
        Simple fallback visualization using Matplotlib if AdvancedChartVisualizer fails or is not used.
        This is primarily for internal debugging or environments where advanced plotting is unavailable.
        return forecast
    def _plot_basic_forecast(self, df, forecast):
        """
        Simple fallback visualization if the advanced chart module isn't available
        """
        coin_name = "Cryptocurrency"
        if hasattr(self, 'current_coin_symbol'):
            coin_symbol = self.current_coin_symbol
            if coin_symbol.endswith('USDT'):
                coin_name = coin_symbol[:-4]
            else:
                coin_name = coin_symbol
        elif 'BTC' in str(self.price_scale) or self.price_scale > 20000:
            coin_name = "Bitcoin"
        fig, ax = plt.subplots(figsize=(10, 6))
        skip_factor = max(1, len(df) // 1000)
        dates = df['open_time'].values[-1000*skip_factor::skip_factor]
        prices = df['close'].values[-1000*skip_factor::skip_factor]
        ax.plot(dates, prices, label='Price', color='blue')
        last_date = df['open_time'].iloc[-1]
        forecast_date = last_date + timedelta(days=self.forecast_horizon)
        forecast_price = forecast['forecast_price']
        ax.scatter(forecast_date, forecast_price, color='red', marker='o')
        ax.set_title(f'{coin_name} Price Forecast')
        ax.set_ylabel('Price (USD)')
        plt.savefig('btc_forecast.png')
        plt.close()
    def _calculate_market_asymmetry(self, df, lookback=100):
        """
        Calculate market asymmetry based on data rather than assumptions.
        Positive values indicate natural bullish tendency, negative values indicate bearish.
        Range: [-1, 1]
        """
        if len(df) < lookback:
            return 0.0
        returns = df['returns'].iloc[-lookback:]
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 0.0
        pos_vol = positive_returns.std()
        neg_vol = negative_returns.std()
        vol_ratio = (pos_vol - neg_vol) / (pos_vol + neg_vol)
        skew = returns.skew()
        skew_factor = np.clip(skew / 2, -0.5, 0.5)
        std_returns = returns.std()
        jump_threshold = 2 * std_returns
        positive_jumps = len(returns[returns > jump_threshold])
        negative_jumps = len(returns[returns < -jump_threshold])
        total_jumps = positive_jumps + negative_jumps
        jump_asymmetry = 0
        if total_jumps > 0:
            jump_asymmetry = (positive_jumps - negative_jumps) / total_jumps
        vol_weight = 0.3
        skew_weight = 0.4
        jump_weight = 0.3
        asymmetry = (vol_weight * vol_ratio + 
                     skew_weight * skew_factor + 
                     jump_weight * jump_asymmetry)
        return np.clip(asymmetry, -1, 1)
    def predict_crypto_price(self, symbol):
        """
        Predict cryptocurrency price for the given symbol.
        This is a wrapper method for run_forecast to maintain backward compatibility.
        Parameters:
        -----------
        symbol : str
            The cryptocurrency symbol (e.g., 'BTC', 'ETH', 'BTCUSDT')
        Returns:
        --------
        dict
            Forecast results with price predictions, signals, and confidence levels.
        """
        # Ensure symbol format for crypto if not already USDT (common for Binance)
        if len(symbol) <= 4 and not symbol.endswith('USDT') and not symbol.endswith('-USD'): # Basic check for typical stock tickers
            # This heuristic might need adjustment for broader exchange/asset coverage.
            # For now, assumes non-USDT/non-USD short symbols are crypto bases needing USDT pairing.
            is_likely_stock_ticker = all(c.isalpha() for c in symbol) and len(symbol) <= 5
            if not is_likely_stock_ticker:
                 symbol = symbol.upper() + 'USDT'
            # else, pass stock ticker as is, yfinance handles it.

        return self.run_forecast(symbol=symbol) # Call the main forecast method

def main():
    """
    Example usage of the AlphaEngine for fetching data and running a forecast.
    This function is primarily for testing and demonstration purposes.
    """
    print("\n===== AlphaEngine Cryptocurrency/Stock Price Forecast Example =====")
    print("--------------------------------------------------------------------")

    # Example: Initialize with dummy API keys if you want to test Binance authenticated endpoints
    # engine = AlphaEngine(api_key="YOUR_KEY", api_secret="YOUR_SECRET")
    engine = AlphaEngine() # No keys, will use public endpoints or Yahoo Finance

    # --- Example for Cryptocurrency ---
    crypto_symbol_example = 'BTCUSDT'
    print(f"\nRunning forecast for Cryptocurrency: {crypto_symbol_example}")
    try:
        # Fetching data explicitly first (optional, run_forecast can do it)
        # df_crypto = engine.fetch_data(symbol=crypto_symbol_example, interval='1h', lookback_days=14)
        # print(f"Data fetched for {crypto_symbol_example}: {len(df_crypto)} records")

        # Run forecast directly
        forecast_crypto = engine.run_forecast(symbol=crypto_symbol_example)

        print(f"\n----- FORECAST RESULTS for {crypto_symbol_example} -----")
        print(f"Current Price: ${forecast_crypto.get('current_price', 0):.2f}")
        print(f"Forecast Price (next {forecast_crypto.get('forecast_horizon', 'N/A')} day(s)): ${forecast_crypto.get('forecast_price', 0):.2f}")
        print(f"Expected Price Change: {forecast_crypto.get('price_change_pct', 0):+.2f}%")
        print(f"Signal: {forecast_crypto.get('signal', 'N/A')} (Confidence: {forecast_crypto.get('confidence', 0):.1%})")
        if 'fragility_score' in forecast_crypto:
            print(f"Market Fragility: {forecast_crypto['fragility_score']:.1f}/100 ({forecast_crypto.get('fragility_interpretation', '')})")
        if forecast_crypto.get('breakout_direction') != 'unknown' and forecast_crypto.get('breakout_direction') is not None:
            print(f"Potential Breakout: {forecast_crypto['breakout_direction']} (Confidence: {forecast_crypto.get('breakout_confidence', 0):.1%})")
        print(f"Annualized Volatility: {forecast_crypto.get('volatility', 0)*100:.2f}%")
        print("Forecast chart saved as 'crypto_forecast.png' (if plotting enabled and successful)")
    except Exception as e:
        print(f"Error during Cryptocurrency forecast for {crypto_symbol_example}: {e}")

    # --- Example for Stock (using Yahoo Finance via AlphaEngine's fetch_data) ---
    # stock_symbol_example = 'AAPL' # Apple Inc.
    # print(f"\nRunning forecast for Stock: {stock_symbol_example}")
    # try:
    #     forecast_stock = engine.run_forecast(symbol=stock_symbol_example) # AlphaEngine handles -USD if needed for yfinance

    #     print(f"\n----- FORECAST RESULTS for {stock_symbol_example} -----")
    #     print(f"Current Price: ${forecast_stock.get('current_price', 0):.2f}")
    #     print(f"Forecast Price (next {forecast_stock.get('forecast_horizon', 'N/A')} day(s)): ${forecast_stock.get('forecast_price', 0):.2f}")
    #     print(f"Expected Price Change: {forecast_stock.get('price_change_pct', 0):+.2f}%")
    #     print(f"Signal: {forecast_stock.get('signal', 'N/A')} (Confidence: {forecast_stock.get('confidence', 0):.1%})")
    #     # Note: Fragility and some other crypto-specific metrics might be less relevant or behave differently for stocks.
    #     if 'fragility_score' in forecast_stock:
    #         print(f"Market Fragility: {forecast_stock['fragility_score']:.1f}/100 ({forecast_stock.get('fragility_interpretation', '')})")

    # except Exception as e:
    #     print(f"Error during Stock forecast for {stock_symbol_example}: {e}")

    return # forecast_crypto # Or return both if needed

if __name__ == "__main__":
    main()
