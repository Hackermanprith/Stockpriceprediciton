"""
Clean Chart Visualization System for AlphaEngine Cryptocurrency Forecasts
Provides elegant, professional-grade visualization with clean visual cues
for different market conditions and analysis types.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, PathPatch
from matplotlib.patheffects import withStroke, Stroke
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib import cm, colors
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
import matplotlib.transforms as transforms
import matplotlib.patheffects as path_effects
class AdvancedChartVisualizer:
    """
    Clean visualization system with adaptive styling based on market conditions,
    utilizing minimalist design principles for clarity and visual impact.
    """
    def __init__(self, theme="dark"):
        self.theme = theme
        self.setup_theme()
        self.chart_types = ["standard", "momentum", "breakout", "volatility", "trend"]
        self.markers = {
            'STRONG_BUY': '^',
            'BUY': '^', 
            'WEAK_BUY': '^', 
            'HOLD': 'o',
            'WEAK_SELL': 'v',  
            'SELL': 'v',
            'STRONG_SELL': 'v'
        }
        self.create_custom_colormaps()
    def setup_theme(self):
        """Configure the visualization theme and color palettes"""
        if self.theme == "dark":
            plt.rcParams.update({
                'figure.facecolor': '#121622',  # Dark background
                'axes.facecolor': '#121622',   # Dark chart area
                'text.color': '#EFEFEF',       # Light text color
                'axes.labelcolor': '#EFEFEF',   # Light label color
                'xtick.color': '#EFEFEF',      # Light tick color
                'ytick.color': '#EFEFEF',      # Light tick color
                'axes.edgecolor': '#2F3A5A',   # Dark blue edge
                'axes.grid': True,
                'grid.color': '#2F3A5A',       # Dark blue grid
                'grid.linestyle': '-',
                'grid.alpha': 0.2,
                'font.family': 'sans-serif',
                'font.weight': 'medium',
            })
            self.palette = {
                'primary': '#5C6BC0',          # Indigo primary
                'secondary': '#7986CB',        # Lighter indigo
                'background': '#121622',       # Dark background
                'panel': '#1E2337',           # Slightly lighter panel
                'text': '#EFEFEF',            # Light text
                'subtext': '#A9A9A9',         # Gray subtext
                'bullish': '#00C853',         # Bright green for bullish
                'very_bullish': '#00B04A',    # Darker green for very bullish
                'extreme_bullish': '#00973F',  # Deepest green for extreme bullish
                'bearish': '#FF3D57',         # Bright red for bearish
                'very_bearish': '#E6354E',    # Darker red for very bearish
                'extreme_bearish': '#CC2E44', # Deepest red for extreme bearish
                'neutral': '#8D97B5',         # Neutral blue-gray
                'highlight': '#FFB74D',       # Orange highlight
                'support': '#4CAF50',         # Support line green
                'resistance': '#F44336',      # Resistance line red
                'golden_ratio': '#FFD54F',    # Golden yellow for golden ratio
                'fibonacci': '#AB47BC',       # Purple for fibonacci
                'pivot': '#FF7043',           # Deep orange for pivot
                'regression': '#26C6DA',      # Cyan for regression
                'vol_low': '#81C784',         # Light green for low volume
                'vol_medium': '#4CAF50',      # Medium green for medium volume
                'vol_high': '#2E7D32',        # Dark green for high volume
                'volatility_shade': '#1A237E', # Dark blue for volatility shade
                'volatility_edge': '#3949AB',  # Medium indigo for volatility edge
                'breakout_up': '#00E676',     # Bright green for upward breakout
                'breakout_down': '#FF1744',   # Bright red for downward breakout
                'divergence_positive': '#1DE9B6', # Teal for positive divergence
                'divergence_negative': '#FF4081', # Pink for negative divergence
                'trend_strong': '#448AFF',    # Strong blue for strong trend
                'trend_weak': '#42A5F5',      # Light blue for weak trend
                'alert': '#FFC107',           # Amber for alert
                'success': '#4CAF50',         # Green for success
                'warning': '#FF9800',         # Orange for warning
                'info': '#2196F3'             # Blue for info
            }
        else:
            plt.rcParams.update({
                'figure.facecolor': '#FFFFFF',  # White background
                'axes.facecolor': '#FFFFFF',   # White chart area
                'text.color': '#333333',       # Dark text color
                'axes.labelcolor': '#333333',   # Dark label color
                'xtick.color': '#333333',      # Dark tick color
                'ytick.color': '#333333',      # Dark tick color
                'axes.edgecolor': '#DDDDDD',   # Light gray edge
                'axes.grid': True,
                'grid.color': '#DDDDDD',       # Light gray grid
                'grid.linestyle': '-',
                'grid.alpha': 0.6,
                'font.family': 'sans-serif',
                'font.weight': 'medium',
            })
            self.palette = {
                'primary': '#3366CC',          # Primary blue
                'secondary': '#6699FF',        # Secondary lighter blue
                'background': '#FFFFFF',       # White background
                'panel': '#F5F5F5',           # Light gray panel
                'text': '#333333',            # Dark gray text
                'subtext': '#666666',         # Medium gray subtext
                'bullish': '#089981',         # Green for bullish
                'very_bullish': '#067A67',    # Darker green for very bullish
                'extreme_bullish': '#045D4E',  # Deepest green for extreme bullish
                'bearish': '#F23645',         # Red for bearish
                'very_bearish': '#D01C29',    # Darker red for very bearish
                'extreme_bearish': '#A61621', # Deepest red for extreme bearish
                'neutral': '#7B8DA6',         # Neutral blue-gray
                'highlight': '#FFA726',       # Orange highlight
                'support': '#2E7D32',         # Support line green
                'resistance': '#C62828',      # Resistance line red
                'golden_ratio': '#FFC107',    # Golden yellow for golden ratio
                'fibonacci': '#9C27B0',       # Purple for fibonacci
                'pivot': '#FF5722',           # Deep orange for pivot
                'regression': '#00ACC1',      # Cyan for regression
                'vol_low': '#A5D6A7',         # Light green for low volume
                'vol_medium': '#66BB6A',      # Medium green for medium volume
                'vol_high': '#2E7D32',        # Dark green for high volume
                'volatility_shade': '#E3F2FD', # Light blue for volatility shade
                'volatility_edge': '#1E88E5',  # Medium blue for volatility edge
                'breakout_up': '#00C853',     # Bright green for upward breakout
                'breakout_down': '#FF1744',   # Bright red for downward breakout
                'divergence_positive': '#00BFA5', # Teal for positive divergence
                'divergence_negative': '#EC407A', # Pink for negative divergence
                'trend_strong': '#1976D2',      # Strong blue for strong trend
                'trend_weak': '#64B5F6',        # Light blue for weak trend
                'alert': '#FFC107',             # Amber for alert
                'success': '#4CAF50',           # Green for success
                'warning': '#FF9800',           # Orange for warning
                'info': '#2196F3'              
            }
    def create_custom_colormaps(self):
        """Create specialized colormaps for different visualization needs"""
        bullish_colors = [
            self.palette['neutral'], 
            self.palette['bullish'], 
            self.palette['very_bullish'], 
            self.palette['extreme_bullish']
        ]
        self.bullish_cmap = LinearSegmentedColormap.from_list(
            'bullish_gradient', bullish_colors, N=100
        )
        bearish_colors = [
            self.palette['neutral'], 
            self.palette['bearish'], 
            self.palette['very_bearish'], 
            self.palette['extreme_bearish']
        ]
        self.bearish_cmap = LinearSegmentedColormap.from_list(
            'bearish_gradient', bearish_colors, N=100
        )
        volatility_colors = [
            self.palette['vol_low'], 
            self.palette['vol_medium'], 
            self.palette['vol_high']
        ]
        self.volatility_cmap = LinearSegmentedColormap.from_list(
            'volatility_gradient', volatility_colors, N=100
        )
        forecast_colors = [
            to_rgba(self.palette['primary'], 0.1),
            to_rgba(self.palette['primary'], 0.5)
        ]
        self.forecast_cmap = LinearSegmentedColormap.from_list(
            'forecast_gradient', forecast_colors, N=100
        )
    def create_forecast_chart(self, df, forecast, simulated_X=None, simulated_phi=None, 
                           chart_type=None, filename='crypto_forecast.png'):
        """
        Generate a premium visualization adapted to market conditions with clear directional guidance
        Parameters:
        -----------
        df : pandas.DataFrame
            Historical OHLCV data with at least 'open_time' and 'close' columns
        forecast : dict
            Forecast data from AlphaEngine including 'forecast_price', 'upper_bound', 'lower_bound', etc.
        simulated_X : numpy.ndarray, optional
            Simulated price path data if available
        simulated_phi : numpy.ndarray, optional
            Simulated volatility path data if available
        chart_type : str, optional
            Visualization style to use ('standard', 'momentum', 'breakout', 'volatility', 'trend')
            If None, the system will auto-select based on market conditions
        filename : str, optional
            Name of output file to save the chart
        """
        coin_name = "Cryptocurrency"
        if 'current_coin_symbol' in forecast:
            coin_symbol = forecast['current_coin_symbol']
            if coin_symbol.endswith('USDT'):
                coin_name = coin_symbol[:-4]
            else:
                coin_name = coin_symbol
        elif forecast.get('asset_name'):
            coin_name = forecast.get('asset_name')
        if chart_type is None:
            volatility = forecast.get('volatility', 0) * 100
            price_change_pct = forecast.get('price_change_pct', 0)
            fragility_score = forecast.get('fragility_score', 50)
            breakout_confidence = forecast.get('breakout_confidence', 0)
            if breakout_confidence > 0.7:
                chart_type = 'breakout'
            elif volatility > 3.0:
                chart_type = 'volatility'
            elif abs(price_change_pct) > 5.0:
                chart_type = 'momentum'
            elif fragility_score > 70:
                chart_type = 'volatility'
            else:
                chart_type = 'standard'
        fig = plt.figure(figsize=(18, 12), dpi=150)
        ax_price = None
        ax_vol = None
        ax_direction = None
        ax_indicators = None
        ax_signals = None
        ax_strategy = None
        if chart_type == 'breakout':
            gs = GridSpec(5, 4, figure=fig, height_ratios=[3, 1, 0.5, 1, 0.5])
            ax_price = fig.add_subplot(gs[0:2, 0:3])
            ax_vol = fig.add_subplot(gs[2, 0:3], sharex=ax_price)
            ax_direction = fig.add_subplot(gs[3, 0:3], sharex=ax_price)
            ax_indicators = fig.add_subplot(gs[0:2, 3])
            ax_signals = fig.add_subplot(gs[2:4, 3])
            ax_strategy = fig.add_subplot(gs[4, 3])
        elif chart_type == 'volatility':
            gs = GridSpec(5, 4, figure=fig, height_ratios=[3, 1.5, 0.5, 0.5, 0.5])
            ax_price = fig.add_subplot(gs[0:2, 0:3])
            ax_vol = fig.add_subplot(gs[2, 0:3], sharex=ax_price)
            ax_indicators = fig.add_subplot(gs[0:2, 3])
            ax_signals = fig.add_subplot(gs[2:4, 3])
            ax_strategy = fig.add_subplot(gs[4, 0:3])
        else:
            gs = GridSpec(4, 4, figure=fig)
            ax_price = fig.add_subplot(gs[0:3, 0:3])
            ax_vol = fig.add_subplot(gs[3, 0:3], sharex=ax_price)
            ax_indicators = fig.add_subplot(gs[0:2, 3])
            ax_signals = fig.add_subplot(gs[2:4, 3])
        axes_list = [ax for ax in [ax_price, ax_vol, ax_indicators, ax_signals, 
                                  ax_direction, ax_strategy] if ax is not None]
        for ax in axes_list:
            ax.set_facecolor(self.palette['panel'])
        skip_factor = max(1, len(df) // 1000)
        dates = df['open_time'].values[-1000*skip_factor::skip_factor]
        prices = df['close'].values[-1000*skip_factor::skip_factor]
        last_date = df['open_time'].iloc[-1]
        forecast_horizon = forecast.get('forecast_horizon', 1)
        forecast_date = last_date + timedelta(days=forecast_horizon)
        price_change_pct = forecast.get('price_change_pct', 0)
        direction_strength = abs(price_change_pct)
        is_bullish = price_change_pct >= 0
        if is_bullish:
            direction_color = self.palette['bullish']
            direction_cmap = self.bullish_cmap
            if direction_strength > 5:
                direction_color = self.palette['very_bullish']
            if direction_strength > 10:
                direction_color = self.palette['extreme_bullish']
        else:
            direction_color = self.palette['bearish']
            direction_cmap = self.bearish_cmap
            if direction_strength > 5:
                direction_color = self.palette['very_bearish']
            if direction_strength > 10:
                direction_color = self.palette['extreme_bearish']
        if len(dates) > 3:
            try:
                xnew = np.linspace(0, len(dates)-1, min(500, len(dates)*2))
                spl = make_interp_spline(np.arange(len(dates)), prices, k=3)
                smooth_prices = spl(xnew)
                smooth_dates = np.array([dates[0] + (dates[-1] - dates[0]) * i / len(xnew) 
                                      for i in range(len(xnew))])
                price_line = ax_price.plot(smooth_dates, smooth_prices, 
                                         color=self.palette['primary'], linewidth=2.5, 
                                         path_effects=[Stroke(linewidth=4, foreground=self.palette['primary'], alpha=0.3)],
                                         label='Price', solid_capstyle='round', alpha=0.95, zorder=5)
                gradient_colors = [to_rgba(self.palette['primary'], 0.01), 
                                 to_rgba(self.palette['primary'], 0.15)]
                gradient_cmap = LinearSegmentedColormap.from_list('price_fill', gradient_colors)
                x = np.array(mdates.date2num(smooth_dates))
                y1 = np.array(smooth_prices)
                y0 = np.ones_like(y1) * min(prices) * 0.95
                ax_price.fill_between(smooth_dates, y0, y1, 
                                   color=self.palette['primary'], alpha=0.05)
            except:
                price_line = ax_price.plot(dates, prices, 
                                        color=self.palette['primary'], linewidth=2.5, 
                                        label='Price', alpha=0.95, zorder=5)
                ax_price.fill_between(dates, prices, min(prices)*0.95, 
                                   color=self.palette['primary'], alpha=0.05)
        else:
            price_line = ax_price.plot(dates, prices, 
                                     color=self.palette['primary'], linewidth=2.5, 
                                     label='Price', alpha=0.95, zorder=5)
            ax_price.fill_between(dates, prices, min(prices)*0.95, 
                               color=self.palette['primary'], alpha=0.05)
        if len(prices) > 20:
            ma20 = pd.Series(prices).rolling(window=20).mean().values
            ax_price.plot(dates, ma20, color=self.palette['secondary'], 
                       linestyle='-', linewidth=1.5, alpha=0.7, 
                       label='20-Day MA', zorder=4)
            if len(prices) > 50:
                ma50 = pd.Series(prices).rolling(window=50).mean().values
                ax_price.plot(dates, ma50, color=self.palette['fibonacci'], 
                           linestyle='-', linewidth=1.3, alpha=0.6, 
                           label='50-Day MA', zorder=3)
                if len(ma20) > 5 and len(ma50) > 5:
                    crosses = []
                    for i in range(1, len(ma20)):
                        if ((ma20[i] > ma50[i] and ma20[i-1] <= ma50[i-1]) or 
                            (ma20[i] < ma50[i] and ma20[i-1] >= ma50[i-1])) and i < len(dates):
                            cross_type = "bullish" if ma20[i] > ma50[i] else "bearish"
                            crosses.append((dates[i], (ma20[i] + ma50[i])/2, cross_type))
                    for date, price, cross_type in crosses[-3:]:
                        color = self.palette['bullish'] if cross_type == "bullish" else self.palette['bearish']
                        marker = "^" if cross_type == "bullish" else "v"
                        ax_price.scatter([date], [price], marker=marker, s=100, 
                                      color=color, edgecolor='white', linewidth=1, 
                                      alpha=0.8, zorder=6)
        if len(prices) > 30:
            try:
                alpha = 0.1
                ewma = pd.Series(prices).ewm(alpha=alpha).mean().values
                ax_price.plot(dates, ewma, color=self.palette['golden_ratio'], 
                           linestyle='-', linewidth=1.2, alpha=0.6, 
                           label='EMA', zorder=2)
                if len(ewma) > 30:
                    recent_slope = (ewma[-1] - ewma[-30]) / 30
                    slope_pct = recent_slope / prices[-1] * 100
                    if slope_pct > 0.2:
                        trend_color = self.palette['very_bullish']
                        trend_linewidth = 2.5
                    elif slope_pct > 0.05:
                        trend_color = self.palette['bullish']
                        trend_linewidth = 2.0
                    elif slope_pct < -0.2:
                        trend_color = self.palette['very_bearish']
                        trend_linewidth = 2.5
                    elif slope_pct < -0.05:
                        trend_color = self.palette['bearish']
                        trend_linewidth = 2.0
                    else:
                        trend_color = self.palette['neutral']
                        trend_linewidth = 1.5
                    trend_start = max(0, len(dates) - 30)
                    ax_price.plot(dates[trend_start:], ewma[trend_start:], 
                               color=trend_color, linewidth=trend_linewidth, 
                               alpha=0.8, zorder=20)
            except:
                pass
        forecast_price = forecast['forecast_price']
        lower_bound = forecast['lower_bound']
        upper_bound = forecast['upper_bound']
        signal = forecast.get('signal', 'HOLD')
        signal_colors = {
            'STRONG_BUY': self.palette['extreme_bullish'],
            'BUY': self.palette['bullish'], 
            'WEAK_BUY': self.palette['bullish'],
            'HOLD': self.palette['neutral'],
            'WEAK_SELL': self.palette['bearish'],
            'SELL': self.palette['bearish'], 
            'STRONG_SELL': self.palette['extreme_bearish']
        }
        signal_color = signal_colors.get(signal, self.palette['neutral'])
        ci_width_pct = (upper_bound - lower_bound) / forecast_price * 100
        future_days = forecast_horizon
        forecast_dates = [last_date + timedelta(days=i/5) for i in range(1, future_days*5+1)]
        # Define cone_dates early to avoid reference errors
        cone_dates = [last_date] + forecast_dates
        
        # Extract volatility from forecast or use a default value
        volatility = forecast.get('volatility', 0.01)
        # Define volatility_level for cone calculations
        volatility_level = volatility * 2.0  # Scale the volatility for better visualization
        
        # Calculate mean path for the forecast cone
        mean_path = [prices[-1]]  # Start with the last observed price
        for i in range(len(forecast_dates)):
            progress = (i+1) / len(forecast_dates)
            mean_price = prices[-1] * (1 + progress * price_change_pct / 100)
            mean_path.append(mean_price)
            
        # Calculate upper and lower bounds for the forecast cone
        upper_path = [prices[-1]]
        lower_path = [prices[-1]]
        for i in range(len(forecast_dates)):
            progress = (i+1) / len(forecast_dates)
            volatility_factor = 1 + (progress * volatility_level * 0.5)
            upper_path.append(mean_path[i+1] * volatility_factor)
            lower_path.append(mean_path[i+1] / volatility_factor)
            
        if simulated_X is not None and hasattr(simulated_X, '__len__') and len(simulated_X) > 0:
            paths = simulated_X
            path_dates = forecast_dates[:len(paths)]
        else:
            num_paths = 30
            paths = []
            current_price = prices[-1]
            for i in range(num_paths):
                price_path = []
                price = current_price
                bias = price_change_pct / 100 / len(forecast_dates)
                direction_bias = bias * (1 + (i % 3))
                for j in range(len(forecast_dates)):
                    rand_factor = np.random.normal(0, volatility)
                    price = price * (1 + direction_bias + rand_factor)
                    price_path.append(price)
                paths.append(price_path)
            path_dates = forecast_dates
        num_display_paths = min(15, len(paths))
        path_sample = np.random.choice(len(paths), num_display_paths, replace=False)
        # Draw the path segments more clearly based on direction
        # For cleaner visualization, we'll remove the individual path lines
        # and just focus on the forecast cone with good gradient
        # Clean forecast cone with elegant gradient
        if price_change_pct < 0:
            # For downward trend, use a smooth, elegant gradient
            from matplotlib.colors import LinearSegmentedColormap
            
            # Create a custom gradient for the forecast cone
            gradient_colors = [(1, 0, 0, 0.02), (1, 0, 0, 0.15)]  # Red with increasing alpha
            custom_cmap = LinearSegmentedColormap.from_list('custom_red', gradient_colors)
            
            # Create a clean, smooth gradient fill
            x = np.array(mdates.date2num(cone_dates))
            y1 = np.array(upper_path)
            y2 = np.array(lower_path)
            
            # Use a polygon for cleaner fill
            ax_price.fill_between(cone_dates, upper_path, lower_path, 
                              color=direction_color, alpha=0.12, 
                              zorder=8, linewidth=0)
            
            # Add subtle edge lines for clarity
            ax_price.plot(cone_dates, upper_path, color=direction_color, 
                       alpha=0.2, linewidth=0.75, linestyle='-', zorder=8)
            ax_price.plot(cone_dates, lower_path, color=direction_color, 
                       alpha=0.2, linewidth=0.75, linestyle='-', zorder=8)
        else:
            # Standard cone fill for upward trends - cleaner version
            ax_price.fill_between(cone_dates, upper_path, lower_path,
                               color=direction_color, alpha=0.10, zorder=8)
            
            # Subtle edge lines
            ax_price.plot(cone_dates, upper_path, color=direction_color, 
                       alpha=0.15, linewidth=0.5, linestyle='-', zorder=8)
            ax_price.plot(cone_dates, lower_path, color=direction_color, 
                       alpha=0.15, linewidth=0.5, linestyle='-', zorder=8)
        # Mean path, upper path, and lower path are already calculated above
            
        # Add clarity to the forecast endpoint
        if price_change_pct < 0:
            # For downward trends, use a more prominent endpoint
            end_marker = 'v' if signal == 'HOLD' else self.markers.get(signal, 'v')
            # Draw a clear ending diamond for the forecast
            ax_price.scatter([forecast_date], [forecast_price], 
                          marker=end_marker, 
                          s=250, color=direction_color, zorder=15, 
                          edgecolor='white', linewidth=2,
                          path_effects=[Stroke(linewidth=4, foreground=direction_color, alpha=0.3)])
        else:
            # Standard endpoint for upward trends
            ax_price.scatter([forecast_date], [forecast_price], 
                          marker=self.markers.get(signal, 'o'),
                          s=200, color=signal_color, zorder=15, 
                          edgecolor='white', linewidth=2,
                          path_effects=[Stroke(linewidth=4, foreground=signal_color, alpha=0.3)])
        
        # Improve visibility of bounds
        # Upper bound
        ax_price.scatter([forecast_date], [upper_bound], marker='_', 
                      s=150, color=direction_color, zorder=14, 
                      linewidth=3.0)
        # Lower bound  
        ax_price.scatter([forecast_date], [lower_bound], marker='_', 
                      s=150, color=direction_color, zorder=14, 
                      linewidth=3.0)
        
        # Single clean price trend line - matching the reference image
        if price_change_pct < 0:
            # Ultra-clean downward trend - single solid line with no extra elements
            ax_price.plot(cone_dates, mean_path, color=direction_color, 
                      linewidth=2.0, alpha=0.95, linestyle='-',
                      zorder=9, solid_capstyle='round')
        else:
            # Upward trend - clean dashed line
            ax_price.plot(cone_dates, mean_path, color=direction_color, 
                      linewidth=2.0, alpha=0.8, linestyle='--',
                      path_effects=[Stroke(linewidth=3, foreground=direction_color, alpha=0.2)],
                      zorder=9, solid_capstyle='round')
                       
        # We already added the forecast endpoint and bound indicators above,
        # so we don't need to add them again here
        # Cleaner price forecast display with better positioning
        if price_change_pct >= 0:
            price_y_offset = 20  # Move up for positive change
            price_text = ax_price.annotate(f"${forecast_price:,.2f}", 
                                        xy=(forecast_date, forecast_price),
                                        xytext=(15, price_y_offset), textcoords='offset points',
                                        fontweight='bold', fontsize=14, color=self.palette['text'],
                                        path_effects=[withStroke(linewidth=4, foreground=self.palette['panel'])],
                                        bbox=dict(boxstyle="round,pad=0.4", fc=self.palette['panel'], 
                                               ec=signal_color, alpha=0.8, linewidth=2),
                                        ha='center', zorder=16)
        else:
            price_y_offset = -35  # Move down for negative change
            price_text = ax_price.annotate(f"${forecast_price:,.2f}", 
                                        xy=(forecast_date, forecast_price),
                                        xytext=(15, price_y_offset), textcoords='offset points',
                                        fontweight='bold', fontsize=14, color=self.palette['text'],
                                        path_effects=[withStroke(linewidth=4, foreground=self.palette['panel'])],
                                        bbox=dict(boxstyle="round,pad=0.4", fc=self.palette['panel'], 
                                               ec=signal_color, alpha=0.8, linewidth=2),
                                        ha='center', zorder=16)

        # Direction indicators - use symbols instead of text for cleaner appearance
        if price_change_pct >= 0:
            direction_arrow = "▲" if price_change_pct > 0 else ""
            change_y_offset = -20  # Position below price for positive change
        else:
            direction_arrow = "▼" if price_change_pct < 0 else ""
            change_y_offset = 20  # Position above price for negative change
            
        # Format with arrow for clearer direction indication
        change_text = f"{direction_arrow} {price_change_pct:+.2f}%"
        
        ax_price.annotate(change_text, 
                       xy=(forecast_date, forecast_price),
                       xytext=(15, change_y_offset), textcoords='offset points',
                       fontweight='bold', fontsize=14, color=direction_color,
                       path_effects=[withStroke(linewidth=4, foreground=self.palette['panel'])],
                       bbox=dict(boxstyle="round,pad=0.4", fc=self.palette['panel'], 
                              ec=direction_color, alpha=0.8, linewidth=2),
                       ha='center', zorder=16)
        if 'volatility' in forecast:
            volatility = forecast.get('volatility', 0) * 100
            
            # Position volatility info differently based on price direction to avoid overlap
            if price_change_pct >= 0:
                vol_y_offset = -65  # Below for positive price change
            else:
                vol_y_offset = 60   # Above for negative price change
                
            # Better volatility labels based on levels
            if volatility > 4.0:
                vol_color = self.palette['vol_high']
                vol_text = f"Volatility: {volatility:.1f}% (High)"
            elif volatility > 2.0:
                vol_color = self.palette['vol_medium']
                vol_text = f"Volatility: {volatility:.1f}% (Medium)"
            else:
                vol_color = self.palette['vol_low']
                vol_text = f"Volatility: {volatility:.1f}% (Low)"
            
            # Enhanced volatility annotation
            ax_price.annotate(vol_text,
                           xy=(forecast_date, forecast_price),
                           xytext=(15, vol_y_offset), textcoords='offset points',
                           fontweight='bold', fontsize=10, color=vol_color,
                           path_effects=[withStroke(linewidth=4, foreground=self.palette['panel'])],
                           bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.05", 
                                  fc=self.palette['panel'], 
                                  ec=vol_color, alpha=0.75, linewidth=1.5),
                           ha='center', zorder=16)
        confidence = forecast.get('confidence', 0.5)
        if abs(price_change_pct) > 10:
            movement_text = "MAJOR MOVE" if confidence > 0.7 else "POTENTIAL MAJOR MOVE"
        elif abs(price_change_pct) > 5:
            movement_text = "SIGNIFICANT CHANGE" if confidence > 0.7 else "POTENTIAL MOVEMENT"
        else:
            movement_text = "FORECAST"
        direction_text = "BULLISH" if price_change_pct > 0 else ("BEARISH" if price_change_pct < 0 else "NEUTRAL")
        if chart_type == 'breakout':
            title = f"{coin_name}: {direction_text} BREAKOUT {movement_text}"
        elif chart_type == 'volatility':
            title = f"{coin_name}: HIGH VOLATILITY {direction_text} {movement_text}"
        elif chart_type == 'momentum':
            title = f"{coin_name}: {direction_text} MOMENTUM {movement_text}"
        else:
            title = f"{coin_name}: {direction_text} {movement_text}"
        ax_price.set_title(title, fontsize=22, pad=20, 
                        fontweight='bold', color=self.palette['text'],
                        path_effects=[withStroke(linewidth=5, foreground=self.palette['panel'])])
        forecast_period = forecast.get('forecast_horizon', 1)
        period_text = "DAY" if forecast_period == 1 else "DAYS"
        subtitle = f"{forecast_period}-{period_text} PRICE PROJECTION • {datetime.now().strftime('%b %d, %Y')}"
        ax_price.annotate(subtitle,
                       xy=(0.5, 0.97),
                       xycoords='axes fraction',
                       ha='center',
                       fontsize=12,
                       color=self.palette['subtext'],
                       fontweight='bold')
        ax_price.set_ylabel('Price (USD)', fontsize=14, color=self.palette['text'], 
                        fontweight='bold')
        ax_price.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.2f}'))
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        date_locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax_price.xaxis.set_major_locator(date_locator)
        ax_price.tick_params(axis='both', which='major', labelsize=12, pad=8, 
                         length=6, width=1, color=self.palette['subtext'])
        ax_price.tick_params(axis='both', which='minor', length=3, 
                         width=0.5, color=self.palette['subtext'])
        ax_price.grid(True, axis='y', linestyle='-', linewidth=0.5, 
                   alpha=0.2, color=self.palette['subtext'])
        time_markers = min(5, len(dates) // 50 + 2)
        for i in range(time_markers):
            idx = len(dates) // time_markers * i
            if idx < len(dates) and idx > 0:
                ax_price.axvline(x=dates[idx], linestyle=':', linewidth=0.7,
                             color=self.palette['subtext'], alpha=0.3, zorder=1)
        if 'returns' in df.columns:
            window_size = min(20, len(df) // 5)
            rolling_vol = df['returns'].rolling(window=window_size).std().values * np.sqrt(252) * 100
            vol_dates = df['open_time'].values
            vol_mean = np.nanmean(rolling_vol)
            vol_std = np.nanstd(rolling_vol)
            low_threshold = max(vol_mean - 0.75 * vol_std, 0)
            high_threshold = vol_mean + 1.0 * vol_std
            low_vol_mask = rolling_vol <= low_threshold
            med_vol_mask = (rolling_vol > low_threshold) & (rolling_vol < high_threshold)
            high_vol_mask = rolling_vol >= high_threshold
            ax_vol.clear()
            ax_vol.set_facecolor(self.palette['panel'])
            ax_vol.plot(vol_dates, rolling_vol, color=self.palette['secondary'], 
                     linewidth=2.5, label='Realized Volatility', alpha=0.9,
                     path_effects=[Stroke(linewidth=4, foreground=self.palette['secondary'], alpha=0.3)])
            if len(vol_dates) > 0:
                ax_vol.fill_between(vol_dates, 0, rolling_vol, where=low_vol_mask, 
                                 color=self.palette['vol_low'], alpha=0.3,
                                 label='Low Volatility')
                ax_vol.fill_between(vol_dates, 0, rolling_vol, where=med_vol_mask, 
                                 color=self.palette['vol_medium'], alpha=0.3,
                                 label='Normal Volatility')
                ax_vol.fill_between(vol_dates, 0, rolling_vol, where=high_vol_mask, 
                                 color=self.palette['vol_high'], alpha=0.3,
                                 label='High Volatility')
                ax_vol.axhline(y=low_threshold, linestyle='--', linewidth=1.5, 
                            color=self.palette['vol_low'], alpha=0.6)
                ax_vol.axhline(y=high_threshold, linestyle='--', linewidth=1.5, 
                            color=self.palette['vol_high'], alpha=0.6)
                ax_vol.annotate(f'Low: {low_threshold:.1f}%', 
                             xy=(vol_dates[len(vol_dates)//10], low_threshold),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=9, color=self.palette['vol_low'],
                             fontweight='bold')
                ax_vol.annotate(f'High: {high_threshold:.1f}%', 
                             xy=(vol_dates[len(vol_dates)//10], high_threshold),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=9, color=self.palette['vol_high'],
                             fontweight='bold')
            forecast_vol = forecast.get('volatility', 0) * 100
            ax_vol.scatter([forecast_date], [forecast_vol], color=self.palette['fibonacci'], s=120, 
                        zorder=15, marker='D', edgecolor='white', linewidth=1.5,
                        path_effects=[Stroke(linewidth=4, foreground=self.palette['fibonacci'], alpha=0.3)])
            vol_color = self.palette['vol_low']
            vol_message = "LOW VOLATILITY FORECAST"
            if forecast_vol > high_threshold:
                vol_color = self.palette['vol_high']
                vol_message = "HIGH VOLATILITY FORECAST"
            elif forecast_vol > low_threshold:
                vol_color = self.palette['vol_medium']
                vol_message = "MODERATE VOLATILITY FORECAST"
            ax_vol.annotate(f"{vol_message}: {forecast_vol:.1f}%", 
                         xy=(forecast_date, forecast_vol),
                         xytext=(15, 10), textcoords='offset points',
                         fontweight='bold', fontsize=11, color=vol_color,
                         path_effects=[withStroke(linewidth=3, foreground=self.palette['panel'])],
                         bbox=dict(boxstyle="round,pad=0.4", fc=self.palette['panel'], 
                                ec=vol_color, alpha=0.8, linewidth=1.5),
                         ha='center', zorder=16)
            if simulated_phi is not None and len(simulated_phi) > 0:
                sim_dates = [last_date + timedelta(days=i/len(simulated_phi)) for i in range(len(simulated_phi))]
                sim_vol = np.sqrt(simulated_phi) * np.sqrt(252) * 100
                points = np.array([mdates.date2num(sim_dates), sim_vol]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                ax_vol.plot(sim_dates, sim_vol, color=self.palette['volatility_edge'], 
                         linestyle='-', linewidth=2.0, alpha=0.7, 
                         label='Projected Volatility',
                         path_effects=[Stroke(linewidth=4, foreground=self.palette['volatility_edge'], alpha=0.3)])
                ax_vol.fill_between(sim_dates, 0, sim_vol, color=self.palette['volatility_edge'], alpha=0.15)
                ax_vol.scatter([sim_dates[-1]], [sim_vol[-1]], color=self.palette['volatility_edge'], 
                            s=80, zorder=11, marker='o', alpha=0.8)
        ax_vol.set_ylabel('VOLATILITY (%)', fontsize=12, color=self.palette['text'], fontweight='bold')
        ax_vol.set_xlabel('DATE', fontsize=12, color=self.palette['text'], fontweight='bold')
        ymax = max(high_threshold * 1.5, forecast.get('volatility', 0) * 100 * 1.3)
        ax_vol.set_ylim(0, ymax)
        ax_vol.grid(True, axis='y', linestyle='-', linewidth=0.5, alpha=0.2)
        ax_vol.tick_params(axis='both', which='major', labelsize=10)
        ax_indicators.clear()
        ax_indicators.axis('off')
        ax_indicators.set_facecolor(self.palette['panel'])
        current_price = forecast.get('current_price', 0)
        forecast_price = forecast.get('forecast_price', 0)
        price_change_pct = forecast.get('price_change_pct', 0)
        forecast_days = forecast.get('forecast_horizon', 1)
        volatility = forecast.get('volatility', 0) * 100
        volume_consistency = forecast.get('volume_consistency', 0)
        fragility_score = forecast.get('fragility_score', 0)
        breakout_direction = forecast.get('breakout_direction', 'unknown')
        breakout_confidence = forecast.get('breakout_confidence', 0)
        if price_change_pct > 10 and breakout_confidence > 0.7:
            signal_strength = 1.0
            signal_color = self.palette['extreme_bullish']
        elif price_change_pct > 5 and breakout_confidence > 0.6:
            signal_strength = 0.7
            signal_color = self.palette['bullish']
        elif price_change_pct < -10 and breakout_confidence > 0.7:
            signal_strength = -1.0
            signal_color = self.palette['extreme_bearish']
        elif price_change_pct < -5 and breakout_confidence > 0.6:
            signal_strength = -0.7
            signal_color = self.palette['bearish']
        else:
            signal_strength = 0
            signal_color = self.palette['neutral']
        bar_width = 0.9
        bar_height = 0.1
        bar_x = 0.05
        bar_y = 0.88
        bar_bg = Rectangle((bar_x, bar_y), bar_width, bar_height, 
                        facecolor=self.palette['background'], alpha=0.4,
                        transform=ax_indicators.transAxes)
        ax_indicators.add_patch(bar_bg)
        if signal_strength != 0:
            if signal_strength > 0:
                fill_x = 0.5
                fill_width = abs(signal_strength) * (bar_width/2)
            else:
                fill_width = abs(signal_strength) * (bar_width/2)
                fill_x = 0.5 - fill_width
            signal_bar = Rectangle((fill_x, bar_y), fill_width, bar_height, 
                            facecolor=signal_color, alpha=0.8,
                            transform=ax_indicators.transAxes)
            ax_indicators.add_patch(signal_bar)
        ax_indicators.plot([0.5, 0.5], [bar_y, bar_y + bar_height], 
                       color=self.palette['subtext'], linewidth=1, alpha=0.5,
                       transform=ax_indicators.transAxes)
        # Ultra-clean, professional price analysis panel
        metrics_bg = FancyBboxPatch((0.05, 0.5), 0.9, 0.35, 
                             boxstyle=f"round,pad=0.2,rounding_size=0.05", 
                             facecolor=self.palette['background'], alpha=0.2, 
                             transform=ax_indicators.transAxes)
        ax_indicators.add_patch(metrics_bg)
        
        # Use more precise color based on price change magnitude
        if price_change_pct > 5:
            price_color = self.palette['very_bullish']
        elif price_change_pct > 0:
            price_color = self.palette['bullish']
        elif price_change_pct < -5:
            price_color = self.palette['very_bearish']
        else:
            price_color = self.palette['bearish']
        
        # Clean, minimal heading with subtle separator line
        y_pos = 0.78
        ax_indicators.text(0.5, y_pos, "PRICE FORECAST", 
                        fontsize=10, fontweight='bold', color=self.palette['text'],
                        ha='center', transform=ax_indicators.transAxes)
        
        # Add subtle separator line below heading
        ax_indicators.plot([0.2, 0.8], [y_pos - 0.03, y_pos - 0.03], 
                       color=self.palette['subtext'], linewidth=0.5, alpha=0.3,
                       transform=ax_indicators.transAxes)
        
        # Create elegant price trajectory visualization
        y_pos -= 0.12
        start_x = 0.15
        end_x = 0.85
        arrow_y = y_pos
        
        # Subtle base line for the arrow
        ax_indicators.plot([start_x, end_x], [arrow_y, arrow_y], 
                       color=self.palette['subtext'], linewidth=0.8, alpha=0.2,
                       transform=ax_indicators.transAxes)
        
        # Starting point marker with subtle glow effect
        current_marker_size = 8
        ax_indicators.scatter([start_x], [arrow_y], 
                          s=current_marker_size+4, color=self.palette['primary'], 
                          alpha=0.15, zorder=9, marker='o',
                          transform=ax_indicators.transAxes)
        ax_indicators.scatter([start_x], [arrow_y], 
                          s=current_marker_size, color=self.palette['primary'], 
                          zorder=10, marker='o', alpha=0.8,
                          transform=ax_indicators.transAxes)
        
        # Target point with precise positioning based on price change
        movement_scale = min(max(0.05, abs(price_change_pct)/20), 0.95)
        if price_change_pct >= 0:
            target_x = start_x + (end_x - start_x) * movement_scale
        else:
            target_x = end_x - (end_x - start_x) * movement_scale
        
        # Ensure target point stays within bounds
        target_x = max(start_x + 0.05, min(end_x - 0.05, target_x))
        
        # Target marker with subtle glow effect
        target_marker_size = current_marker_size + 2
        ax_indicators.scatter([target_x], [arrow_y], 
                          s=target_marker_size+6, color=price_color, 
                          alpha=0.2, zorder=9, marker='o',
                          transform=ax_indicators.transAxes)
        ax_indicators.scatter([target_x], [arrow_y], 
                          s=target_marker_size, color=price_color, 
                          zorder=10, marker='o', alpha=0.9,
                          transform=ax_indicators.transAxes)
        
        # Elegant arrow with adaptive styling based on price change magnitude
        arrow_width = 1.5 + min(abs(price_change_pct) / 10, 1.5)
        arrow_props = dict(arrowstyle='->', color=price_color, 
                         linewidth=arrow_width, shrinkA=4, shrinkB=4)
        ax_indicators.annotate('', xy=(target_x, arrow_y), xytext=(start_x, arrow_y),
                           arrowprops=arrow_props, transform=ax_indicators.transAxes)
        
        # Clean price labels with improved positioning and styling
        ax_indicators.text(start_x, arrow_y - 0.04, f"${current_price:,.2f}", 
                        fontsize=9, color=self.palette['text'],
                        ha='center', va='top', transform=ax_indicators.transAxes)
        
        # Target price with subtle highlight
        ax_indicators.text(target_x, arrow_y - 0.04, f"${forecast_price:,.2f}", 
                        fontsize=10, color=price_color, fontweight='bold',
                        ha='center', va='top', transform=ax_indicators.transAxes,
                        path_effects=[withStroke(linewidth=3, foreground=self.palette['panel'])])
        
        # Clean percentage change with visual enhancement
        pct_y_pos = arrow_y - 0.09
        direction_arrow = "▲" if price_change_pct > 0 else ("▼" if price_change_pct < 0 else "")
        # Add a subtle background for the percentage text
        ax_indicators.add_patch(FancyBboxPatch(
            (0.4, pct_y_pos-0.015), 0.2, 0.03, 
            boxstyle=f"round,pad=0.02,rounding_size=0.02",
            facecolor=self.palette['background'], alpha=0.3, 
            transform=ax_indicators.transAxes)
        )
        ax_indicators.text(0.5, pct_y_pos, f"{direction_arrow} {price_change_pct:+.2f}%", 
                        fontsize=10, fontweight='bold', color=price_color,
                        ha='center', transform=ax_indicators.transAxes)
        # Ultra-minimal risk metrics panel matching reference design
        # Clean background with subtle border
        risk_bg = Rectangle(
            (0.05, 0.16), 0.9, 0.3, 
            facecolor=self.palette['background'], 
            alpha=0.2,
            edgecolor=self.palette['subtext'],
            linewidth=0.5,
            transform=ax_indicators.transAxes
        )
        ax_indicators.add_patch(risk_bg)
        
        # Clean, minimal heading
        y_pos = 0.4
        ax_indicators.text(0.5, y_pos, "RISK METRICS", 
                       fontsize=9, fontweight='bold', color=self.palette['text'],
                       ha='center', transform=ax_indicators.transAxes)
        
        # Define colors using precise color codes for consistency
        green_color = "#089981"  # Clean green
        red_color = "#F23645"    # Clean red
        amber_color = "#FF9800"  # Clean amber
        
        # Set starting position for metrics
        y_pos -= 0.06
        label_offset = 0.15
        value_offset = 0.85
        spacing = 0.05
        
        # Simple volatility indicator
        if volatility > 3.5:
            vol_text = "HIGH"
            vol_color = red_color
        elif volatility > 2:
            vol_text = "MODERATE"
            vol_color = amber_color
        else:
            vol_text = "LOW"
            vol_color = green_color
        
        # Add clean volatility info
        ax_indicators.text(label_offset, y_pos, "Volatility:", 
                       fontsize=9, color=self.palette['text'],
                       ha='left', transform=ax_indicators.transAxes)
        
        ax_indicators.text(value_offset, y_pos, vol_text, 
                       fontsize=9, color=vol_color, fontweight='bold',
                       ha='right', transform=ax_indicators.transAxes)
        
        # Add risk level
        y_pos -= spacing
        
        # Determine risk level based on fragility score
        if fragility_score > 70:
            risk_level = "HIGH"
            risk_color = red_color
        elif fragility_score > 50:
            risk_level = "MODERATE"
            risk_color = amber_color
        else:
            risk_level = "LOW"
            risk_color = green_color
        
        ax_indicators.text(label_offset, y_pos, "Risk Level:", 
                       fontsize=9, color=self.palette['text'],
                       ha='left', transform=ax_indicators.transAxes)
        
        ax_indicators.text(value_offset, y_pos, risk_level, 
                       fontsize=9, color=risk_color, fontweight='bold',
                       ha='right', transform=ax_indicators.transAxes)
        
        # Add market condition
        y_pos -= spacing
        
        # Determine market condition - simplified for consistency
        if fragility_score > 60:
            market_condition = "UNSTABLE"
            condition_color = red_color
        elif fragility_score > 40:
            market_condition = "CAUTIOUS"
            condition_color = amber_color
        else:
            market_condition = "STABLE"
            condition_color = green_color
        
        ax_indicators.text(label_offset, y_pos, "Market Condition:", 
                       fontsize=9, color=self.palette['text'],
                       ha='left', transform=ax_indicators.transAxes)
        
        ax_indicators.text(value_offset, y_pos, market_condition, 
                       fontsize=9, color=condition_color, fontweight='bold',
                       ha='right', transform=ax_indicators.transAxes)
        
        # Add trading environment
        y_pos -= spacing
        
        if price_change_pct > 3 and fragility_score < 50:
            trading_env = "FAVORABLE"
            env_color = green_color
        elif price_change_pct < -3 and fragility_score > 50:
            trading_env = "CHALLENGING"
            env_color = red_color
        else:
            trading_env = "NEUTRAL"
            env_color = self.palette['neutral']
        
        ax_indicators.text(label_offset, y_pos, "Trading Environment:", 
                       fontsize=9, color=self.palette['text'],
                       ha='left', transform=ax_indicators.transAxes)
        
        ax_indicators.text(value_offset, y_pos, trading_env, 
                       fontsize=9, color=env_color, fontweight='bold',
                       ha='right', transform=ax_indicators.transAxes)
        # Minimal breakout alert matching reference design
        if breakout_direction != 'unknown':
            direction_symbol = "▲" if breakout_direction == "up" else "▼"
            
            # Use precise color codes for consistency
            if breakout_direction == "up":
                break_color = "#089981"  # Clean green
            else:
                break_color = "#F23645"  # Clean red
            
            # Create clean breakout alert with rectangular design
            breakout_bg = Rectangle(
                (0.05, 0.04), 0.9, 0.08,
                facecolor=break_color, alpha=0.8,
                transform=ax_indicators.transAxes
            )
            ax_indicators.add_patch(breakout_bg)
            
            # Simple, minimal breakout text
            confidence_pct = int(breakout_confidence * 100)
            break_text = f"BREAKOUT DETECTED {direction_symbol} ({confidence_pct}%)"
            ax_indicators.text(0.5, 0.08, break_text, 
                           fontsize=9, fontweight='bold', color='white',
                           ha='center', transform=ax_indicators.transAxes)
        ax_signals.clear()
        ax_signals.axis('off')
        ax_signals.set_facecolor(self.palette['panel'])
        signal = forecast.get('signal', 'HOLD')
        confidence = forecast.get('confidence', 0)
        upside_prob = forecast.get('upside_probability', 0.5)
        downside_prob = forecast.get('downside_probability', 0.5)
        signal_colors = {
            'STRONG_BUY': self.palette['extreme_bullish'],
            'BUY': self.palette['bullish'], 
            'WEAK_BUY': self.palette['bullish'],
            'HOLD': self.palette['neutral'],
            'WEAK_SELL': self.palette['bearish'],
            'SELL': self.palette['bearish'], 
            'STRONG_SELL': self.palette['extreme_bearish']
        }
        signal_color = signal_colors.get(signal, self.palette['neutral'])
        # Ultra-minimal signal indicator matching reference design
        if signal == 'HOLD':
            # For HOLD signals, use a clean gray bar
            signal_bg = Rectangle(
                (0.05, 0.80), 0.9, 0.12, 
                facecolor=self.palette['neutral'], alpha=0.4,
                edgecolor=self.palette['subtext'],
                linewidth=0.5, 
                transform=ax_signals.transAxes
            )
            ax_signals.add_patch(signal_bg)
            
            # Simple horizontal bar for HOLD
            bar_width = 0.25
            ax_signals.plot([0.5 - bar_width/2, 0.5 + bar_width/2], [0.86, 0.86], 
                        color='white', linewidth=2, solid_capstyle='round',
                        transform=ax_signals.transAxes)
            
            # Clean minimal text
            ax_signals.text(0.5, 0.83, "NEUTRAL POSITION", fontsize=9, 
                        color='white', ha='center', va='center', 
                        transform=ax_signals.transAxes)
        else:
            # For BUY/SELL signals, use clean solid color block
            if 'BUY' in signal:
                signal_color = "#089981"  # Clean green
            else:
                signal_color = "#F23645"  # Clean red
                
            signal_bg = Rectangle(
                (0.05, 0.75), 0.9, 0.2, 
                facecolor=signal_color, alpha=0.7,
                transform=ax_signals.transAxes
            )
            ax_signals.add_patch(signal_bg)
            
            if signal in ['BUY', 'STRONG_BUY']:
                # Simple up arrow
                triangle_size = 0.06
                triangle_points = [
                    [0.5, 0.88],
                    [0.5 - triangle_size, 0.88 - triangle_size*1.5],
                    [0.5 + triangle_size, 0.88 - triangle_size*1.5]
                ]
                triangle = Polygon(triangle_points, closed=True, 
                                facecolor='white', edgecolor=None,
                                transform=ax_signals.transAxes)
                ax_signals.add_patch(triangle)
            elif signal in ['SELL', 'STRONG_SELL']:
                # Simple down arrow
                triangle_size = 0.06
                triangle_points = [
                    [0.5, 0.88 - triangle_size*1.5],
                    [0.5 - triangle_size, 0.88],
                    [0.5 + triangle_size, 0.88]
                ]
                triangle = Polygon(triangle_points, closed=True, 
                                facecolor='white', edgecolor=None,
                                transform=ax_signals.transAxes)
                ax_signals.add_patch(triangle)
            
            # Clean minimal signal text
            refined_signal = signal.replace('STRONG_', '').replace('_', ' ')
            ax_signals.text(0.5, 0.77, refined_signal, fontsize=11, fontweight='bold', 
                        color='white', ha='center', va='center', 
                        transform=ax_signals.transAxes)
        # Clean market condition indicators with small dots instead of blocks
        heat_y = 0.55
        heat_height = 0.12
        
        # Minimal backing for indicators
        ax_signals.add_patch(FancyBboxPatch(
            (0.1, heat_y), 0.8, heat_height, 
            boxstyle="round,pad=0.1,rounding_size=0.02",
            facecolor=self.palette['background'], alpha=0.2,
            transform=ax_signals.transAxes)
        )
        
        ax_signals.text(0.5, heat_y + heat_height + 0.03, "MARKET CONDITIONS", 
                     fontsize=10, color=self.palette['text'],
                     ha='center', transform=ax_signals.transAxes)
        
        # Ultra-minimal market condition indicator layout (matching reference image)
        indicator_width = 0.75
        indicator_x = 0.15
        
        # Set up indicators with extremely clean layout
        if 'fragility_score' in forecast and 'volatility' in forecast:
            # Normalize metrics for visualization
            frag_norm = min(1.0, max(0.0, fragility_score / 100))
            vol_norm = min(1.0, max(0.0, volatility / 5.0))
            volume_consistency = forecast.get('volume_consistency', 0.5)
            
            # Simple panel background for cleaner look
            ax_signals.add_patch(Rectangle(
                (indicator_x - 0.05, heat_y - 0.07), 
                indicator_width + 0.1, heat_height + 0.14, 
                facecolor=self.palette['panel'], 
                edgecolor=self.palette['subtext'],
                linewidth=0.5,
                alpha=0.3, zorder=2,
                transform=ax_signals.transAxes)
            )
            
            # Evenly spaced dots
            dot_spacing = indicator_width / 5
            
            # Minimal styling
            dot_size = 28  # Smaller dots for cleaner look
            dot_edge_color = 'white' 
            dot_edge_width = 0.8
            label_fontsize = 7  # Smaller text for cleaner appearance
            label_y = heat_y - 0.05
            indicators_y = heat_y + heat_height/2
            
            # Use a single consistent colormap for all indicators
            if self.theme == "dark":
                stable_color = "#089981"  # Green
                unstable_color = "#F23645"  # Red
            else:
                stable_color = "#089981"  # Green
                unstable_color = "#F23645"  # Red
                
            # Draw small horizontal line as minimal separators between indicators
            for i in range(1, 5):
                sep_x = indicator_x + dot_spacing * i - dot_spacing/2
                ax_signals.plot([sep_x-0.02, sep_x+0.02], 
                             [indicators_y, indicators_y], 
                             color=self.palette['subtext'], alpha=0.2, linewidth=0.5,
                             transform=ax_signals.transAxes)
                
            # STABILITY - minimal clean dot
            stability_str = "LOW" if frag_norm > 0.6 else "STABLE"
            stability_color = unstable_color if frag_norm > 0.6 else stable_color
            stability_x = indicator_x + dot_spacing * 0.5
            ax_signals.scatter([stability_x], [indicators_y], 
                            s=dot_size, color=stability_color, 
                            edgecolor=dot_edge_color, linewidth=dot_edge_width,
                            alpha=0.9, zorder=10, transform=ax_signals.transAxes)
            ax_signals.text(stability_x, label_y, "Stability", 
                         fontsize=label_fontsize, color=self.palette['text'],
                         ha='center', transform=ax_signals.transAxes)
            
            # VOLATILITY - minimal styling
            vol_str = "HIGH" if vol_norm > 0.6 else "LOW"
            vol_color = unstable_color if vol_norm > 0.6 else stable_color
            vol_x = indicator_x + dot_spacing * 1.5
            ax_signals.scatter([vol_x], [indicators_y], 
                            s=dot_size, color=vol_color, 
                            edgecolor=dot_edge_color, linewidth=dot_edge_width,
                            alpha=0.9, zorder=10, transform=ax_signals.transAxes)
            ax_signals.text(vol_x, label_y, "Volatility", 
                         fontsize=label_fontsize, color=self.palette['text'],
                         ha='center', transform=ax_signals.transAxes)
            
            # TREND - minimal styling
            trend_strength = abs(price_change_pct / 10.0)
            trend_str = "STRONG" if trend_strength > 0.6 else "WEAK"
            trend_color = stable_color if price_change_pct > 0 else unstable_color
            trend_x = indicator_x + dot_spacing * 2.5
            ax_signals.scatter([trend_x], [indicators_y], 
                            s=dot_size, color=trend_color, 
                            edgecolor=dot_edge_color, linewidth=dot_edge_width,
                            alpha=0.9, zorder=10, transform=ax_signals.transAxes)
            ax_signals.text(trend_x, label_y, "Trend", 
                         fontsize=label_fontsize, color=self.palette['text'],
                         ha='center', transform=ax_signals.transAxes)
            
            # LIQUIDITY - minimal styling
            if 'volume_consistency' in forecast:
                liquidity_str = "HIGH" if volume_consistency > 0.6 else "LOW"
                liquidity_color = stable_color if volume_consistency > 0.6 else unstable_color
                liquidity_x = indicator_x + dot_spacing * 3.5
                ax_signals.scatter([liquidity_x], [indicators_y], 
                                s=dot_size, color=liquidity_color, 
                                edgecolor=dot_edge_color, linewidth=dot_edge_width,
                                alpha=0.9, zorder=10, transform=ax_signals.transAxes)
                ax_signals.text(liquidity_x, label_y, "Liquidity", 
                             fontsize=label_fontsize, color=self.palette['text'],
                             ha='center', transform=ax_signals.transAxes)
            
            # BREAKOUT - minimal styling
            if 'breakout_confidence' in forecast:
                breakout_str = "HIGH" if breakout_confidence > 0.6 else "LOW"
                if breakout_direction == 'up':
                    breakout_color = stable_color if breakout_confidence > 0.6 else self.palette['subtext']
                else:
                    breakout_color = unstable_color if breakout_confidence > 0.6 else self.palette['subtext']
                breakout_x = indicator_x + dot_spacing * 4.5
                ax_signals.scatter([breakout_x], [indicators_y], 
                                s=dot_size, color=breakout_color, 
                                edgecolor=dot_edge_color, linewidth=dot_edge_width,
                                alpha=0.9, zorder=10, transform=ax_signals.transAxes)
                ax_signals.text(breakout_x, label_y, "Breakout", 
                             fontsize=label_fontsize, color=self.palette['text'],
                             ha='center', transform=ax_signals.transAxes)
        # Minimalist fragility score meter matching reference design
        frag_y = 0.3
        meter_height = 0.03  # Extra thin for modern look
        
        # Only show if fragility score is available
        if 'fragility_score' in forecast:
            # Simple risk level determination
            if fragility_score > 70:
                risk_text = "HIGH RISK"
                frag_color = "#F23645"  # Red
            elif fragility_score > 50:
                risk_text = "MODERATE RISK" 
                frag_color = "#FF9800"  # Orange
            else:
                risk_text = "LOW RISK"
                frag_color = "#089981"  # Green
            
            # Create clean minimal track
            ax_signals.add_patch(Rectangle(
                (0.1, frag_y), 0.8, meter_height, 
                facecolor=self.palette['background'],
                edgecolor=self.palette['subtext'],
                linewidth=0.5,
                alpha=0.15, 
                transform=ax_signals.transAxes
            ))
            
            # Calculate width for filled portion
            frag_width = 0.8 * (fragility_score / 100)
            
            # Create a simple solid fill (no gradient) for cleaner look
            ax_signals.add_patch(Rectangle(
                (0.1, frag_y), frag_width, meter_height,
                facecolor=frag_color, 
                alpha=0.8, 
                transform=ax_signals.transAxes
            ))
            
            # Add a simple triangle marker at the position
            marker_x = 0.1 + frag_width
            marker_y = frag_y + meter_height/2
            marker_size = 0.01
            marker_points = [
                [marker_x, marker_y + marker_size],
                [marker_x - marker_size, marker_y - marker_size],
                [marker_x + marker_size, marker_y - marker_size]
            ]
            ax_signals.add_patch(Polygon(
                marker_points, 
                closed=True,
                facecolor='white',
                edgecolor=frag_color,
                linewidth=0.5,
                transform=ax_signals.transAxes,
                zorder=11
            ))
            
            # Clean minimalist label for meter
            ax_signals.text(0.5, frag_y + meter_height + 0.03, "MARKET FRAGILITY", 
                         fontsize=8, color=self.palette['text'], fontweight='bold',
                         ha='center', transform=ax_signals.transAxes)
            
            # Score in cleaner format - digits only
            score_display = f"{fragility_score:.0f}/100"
            ax_signals.text(0.17, frag_y + meter_height/2, score_display, 
                         fontsize=8, color='white', fontweight='bold',
                         ha='left', va='center', transform=ax_signals.transAxes,
                         path_effects=[withStroke(linewidth=2, foreground=self.palette['panel'])])
            
            # Risk level with cleaner positioning
            ax_signals.text(0.87, frag_y + meter_height/2, risk_text, 
                         fontsize=8, color='white', fontweight='bold',
                         ha='right', va='center', transform=ax_signals.transAxes,
                         path_effects=[withStroke(linewidth=2, foreground=self.palette['panel'])])
        # Minimal recommended action panel matching reference design
        action_y = 0.1
        action_title = "RECOMMENDED ACTION"
        
        # Clean rectangular background
        action_bg = Rectangle(
            (0.05, action_y), 0.9, 0.15,
            facecolor=self.palette['background'], 
            alpha=0.2,
            edgecolor=self.palette['subtext'],
            linewidth=0.5,
            transform=ax_signals.transAxes
        )
        ax_signals.add_patch(action_bg)
        
        # Simplified action text options
        if signal in ['BUY', 'STRONG_BUY']:
            action_text = "MAINTAIN POSITION"
            sub_action = "Monitor for breakout signals"
        elif signal in ['SELL', 'STRONG_SELL']:
            action_text = "MAINTAIN POSITION"
            sub_action = "Monitor for breakout signals"
        else:
            action_text = "MAINTAIN POSITION"
            sub_action = "Monitor for breakout signals"
        
        # Clean minimal heading
        ax_signals.text(0.5, action_y + 0.10, action_title, 
                     fontsize=8, color=self.palette['text'], 
                     fontweight='bold', ha='center', transform=ax_signals.transAxes)
        
        # Add main action with clean styling
        ax_signals.text(0.5, action_y + 0.055, action_text, 
                     fontsize=10, color=self.palette['text'], 
                     fontweight='bold', ha='center', transform=ax_signals.transAxes)
        
        # Add minimal sub-action text
        ax_signals.text(0.5, action_y + 0.02, sub_action, 
                     fontsize=7, color=self.palette['subtext'], 
                     ha='center', transform=ax_signals.transAxes)
        if chart_type == 'breakout' and ax_direction is not None:
            ax_direction.clear()
            ax_direction.axis('off')
            ax_direction.set_facecolor(self.palette['panel'])
            breakout_direction = forecast.get('breakout_direction', 'unknown')
            breakout_confidence = forecast.get('breakout_confidence', 0.5)
            if breakout_direction != 'unknown':
                if breakout_direction == 'up':
                    arrow_color = self.palette['breakout_up']
                else:
                    arrow_color = self.palette['breakout_down']
                breakout_title = "BREAKOUT"
                ax_direction.text(0.5, 0.8, breakout_title, 
                              fontsize=14, color=arrow_color, fontweight='bold',
                              ha='center', transform=ax_direction.transAxes)
                arrow_width = 0.6
                arrow_height = 0.25
                arrow_x_center = 0.5
                arrow_y_center = 0.4
                if breakout_direction == 'up':
                    triangle_points = [
                        [arrow_x_center, arrow_y_center + arrow_height/2],
                        [arrow_x_center - arrow_width/2, arrow_y_center - arrow_height/2],
                        [arrow_x_center + arrow_width/2, arrow_y_center - arrow_height/2]
                    ]
                else:
                    triangle_points = [
                        [arrow_x_center, arrow_y_center - arrow_height/2],
                        [arrow_x_center - arrow_width/2, arrow_y_center + arrow_height/2],
                        [arrow_x_center + arrow_width/2, arrow_y_center + arrow_height/2]
                    ]
                triangle = Polygon(triangle_points, closed=True, 
                                facecolor=arrow_color, edgecolor=arrow_color, 
                                alpha=0.8, transform=ax_direction.transAxes)
                ax_direction.add_patch(triangle)
                meter_width = 0.6 * breakout_confidence
                meter_y = 0.15
                meter_height = 0.1
                meter_bg = Rectangle((0.2, meter_y), 0.6, meter_height, 
                                  facecolor=self.palette['background'], alpha=0.3, 
                                  transform=ax_direction.transAxes)
                ax_direction.add_patch(meter_bg)
                meter_fill = Rectangle((0.2, meter_y), meter_width, meter_height, 
                                    facecolor=arrow_color, alpha=0.6, 
                                    transform=ax_direction.transAxes)
                ax_direction.add_patch(meter_fill)
                ax_direction.text(0.5, meter_y - 0.1, f"CONFIDENCE: {breakout_confidence:.0%}", 
                              fontsize=10, color=self.palette['text'],
                              ha='center', transform=ax_direction.transAxes)
        if chart_type == 'volatility' and ax_strategy is not None:
            ax_strategy.clear()
            ax_strategy.axis('off')
            ax_strategy.set_facecolor(self.palette['panel'])
            volatility = forecast.get('volatility', 0) * 100
            fragility_score = forecast.get('fragility_score', 50)
            strategy_text = "VOLATILITY STRATEGY: "
            if volatility > 4:
                if fragility_score > 70:
                    strategy = "Avoid directional bets. Consider volatility plays with defined risk."
                    strategy_color = self.palette['vol_high']
                else:
                    strategy = "Use wider stops. Reduce position sizing until volatility subsides."
                    strategy_color = self.palette['vol_medium']
            else:
                if price_change_pct > 0:
                    strategy = "Normal position sizing with standard risk management."
                    strategy_color = self.palette['bullish']
                else:
                    strategy = "Maintain cautious approach with defined risk parameters."
                    strategy_color = self.palette['bearish']
            ax_strategy.text(0.5, 0.6, strategy_text, 
                          fontsize=12, color=self.palette['text'], fontweight='bold',
                          ha='center', transform=ax_strategy.transAxes)
            ax_strategy.text(0.5, 0.3, strategy, 
                          fontsize=10, color=strategy_color, fontweight='bold',
                          ha='center', transform=ax_strategy.transAxes)
        fig.text(0.985, 0.015, "AlphaEngine Premium", fontsize=12, 
               color=self.palette['secondary'], alpha=0.9, ha='right', va='bottom', 
               fontweight='bold', fontfamily='sans-serif',
               path_effects=[withStroke(linewidth=3, foreground=self.palette['background'])])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        fig.text(0.015, 0.015, f"Generated: {timestamp} UTC", fontsize=9, 
               color=self.palette['subtext'], alpha=0.7, ha='left', va='bottom',
               fontfamily='monospace')
        settings_text = f"Chart Type: {chart_type.upper()} • Horizon: {forecast_days}D • Alpha: {forecast.get('alpha', 0):.2f}"
        fig.text(0.5, 0.015, settings_text, fontsize=8, 
               color=self.palette['subtext'], alpha=0.7, ha='center', va='bottom')
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.15, wspace=0.05)
        plt.savefig(filename, facecolor=self.palette['background'], 
                  edgecolor='none', dpi=200, bbox_inches='tight')
        print(f"Chart saved as {filename}")
        plt.close(fig)
        return filename
def plot_enhanced_forecast(df, forecast, simulated_X=None, simulated_phi=None, filename='crypto_forecast.png'):
    """
    Generate and save an advanced chart visualization for crypto price forecasts
    Parameters:
    -----------
    df : pandas.DataFrame
        Historical price data including at minimum 'open_time' and 'close' columns
    forecast : dict
        Dictionary with forecast data including 'forecast_price', 'upper_bound', etc.
    simulated_X : numpy.ndarray, optional
        Simulated price path if available
    simulated_phi : numpy.ndarray, optional 
        Simulated volatility path if available
    filename : str, optional
        Output file name
    Returns:
    --------
    str : Path to the saved chart image
    """
    visualizer = AdvancedChartVisualizer()
    return visualizer.create_forecast_chart(df, forecast, simulated_X, simulated_phi, filename)
if __name__ == '__main__':
    print("Advanced Chart Visualizer for Cryptocurrency Forecasts")
    print("This file is intended to be imported and used by other modules.")
