import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import warnings

class FragilityScoreCalculator:
    """
    A sophisticated module to detect potential breakouts in cryptocurrency prices.
    
    This module analyzes multiple factors including volatility compression, volume anomalies,
    price patterns, liquidity metrics, and market microstructure to quantify the
    likelihood of an imminent price breakout.
    
    The fragility score ranges from 0-100, with higher scores indicating greater
    likelihood of a significant price movement.pip
    """
    
    def __init__(self, sensitivity=1.0):
        """
        Initialize the FragilityScoreCalculator.
        
        Parameters:
        -----------
        sensitivity : float
            Multiplier to adjust the overall sensitivity of the detection.
            Higher values increase sensitivity (may increase false positives).
            Lower values decrease sensitivity (more conservative).
            Default is 1.0 (balanced).
        """
        self.sensitivity = np.clip(sensitivity, 0.5, 2.0)
        self.history = []
        self.last_update = None
        self.false_positive_dampener = 0.8
        
        # Weights for different components
        self.weights = {
            'vol_compression': 20,
            'pattern_recognition': 20,
            'volume_analysis': 15,
            'liquidity_imbalance': 15, 
            'momentum_divergence': 10,
            'support_resistance': 10,
            'microstructure': 10
        }
        
        # Ensure weights sum to 100
        weight_sum = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] = self.weights[k] * 100 / weight_sum
    
    def calculate_fragility_score(self, df, liquidity_profile=None):
        """
        Calculate the overall fragility score based on multiple factors.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        liquidity_profile : dict, optional
            Dictionary containing liquidity metrics from AlphaEngine
            
        Returns:
        --------
        dict: Dictionary containing:
            - overall_score: Overall fragility score (0-100)
            - component_scores: Individual component scores
            - breakout_direction: Predicted direction if breakout occurs ('up', 'down', or 'unknown')
            - confidence: Confidence in the prediction (0-1)
        """
        if len(df) < 100:
            # Not enough data for reliable analysis
            return self._create_default_result()
        
        # Store timestamp of calculation
        self.last_update = datetime.now()
        
        try:
            # Check if we have the necessary columns
            required_cols = ['close', 'returns']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {set(required_cols) - set(df.columns)}")
            
            # 1. Volatility compression component
            vol_score, vol_direction = self._calculate_volatility_compression(df)
            
            # 2. Price pattern recognition component
            pattern_score, pattern_direction = self._identify_price_patterns(df)
            
            # 3. Volume analysis component
            volume_score, volume_direction = self._analyze_volume_patterns(df)
            
            # 4. Liquidity imbalance component
            liquidity_score, liquidity_direction = self._assess_liquidity_imbalance(df, liquidity_profile)
            
            # 5. Momentum divergence component
            momentum_score, momentum_direction = self._detect_momentum_divergence(df)
            
            # 6. Support/resistance proximity component
            sr_score, sr_direction = self._evaluate_support_resistance_proximity(df)
            
            # 7. Market microstructure component
            micro_score, micro_direction = self._analyze_market_microstructure(df)
            
            # Collect all component scores
            component_scores = {
                'volatility_compression': vol_score,
                'price_patterns': pattern_score,
                'volume_analysis': volume_score,
                'liquidity_imbalance': liquidity_score,
                'momentum_divergence': momentum_score,
                'support_resistance': sr_score,
                'market_microstructure': micro_score
            }
            
            # Calculate weighted overall score
            overall_score = (
                self.weights['vol_compression'] * vol_score +
                self.weights['pattern_recognition'] * pattern_score +
                self.weights['volume_analysis'] * volume_score +
                self.weights['liquidity_imbalance'] * liquidity_score +
                self.weights['momentum_divergence'] * momentum_score +
                self.weights['support_resistance'] * sr_score +
                self.weights['microstructure'] * micro_score
            ) / 100.0
            
            # Apply sensitivity adjustment
            overall_score = min(100, overall_score * self.sensitivity)
            
            # Determine breakout direction from component signals
            directions = [
                (vol_direction, self.weights['vol_compression']),
                (pattern_direction, self.weights['pattern_recognition']),
                (volume_direction, self.weights['volume_analysis']),
                (liquidity_direction, self.weights['liquidity_imbalance']),
                (momentum_direction, self.weights['momentum_divergence']),
                (sr_direction, self.weights['support_resistance']),
                (micro_direction, self.weights['microstructure'])
            ]
            
            breakout_direction = self._determine_consensus_direction(directions)
            
            # Calculate confidence in prediction
            confidence = self._calculate_confidence(component_scores, overall_score)
            
            # Apply adaptive thresholding to reduce false positives
            overall_score = self._apply_false_positive_filtering(overall_score, confidence)
            
            # Update history for future reference
            self._update_history(overall_score, breakout_direction, confidence)
            
            return {
                'overall_score': overall_score,
                'component_scores': component_scores,
                'breakout_direction': breakout_direction,
                'confidence': confidence
            }
            
        except Exception as e:
            warnings.warn(f"Error calculating fragility score: {str(e)}")
            return self._create_default_result()
    
    def _calculate_volatility_compression(self, df):
        """
        Calculate volatility compression score.
        
        Volatility compression (periods of abnormally low volatility) often
        precedes significant breakouts.
        """
        try:
            # Calculate volatility over different timeframes
            short_vol = df['returns'].rolling(14).std().iloc[-30:].mean()
            medium_vol = df['returns'].rolling(30).std().iloc[-30:].mean()
            long_vol = df['returns'].rolling(60).std().iloc[-30:].mean()
            
            # Detect volatility compression relative to historical levels
            recent_vol = short_vol
            normal_vol = long_vol
            
            # Volatility compression ratio (lower means more compression)
            vol_ratio = recent_vol / normal_vol if normal_vol > 0 else 1.0
            
            # Calculate variability of recent volatility
            recent_vol_variability = df['returns'].rolling(14).std().iloc[-14:].std()
            vol_stability = 1.0 - min(1.0, recent_vol_variability / (short_vol + 1e-10))
            
            # Low and decreasing volatility scores highest
            compression_score = max(0, min(100, 80 * (1.0 - min(1.0, vol_ratio)) + 20 * vol_stability))
            
            # Detect potential direction
            if len(df) >= 30:
                # Look at price position within recent volatility bands
                close = df['close'].iloc[-1]
                upper_band = df['close'].iloc[-15:].mean() + 2 * short_vol * df['close'].iloc[-15:].mean()
                lower_band = df['close'].iloc[-15:].mean() - 2 * short_vol * df['close'].iloc[-15:].mean()
                
                band_position = (close - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 0 else 0.5
                
                if band_position > 0.7:
                    direction = "up"  # Close to upper band, may break upward
                elif band_position < 0.3:
                    direction = "down"  # Close to lower band, may break downward
                else:
                    direction = "unknown"
            else:
                direction = "unknown"
                
            return compression_score, direction
        
        except Exception as e:
            warnings.warn(f"Error in volatility compression calculation: {str(e)}")
            return 0, "unknown"
    
    def _identify_price_patterns(self, df):
        """
        Identify chart patterns that typically precede breakouts.
        
        Detects patterns like triangles, flags, pennants, and
        rectangle consolidations.
        """
        try:
            if len(df) < 30:
                return 0, "unknown"
                
            # Extract recent price action for pattern detection
            closes = df['close'].values
            highs = df['high'].values if 'high' in df.columns else closes
            lows = df['low'].values if 'low' in df.columns else closes
            
            # Calculate trendlines for highs and lows
            x_high = np.arange(30)
            x_low = np.arange(30)
            y_high = highs[-30:]
            y_low = lows[-30:]
            
            # Calculate slopes of high and low trendlines
            high_slope, high_intercept, _, _, _ = stats.linregress(x_high, y_high)
            low_slope, low_intercept, _, _, _ = stats.linregress(x_low, y_low)
            
            # Normalize slopes to percentage of price
            avg_price = np.mean(closes[-30:])
            norm_high_slope = high_slope / avg_price if avg_price > 0 else 0
            norm_low_slope = low_slope / avg_price if avg_price > 0 else 0
            
            # Detect triangle patterns (converging trendlines)
            convergence = abs(norm_high_slope - norm_low_slope)
            triangle_score = min(100, max(0, 80 * (1.0 - convergence * 100)))
            
            # Detect rectangle patterns (flat trendlines)
            rectangle_score = min(100, max(0, 80 * (1.0 - (abs(norm_high_slope) + abs(norm_low_slope)) * 100)))
            
            # Check for breakout from recent ranges
            recent_range = np.max(highs[-20:]) - np.min(lows[-20:])
            range_percent = recent_range / avg_price if avg_price > 0 else 0
            
            # Lower range indicates tighter consolidation (higher score)
            consolidation_score = min(100, max(0, 100 * (1.0 - min(0.1, range_percent) / 0.1)))
            
            # Find fractal patterns (swing highs/lows)
            num_swings = self._count_price_swings(closes[-40:])
            swing_density = min(10, num_swings) / 10
            swing_score = min(100, max(0, 70 * swing_density))
            
            # Calculate overall pattern score
            pattern_score = 0.4 * triangle_score + 0.3 * rectangle_score + 0.2 * consolidation_score + 0.1 * swing_score
            
            # Determine likely breakout direction based on pattern
            if high_slope > 0 and low_slope > 0:
                direction = "up"  # Ascending triangle or ascending consolidation
            elif high_slope < 0 and low_slope < 0:
                direction = "down"  # Descending triangle or descending consolidation
            elif high_slope < 0 and low_slope > 0:
                direction = "unknown"  # Symmetrical triangle, direction unclear
            else:
                # Look at recent momentum
                recent_returns = np.mean(df['returns'].iloc[-10:])
                direction = "up" if recent_returns > 0 else "down"
            
            return pattern_score, direction
            
        except Exception as e:
            warnings.warn(f"Error in price pattern identification: {str(e)}")
            return 0, "unknown"
    
    def _analyze_volume_patterns(self, df):
        """
        Analyze volume patterns for signs of accumulation or distribution.
        
        Volume often contracts before breakouts and expands at the start of a move.
        """
        try:
            if 'volume' not in df.columns or len(df) < 30:
                return 0, "unknown"
                
            volume = df['volume'].values
            closes = df['close'].values
            
            # Check for volume contraction (often precedes breakouts)
            recent_vol = np.mean(volume[-10:])
            prior_vol = np.mean(volume[-30:-10])
            vol_contraction = 1.0 - min(1.5, recent_vol / prior_vol) / 1.5 if prior_vol > 0 else 0
            
            # Check for volume/price divergence
            price_change = (closes[-1] / closes[-20] - 1) if closes[-20] > 0 else 0
            volume_change = (np.mean(volume[-5:]) / np.mean(volume[-20:-5]) - 1) if np.mean(volume[-20:-5]) > 0 else 0
            
            # Volume increasing while price stagnates can signal accumulation
            divergence = abs(volume_change - price_change)
            divergence_score = min(100, divergence * 200)
            
            # Check for unusual volume spikes
            avg_volume = np.mean(volume[-30:])
            std_volume = np.std(volume[-30:])
            recent_spikes = np.sum(volume[-5:] > (avg_volume + 2 * std_volume))
            spike_score = min(100, recent_spikes * 25)
            
            # Check volume consistency - inconsistent volume often precedes breakouts
            volume_cv = np.std(volume[-15:]) / np.mean(volume[-15:]) if np.mean(volume[-15:]) > 0 else 0
            consistency_score = min(100, volume_cv * 100)
            
            # Calculate overall volume score
            volume_score = 0.4 * vol_contraction * 100 + 0.3 * divergence_score + 0.2 * spike_score + 0.1 * consistency_score
            
            # Determine direction
            if volume_change > 0 and price_change > 0:
                direction = "up"  # Rising prices with rising volume = bullish
            elif volume_change > 0 and price_change < 0:
                direction = "down"  # Falling prices with rising volume = bearish
            elif volume_change < 0 and price_change < 0:
                direction = "up"  # Falling prices with falling volume = potential reversal up
            else:
                # Look at recent price action for direction
                recent_returns = np.sum(df['returns'].iloc[-5:])
                direction = "up" if recent_returns > 0 else "down"
            
            return volume_score, direction
            
        except Exception as e:
            warnings.warn(f"Error in volume analysis: {str(e)}")
            return 0, "unknown"
    
    def _assess_liquidity_imbalance(self, df, liquidity_profile=None):
        """
        Assess liquidity imbalances that could lead to sharp price movements.
        
        Uses liquidity profile information if available.
        """
        try:
            score = 0
            direction = "unknown"
            
            # Use provided liquidity profile if available
            if liquidity_profile is not None:
                # Lower liquidity increases fragility score
                liquidity = min(1.0, max(0.1, liquidity_profile.get('normalized_volume', 0.5) / 10000))
                consistency = liquidity_profile.get('volume_consistency', 0.5)
                spread = liquidity_profile.get('spread_estimate', 0.01)
                slippage = liquidity_profile.get('slippage_factor', 0.02)
                
                # Calculate scores based on liquidity metrics
                liquidity_score = 100 * (1.0 - liquidity)
                consistency_score = 100 * (1.0 - consistency)
                spread_score = min(100, spread * 2000)  # Higher spread = higher score
                slippage_score = min(100, slippage * 1000)  # Higher slippage = higher score
                
                # Combine scores
                score = 0.3 * liquidity_score + 0.2 * consistency_score + 0.3 * spread_score + 0.2 * slippage_score
                
                # Return early with these scores
                return score, direction
            
            # If no liquidity profile, attempt to calculate from price data
            if len(df) < 30 or 'volume' not in df.columns:
                return 50, "unknown"  # Default mid-level score with unknown direction
            
            # Estimate liquidity from available data
            price_range = 0
            if 'high' in df.columns and 'low' in df.columns:
                # Calculate average daily range as percentage
                price_range = np.mean((df['high'].iloc[-20:] - df['low'].iloc[-20:]) / df['close'].iloc[-20:])
            else:
                # Estimate from close prices
                daily_change = np.abs(df['close'].pct_change().iloc[-20:])
                price_range = np.mean(daily_change)
            
            # Higher range implies lower liquidity
            liquidity_score = min(100, price_range * 1000)
            
            # Check volume consistency
            if 'volume' in df.columns:
                volume = df['volume'].iloc[-20:].values
                volume_cv = np.std(volume) / np.mean(volume) if np.mean(volume) > 0 else 0
                consistency_score = min(100, volume_cv * 100)
            else:
                consistency_score = 50
            
            # Combined score
            score = 0.6 * liquidity_score + 0.4 * consistency_score
            
            # Direction - look at recent price trend and volume if available
            if 'volume' in df.columns and len(df) > 20:
                price_change = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1)
                volume_change = (np.mean(df['volume'].iloc[-5:]) / np.mean(df['volume'].iloc[-15:-5]) - 1)
                
                if price_change > 0 and volume_change > 0:
                    direction = "up"
                elif price_change < 0 and volume_change > 0:
                    direction = "down"
                else:
                    direction = "unknown"
            else:
                direction = "unknown"
            
            return score, direction
            
        except Exception as e:
            warnings.warn(f"Error in liquidity assessment: {str(e)}")
            return 0, "unknown"
    
    def _detect_momentum_divergence(self, df):
        """
        Detect divergences between price and momentum indicators.
        
        Divergences often signal potential reversals and breakouts.
        """
        try:
            if len(df) < 40:
                return 0, "unknown"
                
            closes = df['close'].values
            
            # Calculate basic momentum indicators
            returns = df['returns'].values
            
            # Simple momentum (rate of change)
            mom_5 = closes[-1] / closes[-6] - 1 if closes[-6] > 0 else 0
            mom_10 = closes[-1] / closes[-11] - 1 if closes[-11] > 0 else 0
            mom_20 = closes[-1] / closes[-21] - 1 if closes[-21] > 0 else 0
            
            # Calculate RSI (simplified version)
            up_moves = np.array([max(0, ret) for ret in returns[-15:]])
            down_moves = np.array([max(0, -ret) for ret in returns[-15:]])
            
            avg_up = np.mean(up_moves) if len(up_moves) > 0 else 0
            avg_down = np.mean(down_moves) if len(down_moves) > 0 else 0
            
            if avg_down > 0:
                rs = avg_up / avg_down
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100 if avg_up > 0 else 50
            
            # Find price highs/lows for divergence detection
            window = 20  # Look back this many periods to find highs/lows
            
            # Find recent and prior highs/lows in price
            recent_segment = closes[-window:]
            prior_segment = closes[-2*window:-window]
            
            recent_high_idx = np.argmax(recent_segment)
            recent_low_idx = np.argmin(recent_segment)
            prior_high_idx = np.argmax(prior_segment)
            prior_low_idx = np.argmin(prior_segment)
            
            # Calculate momentum at those points (simple price change over 5 periods)
            def momentum_at(data, idx):
                if idx >= 5 and idx < len(data):
                    return data[idx] / data[idx-5] - 1 if data[idx-5] > 0 else 0
                return 0
            
            recent_high_mom = momentum_at(recent_segment, recent_high_idx)
            recent_low_mom = momentum_at(recent_segment, recent_low_idx)
            prior_high_mom = momentum_at(prior_segment, prior_high_idx)
            prior_low_mom = momentum_at(prior_segment, prior_low_idx)
            
            # Check for divergences
            bullish_div = False
            bearish_div = False
            
            if len(recent_segment) > 0 and len(prior_segment) > 0:
                # Bearish divergence: Higher high in price, lower high in momentum
                if recent_segment[recent_high_idx] > prior_segment[prior_high_idx] and recent_high_mom < prior_high_mom:
                    bearish_div = True
                
                # Bullish divergence: Lower low in price, higher low in momentum
                if recent_segment[recent_low_idx] < prior_segment[prior_low_idx] and recent_low_mom > prior_low_mom:
                    bullish_div = True
            
            # Calculate overbought/oversold conditions from RSI
            overbought = rsi > 70
            oversold = rsi < 30
            
            # Calculate momentum score
            divergence_score = 70 if (bullish_div or bearish_div) else 0
            extremity_score = 70 if (overbought or oversold) else 0
            
            # Add in acceleration component
            accel = (mom_5 - mom_10) - (mom_10 - mom_20)
            accel_score = min(100, abs(accel) * 1000)
            
            # Overall momentum divergence score
            mom_score = 0.4 * divergence_score + 0.3 * extremity_score + 0.3 * accel_score
            
            # Determine direction
            direction = "unknown"
            if bullish_div or oversold:
                direction = "up"
            elif bearish_div or overbought:
                direction = "down"
            
            return mom_score, direction
            
        except Exception as e:
            warnings.warn(f"Error in momentum divergence detection: {str(e)}")
            return 0, "unknown"
    
    def _evaluate_support_resistance_proximity(self, df):
        """
        Evaluate proximity to key support/resistance levels.
        
        Breakouts often occur from these levels.
        """
        try:
            if len(df) < 60:
                return 0, "unknown"
                
            closes = df['close'].values
            current_price = closes[-1]
            
            # Identify potential support/resistance levels from price clusters
            price_history = closes[-60:]
            hist, bin_edges = np.histogram(price_history, bins=15)
            
            # Find bins with high frequency (price clusters)
            threshold = np.mean(hist) + 0.5 * np.std(hist)
            high_freq_bins = [i for i, freq in enumerate(hist) if freq > threshold]
            
            if not high_freq_bins:
                return 0, "unknown"
            
            # Calculate levels from bin edges
            levels = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in high_freq_bins]
            
            # Find closest level
            distances = [abs(level - current_price) / current_price for level in levels]
            closest_idx = np.argmin(distances)
            closest_level = levels[closest_idx]
            closest_distance = distances[closest_idx]
            
            # Calculate score based on proximity (closer = higher score)
            proximity_score = max(0, min(100, 100 * (1 - 20 * closest_distance)))
            
            # Determine if price is testing level from above or below (for direction)
            direction = "up" if current_price > closest_level else "down"
            
            # Calculate strength of level based on frequency and historical reactions
            level_bin = high_freq_bins[closest_idx]
            level_strength = hist[level_bin] / np.max(hist)
            
            # Count historical reactions from this level
            reactions = 0
            for i in range(5, len(price_history) - 1):
                if (abs(price_history[i] - closest_level) / closest_level < 0.01 and
                    abs(price_history[i+1] - price_history[i]) / price_history[i] > 0.01):
                    reactions += 1
            
            reaction_score = min(100, reactions * 25)
            
            # Combine scores
            sr_score = 0.7 * proximity_score + 0.2 * level_strength * 100 + 0.1 * reaction_score
            
            # Use recent price action to refine direction
            recent_trend = np.mean(df['returns'].iloc[-5:])
            
            if proximity_score > 70:  # Very close to a level
                # If we're very close to a level, the breakout direction might be counter to recent trend
                if direction == "up" and recent_trend < 0:
                    direction = "down"  # Potential failure at resistance
                elif direction == "down" and recent_trend > 0:
                    direction = "up"  # Potential bounce from support
            
            return sr_score, direction
            
        except Exception as e:
            warnings.warn(f"Error in support/resistance evaluation: {str(e)}")
            return 0, "unknown"
    
    def _analyze_market_microstructure(self, df):
        """
        Analyze market microstructure for signs of imminent breakouts.
        
        Examines price clusters, wick patterns, and candle formations.
        """
        try:
            if len(df) < 30:
                return 0, "unknown"
                
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                return 0, "unknown"
            
            # Examine recent candle patterns
            opens = df['open'].iloc[-10:].values
            highs = df['high'].iloc[-10:].values
            lows = df['low'].iloc[-10:].values
            closes = df['close'].iloc[-10:].values
            
            # Calculate doji patterns (indecision)
            body_sizes = np.abs(closes - opens)
            wick_sizes = (highs - np.maximum(opens, closes)) + (np.minimum(opens, closes) - lows)
            doji_ratio = np.mean(wick_sizes / (body_sizes + 1e-10))
            
            doji_score = min(100, max(0, doji_ratio * 100))
            
            # Look for engulfing patterns
            engulfing_bullish = False
            engulfing_bearish = False
            
            for i in range(1, len(opens)):
                # Bullish engulfing
                if (closes[i-1] < opens[i-1] and  # Prior candle is down
                    closes[i] > opens[i] and      # Current candle is up
                    opens[i] <= closes[i-1] and   # Current open <= prior close
                    closes[i] >= opens[i-1]):     # Current close >= prior open
                    engulfing_bullish = True
                    
                # Bearish engulfing
                if (closes[i-1] > opens[i-1] and  # Prior candle is up
                    closes[i] < opens[i] and      # Current candle is down
                    opens[i] >= closes[i-1] and   # Current open >= prior close
                    closes[i] <= opens[i-1]):     # Current close <= prior open
                    engulfing_bearish = True
            
            engulfing_score = 80 if (engulfing_bullish or engulfing_bearish) else 0
            
            # Examine consecutive candle colors (momentum)
            candle_colors = np.sign(closes - opens)
            consecutive_up = 0
            consecutive_down = 0
            
            for i in range(len(candle_colors) - 1, -1, -1):
                if candle_colors[i] > 0:
                    consecutive_up += 1
                    consecutive_down = 0
                elif candle_colors[i] < 0:
                    consecutive_down += 1
                    consecutive_up = 0
                else:
                    consecutive_up = 0
                    consecutive_down = 0
            
            momentum_score = max(consecutive_up, consecutive_down) * 15
            momentum_score = min(100, momentum_score)
            
            # Overall microstructure score
            micro_score = 0.4 * doji_score + 0.4 * engulfing_score + 0.2 * momentum_score
            
            # Determine direction
            direction = "unknown"
            if engulfing_bullish or consecutive_up >= 3:
                direction = "up"
            elif engulfing_bearish or consecutive_down >= 3:
                direction = "down"
            
            return micro_score, direction
            
        except Exception as e:
            warnings.warn(f"Error in market microstructure analysis: {str(e)}")
            return 0, "unknown"
    
    def _count_price_swings(self, prices, min_swing_size=0.01):
        """Count the number of price swings (changes in direction)"""
        if len(prices) < 3:
            return 0
            
        swings = 0
        prev_direction = np.sign(prices[1] - prices[0])
        
        for i in range(2, len(prices)):
            curr_direction = np.sign(prices[i] - prices[i-1])
            if curr_direction != prev_direction and curr_direction != 0:
                # Check if swing size is significant
                swing_size = abs(prices[i] - prices[i-1]) / prices[i-1]
                if swing_size >= min_swing_size:
                    swings += 1
                    prev_direction = curr_direction
                    
        return swings
    
    def _determine_consensus_direction(self, directions):
        """
        Determine the consensus direction from multiple component signals.
        
        Parameters:
        -----------
        directions : list of (direction, weight) tuples
        
        Returns:
        --------
        str : "up", "down", or "unknown"
        """
        up_weight = sum(weight for direction, weight in directions if direction == "up")
        down_weight = sum(weight for direction, weight in directions if direction == "down")
        
        if up_weight > down_weight * 1.5:
            return "up"
        elif down_weight > up_weight * 1.5:
            return "down"
        elif up_weight > down_weight:
            return "up"
        elif down_weight > up_weight:
            return "down"
        else:
            return "unknown"
    
    def _calculate_confidence(self, component_scores, overall_score):
        """
        Calculate confidence in the breakout prediction.
        
        Higher scores and more agreement between components = higher confidence.
        """
        # Calculate standard deviation of component scores (lower = more agreement)
        values = list(component_scores.values())
        score_std = np.std(values)
        
        # Scale standard deviation to 0-1 range (inverted, so lower std = higher confidence)
        agreement_factor = max(0, min(1, 1 - score_std / 50))
        
        # Overall score factor (higher score = higher confidence)
        score_factor = min(1, overall_score / 75)
        
        # Combine factors with weights
        confidence = 0.6 * score_factor + 0.4 * agreement_factor
        
        return confidence
    
    def _apply_false_positive_filtering(self, score, confidence):
        """
        Apply adaptive thresholding to reduce false positives.
        
        This introduces a non-linear transformation that requires a stronger
        signal for high scores, reducing false positives.
        """
        # Apply confidence-based correction
        confidence_factor = 0.5 + 0.5 * confidence
        adjusted_score = score * confidence_factor
        
        # Apply non-linear transformation to create a higher bar for extreme scores
        if adjusted_score > 60:
            # Higher scores require exponentially more signal
            threshold_factor = self.false_positive_dampener * np.exp((adjusted_score - 60) / 40)
            adjusted_score = 60 + (adjusted_score - 60) / threshold_factor
        
        # Check history if available
        if len(self.history) > 0:
            # If score suddenly spikes, be more cautious
            last_score = self.history[-1]['score']
            if adjusted_score > last_score * 2:
                adjusted_score = (adjusted_score + last_score * 2) / 3
        
        return adjusted_score
    
    def _update_history(self, score, direction, confidence):
        """Update historical record of fragility scores"""
        self.history.append({
            'timestamp': datetime.now(),
            'score': score,
            'direction': direction,
            'confidence': confidence
        })
        
        # Keep history at a manageable size
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def _create_default_result(self):
        """Create a default fragility score result when analysis fails"""
        return {
            'overall_score': 0,
            'component_scores': {
                'volatility_compression': 0,
                'price_patterns': 0,
                'volume_analysis': 0,
                'liquidity_imbalance': 0, 
                'momentum_divergence': 0,
                'support_resistance': 0,
                'market_microstructure': 0
            },
            'breakout_direction': 'unknown',
            'confidence': 0
        }

    def get_score_interpretation(self, score):
        """
        Get interpretation of a fragility score value.
        
        Parameters:
        -----------
        score : float
            Fragility score from 0-100
            
        Returns:
        --------
        str: Verbal interpretation of the score
        """
        if score < 20:
            return "Very low breakout potential. Market appears stable."
        elif score < 40:
            return "Low breakout potential. Some minor signs of instability."
        elif score < 60:
            return "Moderate breakout potential. Market showing some fragility signs."
        elif score < 80:
            return "High breakout potential. Multiple indicators suggest imminent movement."
        else:
            return "Very high breakout potential. Strong evidence of imminent significant move."
