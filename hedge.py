"""
Cryptocurrency Dynamic Hedge Scanner
This script uses the AlphaEngine forecasting model to scan random cryptocurrency
coins from Binance (or alternative sources) to find high-value investment opportunities.
It logs coins with high potential returns, high fragility scores, or strong buy signals.
"""
import os
import sys
import time
import random
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from binance.client import Client
import yfinance as yf
from alpha import AlphaEngine
import colorama
from colorama import Fore, Back, Style
import threading
import itertools
def spinner_task(stop_event, message, delay=0.1):
    """Display an animated spinner with the given message"""
    spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    while not stop_event.is_set():
        sys.stdout.write(f'\r{Fore.BLUE}{next(spinner)} {message}{Style.RESET_ALL}')
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write('\r' + ' ' * (len(message) + 10) + '\r')
    sys.stdout.flush()
colorama.init(autoreset=True)
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE
    }
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = self.COLORS[levelname] + levelname + Style.RESET_ALL
            if 'OPPORTUNITY FOUND' in str(record.msg):
                record.msg = Fore.MAGENTA + Style.BRIGHT + str(record.msg) + Style.RESET_ALL
            elif 'Analyzing' in str(record.msg):
                record.msg = Fore.BLUE + str(record.msg) + Style.RESET_ALL
            elif 'Investment opportunity' in str(record.msg):
                record.msg = Fore.GREEN + Style.BRIGHT + str(record.msg) + Style.RESET_ALL
        return super().format(record)
file_handler = logging.FileHandler("hedge_scanner.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter('%(message)s'))
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
opportunity_logger = logging.getLogger('opportunities')
opportunity_logger.setLevel(logging.INFO)
opportunity_handler = logging.FileHandler('investment_opportunities.log')
opportunity_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
opportunity_logger.addHandler(opportunity_handler)
opportunity_logger.propagate = False
class HedgeScanner:
    def __init__(self, api_key=None, api_secret=None, min_price=0.00001, min_volume=100000):
        """
        Initialize the hedge scanner with API credentials and filtering criteria.
        Parameters:
        -----------
        api_key : str, optional
            Binance API key
        api_secret : str, optional
            Binance API secret
        min_price : float, optional
            Minimum coin price in USD to consider
        min_volume : float, optional
            Minimum daily trading volume in USD to consider
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.min_price = min_price
        self.min_volume = min_volume
        self.client = None
        self.all_symbols = []
        self.usdt_symbols = []
        self.blacklist = set(['USDT', 'BUSD', 'USDC', 'DAI', 'TUSD'])
        self.analyzed_coins = set()
        self.rate_limited = False
        self.last_api_call = 0
        self.api_call_limit = 1.0
        try:
            self.client = Client(api_key, api_secret)
            logging.info("Connected to Binance API")
        except Exception as e:
            logging.warning(f"Failed to connect to Binance API: {e}")
            self.client = None
    def fetch_all_symbols(self):
        """Fetch all available trading symbols from Binance"""
        if self.rate_limited or not self.client:
            return self._fetch_symbols_fallback()
        try:
            time_since_last_call = time.time() - self.last_api_call
            if time_since_last_call < self.api_call_limit:
                time.sleep(self.api_call_limit - time_since_last_call)
            self.last_api_call = time.time()
            exchange_info = self.client.get_exchange_info()
            symbols = []
            usdt_symbols = []
            for s in exchange_info['symbols']:
                if s['status'] == 'TRADING':
                    symbol = s['symbol']
                    symbols.append(symbol)
                    if symbol.endswith('USDT'):
                        base_asset = s['baseAsset']
                        if base_asset not in self.blacklist:
                            usdt_symbols.append(symbol)
            self.all_symbols = symbols
            self.usdt_symbols = usdt_symbols
            logging.info(f"🔄 Ready: {len(self.usdt_symbols)} USDT trading pairs available")
            return True
        except Exception as e:
            logging.error(f"⚠️ Binance API error: {str(e)[:50]}...")
            self.rate_limited = True
            return self._fetch_symbols_fallback()
    def _fetch_symbols_fallback(self):
        """Fallback method to get symbols if Binance API is unavailable"""
        try:
            logging.info("Using CoinGecko API as fallback for symbols")
            response = requests.get('https://api.coingecko.com/api/v3/coins/markets', 
                                   params={'vs_currency': 'usd', 'per_page': 250, 'page': 1})
            if response.status_code == 200:
                coins = response.json()
                usdt_symbols = []
                for coin in coins:
                    symbol = coin['symbol'].upper() + 'USDT'
                    if coin['symbol'].upper() not in self.blacklist:
                        usdt_symbols.append(symbol)
                self.usdt_symbols = usdt_symbols
                self.all_symbols = usdt_symbols
                logging.info(f"Fetched {len(self.usdt_symbols)} symbols from CoinGecko")
                return True
            else:
                logging.warning(f"CoinGecko API returned status code {response.status_code}")
                self.usdt_symbols = [
                    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 
                    'DOGEUSDT', 'SOLUSDT', 'DOTUSDT', 'AVAXUSDT', 'TRXUSDT',
                    'LINKUSDT', 'UNIUSDT', 'MATICUSDT', 'ATOMUSDT', 'LTCUSDT'
                ]
                self.all_symbols = self.usdt_symbols
                logging.warning(f"Using hardcoded list with {len(self.usdt_symbols)} common symbols")
                return True
        except Exception as e:
            logging.error(f"Error in fallback symbol fetch: {e}")
            self.usdt_symbols = [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 
                'DOGEUSDT', 'SOLUSDT', 'DOTUSDT', 'AVAXUSDT', 'TRXUSDT'
            ]
            self.all_symbols = self.usdt_symbols
            logging.warning(f"Using minimal hardcoded list with {len(self.usdt_symbols)} symbols")
            return True
    def filter_eligible_symbols(self):
        """Filter symbols based on price and volume criteria"""
        if not self.usdt_symbols:
            self.fetch_all_symbols()
        eligible_symbols = []
        if self.client and not self.rate_limited:
            try:
                time_since_last_call = time.time() - self.last_api_call
                if time_since_last_call < self.api_call_limit:
                    time.sleep(self.api_call_limit - time_since_last_call)
                self.last_api_call = time.time()
                tickers = self.client.get_ticker()
                ticker_data = {t['symbol']: t for t in tickers}
                for symbol in self.usdt_symbols:
                    if symbol in ticker_data:
                        price = float(ticker_data[symbol]['lastPrice'])
                        volume = float(ticker_data[symbol]['quoteVolume'])
                        if price >= self.min_price and volume >= self.min_volume:
                            eligible_symbols.append(symbol)
                logging.info(f"Found {len(eligible_symbols)} eligible symbols after filtering")
                return eligible_symbols
            except Exception as e:
                logging.error(f"Error filtering symbols: {e}")
                self.rate_limited = True
                return self.usdt_symbols
        else:
            logging.warning("No price/volume data available for filtering, using all USDT symbols")
            return self.usdt_symbols
    def select_random_symbol(self):
        """Select a random symbol from the eligible symbols that hasn't been analyzed yet"""
        eligible_symbols = self.filter_eligible_symbols()
        remaining_symbols = [s for s in eligible_symbols if s not in self.analyzed_coins]
        if not remaining_symbols:
            logging.info(f"🔄 Resetting analysis history ({len(eligible_symbols)} symbols)")
            self.analyzed_coins = set()
            remaining_symbols = eligible_symbols
        if remaining_symbols:
            symbol = random.choice(remaining_symbols)
            self.analyzed_coins.add(symbol)
            return symbol
        else:
            logging.warning(f"{Fore.YELLOW}⚠️ No eligible symbols found, using BTC fallback{Style.RESET_ALL}")
            return 'BTCUSDT'
    def analyze_symbol(self, symbol, max_retries=3):
        """
        Analyze a cryptocurrency symbol using AlphaEngine
        Parameters:
        -----------
        symbol : str
            The symbol to analyze (e.g., 'BTCUSDT')
        max_retries : int
            Maximum number of retry attempts if analysis fails
        Returns:
        --------
        dict or None
            Forecast results if successful, None if failed
        """
        engine = AlphaEngine(self.api_key, self.api_secret)
        coin_name = symbol.replace('USDT', '')
        logging.info(f"{Fore.CYAN}⏳ Analyzing {Style.BRIGHT}{coin_name}{Style.RESET_ALL}{Fore.CYAN}...{Style.RESET_ALL}")
        retries = 0
        while retries < max_retries:
            try:
                start_time = time.time()
                # Explicitly set the symbol for data fetching and make sure it's stored
                df = engine.fetch_data(symbol=symbol, lookback_days=14)
                # Ensure the current_coin_symbol is set before running forecast
                engine.current_coin_symbol = symbol
                # Pass the symbol explicitly to run_forecast to ensure it uses the right coin
                forecast = engine.run_forecast(symbol=symbol)
                duration = time.time() - start_time
                current_price = forecast.get('current_price', 0)
                forecast_price = forecast.get('forecast_price', 0)
                if current_price >= 1000:
                    price_format = "${:.2f}"
                elif current_price >= 1:
                    price_format = "${:.4f}"
                elif current_price >= 0.01:
                    price_format = "${:.6f}"
                else:
                    price_format = "${:.8f}"
                price_diff = forecast_price - current_price
                pct_diff = (price_diff / current_price) * 100 if current_price > 0 else 0
                diff_color = Fore.GREEN if pct_diff >= 0 else Fore.RED
                diff_symbol = "↗" if pct_diff >= 0 else "↘"
                message = (
                    f"✅ {Style.BRIGHT}{coin_name}{Style.RESET_ALL} | "
                    f"Now: {price_format.format(current_price)} | "
                    f"Forecast: {price_format.format(forecast_price)} | "
                    f"{diff_color}{diff_symbol} {pct_diff:+.2f}%{Style.RESET_ALL} | "
                    f"⌚ {duration:.2f}s"
                )
                logging.info(message)
                return forecast
            except Exception as e:
                retries += 1
                error_msg = str(e)
                if len(error_msg) > 50:
                    error_msg = error_msg[:47] + "..."
                logging.error(f"❌ {coin_name} error ({retries}/{max_retries}): {error_msg}")
                if retries < max_retries:
                    logging.info(f"🔄 Retrying {coin_name} in 3s...")
                    time.sleep(3)
                else:
                    logging.error(f"💔 Failed to analyze {coin_name} after {max_retries} attempts")
                    return None
    def evaluate_opportunity(self, symbol, forecast):
        """
        Evaluate if a coin represents a good investment opportunity
        Parameters:
        -----------
        symbol : str
            The cryptocurrency symbol
        forecast : dict
            The forecast results from AlphaEngine
        Returns:
        --------
        bool
            True if the coin is a good opportunity, False otherwise
        """
        if not forecast:
            return False
        price_change_pct = forecast.get('price_change_pct', 0)
        signal = forecast.get('signal', 'HOLD')
        confidence = forecast.get('confidence', 0)
        fragility_score = forecast.get('fragility_score', 0)
        breakout_direction = forecast.get('breakout_direction', 'unknown')
        breakout_confidence = forecast.get('breakout_confidence', 0)
        current_price = forecast.get('current_price', 0)
        forecast_price = forecast.get('forecast_price', 0)
        is_opportunity = False
        opportunity_reasons = []
        signal_color = {
            "BUY": f"{Fore.GREEN}BUY{Style.RESET_ALL}",
            "SELL": f"{Fore.RED}SELL{Style.RESET_ALL}",
            "HOLD": f"{Fore.YELLOW}HOLD{Style.RESET_ALL}"
        }
        if price_change_pct >= 5.0:
            is_opportunity = True
            opportunity_reasons.append(f"Growth: {price_change_pct:.2f}%")
        if signal == 'BUY' and confidence >= 0.65:
            is_opportunity = True
            opportunity_reasons.append(f"Strong BUY ({confidence:.1%})")
        if (fragility_score >= 70 and 
            breakout_direction == 'up' and 
            breakout_confidence >= 0.6):
            is_opportunity = True
            opportunity_reasons.append(f"Breakout ({breakout_confidence:.1%})")
        status_parts = []
        change_color = Fore.GREEN if price_change_pct >= 0 else Fore.RED
        status_parts.append(f"Growth: {change_color}{price_change_pct:+.2f}%{Style.RESET_ALL}")
        signal_display = signal_color.get(signal, signal)
        status_parts.append(f"Signal: {signal_display} ({confidence:.1%})")
        if fragility_score > 0:
            if fragility_score >= 70:
                frag_color = Fore.RED
            elif fragility_score >= 50:
                frag_color = Fore.YELLOW
            else:
                frag_color = Fore.GREEN
            status_parts.append(f"Fragility: {frag_color}{fragility_score:.1f}{Style.RESET_ALL}")
            if breakout_direction != 'unknown':
                direction_symbol = "↑" if breakout_direction == "up" else "↓"
                status_parts.append(f"Breakout: {direction_symbol} ({breakout_confidence:.1%})")
        if is_opportunity:
            opportunity_details = {
                "symbol": symbol,
                "current_price": current_price,
                "forecast_price": forecast_price,
                "price_change_pct": price_change_pct,
                "signal": signal,
                "confidence": confidence,
                "fragility_score": fragility_score,
                "breakout_direction": breakout_direction,
                "breakout_confidence": breakout_confidence,
                "opportunity_reasons": opportunity_reasons,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            opportunity_logger.info(f"OPPORTUNITY FOUND: {symbol}")
            opportunity_logger.info(json.dumps(opportunity_details, indent=2))
            opportunity_logger.info("-" * 50)
            coin_name = symbol.replace('USDT', '')
            opportunity_msg = (
                f"\n{Back.MAGENTA}{Fore.WHITE} 💰 INVESTMENT OPPORTUNITY {Style.RESET_ALL}\n"
                f"{Fore.MAGENTA}┌──[ {Style.BRIGHT}{coin_name}{Style.RESET_ALL}{Fore.MAGENTA} ]"
                f"{'─' * (45 - len(coin_name))}\n"
                f"│ {', '.join(opportunity_reasons)}\n"
                f"└{'─' * 50}{Style.RESET_ALL}"
            )
            logging.info(opportunity_msg)
            metrics_line = " | ".join(status_parts)
            logging.info(f"{Fore.YELLOW}➤ Details: {metrics_line}{Style.RESET_ALL}")
            return True
        else:
            logging.info(f"⏹️ SKIPPED: {symbol} | " + " | ".join(status_parts))
            return False
    def scan_random_coins(self, num_coins=5, cooldown=5):
        """
        Scan a number of random coins for investment opportunities
        Parameters:
        -----------
        num_coins : int
            Number of random coins to scan
        cooldown : int
            Seconds to wait between coin analysis to avoid API rate limits
        """
        opportunities = 0
        scanned = 0
        batch_start_time = time.time()
        logging.info(f"🔍 Starting scan batch ({num_coins} coins)")
        while scanned < num_coins:
            try:
                symbol = self.select_random_symbol()
                forecast = self.analyze_symbol(symbol)
                if forecast:
                    is_opportunity = self.evaluate_opportunity(symbol, forecast)
                    if is_opportunity:
                        opportunities += 1
                scanned += 1
                if scanned < num_coins:
                    time.sleep(cooldown)
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                if len(error_msg) > 80:
                    short_error = error_msg[:77] + "..."
                else:
                    short_error = error_msg
                logging.error(f"{Fore.RED}❌ Error scanning {symbol}: [{error_type}] {short_error}{Style.RESET_ALL}")
                logging.debug(f"Full error details for {symbol}: {error_msg}")
                scanned += 1
                time.sleep(cooldown)
        batch_duration = time.time() - batch_start_time
        mins, secs = divmod(int(batch_duration), 60)
        duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        logging.info(f"{Fore.CYAN}✅ Batch complete: {Style.BRIGHT}{opportunities}{Style.RESET_ALL}{Fore.CYAN} opportunities found in {duration_str}{Style.RESET_ALL}")
        return opportunities
def main():
    """Main function to run the hedge scanner"""
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')
    coins_per_batch = 10
    cooldown_between_coins = 10
    cooldown_between_batches = 300
    scanner = HedgeScanner(api_key=api_key, api_secret=api_secret)
    print(f"\n{Back.BLUE}{Fore.WHITE}{Style.BRIGHT} AlphaEngine Crypto Hedge Scanner {Style.RESET_ALL}")
    print(f"""
{Fore.CYAN}╔════════════════════════════════════════════════╗
║ {Style.BRIGHT}╔═╗╦  ╔═╗╦ ╦╔═╗╔═╗╔╗╔╔═╗╦╔╗╔╔═╗{Style.RESET_ALL}{Fore.CYAN} ┃ {Style.BRIGHT}╔═╗╔═╗╔═╗╔╗╔{Style.RESET_ALL}{Fore.CYAN} ║
║ {Style.BRIGHT}╠═╣║  ╠═╝╠═╣╠═╣║╣ ║║║║ ╦║║║║║╣ {Style.RESET_ALL}{Fore.CYAN} ┃ {Style.BRIGHT}╚═╗║  ╠═╣║║║{Style.RESET_ALL}{Fore.CYAN} ║
║ {Style.BRIGHT}╩ ╩╩═╝╩  ╩ ╩╩ ╩╚═╝╝╚╝╚═╝╩╝╚╝╚═╝{Style.RESET_ALL}{Fore.CYAN} ┃ {Style.BRIGHT}╚═╝╚═╝╩ ╩╝╚╝{Style.RESET_ALL}{Fore.CYAN} ║
╚════════════════════════════════════════════════╝{Style.RESET_ALL}""")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{Fore.YELLOW}┌─ Started at: {current_time}")
    print(f"└─ Scanning for high-value investment opportunities...{Style.RESET_ALL}\n")
    start_time = time.time()
    total_opportunities = 0
    total_coins_scanned = 0
    batch_count = 0
    try:
        while True:
            batch_count += 1
            batch_start = time.time()
            if batch_count > 1 and batch_count % 5 == 0:
                runtime = time.time() - start_time
                hours, remainder = divmod(int(runtime), 3600)
                minutes, seconds = divmod(remainder, 60)
                runtime_str = f"{hours}h {minutes}m {seconds}s"
                opportunity_rate = (total_opportunities/max(1, total_coins_scanned)*100)
                rate_color = Fore.GREEN if opportunity_rate >= 5 else Fore.YELLOW if opportunity_rate >= 2 else Fore.RED
                logging.info(f"\n{Fore.CYAN}{'═'*50}")
                logging.info(f"{Fore.CYAN}{Style.BRIGHT}📊 SESSION STATS AFTER BATCH {batch_count}{Style.RESET_ALL}")
                logging.info(f"{Fore.CYAN}┌─ Runtime: {Fore.WHITE}{runtime_str}{Style.RESET_ALL}")
                logging.info(f"{Fore.CYAN}├─ Coins analyzed: {Fore.WHITE}{total_coins_scanned}{Style.RESET_ALL}")
                logging.info(f"{Fore.CYAN}├─ Opportunities found: {Fore.MAGENTA}{Style.BRIGHT}{total_opportunities}{Style.RESET_ALL}")
                logging.info(f"{Fore.CYAN}└─ Opportunity rate: {rate_color}{opportunity_rate:.2f}%{Style.RESET_ALL}")
                logging.info(f"{Fore.CYAN}{'═'*50}{Style.RESET_ALL}\n")
            current_time = datetime.now().strftime("%H:%M:%S")
            logging.info(f"{Fore.YELLOW}{Style.BRIGHT}⚡ Batch {batch_count} starting at {current_time}{Style.RESET_ALL}")
            try:
                opportunities = scanner.scan_random_coins(num_coins=coins_per_batch, cooldown=cooldown_between_coins)
                total_opportunities += opportunities
                total_coins_scanned += coins_per_batch
                batch_duration = time.time() - batch_start
                cooldown_needed = max(1, cooldown_between_batches - int(batch_duration))
                if cooldown_needed > 10:
                    logging.info(f"{Fore.BLUE}⏳ Cooldown period: {cooldown_needed} seconds{Style.RESET_ALL}")
                    stop_event = threading.Event()
                    message = f"Waiting for next batch... {cooldown_needed}s remaining"
                    spinner_thread = threading.Thread(target=spinner_task, args=(stop_event, message))
                    spinner_thread.daemon = True
                    spinner_thread.start()
                    for i in range(cooldown_needed, 0, -1):
                        if i % 10 == 0 or i <= 5:
                            stop_event.set()
                            time.sleep(0.2)
                            message = f"Waiting for next batch... {i}s remaining"
                            stop_event = threading.Event()
                            spinner_thread = threading.Thread(target=spinner_task, args=(stop_event, message))
                            spinner_thread.daemon = True
                            spinner_thread.start()
                        time.sleep(1)
                    stop_event.set()
                    spinner_thread.join(timeout=1.0)
                else:
                    time.sleep(cooldown_needed)
            except Exception as e:
                logging.error(f"{Fore.RED}💥 Error in batch {batch_count}: {str(e)[:100]}{Style.RESET_ALL}")
                logging.info(f"{Fore.YELLOW}⚠️ Restarting after 60 seconds...{Style.RESET_ALL}")
                time.sleep(60)
    except KeyboardInterrupt:
        runtime = time.time() - start_time
        hours, remainder = divmod(int(runtime), 3600)
        minutes, seconds = divmod(remainder, 60)
        runtime_str = f"{hours}h {minutes}m {seconds}s"
        opportunity_rate = (total_opportunities/max(1, total_coins_scanned)*100)
        rate_color = Fore.GREEN if opportunity_rate >= 5 else Fore.YELLOW if opportunity_rate >= 2 else Fore.RED
        print(f"\n{Back.YELLOW}{Fore.BLACK} SCAN COMPLETE {Style.RESET_ALL} {Fore.YELLOW}Interrupted by user{Style.RESET_ALL}")
        print(f"\n{Back.CYAN}{Fore.BLACK} SUMMARY {Style.RESET_ALL}")
        print(f"""
{Fore.GREEN}╔════════════════════════════════════════════╗
║ {Fore.WHITE}📈 AlphaEngine Hedge Scanner Results        {Fore.GREEN}║
╠════════════════════════════════════════════╣
║ {Fore.CYAN}⏱️  Runtime:           {Fore.WHITE}{runtime_str:<18} {Fore.GREEN}║
║ {Fore.BLUE}🔍 Batches run:        {Fore.WHITE}{batch_count:<18} {Fore.GREEN}║
║ {Fore.BLUE}🪙 Coins analyzed:     {Fore.WHITE}{total_coins_scanned:<18} {Fore.GREEN}║
║ {Fore.MAGENTA}💰 Opportunities:      {Style.BRIGHT}{total_opportunities:<18}{Style.RESET_ALL} {Fore.GREEN}║
║ {Fore.YELLOW}📊 Opportunity rate:   {rate_color}{opportunity_rate:.2f}%{' '*(18-len(f"{opportunity_rate:.2f}%"))} {Fore.GREEN}║
╚════════════════════════════════════════════╝{Style.RESET_ALL}
""")
        print(f"{Fore.YELLOW}📝 Investment opportunities saved to: {Style.BRIGHT}investment_opportunities.log{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📜 Full scan log saved to: {Style.BRIGHT}hedge_scanner.log{Style.RESET_ALL}")
    except Exception as e:
        error_msg = str(e)
        import traceback
        tb_str = traceback.format_exc()
        print(f"\n{Back.RED}{Fore.WHITE} CRITICAL ERROR {Style.RESET_ALL}")
        print(f"{Fore.RED}{'='*60}")
        print(f"{Fore.RED}An unrecoverable error has occurred:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{error_msg}{Style.RESET_ALL}")
        print(f"{Fore.RED}{'='*60}{Style.RESET_ALL}")
        logging.critical("Unrecoverable error occurred. Full traceback:")
        for line in tb_str.split('\n'):
            logging.critical(line)
        logging.info("Exiting due to unrecoverable error.")
        print(f"\n{Fore.YELLOW}Check hedge_scanner.log for full error details{Style.RESET_ALL}")
if __name__ == "__main__":
    main()
