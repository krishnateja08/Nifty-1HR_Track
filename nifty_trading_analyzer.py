"""
Nifty Option Chain & Technical Analysis for Day Trading
COMPLETE VERSION - Both 1H and 5H Momentum Side-by-Side
1-HOUR TIMEFRAME with WILDER'S RSI (matches TradingView)
Enhanced with Pivot Points + Dual Momentum Analysis + Top 10 OI Display
EXPIRY: Weekly TUESDAY expiry with 3:30 PM IST cutoff logic
FIXED: Using curl-cffi for NSE API to bypass anti-scraping
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import yfinance as yf
from curl_cffi import requests  # Using curl-cffi instead of requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
import yaml
import os
import logging
import time

warnings.filterwarnings('ignore')


class NiftyAnalyzer:
    def __init__(self, config_path='config.yml'):
        """Initialize analyzer with YAML configuration"""
        self.config = self.load_config(config_path)
        self.setup_logging()

        # IST timezone
        self.ist = pytz.timezone('Asia/Kolkata')

        self.nifty_symbol = "^NSEI"
        # Using correct v3 API endpoint
        self.option_chain_base_url = (
            "https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol=NIFTY&expiry="
        )

        # Headers that work with NSE
        self.headers = {
            "authority": "www.nseindia.com",
            "accept": "application/json, text/plain, */*",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/121.0.0.0 Safari/537.36",
            "referer": "https://www.nseindia.com/option-chain",
            "accept-language": "en-US,en;q=0.9",
        }

    # ---------------------------------------------------------------------
    # Time & Config
    # ---------------------------------------------------------------------
    def get_next_expiry_date(self):
        """
        Calculate the next NIFTY expiry date (Weekly Tuesday)
        If today is Tuesday after 3:30 PM, return next week's Tuesday
        """
        now_ist = self.get_ist_time()
        current_day = now_ist.weekday()  # 0=Mon,1=Tue,...,6=Sun

        if current_day == 1:
            current_hour = now_ist.hour
            current_minute = now_ist.minute
            if current_hour < 15 or (current_hour == 15 and current_minute < 30):
                days_until_tuesday = 0
                self.logger.info("📅 Today is Tuesday before 3:30 PM - Using today as expiry")
            else:
                days_until_tuesday = 7
                self.logger.info("📅 Tuesday after 3:30 PM - Moving to next Tuesday")
        elif current_day == 0:
            days_until_tuesday = 1
        else:
            days_until_tuesday = (1 - current_day) % 7
            if days_until_tuesday == 0:
                days_until_tuesday = 7

        expiry_date = now_ist + timedelta(days=days_until_tuesday)
        expiry_str = expiry_date.strftime('%d-%b-%Y')
        self.logger.info(f"📅 Next NIFTY Expiry: {expiry_str} ({expiry_date.strftime('%A')})")
        return expiry_str

    def get_ist_time(self):
        """Get current time in IST"""
        return datetime.now(self.ist)

    def format_ist_time(self, dt=None):
        """Format datetime in IST"""
        if dt is None:
            dt = self.get_ist_time()
        elif dt.tzinfo is None:
            dt = self.ist.localize(dt)
        else:
            dt = dt.astimezone(self.ist)
        return dt.strftime("%Y-%m-%d %H:%M:%S IST")

    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠️ Config file not found: {config_path}")
            print("Using default configuration...")
            return self.get_default_config()
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return self.get_default_config()

    def get_default_config(self):
        """Return default configuration"""
        return {
            'email': {
                'recipient': 'your_email@gmail.com',
                'sender': 'your_email@gmail.com',
                'app_password': 'your_app_password',
                'subject_prefix': 'Nifty Day Trading Report',
                'send_on_failure': False
            },
            'technical': {
                'timeframe': '1h',
                'period': '6mo',
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'ema_short': 20,
                'ema_long': 50,
                'num_support_levels': 2,
                'num_resistance_levels': 2,
                'momentum_threshold_strong': 0.5,
                'momentum_threshold_moderate': 0.2
            },
            'option_chain': {
                'pcr_bullish': 1.0,
                'pcr_very_bullish': 1.2,
                'pcr_bearish': 1.0,
                'pcr_very_bearish': 0.8,
                'strike_range': 500,
                'min_oi': 100000,
                'top_strikes_count': 5
            },
            'recommendation': {
                'strong_buy_threshold': 3,
                'buy_threshold': 1,
                'sell_threshold': -1,
                'strong_sell_threshold': -3,
                'momentum_5h_weight': 2,
                'momentum_1h_weight': 1
            },
            'report': {
                'title': 'NIFTY DAY TRADING ANALYSIS (1H)',
                'save_local': True,
                'local_dir': './reports',
                'filename_format': 'nifty_analysis_%Y%m%d_%H%M%S.html'
            },
            'data_source': {
                'option_chain_source': 'nse',
                'technical_source': 'yahoo',
                'max_retries': 3,
                'retry_delay': 2,
                'timeout': 30,
                'fallback_to_sample': True
            },
            'logging': {
                'level': 'INFO',
                'log_to_file': True,
                'log_file': './logs/nifty_analyzer.log',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            },
            'advanced': {
                'verbose': True,
                'debug': False,
                'validate_data': True,
                'min_data_points': 100,
                'use_momentum_filter': True
            }
        }

    def setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))

        self.logger = logging.getLogger('NiftyAnalyzer')
        self.logger.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if log_config.get('log_to_file', True):
            log_file = log_config.get('log_file', './logs/nifty_analyzer.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    # ---------------------------------------------------------------------
    # Option Chain
    # ---------------------------------------------------------------------
    def fetch_option_chain(self):
        """Fetch Nifty option chain data from NSE using curl-cffi"""
        if self.config['data_source']['option_chain_source'] == 'sample':
            self.logger.info("Using sample option chain data")
            return None, None

        expiry_date = self.get_next_expiry_date()
        symbol = "NIFTY"

        api_url = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={symbol}&expiry={expiry_date}"
        base_url = "https://www.nseindia.com/"

        max_retries = self.config['data_source']['max_retries']
        retry_delay = self.config['data_source']['retry_delay']
        timeout = self.config['data_source']['timeout']

        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"Fetching option chain data for expiry {expiry_date} "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )

                session = requests.Session()
                session.get(base_url, headers=self.headers, impersonate="chrome", timeout=15)
                time.sleep(1)

                response = session.get(
                    api_url, headers=self.headers, impersonate="chrome", timeout=timeout
                )

                if response.status_code == 200:
                    data = response.json()
                    if 'records' in data and 'data' in data['records']:
                        option_data = data['records']['data']
                        current_price = data['records']['underlyingValue']

                        if not option_data:
                            self.logger.warning(f"No option data for expiry {expiry_date}")
                            continue

                        calls_data = []
                        puts_data = []

                        for item in option_data:
                            strike = item.get('strikePrice', 0)

                            if 'CE' in item:
                                ce = item['CE']
                                calls_data.append({
                                    'Strike': strike,
                                    'Call_OI': ce.get('openInterest', 0),
                                    'Call_Chng_OI': ce.get('changeinOpenInterest', 0),
                                    'Call_Volume': ce.get('totalTradedVolume', 0),
                                    'Call_IV': ce.get('impliedVolatility', 0),
                                    'Call_LTP': ce.get('lastPrice', 0)
                                })

                            if 'PE' in item:
                                pe = item['PE']
                                puts_data.append({
                                    'Strike': strike,
                                    'Put_OI': pe.get('openInterest', 0),
                                    'Put_Chng_OI': pe.get('changeinOpenInterest', 0),
                                    'Put_Volume': pe.get('totalTradedVolume', 0),
                                    'Put_IV': pe.get('impliedVolatility', 0),
                                    'Put_LTP': pe.get('lastPrice', 0)
                                })

                        calls_df = pd.DataFrame(calls_data)
                        puts_df = pd.DataFrame(puts_data)

                        oc_df = pd.merge(calls_df, puts_df, on='Strike', how='outer')
                        oc_df = oc_df.fillna(0)
                        oc_df = oc_df.sort_values('Strike')

                        self.logger.info(
                            f"✅ Option chain data fetched successfully | "
                            f"Spot: ₹{current_price} | Expiry: {expiry_date}"
                        )
                        self.logger.info(f"✅ Total strikes fetched: {len(oc_df)}")
                        return oc_df, current_price
                    else:
                        self.logger.warning("Invalid response structure from NSE API")
                else:
                    self.logger.warning(
                        f"NSE API returned status code: {response.status_code}"
                    )

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        if self.config['data_source']['fallback_to_sample']:
            self.logger.warning("All NSE API attempts failed, using sample data")

        return None, None

    def get_top_strikes_by_oi(self, oc_df, spot_price):
        """Get top strikes by Open Interest for CE and PE"""
        if oc_df is None or oc_df.empty:
            return {'top_ce_strikes': [], 'top_pe_strikes': []}

        top_count = self.config['option_chain'].get('top_strikes_count', 5)

        ce_data = oc_df[oc_df['Call_OI'] > 0].copy()
        ce_data = ce_data.sort_values('Call_OI', ascending=False).head(top_count)
        top_ce_strikes = []
        for _, row in ce_data.iterrows():
            strike_type = (
                'ITM' if row['Strike'] < spot_price
                else ('ATM' if row['Strike'] == spot_price else 'OTM')
            )
            top_ce_strikes.append({
                'strike': row['Strike'],
                'oi': int(row['Call_OI']),
                'ltp': row['Call_LTP'],
                'iv': row['Call_IV'],
                'type': strike_type,
                'chng_oi': int(row['Call_Chng_OI']),
                'volume': int(row['Call_Volume'])
            })

        pe_data = oc_df[oc_df['Put_OI'] > 0].copy()
        pe_data = pe_data.sort_values('Put_OI', ascending=False).head(top_count)
        top_pe_strikes = []
        for _, row in pe_data.iterrows():
            strike_type = (
                'ITM' if row['Strike'] > spot_price
                else ('ATM' if row['Strike'] == spot_price else 'OTM')
            )
            top_pe_strikes.append({
                'strike': row['Strike'],
                'oi': int(row['Put_OI']),
                'ltp': row['Put_LTP'],
                'iv': row['Put_IV'],
                'type': strike_type,
                'chng_oi': int(row['Put_Chng_OI']),
                'volume': int(row['Put_Volume'])
            })

        return {'top_ce_strikes': top_ce_strikes, 'top_pe_strikes': top_pe_strikes}

    def analyze_option_chain(self, oc_df, spot_price):
        """Analyze option chain for trading signals"""
        if oc_df is None or oc_df.empty:
            self.logger.warning("No option chain data, using sample analysis")
            return self.get_sample_oc_analysis()

        config = self.config['option_chain']

        total_call_oi = oc_df['Call_OI'].sum()
        total_put_oi = oc_df['Put_OI'].sum()
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0

        oc_df['Call_Pain'] = oc_df.apply(
            lambda row: row['Call_OI'] * max(0, spot_price - row['Strike']), axis=1
        )
        oc_df['Put_Pain'] = oc_df.apply(
            lambda row: row['Put_OI'] * max(0, row['Strike'] - spot_price), axis=1
        )
        oc_df['Total_Pain'] = oc_df['Call_Pain'] + oc_df['Put_Pain']

        max_pain_strike = oc_df.loc[oc_df['Total_Pain'].idxmax(), 'Strike']

        strike_range = config['strike_range']
        nearby_strikes = oc_df[
            (oc_df['Strike'] >= spot_price - strike_range) &
            (oc_df['Strike'] <= spot_price + strike_range)
        ].copy()

        num_resistance = self.config['technical']['num_resistance_levels']
        num_support = self.config['technical']['num_support_levels']

        resistance_df = nearby_strikes[nearby_strikes['Strike'] > spot_price].nlargest(
            num_resistance, 'Call_OI'
        )
        resistances = resistance_df['Strike'].tolist()

        support_df = nearby_strikes[nearby_strikes['Strike'] < spot_price].nlargest(
            num_support, 'Put_OI'
        )
        supports = support_df['Strike'].tolist()

        total_call_buildup = oc_df['Call_Chng_OI'].sum()
        total_put_buildup = oc_df['Put_Chng_OI'].sum()

        avg_call_iv = oc_df['Call_IV'].mean()
        avg_put_iv = oc_df['Put_IV'].mean()

        top_strikes = self.get_top_strikes_by_oi(oc_df, spot_price)

        return {
            'pcr': round(pcr, 2),
            'max_pain': max_pain_strike,
            'resistances': sorted(resistances, reverse=True),
            'supports': sorted(supports, reverse=True),
            'call_buildup': total_call_buildup,
            'put_buildup': total_put_buildup,
            'avg_call_iv': round(avg_call_iv, 2),
            'avg_put_iv': round(avg_put_iv, 2),
            'oi_sentiment': 'Bullish' if total_put_buildup > total_call_buildup else 'Bearish',
            'top_ce_strikes': top_strikes['top_ce_strikes'],
            'top_pe_strikes': top_strikes['top_pe_strikes']
        }

    def get_sample_oc_analysis(self):
        """Return sample option chain analysis"""
        return {
            'pcr': 1.15,
            'max_pain': 24500,
            'resistances': [24650, 24600],
            'supports': [24400, 24350],
            'call_buildup': 5000000,
            'put_buildup': 6000000,
            'avg_call_iv': 15.5,
            'avg_put_iv': 16.2,
            'oi_sentiment': 'Bullish',
            'top_ce_strikes': [
                {'strike': 24500, 'oi': 5000000, 'ltp': 120, 'iv': 16.5,
                 'type': 'ATM', 'chng_oi': 500000, 'volume': 125000},
                {'strike': 24600, 'oi': 4500000, 'ltp': 80, 'iv': 15.8,
                 'type': 'OTM', 'chng_oi': 450000, 'volume': 110000},
                {'strike': 24550, 'oi': 4200000, 'ltp': 95, 'iv': 16.0,
                 'type': 'OTM', 'chng_oi': 420000, 'volume': 105000},
                {'strike': 24450, 'oi': 3800000, 'ltp': 145, 'iv': 16.8,
                 'type': 'ITM', 'chng_oi': 380000, 'volume': 95000},
                {'strike': 24400, 'oi': 3500000, 'ltp': 170, 'iv': 17.0,
                 'type': 'ITM', 'chng_oi': 350000, 'volume': 90000},
            ],
            'top_pe_strikes': [
                {'strike': 24500, 'oi': 5500000, 'ltp': 110, 'iv': 16.0,
                 'type': 'ATM', 'chng_oi': 550000, 'volume': 130000},
                {'strike': 24400, 'oi': 5000000, 'ltp': 75, 'iv': 15.5,
                 'type': 'OTM', 'chng_oi': 500000, 'volume': 120000},
                {'strike': 24450, 'oi': 4700000, 'ltp': 90, 'iv': 15.7,
                 'type': 'OTM', 'chng_oi': 470000, 'volume': 115000},
                {'strike': 24550, 'oi': 4300000, 'ltp': 135, 'iv': 16.5,
                 'type': 'ITM', 'chng_oi': 430000, 'volume': 100000},
                {'strike': 24600, 'oi': 4000000, 'ltp': 160, 'iv': 16.8,
                 'type': 'ITM', 'chng_oi': 400000, 'volume': 95000},
            ]
        }

    # ---------------------------------------------------------------------
    # Technicals
    # ---------------------------------------------------------------------
    def fetch_technical_data(self):
        """Fetch historical data for technical analysis - ALWAYS 1 HOUR"""
        if self.config['data_source']['technical_source'] == 'sample':
            self.logger.info("Using sample technical data")
            return None

        period = self.config['technical']['period']
        interval = '1h'

        try:
            self.logger.info(f"Fetching 1-HOUR technical data ({period})...")
            ticker = yf.Ticker(self.nifty_symbol)
            df = ticker.history(period=period, interval=interval)

            if self.config['advanced']['validate_data']:
                min_points = self.config['advanced']['min_data_points']
                if len(df) < min_points:
                    self.logger.warning(
                        f"Insufficient data points: {len(df)} < {min_points}"
                    )
                    return None

            self.logger.info(f"✅ 1-HOUR data fetched | {len(df)} bars")
            self.logger.info(
                f"Price: ₹{df['Close'].iloc[-1]:.2f} | Last candle: {df.index[-1]}"
            )
            return df
        except Exception as e:
            self.logger.error(f"Error fetching technical data: {e}")
            return None

    def calculate_pivot_points(self, df, current_price):
        """
        Calculate Traditional Pivot Points (30-minute timeframe)
        Uses previous 30-min candle's OHLC for pivot calculation
        """
        try:
            ticker = yf.Ticker(self.nifty_symbol)
            min_30_df = ticker.history(period='5d', interval='30m')

            if len(min_30_df) >= 2:
                prev_high = min_30_df['High'].iloc[-2]
                prev_low = min_30_df['Low'].iloc[-2]
                prev_close = min_30_df['Close'].iloc[-2]
            else:
                prev_high = df['High'].max()
                prev_low = df['Low'].min()
                prev_close = df['Close'].iloc[-1]

            pivot = (prev_high + prev_low + prev_close) / 3

            r1 = (2 * pivot) - prev_low
            r2 = pivot + (prev_high - prev_low)
            r3 = prev_high + 2 * (pivot - prev_low)

            s1 = (2 * pivot) - prev_high
            s2 = pivot - (prev_high - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)

            self.logger.info(f"📍 Pivot Points (30m) calculated | PP: ₹{pivot:.2f}")

            return {
                'pivot': round(pivot, 2),
                'r1': round(r1, 2),
                'r2': round(r2, 2),
                'r3': round(r3, 2),
                's1': round(s1, 2),
                's2': round(s2, 2),
                's3': round(s3, 2),
                'prev_high': round(prev_high, 2),
                'prev_low': round(prev_low, 2),
                'prev_close': round(prev_close, 2)
            }

        except Exception as e:
            self.logger.error(f"Error calculating pivot points: {e}")
            return {
                'pivot': 24520.00,
                'r1': 24590.00,
                'r2': 24650.00,
                'r3': 24720.00,
                's1': 24450.00,
                's2': 24390.00,
                's3': 24320.00,
                'prev_high': 24580.00,
                'prev_low': 24420.00,
                'prev_close': 24500.00
            }

    def calculate_rsi(self, data, period=None):
        """
        Calculate RSI using Wilder's smoothing method (RMA)
        This matches TradingView-style RMA RSI
        """
        if period is None:
            period = self.config['technical']['rsi_period']

        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_support_resistance(self, df, current_price):
        """Calculate nearest support and resistance levels from price action"""
        recent_data = df.tail(300)

        pivots_high = []
        pivots_low = []

        for i in range(5, len(recent_data) - 5):
            high = recent_data['High'].iloc[i]
            low = recent_data['Low'].iloc[i]

            if high == max(recent_data['High'].iloc[i - 5:i + 6]):
                pivots_high.append(high)

            if low == min(recent_data['Low'].iloc[i - 5:i + 6]):
                pivots_low.append(low)

        resistances = sorted([p for p in pivots_high if p > current_price])
        resistances = list(dict.fromkeys(resistances))

        supports = sorted([p for p in pivots_low if p < current_price], reverse=True)
        supports = list(dict.fromkeys(supports))

        num_resistance = self.config['technical']['num_resistance_levels']
        num_support = self.config['technical']['num_support_levels']

        return {
            'resistances': (
                resistances[:num_resistance]
                if len(resistances) >= num_resistance else resistances
            ),
            'supports': (
                supports[:num_support]
                if len(supports) >= num_support else supports
            )
        }

    def get_momentum_signal(self, momentum_pct):
        """Get momentum signal, bias, and CSS color variables based on percentage"""
        strong_threshold = self.config['technical'].get('momentum_threshold_strong', 0.5)
        moderate_threshold = self.config['technical'].get('momentum_threshold_moderate', 0.2)

        if momentum_pct > strong_threshold:
            return "Strong Upward", "Bullish", {
                'bg': '#1e7e34',
                'bg_dark': '#155724',
                'text': '#ffffff',
                'border': '#28a745'
            }
        elif momentum_pct > moderate_threshold:
            return "Moderate Upward", "Bullish", {
                'bg': '#28a745',
                'bg_dark': '#218838',
                'text': '#ffffff',
                'border': '#1e7e34'
            }
        elif momentum_pct < -strong_threshold:
            return "Strong Downward", "Bearish", {
                'bg': '#c82333',
                'bg_dark': '#bd2130',
                'text': '#ffffff',
                'border': '#dc3545'
            }
        elif momentum_pct < -moderate_threshold:
            return "Moderate Downward", "Bearish", {
                'bg': '#fd7e14',
                'bg_dark': '#e8590c',
                'text': '#ffffff',
                'border': '#dc3545'
            }
        else:
            return "Sideways/Weak", "Neutral", {
                'bg': '#6c757d',
                'bg_dark': '#5a6268',
                'text': '#ffffff',
                'border': '#495057'
            }

    def technical_analysis(self, df):
        """Perform complete technical analysis - 1 HOUR TIMEFRAME with DUAL MOMENTUM"""
        if df is None or df.empty:
            self.logger.warning("No technical data, using sample analysis")
            return self.get_sample_tech_analysis()

        current_price = df['Close'].iloc[-1]

        # 1-HOUR MOMENTUM
        if len(df) > 1:
            price_1h_ago = df['Close'].iloc[-2]
            price_change_1h = current_price - price_1h_ago
            price_change_pct_1h = (price_change_1h / price_1h_ago * 100)
        else:
            price_change_1h = 0
            price_change_pct_1h = 0

        momentum_1h_signal, momentum_1h_bias, momentum_1h_colors = self.get_momentum_signal(
            price_change_pct_1h
        )

        # 5-HOUR MOMENTUM
        if len(df) >= 5:
            price_5h_ago = df['Close'].iloc[-5]
            momentum_5h = current_price - price_5h_ago
            momentum_5h_pct = (momentum_5h / price_5h_ago * 100)
        else:
            momentum_5h = 0
            momentum_5h_pct = 0

        momentum_5h_signal, momentum_5h_bias, momentum_5h_colors = self.get_momentum_signal(
            momentum_5h_pct
        )

        self.logger.info(f"📊 1H Momentum: {price_change_pct_1h:+.2f}% - {momentum_1h_signal}")
        self.logger.info(f"📊 5H Momentum: {momentum_5h_pct:+.2f}% - {momentum_5h_signal}")

        df['RSI'] = self.calculate_rsi(df['Close'])
        current_rsi = df['RSI'].iloc[-1]
        self.logger.info(f"🎯 RSI(14) calculated: {current_rsi:.2f} (Wilder's method)")

        ema_short = self.config['technical']['ema_short']
        ema_long = self.config['technical']['ema_long']

        df['EMA_Short'] = df['Close'].ewm(span=ema_short, adjust=False).mean()
        df['EMA_Long'] = df['Close'].ewm(span=ema_long, adjust=False).mean()

        ema_short_val = df['EMA_Short'].iloc[-1]
        ema_long_val = df['EMA_Long'].iloc[-1]

        sr_levels = self.calculate_support_resistance(df, current_price)
        pivot_points = self.calculate_pivot_points(df, current_price)

        if current_price > ema_short_val > ema_long_val:
            trend = "Strong Uptrend"
        elif current_price > ema_short_val:
            trend = "Uptrend"
        elif current_price < ema_short_val < ema_long_val:
            trend = "Strong Downtrend"
        elif current_price < ema_short_val:
            trend = "Downtrend"
        else:
            trend = "Sideways"

        rsi_ob = self.config['technical']['rsi_overbought']
        rsi_os = self.config['technical']['rsi_oversold']

        if current_rsi > rsi_ob:
            rsi_signal = "Overbought - Bearish"
        elif current_rsi < rsi_os:
            rsi_signal = "Oversold - Bullish"
        else:
            rsi_signal = "Neutral"

        return {
            'price': round(current_price, 2),
            'trend': trend,
            'rsi': round(current_rsi, 2),
            'rsi_signal': rsi_signal,
            'ema_short': round(ema_short_val, 2),
            'ema_long': round(ema_long_val, 2),
            'supports': [round(x, 2) for x in sr_levels['supports']],
            'resistances': [round(x, 2) for x in sr_levels['resistances']],
            'pivot_points': pivot_points,
            'momentum_1h_pct': round(price_change_pct_1h, 2),
            'momentum_1h_signal': momentum_1h_signal,
            'momentum_1h_bias': momentum_1h_bias,
            'momentum_1h_colors': momentum_1h_colors,
            'momentum_5h_pct': round(momentum_5h_pct, 2),
            'momentum_5h_signal': momentum_5h_signal,
            'momentum_5h_bias': momentum_5h_bias,
            'momentum_5h_colors': momentum_5h_colors
        }

    def get_sample_tech_analysis(self):
        """Sample technical analysis"""
        return {
            'price': 24500.0,
            'trend': 'Uptrend',
            'rsi': 58.5,
            'rsi_signal': 'Neutral',
            'ema_short': 24480.0,
            'ema_long': 24350.0,
            'supports': [24400.0, 24350.0],
            'resistances': [24550.0, 24600.0],
            'pivot_points': {
                'pivot': 24520.0,
                'r1': 24590.0,
                'r2': 24650.0,
                'r3': 24720.0,
                's1': 24450.0,
                's2': 24390.0,
                's3': 24320.0,
                'prev_high': 24580.0,
                'prev_low': 24420.0,
                'prev_close': 24500.0
            },
            'momentum_1h_pct': 0.35,
            'momentum_1h_signal': 'Moderate Upward',
            'momentum_1h_bias': 'Bullish',
            'momentum_1h_colors': {
                'bg': '#28a745',
                'bg_dark': '#218838',
                'text': '#ffffff',
                'border': '#1e7e34'
            },
            'momentum_5h_pct': 0.85,
            'momentum_5h_signal': 'Strong Upward',
            'momentum_5h_bias': 'Bullish',
            'momentum_5h_colors': {
                'bg': '#1e7e34',
                'bg_dark': '#155724',
                'text': '#ffffff',
                'border': '#28a745'
            }
        }

    # ---------------------------------------------------------------------
    # Recommendation
    # ---------------------------------------------------------------------
    def generate_recommendation(self, tech, oc):
        """
        Combine technicals + option chain into a simple score-based recommendation.
        Logic is intentionally simple and transparent.
        """
        cfg = self.config['recommendation']

        score = 0
        details = []

        # Momentum weighting
        m5 = tech['momentum_5h_pct']
        m1 = tech['momentum_1h_pct']

        if m5 > 0:
            score += cfg['momentum_5h_weight']
            details.append("5H momentum positive")
        elif m5 < 0:
            score -= cfg['momentum_5h_weight']
            details.append("5H momentum negative")

        if m1 > 0:
            score += cfg['momentum_1h_weight']
            details.append("1H momentum positive")
        elif m1 < 0:
            score -= cfg['momentum_1h_weight']
            details.append("1H momentum negative")

        # Trend
        if "Strong Uptrend" in tech['trend']:
            score += 2
            details.append("Strong uptrend")
        elif "Uptrend" in tech['trend']:
            score += 1
            details.append("Uptrend")
        elif "Strong Downtrend" in tech['trend']:
            score -= 2
            details.append("Strong downtrend")
        elif "Downtrend" in tech['trend']:
            score -= 1
            details.append("Downtrend")

        # RSI
        if "Overbought" in tech['rsi_signal']:
            score -= 1
            details.append("RSI overbought")
        elif "Oversold" in tech['rsi_signal']:
            score += 1
            details.append("RSI oversold")

        # PCR / OI sentiment
        if oc['oi_sentiment'] == 'Bullish':
            score += 1
            details.append("OI sentiment bullish")
        else:
            score -= 1
            details.append("OI sentiment bearish")

        if oc['pcr'] >= self.config['option_chain']['pcr_very_bullish']:
            score += 2
            details.append("PCR very bullish")
        elif oc['pcr'] >= self.config['option_chain']['pcr_bullish']:
            score += 1
            details.append("PCR mildly bullish")
        elif oc['pcr'] <= self.config['option_chain']['pcr_very_bearish']:
            score -= 2
            details.append("PCR very bearish")
        elif oc['pcr'] <= self.config['option_chain']['pcr_bearish']:
            score -= 1
            details.append("PCR mildly bearish")

        # Final label
        if score >= cfg['strong_buy_threshold']:
            label = "STRONG BUY (Intraday Bias: Long)"
            color = "#0c7a29"
        elif score >= cfg['buy_threshold']:
            label = "BUY (Intraday Bias: Long)"
            color = "#198754"
        elif score <= cfg['strong_sell_threshold']:
            label = "STRONG SELL (Intraday Bias: Short)"
            color = "#b02a37"
        elif score <= cfg['sell_threshold']:
            label = "SELL (Intraday Bias: Short)"
            color = "#dc3545"
        else:
            label = "NEUTRAL / WAIT"
            color = "#6c757d"

        return {
            'score': score,
            'label': label,
            'color': color,
            'details': details
        }

    # ---------------------------------------------------------------------
    # HTML REPORT (Light Blue Professional Theme)
    # ---------------------------------------------------------------------
    def create_html_report(self, tech, oc, recommendation, run_meta):
        """
        Build a light-blue professional dashboard-style HTML report.
        """
        now_str = self.format_ist_time()
        title = self.config['report']['title']

        # Momentum color blocks
        m1 = tech['momentum_1h_colors']
        m5 = tech['momentum_5h_colors']

        pivot = tech['pivot_points']

        def format_list(lst):
            if not lst:
                return "—"
            return ", ".join([f"₹{x:,.2f}" for x in lst])

        top_ce_rows = ""
        for row in oc['top_ce_strikes']:
            top_ce_rows += f"""
            <tr>
                <td>{int(row['strike'])}</td>
                <td>{row['type']}</td>
                <td>{row['oi']:,}</td>
                <td>{row['chng_oi']:,}</td>
                <td>{row['volume']:,}</td>
                <td>{row['iv']:.2f}</td>
                <td>{row['ltp']:.2f}</td>
            </tr>
            """

        top_pe_rows = ""
        for row in oc['top_pe_strikes']:
            top_pe_rows += f"""
            <tr>
                <td>{int(row['strike'])}</td>
                <td>{row['type']}</td>
                <td>{row['oi']:,}</td>
                <td>{row['chng_oi']:,}</td>
                <td>{row['volume']:,}</td>
                <td>{row['iv']:.2f}</td>
                <td>{row['ltp']:.2f}</td>
            </tr>
            """

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
    body {{
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background: #e3f2fd; /* light blue background */
        color: #1f2933;
    }}
    .page-wrapper {{
        min-height: 100vh;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        padding: 24px 12px;
    }}
    .container {{
        max-width: 1100px;
        width: 100%;
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
        overflow: hidden;
        border: 1px solid #cfe2ff;
    }}
    .header {{
        background: linear-gradient(135deg, #0d47a1, #1976d2);
        color: #ffffff;
        padding: 20px 24px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
    }}
    .header-title {{
        font-size: 20px;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}
    .header-subtitle {{
        font-size: 13px;
        opacity: 0.9;
    }}
    .header-meta {{
        text-align: right;
        font-size: 12px;
        opacity: 0.95;
    }}
    .badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.35);
        margin-top: 6px;
    }}
    .content {{
        padding: 20px 24px 24px 24px;
        background: linear-gradient(180deg, #ffffff 0%, #f5f9ff 100%);
    }}
    .section-title {{
        font-size: 15px;
        font-weight: 600;
        color: #0d47a1;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .section-title span.icon {{
        width: 18px;
        height: 18px;
        border-radius: 6px;
        background: #e3f2fd;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        color: #0d47a1;
    }}
    .section {{
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #d0e2ff;
        padding: 14px 16px 16px 16px;
        margin-bottom: 14px;
    }}
    .section-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 12px;
    }}
    .metric-row {{
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 4px;
    }}
    .metric-label {{
        color: #4b5563;
    }}
    .metric-value {{
        font-weight: 600;
        color: #111827;
    }}
    .metric-value-strong {{
        font-weight: 700;
        color: #0d47a1;
    }}
    .pill {{
        display: inline-flex;
        align-items: center;
        padding: 3px 9px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        border: 1px solid #c7d2fe;
        background: #eef2ff;
        color: #1d4ed8;
    }}
    .pill-neutral {{
        background: #e5e7eb;
        border-color: #d1d5db;
        color: #374151;
    }}
    .pill-bullish {{
        background: #e6f4ea;
        border-color: #a7f3d0;
        color: #166534;
    }}
    .pill-bearish {{
        background: #fee2e2;
        border-color: #fecaca;
        color: #b91c1c;
    }}
    .momentum-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 10px;
        margin-top: 6px;
    }}
    .momentum-card {{
        border-radius: 12px;
        padding: 10px 12px;
        color: #ffffff;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.25);
        border-width: 1px;
        border-style: solid;
    }}
    .momentum-title {{
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.9;
        margin-bottom: 4px;
    }}
    .momentum-main {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 4px;
    }}
    .momentum-pct {{
        font-size: 20px;
        font-weight: 700;
    }}
    .momentum-signal {{
        font-size: 12px;
        font-weight: 600;
        text-align: right;
    }}
    .momentum-bias {{
        font-size: 11px;
        opacity: 0.9;
    }}
    .momentum-footer {{
        font-size: 11px;
        opacity: 0.85;
        display: flex;
        justify-content: space-between;
    }}
    .recommendation-card {{
        border-radius: 14px;
        padding: 12px 14px;
        margin-top: 4px;
        background: #e3f2fd;
        border: 1px solid #90caf9;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }}
    .recommendation-label {{
        font-size: 14px;
        font-weight: 700;
    }}
    .recommendation-score {{
        font-size: 12px;
        color: #374151;
    }}
    .recommendation-details {{
        font-size: 11px;
        color: #4b5563;
        line-height: 1.4;
    }}
    .table-wrapper {{
        margin-top: 6px;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #d0e2ff;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }}
    thead {{
        background: linear-gradient(90deg, #0d47a1, #1976d2);
        color: #ffffff;
    }}
    th, td {{
        padding: 6px 8px;
        text-align: right;
        border-bottom: 1px solid #e5e7eb;
    }}
    th:first-child, td:first-child {{
        text-align: left;
    }}
    tbody tr:nth-child(even) {{
        background: #f3f4ff;
    }}
    tbody tr:nth-child(odd) {{
        background: #ffffff;
    }}
    .footer {{
        font-size: 11px;
        color: #6b7280;
        text-align: right;
        margin-top: 10px;
    }}
    @media (max-width: 640px) {{
        .header {{
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
        }}
        .header-meta {{
            text-align: left;
        }}
        .content {{
            padding: 16px 14px 18px 14px;
        }}
    }}
</style>
</head>
<body>
<div class="page-wrapper">
  <div class="container">
    <div class="header">
      <div>
        <div class="header-title">{title}</div>
        <div class="header-subtitle">
          NIFTY 50 • 1H Timeframe • Dual Momentum • Option Chain + Pivot Points
        </div>
        <div class="badge">
          LIVE INTRADAY VIEW • {run_meta.get('expiry', 'Weekly Expiry')}
        </div>
      </div>
      <div class="header-meta">
        <div><strong>Generated:</strong> {now_str}</div>
        <div><strong>Spot:</strong> ₹{tech['price']:,.2f}</div>
        <div><strong>Source:</strong> NSE (Options) • Yahoo (Price)</div>
      </div>
    </div>

    <div class="content">

      <!-- Top Summary: Trend + Recommendation -->
      <div class="section">
        <div class="section-title">
          <span class="icon">⏱</span>
          Market snapshot & intraday bias
        </div>
        <div class="section-grid">
          <div>
            <div class="metric-row">
              <div class="metric-label">Current Price</div>
              <div class="metric-value-strong">₹{tech['price']:,.2f}</div>
            </div>
            <div class="metric-row">
              <div class="metric-label">Trend (1H EMAs)</div>
              <div class="metric-value">{tech['trend']}</div>
            </div>
            <div class="metric-row">
              <div class="metric-label">RSI (14, Wilder)</div>
              <div class="metric-value">
                {tech['rsi']:.2f}
                <span class="pill pill-neutral" style="margin-left:6px;">{tech['rsi_signal']}</span>
              </div>
            </div>
            <div class="metric-row">
              <div class="metric-label">EMA Short / Long</div>
              <div class="metric-value">
                ₹{tech['ema_short']:,.2f} / ₹{tech['ema_long']:,.2f}
              </div>
            </div>
            <div class="metric-row">
              <div class="metric-label">Price vs Pivot</div>
              <div class="metric-value">
                Pivot ₹{pivot['pivot']:,.2f}
              </div>
            </div>
          </div>
          <div>
            <div class="recommendation-card">
              <div class="recommendation-label" style="color:{recommendation['color']};">
                {recommendation['label']}
              </div>
              <div class="recommendation-score">
                Composite Score: <strong>{recommendation['score']:+.0f}</strong>
              </div>
              <div class="recommendation-details">
                {"; ".join(recommendation['details'])}
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Dual Momentum -->
      <div class="section">
        <div class="section-title">
          <span class="icon">⚡</span>
          Dual momentum (1H vs 5H)
        </div>
        <div class="momentum-grid">
          <div class="momentum-card"
               style="background:{m1['bg']};border-color:{m1['border']};">
            <div class="momentum-title">1-HOUR MOMENTUM</div>
            <div class="momentum-main">
              <div class="momentum-pct">
                {tech['momentum_1h_pct']:+.2f}%
              </div>
              <div class="momentum-signal">
                {tech['momentum_1h_signal']}
              </div>
            </div>
            <div class="momentum-footer">
              <div class="momentum-bias">
                Bias: {tech['momentum_1h_bias']}
              </div>
              <div>
                Last candle vs previous close
              </div>
            </div>
          </div>

          <div class="momentum-card"
               style="background:{m5['bg']};border-color:{m5['border']};">
            <div class="momentum-title">5-HOUR MOMENTUM</div>
            <div class="momentum-main">
              <div class="momentum-pct">
                {tech['momentum_5h_pct']:+.2f}%
              </div>
              <div class="momentum-signal">
                {tech['momentum_5h_signal']}
              </div>
            </div>
            <div class="momentum-footer">
              <div class="momentum-bias">
                Bias: {tech['momentum_5h_bias']}
              </div>
              <div>
                Last 5 candles vs 5H ago
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Support / Resistance & Pivot -->
      <div class="section">
        <div class="section-title">
          <span class="icon">📍</span>
          Price action levels & pivot points
        </div>
        <div class="section-grid">
          <div>
            <div class="metric-row">
              <div class="metric-label">Supports (Price Action)</div>
              <div class="metric-value">{format_list(tech['supports'])}</div>
            </div>
            <div class="metric-row">
              <div class="metric-label">Resistances (Price Action)</div>
              <div class="metric-value">{format_list(tech['resistances'])}</div>
            </div>
            <div class="metric-row">
              <div class="metric-label">Max Pain (Options)</div>
              <div class="metric-value">₹{oc['max_pain']:,.0f}</div>
            </div>
            <div class="metric-row">
              <div class="metric-label">OI Supports (Options)</div>
              <div class="metric-value">{format_list(oc['supports'])}</div>
            </div>
            <div class="metric-row">
              <div class="metric-label">OI Resistances (Options)</div>
              <div class="metric-value">{format_list(oc['resistances'])}</div>
            </div>
          </div>
          <div>
            <div class="metric-row">
              <div class="metric-label">Pivot (PP)</div>
              <div class="metric-value">₹{pivot['pivot']:,.2f}</div>
            </div>
            <div class="metric-row">
              <div class="metric-label">R1 / R2 / R3</div>
              <div class="metric-value">
                ₹{pivot['r1']:,.2f} / ₹{pivot['r2']:,.2f} / ₹{pivot['r3']:,.2f}
              </div>
            </div>
            <div class="metric-row">
              <div class="metric-label">S1 / S2 / S3</div>
              <div class="metric-value">
                ₹{pivot['s1']:,.2f} / ₹{pivot['s2']:,.2f} / ₹{pivot['s3']:,.2f}
              </div>
            </div>
            <div class="metric-row">
              <div class="metric-label">Prev 30m High / Low</div>
              <div class="metric-value">
                ₹{pivot['prev_high']:,.2f} / ₹{pivot['prev_low']:,.2f}
              </div>
            </div>
            <div class="metric-row">
              <div class="metric-label">Prev 30m Close</div>
              <div class="metric-value">₹{pivot['prev_close']:,.2f}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Option Chain Summary -->
      <div class="section">
        <div class="section-title">
          <span class="icon">📊</span>
          Option chain overview (NIFTY Weekly)
        </div>
        <div class="section-grid">
          <div>
            <div class="metric-row">
              <div class="metric-label">Put/Call Ratio (PCR)</div>
              <div class="metric-value">
                {oc['pcr']:.2f}
                <span class="pill {'pill-bullish' if oc['oi_sentiment']=='Bullish' else 'pill-bearish'}"
                      style="margin-left:6px;">
                  {oc['oi_sentiment']}
                </span>
              </div>
            </div>
            <div class="metric-row">
              <div class="metric-label">Total Call Buildup (ΔOI)</div>
              <div class="metric-value">{oc['call_buildup']:,}</div>
            </div>
            <div class="metric-row">
              <div class="metric-label">Total Put Buildup (ΔOI)</div>
              <div class="metric-value">{oc['put_buildup']:,}</div>
            </div>
            <div class="metric-row">
              <div class="metric-label">Avg Call IV</div>
              <div class="metric-value">{oc['avg_call_iv']:.2f}%</div>
            </div>
            <div class="metric-row">
              <div class="metric-label">Avg Put IV</div>
              <div class="metric-value">{oc['avg_put_iv']:.2f}%</div>
            </div>
          </div>
          <div>
            <div class="metric-row">
              <div class="metric-label">Expiry</div>
              <div class="metric-value">{run_meta.get('expiry', 'Weekly')}</div>
            </div>
            <div class="metric-row">
              <div class="metric-label">Spot vs Max Pain</div>
              <div class="metric-value">
                Spot ₹{tech['price']:,.2f} / Max Pain ₹{oc['max_pain']:,.0f}
              </div>
            </div>
            <div class="metric-row">
              <div class="metric-label">OI-Based Supports</div>
              <div class="metric-value">{format_list(oc['supports'])}</div>
            </div>
            <div class="metric-row">
              <div class="metric-label">OI-Based Resistances</div>
              <div class="metric-value">{format_list(oc['resistances'])}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Top OI Strikes -->
      <div class="section">
        <div class="section-title">
          <span class="icon">🏦</span>
          Top strikes by open interest
        </div>
        <div class="section-grid">
          <div>
            <div class="metric-row" style="margin-bottom:6px;">
              <div class="metric-label">Top Call OI Strikes</div>
              <div class="metric-value">
                <span class="pill pill-bearish">CALLS</span>
              </div>
            </div>
            <div class="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Strike</th>
                    <th>Type</th>
                    <th>OI</th>
                    <th>ΔOI</th>
                    <th>Volume</th>
                    <th>IV</th>
                    <th>LTP</th>
                  </tr>
                </thead>
                <tbody>
                  {top_ce_rows}
                </tbody>
              </table>
            </div>
          </div>
          <div>
            <div class="metric-row" style="margin-bottom:6px;">
              <div class="metric-label">Top Put OI Strikes</div>
              <div class="metric-value">
                <span class="pill pill-bullish">PUTS</span>
              </div>
            </div>
            <div class="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Strike</th>
                    <th>Type</th>
                    <th>OI</th>
                    <th>ΔOI</th>
                    <th>Volume</th>
                    <th>IV</th>
                    <th>LTP</th>
                  </tr>
                </thead>
                <tbody>
                  {top_pe_rows}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div class="footer">
        Generated automatically for intraday reference only. Not investment advice.
      </div>

    </div>
  </div>
</div>
</body>
</html>
"""
        return html

    # ---------------------------------------------------------------------
    # Email + Save
    # ---------------------------------------------------------------------
    def save_report(self, html_content):
        report_cfg = self.config['report']
        if not report_cfg.get('save_local', True):
            return None

        directory = report_cfg.get('local_dir', './reports')
        os.makedirs(directory, exist_ok=True)
        filename_format = report_cfg.get('filename_format', 'nifty_analysis_%Y%m%d_%H%M%S.html')
        filename = datetime.now().strftime(filename_format)
        path = os.path.join(directory, filename)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"📄 Report saved to {path}")
        return path

    def send_email(self, html_content, subject_suffix=""):
        email_cfg = self.config['email']
        sender = email_cfg['sender']
        recipient = email_cfg['recipient']
        app_password = email_cfg['app_password']
        subject_prefix = email_cfg.get('subject_prefix', 'Nifty Day Trading Report')

        subject = subject_prefix
        if subject_suffix:
            subject += f" - {subject_suffix}"

        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = recipient

        part = MIMEText(html_content, 'html')
        msg.attach(part)

        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender, app_password)
                server.sendmail(sender, [recipient], msg.as_string())
            self.logger.info(f"📧 Email sent to {recipient}")
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")

    # ---------------------------------------------------------------------
    # Main Orchestration
    # ---------------------------------------------------------------------
    def run(self):
        self.logger.info("🚀 Starting Nifty Analyzer run...")

        # Fetch data
        oc_df, spot_price = self.fetch_option_chain()
        tech_df = self.fetch_technical_data()

        if spot_price is None and tech_df is not None:
            spot_price = tech_df['Close'].iloc[-1]

        if tech_df is None:
            tech = self.get_sample_tech_analysis()
        else:
            tech = self.technical_analysis(tech_df)

        if oc_df is None or spot_price is None:
            oc = self.get_sample_oc_analysis()
        else:
            oc = self.analyze_option_chain(oc_df, spot_price)

        expiry = self.get_next_expiry_date()
        run_meta = {
            'expiry': expiry
        }

        recommendation = self.generate_recommendation(tech, oc)

        html = self.create_html_report(tech, oc, recommendation, run_meta)
        path = self.save_report(html)

        # Optional email
        if self.config['email'].get('recipient') and self.config['email'].get('sender'):
            self.send_email(html_content=html, subject_suffix=f"Spot {tech['price']:,.2f}")

        self.logger.info("✅ Nifty Analyzer run completed.")
        return path


if __name__ == "__main__":
    analyzer = NiftyAnalyzer(config_path='config.yml')
    analyzer.run()
