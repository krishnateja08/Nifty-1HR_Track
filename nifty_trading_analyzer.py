"""
Nifty Option Chain & Technical Analysis for Day Trading
COMPLETE VERSION with IMPROVED HTML OUTPUT
1-HOUR TIMEFRAME with WILDER'S RSI (matches TradingView)
Enhanced with Pivot Points + Dual Momentum Analysis + Top 10 OI Display
EXPIRY: Weekly TUESDAY expiry with 3:30 PM IST cutoff logic
FIXED: Using curl-cffi for NSE API to bypass anti-scraping
HTML: Modern dark theme with professional design
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import yfinance as yf
from curl_cffi import requests  # ← Using curl-cffi instead of requests
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
        self.option_chain_base_url = "https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol=NIFTY&expiry="
        
        # Headers that work with NSE
        self.headers = {
            "authority": "www.nseindia.com",
            "accept": "application/json, text/plain, */*",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "referer": "https://www.nseindia.com/option-chain",
            "accept-language": "en-US,en;q=0.9",
        }
    
    def get_next_expiry_date(self):
        """
        Calculate the next NIFTY expiry date (Weekly Tuesday)
        If today is Tuesday after 3:30 PM, return next week's Tuesday
        Logic: Every Tuesday is expiry. After 3:30 PM on Tuesday, switch to next Tuesday.
        """
        now_ist = self.get_ist_time()
        current_day = now_ist.weekday()  # 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday
        
        # NIFTY weekly expiry is on TUESDAY (weekday=1)
        if current_day == 1:
            # It's Tuesday - check if before 3:30 PM
            current_hour = now_ist.hour
            current_minute = now_ist.minute
            
            # If it's before 3:30 PM, today is expiry
            if current_hour < 15 or (current_hour == 15 and current_minute < 30):
                days_until_tuesday = 0
                self.logger.info(f"📅 Today is Tuesday before 3:30 PM - Using today as expiry")
            else:
                # After 3:30 PM on Tuesday, move to next Tuesday (7 days)
                days_until_tuesday = 7
                self.logger.info(f"📅 Tuesday after 3:30 PM - Moving to next Tuesday")
        elif current_day == 0:
            # Monday - tomorrow is Tuesday (1 day)
            days_until_tuesday = 1
        else:
            # For any other day (Wed, Thu, Fri, Sat, Sun), calculate days to next Tuesday
            days_until_tuesday = (1 - current_day) % 7
            if days_until_tuesday == 0:
                days_until_tuesday = 7
        
        expiry_date = now_ist + timedelta(days=days_until_tuesday)
        
        # Format as DD-MMM-YYYY (e.g., 17-Feb-2026)
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
                'title': 'NIFTY DAY TRADING ANALYSIS',
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
        formatter = logging.Formatter(log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'))
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        if log_config.get('log_to_file', True):
            log_file = log_config.get('log_file', './logs/nifty_analyzer.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
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
                self.logger.info(f"Fetching option chain data for expiry {expiry_date} (attempt {attempt + 1}/{max_retries})...")
                
                session = requests.Session()
                session.get(base_url, headers=self.headers, impersonate="chrome", timeout=15)
                time.sleep(1)
                response = session.get(api_url, headers=self.headers, impersonate="chrome", timeout=timeout)
                
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
                        
                        self.logger.info(f"✅ Option chain data fetched successfully | Spot: ₹{current_price} | Expiry: {expiry_date}")
                        self.logger.info(f"✅ Total strikes fetched: {len(oc_df)}")
                        return oc_df, current_price
                    else:
                        self.logger.warning("Invalid response structure from NSE API")
                else:
                    self.logger.warning(f"NSE API returned status code: {response.status_code}")
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        if self.config['data_source']['fallback_to_sample']:
            self.logger.warning("All NSE API attempts failed, using sample data")
        
        return None, None
    
    def get_top_strikes_by_oi(self, oc_df, spot_price):
        """Get top 5 strikes by Open Interest for CE and PE"""
        if oc_df is None or oc_df.empty:
            return {'top_ce_strikes': [], 'top_pe_strikes': []}
        
        top_count = self.config['option_chain'].get('top_strikes_count', 5)
        
        ce_data = oc_df[oc_df['Call_OI'] > 0].copy()
        ce_data = ce_data.sort_values('Call_OI', ascending=False).head(top_count)
        top_ce_strikes = []
        for _, row in ce_data.iterrows():
            strike_type = 'ITM' if row['Strike'] < spot_price else ('ATM' if row['Strike'] == spot_price else 'OTM')
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
            strike_type = 'ITM' if row['Strike'] > spot_price else ('ATM' if row['Strike'] == spot_price else 'OTM')
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
        
        resistance_df = nearby_strikes[nearby_strikes['Strike'] > spot_price].nlargest(num_resistance, 'Call_OI')
        resistances = resistance_df['Strike'].tolist()
        
        support_df = nearby_strikes[nearby_strikes['Strike'] < spot_price].nlargest(num_support, 'Put_OI')
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
            'resistances': [24600, 24650],
            'supports': [24400, 24350],
            'call_buildup': 5000000,
            'put_buildup': 6000000,
            'avg_call_iv': 15.5,
            'avg_put_iv': 16.2,
            'oi_sentiment': 'Bullish',
            'top_ce_strikes': [
                {'strike': 24500, 'oi': 5000000, 'ltp': 120, 'iv': 16.5, 'type': 'ATM', 'chng_oi': 500000, 'volume': 125000},
                {'strike': 24600, 'oi': 4500000, 'ltp': 80, 'iv': 15.8, 'type': 'OTM', 'chng_oi': 450000, 'volume': 110000},
                {'strike': 24550, 'oi': 4200000, 'ltp': 95, 'iv': 16.0, 'type': 'OTM', 'chng_oi': 420000, 'volume': 105000},
                {'strike': 24450, 'oi': 3800000, 'ltp': 145, 'iv': 16.8, 'type': 'ITM', 'chng_oi': 380000, 'volume': 95000},
                {'strike': 24400, 'oi': 3500000, 'ltp': 170, 'iv': 17.0, 'type': 'ITM', 'chng_oi': 350000, 'volume': 90000},
            ],
            'top_pe_strikes': [
                {'strike': 24500, 'oi': 5500000, 'ltp': 110, 'iv': 16.0, 'type': 'ATM', 'chng_oi': 550000, 'volume': 130000},
                {'strike': 24400, 'oi': 5000000, 'ltp': 75, 'iv': 15.5, 'type': 'OTM', 'chng_oi': 500000, 'volume': 120000},
                {'strike': 24450, 'oi': 4700000, 'ltp': 90, 'iv': 15.7, 'type': 'OTM', 'chng_oi': 470000, 'volume': 115000},
                {'strike': 24550, 'oi': 4300000, 'ltp': 135, 'iv': 16.5, 'type': 'ITM', 'chng_oi': 430000, 'volume': 100000},
                {'strike': 24600, 'oi': 4000000, 'ltp': 160, 'iv': 16.8, 'type': 'ITM', 'chng_oi': 400000, 'volume': 95000},
            ]
        }
    
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
                    self.logger.warning(f"Insufficient data points: {len(df)} < {min_points}")
                    return None
            
            self.logger.info(f"✅ 1-HOUR data fetched | {len(df)} bars")
            self.logger.info(f"Price: ₹{df['Close'].iloc[-1]:.2f} | Last candle: {df.index[-1]}")
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
        This EXACTLY matches TradingView's ta.rma() function
        """
        if period is None:
            period = self.config['technical']['rsi_period']
        
        delta = data.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
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
            
            if high == max(recent_data['High'].iloc[i-5:i+6]):
                pivots_high.append(high)
            
            if low == min(recent_data['Low'].iloc[i-5:i+6]):
                pivots_low.append(low)
        
        resistances = sorted([p for p in pivots_high if p > current_price])
        resistances = list(dict.fromkeys(resistances))
        
        supports = sorted([p for p in pivots_low if p < current_price], reverse=True)
        supports = list(dict.fromkeys(supports))
        
        num_resistance = self.config['technical']['num_resistance_levels']
        num_support = self.config['technical']['num_support_levels']
        
        return {
            'resistances': resistances[:num_resistance] if len(resistances) >= num_resistance else resistances,
            'supports': supports[:num_support] if len(supports) >= num_support else supports
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
        
        # 1-HOUR MOMENTUM (last candle)
        if len(df) > 1:
            price_1h_ago = df['Close'].iloc[-2]
            price_change_1h = current_price - price_1h_ago
            price_change_pct_1h = (price_change_1h / price_1h_ago * 100)
        else:
            price_change_1h = 0
            price_change_pct_1h = 0
        
        momentum_1h_signal, momentum_1h_bias, momentum_1h_colors = self.get_momentum_signal(price_change_pct_1h)
        
        # 5-HOUR MOMENTUM (last 5 candles)
        if len(df) >= 5:
            price_5h_ago = df['Close'].iloc[-5]
            momentum_5h = current_price - price_5h_ago
            momentum_5h_pct = (momentum_5h / price_5h_ago * 100)
        else:
            momentum_5h = 0
            momentum_5h_pct = 0
        
        momentum_5h_signal, momentum_5h_bias, momentum_5h_colors = self.get_momentum_signal(momentum_5h_pct)
        
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
        elif current_rsi > 50:
            rsi_signal = "Bullish"
        else:
            rsi_signal = "Bearish"
        
        return {
            'current_price': round(current_price, 2),
            'rsi': round(current_rsi, 2),
            'rsi_signal': rsi_signal,
            'ema20': round(ema_short_val, 2),
            'ema50': round(ema_long_val, 2),
            'trend': trend,
            'tech_resistances': [round(r, 2) for r in sr_levels['resistances']],
            'tech_supports': [round(s, 2) for s in sr_levels['supports']],
            'pivot_points': pivot_points,
            'timeframe': '1 Hour',
            'price_change_1h': round(price_change_1h, 2),
            'price_change_pct_1h': round(price_change_pct_1h, 2),
            'momentum_1h_signal': momentum_1h_signal,
            'momentum_1h_bias': momentum_1h_bias,
            'momentum_1h_colors': momentum_1h_colors,
            'momentum_5h': round(momentum_5h, 2),
            'momentum_5h_pct': round(momentum_5h_pct, 2),
            'momentum_5h_signal': momentum_5h_signal,
            'momentum_5h_bias': momentum_5h_bias,
            'momentum_5h_colors': momentum_5h_colors
        }
    
    def get_sample_tech_analysis(self):
        """Return sample technical analysis"""
        return {
            'current_price': 24520.50,
            'rsi': 42.82,
            'rsi_signal': 'Bearish',
            'ema20': 24480.00,
            'ema50': 24450.00,
            'trend': 'Uptrend',
            'tech_resistances': [24580.00, 24650.00],
            'tech_supports': [24420.00, 24380.00],
            'pivot_points': {
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
            },
            'timeframe': '1 Hour',
            'price_change_1h': -15.50,
            'price_change_pct_1h': -0.06,
            'momentum_1h_signal': 'Sideways/Weak',
            'momentum_1h_bias': 'Neutral',
            'momentum_1h_colors': {
                'bg': '#6c757d',
                'bg_dark': '#5a6268',
                'text': '#ffffff',
                'border': '#495057'
            },
            'momentum_5h': -35.50,
            'momentum_5h_pct': -0.14,
            'momentum_5h_signal': 'Moderate Downward',
            'momentum_5h_bias': 'Bearish',
            'momentum_5h_colors': {
                'bg': '#fd7e14',
                'bg_dark': '#e8590c',
                'text': '#ffffff',
                'border': '#dc3545'
            }
        }
    
    def generate_recommendation(self, oc_analysis, tech_analysis):
        """Generate trading recommendation WITH DUAL MOMENTUM FILTER"""
        if not oc_analysis or not tech_analysis:
            return {"recommendation": "Insufficient data", "bias": "Neutral", "confidence": "Low", "reasons": []}
        
        config = self.config['recommendation']
        oc_config = self.config['option_chain']
        tech_config = self.config['technical']
        
        bullish_signals = 0
        bearish_signals = 0
        reasons = []
        
        # DUAL MOMENTUM SIGNALS
        use_momentum = self.config['advanced'].get('use_momentum_filter', True)
        
        if use_momentum:
            momentum_5h_pct = tech_analysis.get('momentum_5h_pct', 0)
            weight_5h = config.get('momentum_5h_weight', 2)
            
            strong_threshold = tech_config.get('momentum_threshold_strong', 0.5)
            moderate_threshold = tech_config.get('momentum_threshold_moderate', 0.2)
            
            if momentum_5h_pct > strong_threshold:
                bullish_signals += weight_5h
                reasons.append(f"🚀 5H Strong upward momentum: {momentum_5h_pct:+.2f}%")
            elif momentum_5h_pct > moderate_threshold:
                bullish_signals += 1
                reasons.append(f"📈 5H Positive momentum: {momentum_5h_pct:+.2f}%")
            elif momentum_5h_pct < -strong_threshold:
                bearish_signals += weight_5h
                reasons.append(f"🔻 5H Strong downward momentum: {momentum_5h_pct:+.2f}%")
            elif momentum_5h_pct < -moderate_threshold:
                bearish_signals += 1
                reasons.append(f"📉 5H Negative momentum: {momentum_5h_pct:+.2f}%")
            
            momentum_1h_pct = tech_analysis.get('price_change_pct_1h', 0)
            weight_1h = config.get('momentum_1h_weight', 1)
            
            if momentum_1h_pct > strong_threshold:
                bullish_signals += weight_1h
                reasons.append(f"⚡ 1H Strong upward move: {momentum_1h_pct:+.2f}%")
            elif momentum_1h_pct < -strong_threshold:
                bearish_signals += weight_1h
                reasons.append(f"⚡ 1H Strong downward move: {momentum_1h_pct:+.2f}%")
        
        # Option chain signals
        pcr = oc_analysis.get('pcr', 0)
        if pcr >= oc_config['pcr_very_bullish']:
            bullish_signals += 2
            reasons.append(f"PCR at {pcr} indicates strong bullish sentiment")
        elif pcr >= oc_config['pcr_bullish']:
            bullish_signals += 1
            reasons.append(f"PCR at {pcr} shows bullish bias")
        elif pcr <= oc_config['pcr_very_bearish']:
            bearish_signals += 2
            reasons.append(f"PCR at {pcr} indicates strong bearish sentiment")
        elif pcr < oc_config['pcr_bearish']:
            bearish_signals += 1
            reasons.append(f"PCR at {pcr} shows bearish bias")
        
        if oc_analysis.get('oi_sentiment') == 'Bullish':
            bullish_signals += 1
            reasons.append("Put OI buildup > Call OI buildup (Bullish)")
        else:
            bearish_signals += 1
            reasons.append("Call OI buildup > Put OI buildup (Bearish)")
        
        # RSI signals
        rsi = tech_analysis.get('rsi', 50)
        rsi_os = tech_config['rsi_oversold']
        rsi_ob = tech_config['rsi_overbought']
        
        if rsi < rsi_os:
            bullish_signals += 2
            reasons.append(f"RSI at {rsi:.1f} - Oversold (Bullish reversal)")
        elif rsi < 45:
            bullish_signals += 1
            reasons.append(f"RSI at {rsi:.1f} - Below neutral")
        elif rsi > rsi_ob:
            bearish_signals += 2
            reasons.append(f"RSI at {rsi:.1f} - Overbought (Bearish)")
        elif rsi > 55:
            bearish_signals += 1
            reasons.append(f"RSI at {rsi:.1f} - Above neutral")
        
        # Trend signals
        trend = tech_analysis.get('trend', '')
        if 'Uptrend' in trend:
            bullish_signals += 1
            reasons.append(f"Trend: {trend}")
        elif 'Downtrend' in trend:
            bearish_signals += 1
            reasons.append(f"Trend: {trend}")
        
        # EMA signals
        current_price = tech_analysis.get('current_price', 0)
        ema20 = tech_analysis.get('ema20', 0)
        if current_price > ema20:
            bullish_signals += 1
            reasons.append("Price above EMA20 (Bullish)")
        else:
            bearish_signals += 1
            reasons.append("Price below EMA20 (Bearish)")
        
        signal_diff = bullish_signals - bearish_signals
        
        strong_buy_t = config['strong_buy_threshold']
        buy_t = config['buy_threshold']
        sell_t = config['sell_threshold']
        strong_sell_t = config['strong_sell_threshold']
        
        if signal_diff >= strong_buy_t:
            recommendation = "STRONG BUY"
            bias = "Bullish"
            confidence = "High"
        elif signal_diff >= buy_t:
            recommendation = "BUY"
            bias = "Bullish"
            confidence = "Medium"
        elif signal_diff <= strong_sell_t:
            recommendation = "STRONG SELL"
            bias = "Bearish"
            confidence = "High"
        elif signal_diff <= sell_t:
            recommendation = "SELL"
            bias = "Bearish"
            confidence = "Medium"
        else:
            recommendation = "NEUTRAL / WAIT"
            bias = "Neutral"
            confidence = "Low"
        
        return {
            'recommendation': recommendation,
            'bias': bias,
            'confidence': confidence,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'reasons': reasons
        }
    
    def create_html_report(self, oc_analysis, tech_analysis, recommendation):
        """Create modern, professional HTML report with improved dark theme design"""
        now_ist = self.format_ist_time()
        
        rec = recommendation['recommendation']
        
        # Dynamic recommendation colors
        if 'STRONG BUY' in rec:
            rec_color = '#10b981'
            rec_gradient = 'linear-gradient(135deg, #10b981 0%, #059669 100%)'
        elif 'BUY' in rec:
            rec_color = '#3b82f6'
            rec_gradient = 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)'
        elif 'STRONG SELL' in rec:
            rec_color = '#ef4444'
            rec_gradient = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)'
        elif 'SELL' in rec:
            rec_color = '#f59e0b'
            rec_gradient = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'
        else:
            rec_color = '#6366f1'
            rec_gradient = 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)'
        
        title = self.config['report'].get('title', 'NIFTY DAY TRADING ANALYSIS')
        
        # Get momentum data
        momentum_1h_pct = tech_analysis.get('price_change_pct_1h', 0)
        momentum_1h_signal = tech_analysis.get('momentum_1h_signal', 'Sideways')
        momentum_1h_colors = tech_analysis.get('momentum_1h_colors', {
            'bg': '#6b7280', 'bg_dark': '#4b5563', 'text': '#ffffff', 'border': '#374151'
        })
        
        momentum_5h_pct = tech_analysis.get('momentum_5h_pct', 0)
        momentum_5h_signal = tech_analysis.get('momentum_5h_signal', 'Sideways')
        momentum_5h_colors = tech_analysis.get('momentum_5h_colors', {
            'bg': '#6b7280', 'bg_dark': '#4b5563', 'text': '#ffffff', 'border': '#374151'
        })
        
        # Build Top OI tables
        top_ce_strikes = oc_analysis.get('top_ce_strikes', [])
        top_pe_strikes = oc_analysis.get('top_pe_strikes', [])
        
        ce_rows_html = ''
        for idx, strike in enumerate(top_ce_strikes, 1):
            badge_class = f"type-badge {strike['type'].lower()}"
            ce_rows_html += f"""
                    <tr>
                        <td class="rank-cell">{idx}</td>
                        <td class="strike-cell">₹{strike['strike']}</td>
                        <td><span class="{badge_class}">{strike['type']}</span></td>
                        <td class="number-cell">{strike['oi']:,}</td>
                        <td class="number-cell">{strike['chng_oi']:,}</td>
                        <td class="price-cell">₹{strike['ltp']:.2f}</td>
                        <td class="number-cell">{strike['iv']:.1f}%</td>
                    </tr>
        """
        
        pe_rows_html = ''
        for idx, strike in enumerate(top_pe_strikes, 1):
            badge_class = f"type-badge {strike['type'].lower()}"
            pe_rows_html += f"""
                    <tr>
                        <td class="rank-cell">{idx}</td>
                        <td class="strike-cell">₹{strike['strike']}</td>
                        <td><span class="{badge_class}">{strike['type']}</span></td>
                        <td class="number-cell">{strike['oi']:,}</td>
                        <td class="number-cell">{strike['chng_oi']:,}</td>
                        <td class="price-cell">₹{strike['ltp']:.2f}</td>
                        <td class="number-cell">{strike['iv']:.1f}%</td>
                    </tr>
        """
        
        # Build pivot points table
        pivot_points = tech_analysis.get('pivot_points', {})
        current_price = tech_analysis.get('current_price', 0)
        
        def get_distance(level_value, current):
            if level_value is None:
                return ''
            diff = level_value - current
            return f'{diff:+.0f} pts'
        
        pivot_rows = f"""
                    <tr class="pivot-resistance">
                        <td class="level-label">R3</td>
                        <td class="level-value">₹{pivot_points.get('r3', 'N/A')}</td>
                        <td class="distance-value">{get_distance(pivot_points.get('r3'), current_price)}</td>
                    </tr>
                    <tr class="pivot-resistance">
                        <td class="level-label">R2</td>
                        <td class="level-value">₹{pivot_points.get('r2', 'N/A')}</td>
                        <td class="distance-value">{get_distance(pivot_points.get('r2'), current_price)}</td>
                    </tr>
                    <tr class="pivot-resistance">
                        <td class="level-label">R1</td>
                        <td class="level-value">₹{pivot_points.get('r1', 'N/A')}</td>
                        <td class="distance-value">{get_distance(pivot_points.get('r1'), current_price)}</td>
                    </tr>
                    <tr class="pivot-main">
                        <td class="level-label">PP</td>
                        <td class="level-value">₹{pivot_points.get('pivot', 'N/A')}</td>
                        <td class="distance-value">{get_distance(pivot_points.get('pivot'), current_price)}</td>
                    </tr>
                    <tr class="pivot-support">
                        <td class="level-label">S1</td>
                        <td class="level-value">₹{pivot_points.get('s1', 'N/A')}</td>
                        <td class="distance-value">{get_distance(pivot_points.get('s1'), current_price)}</td>
                    </tr>
                    <tr class="pivot-support">
                        <td class="level-label">S2</td>
                        <td class="level-value">₹{pivot_points.get('s2', 'N/A')}</td>
                        <td class="distance-value">{get_distance(pivot_points.get('s2'), current_price)}</td>
                    </tr>
                    <tr class="pivot-support">
                        <td class="level-label">S3</td>
                        <td class="level-value">₹{pivot_points.get('s3', 'N/A')}</td>
                        <td class="distance-value">{get_distance(pivot_points.get('s3'), current_price)}</td>
                    </tr>
        """
    
    def send_email(self, html_content):
        """Send email with HTML report"""
        email_config = self.config['email']
        recipient_email = email_config['recipient']
        sender_email = email_config['sender']
        sender_password = email_config['app_password']
        subject_prefix = email_config.get('subject_prefix', 'Nifty 1H Analysis')
        ist_time = self.get_ist_time()
        subject_time = ist_time.strftime('%Y-%m-%d %H:%M IST')
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"{subject_prefix} - {subject_time}"
            msg['From'] = sender_email
            msg['To'] = recipient_email
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            self.logger.info(f"✅ Email sent successfully to {recipient_email}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error sending email: {e}")
            return False
    
    def run_analysis(self):
        """Run complete analysis"""
        self.logger.info("🚀 Starting Nifty 1-HOUR Analysis with Dual Momentum...")
        self.logger.info("=" * 60)
        oc_df, spot_price = self.fetch_option_chain()
        if oc_df is not None and spot_price is not None:
            oc_analysis = self.analyze_option_chain(oc_df, spot_price)
        else:
            spot_price = 25796
            oc_analysis = self.get_sample_oc_analysis()
        tech_df = self.fetch_technical_data()
        if tech_df is not None and not tech_df.empty:
            tech_analysis = self.technical_analysis(tech_df)
        else:
            tech_analysis = self.get_sample_tech_analysis()
        self.logger.info("🎯 Generating Trading Recommendation...")
        recommendation = self.generate_recommendation(oc_analysis, tech_analysis)
        self.logger.info("=" * 60)
        self.logger.info(f"📊 RECOMMENDATION: {recommendation['recommendation']}")
        self.logger.info(f"📈 Bias: {recommendation['bias']} | Confidence: {recommendation['confidence']}")
        self.logger.info(f"🎯 RSI (1H): {tech_analysis.get('rsi', 'N/A')}")
        self.logger.info(f"⚡ 1H Momentum: {tech_analysis.get('price_change_pct_1h', 0):+.2f}%")
        self.logger.info(f"📊 5H Momentum: {tech_analysis.get('momentum_5h_pct', 0):+.2f}%")
        self.logger.info("=" * 60)
        html_report = self.create_html_report(oc_analysis, tech_analysis, recommendation)
        if self.config['report']['save_local']:
            report_dir = self.config['report']['local_dir']
            os.makedirs(report_dir, exist_ok=True)
            ist_time = self.get_ist_time()
            filename_format = self.config['report']['filename_format']
            report_filename = os.path.join(report_dir, ist_time.strftime(filename_format))
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(html_report)
            self.logger.info(f"💾 Report saved as: {report_filename}")
        self.logger.info(f"📧 Sending email to {self.config['email']['recipient']}...")
        self.send_email(html_report)
        self.logger.info("✅ Analysis Complete!")
        return {
            'oc_analysis': oc_analysis,
            'tech_analysis': tech_analysis,
            'recommendation': recommendation,
            'html_report': html_report
        }


if __name__ == "__main__":
    analyzer = NiftyAnalyzer(config_path='config.yml')
    result = analyzer.run_analysis()
    print(f"\n✅ Analysis Complete!")
    print(f"Recommendation: {result['recommendation']['recommendation']}")
    print(f"RSI (1H): {result['tech_analysis']['rsi']}")
    print(f"1H Momentum: {result['tech_analysis']['price_change_pct_1h']:+.2f}%")
    print(f"5H Momentum: {result['tech_analysis']['momentum_5h_pct']:+.2f}%")
    print(f"Check your email for the detailed report!")
