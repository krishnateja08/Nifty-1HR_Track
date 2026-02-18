"""
Nifty Option Chain & Technical Analysis for Day Trading
PROFESSIONAL VERSION - Enhanced Design with Better Contrast
1-HOUR TIMEFRAME with WILDER'S RSI (matches TradingView)
Enhanced with Pivot Points + Dual Momentum Analysis + Top 10 OI Display
EXPIRY: Weekly TUESDAY expiry with 3:30 PM IST cutoff logic
FIXED: Using curl-cffi for NSE API to bypass anti-scraping
UPDATED: Professional grey theme with improved readability
BUGFIX: ValueError: Unknown format code 'f' for object of type 'str' - fixed in create_html_report
UPDATED: Terminal Split Panel Pivot Points Widget (v2)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import yfinance as yf
from curl_cffi import requests
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
        self.option_chain_base_url = "https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol=NIFTY&expiry="
        
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
        """
        now_ist = self.get_ist_time()
        current_day = now_ist.weekday()  # 0=Monday, 1=Tuesday, ...
        
        if current_day == 1:
            current_hour = now_ist.hour
            current_minute = now_ist.minute
            if current_hour < 15 or (current_hour == 15 and current_minute < 30):
                days_until_tuesday = 0
                self.logger.info(f"📅 Today is Tuesday before 3:30 PM - Using today as expiry")
            else:
                days_until_tuesday = 7
                self.logger.info(f"📅 Tuesday after 3:30 PM - Moving to next Tuesday")
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
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
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
                        
                        self.logger.info(f"✅ Option chain fetched | Spot: ₹{current_price} | Expiry: {expiry_date}")
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
                {'strike': 24600, 'oi': 4500000, 'ltp': 80,  'iv': 15.8, 'type': 'OTM', 'chng_oi': 450000, 'volume': 110000},
                {'strike': 24550, 'oi': 4200000, 'ltp': 95,  'iv': 16.0, 'type': 'OTM', 'chng_oi': 420000, 'volume': 105000},
                {'strike': 24450, 'oi': 3800000, 'ltp': 145, 'iv': 16.8, 'type': 'ITM', 'chng_oi': 380000, 'volume': 95000},
                {'strike': 24400, 'oi': 3500000, 'ltp': 170, 'iv': 17.0, 'type': 'ITM', 'chng_oi': 350000, 'volume': 90000},
            ],
            'top_pe_strikes': [
                {'strike': 24500, 'oi': 5500000, 'ltp': 110, 'iv': 16.0, 'type': 'ATM', 'chng_oi': 550000, 'volume': 130000},
                {'strike': 24400, 'oi': 5000000, 'ltp': 75,  'iv': 15.5, 'type': 'OTM', 'chng_oi': 500000, 'volume': 120000},
                {'strike': 24450, 'oi': 4700000, 'ltp': 90,  'iv': 15.7, 'type': 'OTM', 'chng_oi': 470000, 'volume': 115000},
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
        """Calculate Traditional Pivot Points (30-minute timeframe)"""
        try:
            ticker = yf.Ticker(self.nifty_symbol)
            min_30_df = ticker.history(period='5d', interval='30m')
            
            if len(min_30_df) >= 2:
                prev_high  = min_30_df['High'].iloc[-2]
                prev_low   = min_30_df['Low'].iloc[-2]
                prev_close = min_30_df['Close'].iloc[-2]
            else:
                prev_high  = df['High'].max()
                prev_low   = df['Low'].min()
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
                'pivot':      round(pivot, 2),
                'r1':         round(r1, 2),
                'r2':         round(r2, 2),
                'r3':         round(r3, 2),
                's1':         round(s1, 2),
                's2':         round(s2, 2),
                's3':         round(s3, 2),
                'prev_high':  round(prev_high, 2),
                'prev_low':   round(prev_low, 2),
                'prev_close': round(prev_close, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating pivot points: {e}")
            return {
                'pivot': 24520.00, 'r1': 24590.00, 'r2': 24650.00, 'r3': 24720.00,
                's1': 24450.00,    's2': 24390.00,  's3': 24320.00,
                'prev_high': 24580.00, 'prev_low': 24420.00, 'prev_close': 24500.00
            }
    
    def calculate_rsi(self, data, period=None):
        """Calculate RSI using Wilder's smoothing method (matches TradingView)"""
        if period is None:
            period = self.config['technical']['rsi_period']
        
        delta = data.diff()
        gain  = delta.where(delta > 0, 0)
        loss  = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs  = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_support_resistance(self, df, current_price):
        """Calculate nearest support and resistance levels from price action"""
        recent_data = df.tail(300)
        
        pivots_high = []
        pivots_low  = []
        
        for i in range(5, len(recent_data) - 5):
            high = recent_data['High'].iloc[i]
            low  = recent_data['Low'].iloc[i]
            
            if high == max(recent_data['High'].iloc[i-5:i+6]):
                pivots_high.append(high)
            if low == min(recent_data['Low'].iloc[i-5:i+6]):
                pivots_low.append(low)
        
        resistances = sorted([p for p in pivots_high if p > current_price])
        resistances = list(dict.fromkeys(resistances))
        
        supports = sorted([p for p in pivots_low if p < current_price], reverse=True)
        supports = list(dict.fromkeys(supports))
        
        num_resistance = self.config['technical']['num_resistance_levels']
        num_support    = self.config['technical']['num_support_levels']
        
        return {
            'resistances': resistances[:num_resistance],
            'supports':    supports[:num_support]
        }
    
    def get_momentum_signal(self, momentum_pct):
        """Get momentum signal, bias, and CSS color variables based on percentage"""
        strong_threshold   = self.config['technical'].get('momentum_threshold_strong', 0.5)
        moderate_threshold = self.config['technical'].get('momentum_threshold_moderate', 0.2)
        
        if momentum_pct > strong_threshold:
            return "Strong Upward", "Bullish", {
                'bg': '#1e7e34', 'bg_dark': '#155724', 'text': '#ffffff', 'border': '#28a745'
            }
        elif momentum_pct > moderate_threshold:
            return "Moderate Upward", "Bullish", {
                'bg': '#28a745', 'bg_dark': '#218838', 'text': '#ffffff', 'border': '#1e7e34'
            }
        elif momentum_pct < -strong_threshold:
            return "Strong Downward", "Bearish", {
                'bg': '#c82333', 'bg_dark': '#bd2130', 'text': '#ffffff', 'border': '#dc3545'
            }
        elif momentum_pct < -moderate_threshold:
            return "Moderate Downward", "Bearish", {
                'bg': '#fd7e14', 'bg_dark': '#e8590c', 'text': '#ffffff', 'border': '#dc3545'
            }
        else:
            return "Sideways/Weak", "Neutral", {
                'bg': '#6c757d', 'bg_dark': '#5a6268', 'text': '#ffffff', 'border': '#495057'
            }
    
    def technical_analysis(self, df):
        """Perform complete technical analysis - 1 HOUR TIMEFRAME with DUAL MOMENTUM"""
        if df is None or df.empty:
            self.logger.warning("No technical data, using sample analysis")
            return self.get_sample_tech_analysis()
        
        current_price = df['Close'].iloc[-1]
        
        # 1-HOUR MOMENTUM
        if len(df) > 1:
            price_1h_ago        = df['Close'].iloc[-2]
            price_change_1h     = current_price - price_1h_ago
            price_change_pct_1h = (price_change_1h / price_1h_ago * 100)
        else:
            price_change_1h     = 0
            price_change_pct_1h = 0
        
        momentum_1h_signal, momentum_1h_bias, momentum_1h_colors = self.get_momentum_signal(price_change_pct_1h)
        
        # 5-HOUR MOMENTUM
        if len(df) >= 5:
            price_5h_ago  = df['Close'].iloc[-5]
            momentum_5h   = current_price - price_5h_ago
            momentum_5h_pct = (momentum_5h / price_5h_ago * 100)
        else:
            momentum_5h     = 0
            momentum_5h_pct = 0
        
        momentum_5h_signal, momentum_5h_bias, momentum_5h_colors = self.get_momentum_signal(momentum_5h_pct)
        
        self.logger.info(f"📊 1H Momentum: {price_change_pct_1h:+.2f}% - {momentum_1h_signal}")
        self.logger.info(f"📊 5H Momentum: {momentum_5h_pct:+.2f}% - {momentum_5h_signal}")
        
        df['RSI'] = self.calculate_rsi(df['Close'])
        current_rsi = df['RSI'].iloc[-1]
        self.logger.info(f"🎯 RSI(14) calculated: {current_rsi:.2f} (Wilder's method)")
        
        ema_short = self.config['technical']['ema_short']
        ema_long  = self.config['technical']['ema_long']
        
        df['EMA_Short'] = df['Close'].ewm(span=ema_short, adjust=False).mean()
        df['EMA_Long']  = df['Close'].ewm(span=ema_long,  adjust=False).mean()
        
        ema_short_val = df['EMA_Short'].iloc[-1]
        ema_long_val  = df['EMA_Long'].iloc[-1]
        
        sr_levels    = self.calculate_support_resistance(df, current_price)
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
            'current_price':       round(current_price, 2),
            'rsi':                 round(current_rsi, 2),
            'rsi_signal':          rsi_signal,
            'ema20':               round(ema_short_val, 2),
            'ema50':               round(ema_long_val, 2),
            'trend':               trend,
            'tech_resistances':    [round(r, 2) for r in sr_levels['resistances']],
            'tech_supports':       [round(s, 2) for s in sr_levels['supports']],
            'pivot_points':        pivot_points,
            'timeframe':           '1 Hour',
            'price_change_1h':     round(price_change_1h, 2),
            'price_change_pct_1h': round(price_change_pct_1h, 2),
            'momentum_1h_signal':  momentum_1h_signal,
            'momentum_1h_bias':    momentum_1h_bias,
            'momentum_1h_colors':  momentum_1h_colors,
            'momentum_5h':         round(momentum_5h, 2),
            'momentum_5h_pct':     round(momentum_5h_pct, 2),
            'momentum_5h_signal':  momentum_5h_signal,
            'momentum_5h_bias':    momentum_5h_bias,
            'momentum_5h_colors':  momentum_5h_colors
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
            'tech_supports':    [24420.00, 24380.00],
            'pivot_points': {
                'pivot': 24520.00, 'r1': 24590.00, 'r2': 24650.00, 'r3': 24720.00,
                's1': 24450.00,    's2': 24390.00,  's3': 24320.00,
                'prev_high': 24580.00, 'prev_low': 24420.00, 'prev_close': 24500.00
            },
            'timeframe': '1 Hour',
            'price_change_1h':     -15.50,
            'price_change_pct_1h': -0.06,
            'momentum_1h_signal':  'Sideways/Weak',
            'momentum_1h_bias':    'Neutral',
            'momentum_1h_colors':  {'bg': '#6c757d', 'bg_dark': '#5a6268', 'text': '#ffffff', 'border': '#495057'},
            'momentum_5h':         -35.50,
            'momentum_5h_pct':     -0.14,
            'momentum_5h_signal':  'Moderate Downward',
            'momentum_5h_bias':    'Bearish',
            'momentum_5h_colors':  {'bg': '#fd7e14', 'bg_dark': '#e8590c', 'text': '#ffffff', 'border': '#dc3545'}
        }
    
    def generate_recommendation(self, oc_analysis, tech_analysis):
        """Generate trading recommendation WITH DUAL MOMENTUM FILTER"""
        if not oc_analysis or not tech_analysis:
            return {"recommendation": "Insufficient data", "bias": "Neutral",
                    "confidence": "Low", "reasons": [],
                    "bullish_signals": 0, "bearish_signals": 0}
        
        config     = self.config['recommendation']
        oc_config  = self.config['option_chain']
        tech_config = self.config['technical']
        
        bullish_signals = 0
        bearish_signals = 0
        reasons = []
        
        use_momentum = self.config['advanced'].get('use_momentum_filter', True)
        
        if use_momentum:
            momentum_5h_pct    = tech_analysis.get('momentum_5h_pct', 0)
            weight_5h          = config.get('momentum_5h_weight', 2)
            strong_threshold   = tech_config.get('momentum_threshold_strong', 0.5)
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
            weight_1h       = config.get('momentum_1h_weight', 1)
            
            if momentum_1h_pct > strong_threshold:
                bullish_signals += weight_1h
                reasons.append(f"⚡ 1H Strong upward move: {momentum_1h_pct:+.2f}%")
            elif momentum_1h_pct < -strong_threshold:
                bearish_signals += weight_1h
                reasons.append(f"⚡ 1H Strong downward move: {momentum_1h_pct:+.2f}%")
        
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
        
        rsi    = tech_analysis.get('rsi', 50)
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
        
        trend = tech_analysis.get('trend', '')
        if 'Uptrend' in trend:
            bullish_signals += 1
            reasons.append(f"Trend: {trend}")
        elif 'Downtrend' in trend:
            bearish_signals += 1
            reasons.append(f"Trend: {trend}")
        
        current_price = tech_analysis.get('current_price', 0)
        ema20         = tech_analysis.get('ema20', 0)
        if current_price > ema20:
            bullish_signals += 1
            reasons.append("Price above EMA20 (Bullish)")
        else:
            bearish_signals += 1
            reasons.append("Price below EMA20 (Bearish)")
        
        signal_diff    = bullish_signals - bearish_signals
        strong_buy_t   = config['strong_buy_threshold']
        buy_t          = config['buy_threshold']
        sell_t         = config['sell_threshold']
        strong_sell_t  = config['strong_sell_threshold']
        
        if signal_diff >= strong_buy_t:
            recommendation = "STRONG BUY"
            bias, confidence = "Bullish", "High"
        elif signal_diff >= buy_t:
            recommendation = "BUY"
            bias, confidence = "Bullish", "Medium"
        elif signal_diff <= strong_sell_t:
            recommendation = "STRONG SELL"
            bias, confidence = "Bearish", "High"
        elif signal_diff <= sell_t:
            recommendation = "SELL"
            bias, confidence = "Bearish", "Medium"
        else:
            recommendation = "NEUTRAL / WAIT"
            bias, confidence = "Neutral", "Low"
        
        return {
            'recommendation':  recommendation,
            'bias':            bias,
            'confidence':      confidence,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'reasons':         reasons
        }
    
    def get_options_strategies(self, recommendation, oc_analysis, tech_analysis):
        """Generate options trading strategy recommendations"""
        bias    = recommendation['bias']
        avg_iv  = (oc_analysis.get('avg_call_iv', 15) + oc_analysis.get('avg_put_iv', 15)) / 2
        
        high_volatility = avg_iv > 18
        strategies = []
        
        if bias == 'Bullish':
            strategies.append({
                'name':        'Long Call',
                'type':        'Bullish - Aggressive',
                'setup':       'Buy ATM or slightly OTM Call option',
                'profit':      'Unlimited upside',
                'risk':        'Limited to premium paid',
                'best_when':   'Strong upward move expected, low IV',
                'recommended': '⭐⭐⭐⭐⭐' if recommendation['confidence'] == 'High' and not high_volatility else '⭐⭐⭐'
            })
            strategies.append({
                'name':        'Bull Call Spread',
                'type':        'Bullish - Moderate',
                'setup':       'Buy ITM Call + Sell OTM Call',
                'profit':      'Limited (Strike difference - Net premium)',
                'risk':        'Limited to net premium paid',
                'best_when':   'Moderately bullish, reduce cost',
                'recommended': '⭐⭐⭐⭐⭐' if recommendation['confidence'] == 'Medium' else '⭐⭐⭐⭐'
            })
        
        elif bias == 'Bearish':
            strategies.append({
                'name':        'Long Put',
                'type':        'Bearish - Aggressive',
                'setup':       'Buy ATM or slightly OTM Put option',
                'profit':      'High (Strike - Stock price - Premium)',
                'risk':        'Limited to premium paid',
                'best_when':   'Strong downward move expected, low IV',
                'recommended': '⭐⭐⭐⭐⭐' if recommendation['confidence'] == 'High' and not high_volatility else '⭐⭐⭐'
            })
            strategies.append({
                'name':        'Bear Put Spread',
                'type':        'Bearish - Debit Strategy',
                'setup':       'Buy ITM Put + Sell OTM Put',
                'profit':      'Limited (Strike difference - Net premium)',
                'risk':        'Limited to net premium paid',
                'best_when':   'Moderately bearish, reduce cost',
                'recommended': '⭐⭐⭐⭐⭐' if recommendation['confidence'] == 'Medium' else '⭐⭐⭐⭐'
            })
        
        else:
            if high_volatility:
                strategies.append({
                    'name':        'Long Straddle',
                    'type':        'Neutral - High Volatility Expected',
                    'setup':       'Buy ATM Call + Buy ATM Put',
                    'profit':      'Unlimited (either direction)',
                    'risk':        'Limited to total premium paid',
                    'best_when':   'Expect big move, unsure of direction',
                    'recommended': '⭐⭐⭐⭐⭐'
                })
            else:
                strategies.append({
                    'name':        'Short Strangle',
                    'type':        'Neutral - Low Volatility Expected',
                    'setup':       'Sell OTM Call + Sell OTM Put',
                    'profit':      'Limited to total premium collected',
                    'risk':        'Unlimited (either direction)',
                    'best_when':   'Expect range-bound, less risk than straddle',
                    'recommended': '⭐⭐⭐⭐⭐'
                })
        
        return strategies
    
    def get_detailed_strike_recommendations(self, oc_analysis, tech_analysis, recommendation):
        """Generate detailed strike price recommendations with LTP and profit calculations"""
        current_price   = tech_analysis.get('current_price', 0)
        bias            = recommendation['bias']
        atm_strike      = round(current_price / 50) * 50
        top_ce_strikes  = oc_analysis.get('top_ce_strikes', [])
        top_pe_strikes  = oc_analysis.get('top_pe_strikes', [])
        
        def find_closest_strike(target_strike, strike_list):
            if not strike_list:
                return None
            return min(strike_list, key=lambda x: abs(x['strike'] - target_strike))
        
        strike_recommendations = []
        
        if bias == 'Bullish':
            atm_ce = find_closest_strike(atm_strike, top_ce_strikes)
            if atm_ce:
                actual_strike = atm_ce['strike']
                strike_recommendations.append({
                    'strategy':           'Long Call (ATM)',
                    'action':             'BUY',
                    'strike':             actual_strike,
                    'type':               'CE',
                    'ltp':                atm_ce['ltp'],
                    'option_type':        'ATM',
                    'target_1':           actual_strike + 100,
                    'target_2':           actual_strike + 200,
                    'stop_loss':          atm_ce['ltp'] * 0.3,
                    'max_loss':           atm_ce['ltp'],
                    'profit_at_target_1': 100 - atm_ce['ltp'],
                    'profit_at_target_2': 200 - atm_ce['ltp'],
                    'oi':                 atm_ce['oi'],
                    'volume':             atm_ce['volume']
                })
            
            otm_target = atm_strike + 50
            otm_ce     = find_closest_strike(otm_target, top_ce_strikes)
            if otm_ce and otm_ce['strike'] != (atm_ce['strike'] if atm_ce else None):
                actual_strike = otm_ce['strike']
                strike_recommendations.append({
                    'strategy':           'Long Call (OTM)',
                    'action':             'BUY',
                    'strike':             actual_strike,
                    'type':               'CE',
                    'ltp':                otm_ce['ltp'],
                    'option_type':        'OTM',
                    'target_1':           actual_strike + 100,
                    'target_2':           actual_strike + 150,
                    'stop_loss':          otm_ce['ltp'] * 0.3,
                    'max_loss':           otm_ce['ltp'],
                    'profit_at_target_1': 100 - otm_ce['ltp'],
                    'profit_at_target_2': 150 - otm_ce['ltp'],
                    'oi':                 otm_ce['oi'],
                    'volume':             otm_ce['volume']
                })
            
            itm_target = atm_strike - 50
            itm_ce     = find_closest_strike(itm_target, top_ce_strikes)
            if itm_ce and otm_ce and len(strike_recommendations) >= 1:
                itm_k       = itm_ce['strike']
                otm_k       = otm_ce['strike']
                net_premium = itm_ce['ltp'] - otm_ce['ltp']
                max_profit  = (otm_k - itm_k) - net_premium
                strike_recommendations.append({
                    'strategy':           'Bull Call Spread',
                    'action':             f"BUY {itm_k} CE + SELL {otm_k} CE",
                    'strike':             f"{itm_k}/{otm_k}",
                    'type':               'Spread',
                    'ltp':                net_premium,
                    'option_type':        'ITM/OTM',
                    'target_1':           itm_k + 25,
                    'target_2':           otm_k,
                    'stop_loss':          net_premium * 0.4,
                    'max_loss':           net_premium,
                    'profit_at_target_1': 25 - net_premium,
                    'profit_at_target_2': max_profit,
                    'oi':                 f"{itm_ce['oi']:,} / {otm_ce['oi']:,}",
                    'volume':             f"{itm_ce['volume']:,} / {otm_ce['volume']:,}"
                })
        
        elif bias == 'Bearish':
            atm_pe = find_closest_strike(atm_strike, top_pe_strikes)
            if atm_pe:
                actual_strike = atm_pe['strike']
                strike_recommendations.append({
                    'strategy':           'Long Put (ATM)',
                    'action':             'BUY',
                    'strike':             actual_strike,
                    'type':               'PE',
                    'ltp':                atm_pe['ltp'],
                    'option_type':        'ATM',
                    'target_1':           actual_strike - 100,
                    'target_2':           actual_strike - 200,
                    'stop_loss':          atm_pe['ltp'] * 0.3,
                    'max_loss':           atm_pe['ltp'],
                    'profit_at_target_1': 100 - atm_pe['ltp'],
                    'profit_at_target_2': 200 - atm_pe['ltp'],
                    'oi':                 atm_pe['oi'],
                    'volume':             atm_pe['volume']
                })
            
            otm_target = atm_strike - 50
            otm_pe     = find_closest_strike(otm_target, top_pe_strikes)
            if otm_pe and otm_pe['strike'] != (atm_pe['strike'] if atm_pe else None):
                actual_strike = otm_pe['strike']
                strike_recommendations.append({
                    'strategy':           'Long Put (OTM)',
                    'action':             'BUY',
                    'strike':             actual_strike,
                    'type':               'PE',
                    'ltp':                otm_pe['ltp'],
                    'option_type':        'OTM',
                    'target_1':           actual_strike - 100,
                    'target_2':           actual_strike - 150,
                    'stop_loss':          otm_pe['ltp'] * 0.3,
                    'max_loss':           otm_pe['ltp'],
                    'profit_at_target_1': 100 - otm_pe['ltp'],
                    'profit_at_target_2': 150 - otm_pe['ltp'],
                    'oi':                 otm_pe['oi'],
                    'volume':             otm_pe['volume']
                })
            
            itm_target = atm_strike + 50
            itm_pe     = find_closest_strike(itm_target, top_pe_strikes)
            if itm_pe and otm_pe and len(strike_recommendations) >= 1:
                itm_k       = itm_pe['strike']
                otm_k       = otm_pe['strike']
                net_premium = itm_pe['ltp'] - otm_pe['ltp']
                max_profit  = (itm_k - otm_k) - net_premium
                strike_recommendations.append({
                    'strategy':           'Bear Put Spread',
                    'action':             f"BUY {itm_k} PE + SELL {otm_k} PE",
                    'strike':             f"{itm_k}/{otm_k}",
                    'type':               'Spread',
                    'ltp':                net_premium,
                    'option_type':        'ITM/OTM',
                    'target_1':           itm_k - 25,
                    'target_2':           otm_k,
                    'stop_loss':          net_premium * 0.4,
                    'max_loss':           net_premium,
                    'profit_at_target_1': 25 - net_premium,
                    'profit_at_target_2': max_profit,
                    'oi':                 f"{itm_pe['oi']:,} / {otm_pe['oi']:,}",
                    'volume':             f"{itm_pe['volume']:,} / {otm_pe['volume']:,}"
                })
        
        else:  # Neutral
            atm_ce = find_closest_strike(atm_strike, top_ce_strikes)
            atm_pe = find_closest_strike(atm_strike, top_pe_strikes)
            
            if atm_ce and atm_pe:
                actual_strike = atm_ce['strike']
                total_premium = atm_ce['ltp'] + atm_pe['ltp']
                strike_recommendations.append({
                    'strategy':           'Long Straddle (ATM)',
                    'action':             f"BUY {actual_strike} CE + BUY {actual_strike} PE",
                    'strike':             actual_strike,
                    'type':               'Straddle',
                    'ltp':                total_premium,
                    'option_type':        'ATM/ATM',
                    'target_1':           actual_strike + total_premium,
                    'target_2':           actual_strike - total_premium,
                    'stop_loss':          total_premium * 0.5,
                    'max_loss':           total_premium,
                    'profit_at_target_1': f"Profit if moves ±{total_premium:.0f} points",
                    'profit_at_target_2': 'Unlimited both sides',
                    'oi':                 f"{atm_ce['oi']:,} / {atm_pe['oi']:,}",
                    'volume':             f"{atm_ce['volume']:,} / {atm_pe['volume']:,}"
                })
        
        if strike_recommendations:
            self.logger.info(f"✅ Generated {len(strike_recommendations)} strike recommendations")
        else:
            self.logger.warning(
                f"⚠️ No strike recommendations generated. ATM={atm_strike}, "
                f"Available CE={[s['strike'] for s in top_ce_strikes[:3]]}, "
                f"Available PE={[s['strike'] for s in top_pe_strikes[:3]]}"
            )
        
        return strike_recommendations
    
    def find_nearest_levels(self, current_price, pivot_points):
        """Find nearest support and resistance from pivot points"""
        all_resistances = [pivot_points['r1'], pivot_points['r2'], pivot_points['r3']]
        all_supports    = [pivot_points['s1'], pivot_points['s2'], pivot_points['s3']]
        
        resistances_above = [r for r in all_resistances if r > current_price]
        nearest_resistance = min(resistances_above) if resistances_above else None
        
        supports_below = [s for s in all_supports if s < current_price]
        nearest_support = max(supports_below) if supports_below else None
        
        return {'nearest_resistance': nearest_resistance, 'nearest_support': nearest_support}
    
    # =========================================================================
    # HELPER: safely format a profit value for display (FIX for ValueError)
    # =========================================================================
    @staticmethod
    def _fmt_profit(value, multiplier=1):
        """Return a formatted profit string; handles both numeric and string values."""
        if isinstance(value, (int, float)):
            return f"₹{value * multiplier:.0f}"
        return str(value)

    @staticmethod
    def _fmt_profit_label(value, multiplier=1):
        """Return 'Profit: ₹X' or the raw string."""
        if isinstance(value, (int, float)):
            return f"Profit: ₹{value * multiplier:.0f}"
        return str(value)

    # =========================================================================
    # PIVOT POINTS WIDGET — Terminal Split Panel
    # =========================================================================
    def _build_pivot_widget(self, pivot_points, current_price, nearest_levels):
        """
        Build the Terminal Green Split Panel pivot widget HTML.
        Returns an HTML string to be embedded in the report.
        """
        pp  = pivot_points

        def dist(val):
            if val is None:
                return 'N/A'
            d = val - current_price
            return f"{d:+.2f}"

        def is_nearest_r(val):
            return val == nearest_levels.get('nearest_resistance')

        def is_nearest_s(val):
            return val == nearest_levels.get('nearest_support')

        # Determine zone text
        nr = nearest_levels.get('nearest_resistance')
        ns = nearest_levels.get('nearest_support')
        if nr and ns:
            zone_text  = f"Between {self._nearest_level_name(pp, ns)} and {self._nearest_level_name(pp, nr)}"
            above_dist = current_price - (pp.get('pivot', current_price))
            if above_dist >= 0:
                zone_detail = f"+{above_dist:.2f} above PP"
            else:
                zone_detail = f"{above_dist:.2f} below PP"
        elif nr:
            zone_text   = f"Below {self._nearest_level_name(pp, nr)}"
            zone_detail = f"Next R: &#8377;{nr}"
        elif ns:
            zone_text   = f"Above {self._nearest_level_name(pp, ns)}"
            zone_detail = f"Next S: &#8377;{ns}"
        else:
            zone_text   = "At Pivot Zone"
            zone_detail = f"PP: &#8377;{pp.get('pivot','N/A')}"

        # Gauge dot position: map current price between s1 and r1
        s1_val = pp.get('s1', current_price - 100)
        r1_val = pp.get('r1', current_price + 100)
        total_range = r1_val - s1_val
        if total_range > 0:
            dot_pct = ((current_price - s1_val) / total_range) * 100
            dot_pct = max(5, min(95, dot_pct))
        else:
            dot_pct = 50

        # Build resistance rows (R3 top → R1 bottom, dimming by distance)
        res_levels = [
            ('R3', pp.get('r3'), 'r3', False),
            ('R2', pp.get('r2'), 'r2', False),
            ('R1', pp.get('r1'), 'r1', True),
        ]
        sup_levels = [
            ('S1', pp.get('s1'), 's1', True),
            ('S2', pp.get('s2'), 's2', False),
            ('S3', pp.get('s3'), 's3', False),
        ]

        def res_row(lbl, val, cls, is_r1):
            near_tag = ' <span class="pv-near-tag">NEAREST R</span>' if is_nearest_r(val) else ''
            icon_html = '<span class="pv-icon pv-icon-r">&#9650;</span>' if is_r1 else '<span class="pv-icon-spacer"></span>'
            border = 'border-left:3px solid rgba(255,68,68,0.5);' if is_r1 else ''
            bg = 'background:rgba(255,68,68,0.025);' if is_r1 else ''
            return f'''
                <div class="pv-lv-row pv-{cls}" style="{border}{bg}">
                    <span class="pv-lbl">{lbl}</span>
                    <span class="pv-price">&#8377;{val}{near_tag}</span>
                    {icon_html}
                </div>'''

        def sup_row(lbl, val, cls, is_s1):
            near_tag = ' <span class="pv-near-tag pv-near-tag-s">NEAREST S</span>' if is_nearest_s(val) else ''
            icon_html = '<span class="pv-icon pv-icon-s">&#9660;</span>' if is_s1 else '<span class="pv-icon-spacer"></span>'
            border = 'border-right:3px solid rgba(0,255,65,0.5);' if is_s1 else ''
            bg = 'background:rgba(0,255,65,0.025);' if is_s1 else ''
            return f'''
                <div class="pv-lv-row pv-{cls}" style="{border}{bg}">
                    {icon_html}
                    <span class="pv-price" style="text-align:right">&#8377;{val}{near_tag}</span>
                    <span class="pv-lbl" style="text-align:right">{lbl}</span>
                </div>'''

        res_rows_html = ''.join([res_row(l, v, c, n) for l, v, c, n in res_levels])
        sup_rows_html = ''.join([sup_row(l, v, c, n) for l, v, c, n in sup_levels])

        widget_html = f'''
        <!-- ═══ PIVOT POINTS WIDGET — Terminal Split Panel ═══ -->
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

            .pv-widget {{
                background: #000a00;
                border: 1px solid #003300;
                border-radius: 10px;
                overflow: hidden;
                font-family: 'IBM Plex Mono', 'Courier New', monospace;
                box-shadow: 0 0 0 1px #001a00, 0 0 40px rgba(0,150,0,.07), 0 16px 50px rgba(0,0,0,.9);
                position: relative;
                width: 100%;
            }}
            .pv-widget::before {{
                content: '';
                position: absolute; inset: 0;
                background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,0,.006) 2px, rgba(0,255,0,.006) 4px);
                pointer-events: none; z-index: 20; border-radius: inherit;
            }}
            /* Header */
            .pv-hdr {{
                display: flex; align-items: center; justify-content: space-between;
                padding: 11px 18px; background: #000d00; border-bottom: 1px solid #002200;
            }}
            .pv-hdr-l {{ display: flex; align-items: center; gap: 10px; }}
            .pv-icon-wrap {{
                width: 32px; height: 32px; border-radius: 7px;
                background: #001800; border: 1px solid #009900;
                display: flex; align-items: center; justify-content: center;
                font-size: .9rem; box-shadow: 0 0 10px rgba(0,170,0,.22);
            }}
            .pv-title {{ font-size: .78rem; font-weight: 700; color: #00ff41; letter-spacing: 1px; }}
            .pv-sub   {{ font-size: .46rem; color: #004400; margin-top: 1px; letter-spacing: .5px; }}
            .pv-badge {{
                font-size: .52rem; padding: 3px 10px; border-radius: 20px;
                background: #001000; border: 1px solid #005500;
                color: #009900; letter-spacing: 1.5px; font-weight: 700;
            }}
            /* Gauge */
            .pv-gauge {{ padding: 10px 18px 8px; border-bottom: 1px solid #001800; background: #000c00; }}
            .pv-gauge-track {{
                height: 7px; border-radius: 20px; position: relative; overflow: visible;
                background: #001000; border: 1px solid #003300;
            }}
            .pv-gfl {{ position:absolute; left:0; top:0; bottom:0; width:{dot_pct:.1f}%; border-radius:20px 0 0 20px; background:linear-gradient(90deg,rgba(0,255,65,.25),transparent); }}
            .pv-gfr {{ position:absolute; right:0; top:0; bottom:0; width:{100-dot_pct:.1f}%; border-radius:0 20px 20px 0; background:linear-gradient(90deg,transparent,rgba(255,68,68,.18)); }}
            .pv-gdot {{
                position:absolute; left:{dot_pct:.1f}%; top:50%; transform:translate(-50%,-50%);
                width:14px; height:14px; background:#00ff41; border-radius:50%;
                border:2px solid #000; z-index:2;
                box-shadow: 0 0 0 2px #00ff41, 0 0 16px rgba(0,255,65,.65);
                animation: pv-pulse 2s ease-in-out infinite;
            }}
            @keyframes pv-pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.35}} }}
            .pv-glabels {{ display:flex; justify-content:space-between; margin-top:5px; font-size:.56rem; }}
            .pv-gl-s  {{ color:#00aa00; }}
            .pv-gl-ltp{{ color:#00ff41; font-weight:700; }}
            .pv-gl-r  {{ color:#ff4444; }}
            /* Zone */
            .pv-zone {{
                margin: 8px 14px;
                padding: 6px 12px;
                background: rgba(100,130,0,.1); border: 1px solid #334400; border-radius: 4px;
                display: flex; align-items: center; gap: 8px;
            }}
            .pv-zdot  {{ width:7px; height:7px; border-radius:50%; background:#ccff00; flex-shrink:0; box-shadow:0 0 7px #ccff00; animation:pv-pulse 2s ease-in-out infinite; }}
            .pv-ztxt  {{ font-size:.64rem; color:#ccff00; font-weight:700; }}
            .pv-zval  {{ margin-left:auto; font-size:.54rem; color:#556600; white-space:nowrap; }}
            /* Prev candle */
            .pv-candle {{ display:flex; margin:0 14px 8px; border:1px solid #002200; border-radius:4px; overflow:hidden; background:#000d00; }}
            .pv-ci {{ flex:1; padding:6px 10px; border-right:1px solid #001800; }}
            .pv-ci:last-child {{ border-right:none; }}
            .pv-ci-lbl {{ font-size:.46rem; color:#003300; letter-spacing:1.5px; margin-bottom:2px; text-transform:uppercase; }}
            .pv-ci-val {{ font-size:.76rem; font-weight:700; }}
            .pv-ci-h .pv-ci-val {{ color:#ff4444; }}
            .pv-ci-l .pv-ci-val {{ color:#00ff41; }}
            .pv-ci-c .pv-ci-val {{ color:#00cc33; }}
            /* Column headers */
            .pv-col-hdr {{
                display: grid; grid-template-columns: 1fr 130px 1fr;
                border-top: 1px solid #002200; border-bottom: 1px solid #002200;
                background: #000d00;
            }}
            .pv-hdr-res, .pv-hdr-sup {{
                display: grid; grid-template-columns: 60px 1fr 80px;
                padding: 6px 14px; gap: 6px;
            }}
            .pv-hdr-res {{ border-right: 1px solid #002200; }}
            .pv-hdr-sup {{ direction: rtl; }}
            .pv-hdr-res span, .pv-hdr-sup span, .pv-hdr-pp span {{
                font-size: .46rem; color: #003300; text-transform: uppercase; letter-spacing: 1.5px;
            }}
            .pv-hdr-pp {{
                display: flex; align-items: center; justify-content: center;
                border-left: 1px solid #002200; border-right: 1px solid #002200;
                padding: 6px 8px;
            }}
            /* Split panel */
            .pv-split {{ display: grid; grid-template-columns: 1fr 130px 1fr; }}
            .pv-col-res {{ display:flex; flex-direction:column; border-right:1px solid #002200; }}
            .pv-col-sup {{ display:flex; flex-direction:column; }}
            /* Level rows */
            .pv-lv-row {{
                display: grid; grid-template-columns: 55px 1fr 26px;
                align-items: center; height: 42px; padding: 0 14px; gap: 6px;
                border-bottom: 1px solid #000f00; transition: background .15s; cursor: default;
            }}
            .pv-lv-row:last-child {{ border-bottom: none; }}
            .pv-lv-row:hover {{ background: #001600 !important; }}
            /* Support rows reverse */
            .pv-col-sup .pv-lv-row {{ grid-template-columns: 26px 1fr 55px; }}
            .pv-lbl   {{ font-size: .66rem; font-weight: 700; letter-spacing: 1px; }}
            .pv-price {{ font-size: .72rem; font-weight: 600; display:flex; align-items:center; gap:5px; flex-wrap:wrap; }}
            .pv-icon  {{ width:22px; height:22px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:.5rem; font-weight:800; flex-shrink:0; }}
            .pv-icon-spacer {{ width:22px; flex-shrink:0; }}
            .pv-icon-r {{ background:rgba(255,68,68,.12); color:#ff4444; border:1px solid rgba(255,68,68,.4); }}
            .pv-icon-s {{ background:rgba(0,255,65,.12); color:#00ff41; border:1px solid rgba(0,255,65,.4); }}
            .pv-near-tag {{
                font-size:.4rem; padding:1px 5px; border-radius:8px; border:1px solid;
                font-weight:800; letter-spacing:.5px; white-space:nowrap;
                background:rgba(255,68,68,.1); color:#ff4444; border-color:rgba(255,68,68,.4);
            }}
            .pv-near-tag-s {{ background:rgba(0,255,65,.1); color:#00ff41; border-color:rgba(0,255,65,.4); }}
            /* Resistance row colours */
            .pv-r3 .pv-lbl, .pv-r3 .pv-price {{ color:rgba(255,68,68,.32); }}
            .pv-r2 .pv-lbl, .pv-r2 .pv-price {{ color:rgba(255,68,68,.58); }}
            .pv-r1 .pv-lbl, .pv-r1 .pv-price {{ color:#ff8080; }}
            .pv-r1 .pv-lbl {{ color:#ff4444; }}
            /* Support row colours */
            .pv-s1 .pv-lbl, .pv-s1 .pv-price {{ color:#80ff90; }}
            .pv-s1 .pv-lbl {{ color:#00ff41; }}
            .pv-s2 .pv-lbl, .pv-s2 .pv-price {{ color:rgba(0,255,65,.58); }}
            .pv-s3 .pv-lbl, .pv-s3 .pv-price {{ color:rgba(0,255,65,.32); }}
            /* Pivot center column */
            .pv-col-pp {{
                display: flex; flex-direction: column; align-items: center; justify-content: center;
                gap: 4px; padding: 12px 8px;
                border-left: 1px solid #002200; border-right: 1px solid #002200;
                background: rgba(100,130,0,.05);
            }}
            .pv-pp-tag   {{ font-size:.44rem; color:#556600; letter-spacing:2px; text-transform:uppercase; }}
            .pv-pp-val   {{ font-size:.95rem; font-weight:700; color:#ccff00; line-height:1; }}
            .pv-pp-dist  {{ font-size:.52rem; color:#668800; }}
            .pv-pp-line  {{ width:26px; height:1px; background:#334400; margin:3px 0; }}
            .pv-ltp-block {{
                display: flex; flex-direction: column; align-items: center; gap: 2px;
                background: rgba(0,60,0,.35); border: 1px solid #003300;
                border-radius: 4px; padding: 4px 12px; margin-top: 1px;
            }}
            .pv-ltp-tag {{ font-size:.4rem; color:#005500; letter-spacing:2px; text-transform:uppercase; }}
            .pv-ltp-val {{ font-size:.72rem; font-weight:700; color:#00ff41; }}
            .pv-ltp-dot {{ width:5px; height:5px; border-radius:50%; background:#00ff41; animation:pv-pulse 2s ease-in-out infinite; box-shadow:0 0 5px #00ff41; }}
            /* Footer */
            .pv-footer {{
                display: flex; justify-content: space-between; align-items: center;
                padding: 8px 18px; background: #000d00; border-top: 1px solid #002200;
            }}
            .pv-footer .pv-f-mthd {{ font-size:.48rem; color:#002800; letter-spacing:1.5px; }}
            .pv-footer .pv-f-ltp  {{ font-size:.62rem; font-weight:700; color:#00ff41; letter-spacing:.5px; }}
        </style>

        <div class="pv-widget">
            <!-- Header -->
            <div class="pv-hdr">
                <div class="pv-hdr-l">
                    <div class="pv-icon-wrap">&#128205;</div>
                    <div>
                        <div class="pv-title">Pivot Points</div>
                        <div class="pv-sub">Traditional Method &middot; Auto-calculated</div>
                    </div>
                </div>
                <div class="pv-badge">30 MIN</div>
            </div>

            <!-- Gauge -->
            <div class="pv-gauge">
                <div class="pv-gauge-track">
                    <div class="pv-gfl"></div>
                    <div class="pv-gfr"></div>
                    <div class="pv-gdot"></div>
                </div>
                <div class="pv-glabels">
                    <span class="pv-gl-s">S1 &#8377;{pp.get('s1','N/A')}</span>
                    <span class="pv-gl-ltp">&#9650; &#8377;{current_price} LTP</span>
                    <span class="pv-gl-r">R1 &#8377;{pp.get('r1','N/A')}</span>
                </div>
            </div>

            <!-- Zone -->
            <div class="pv-zone">
                <div class="pv-zdot"></div>
                <span class="pv-ztxt">{zone_text}</span>
                <span class="pv-zval">{zone_detail}</span>
            </div>

            <!-- Prev Candle -->
            <div class="pv-candle">
                <div class="pv-ci pv-ci-h">
                    <div class="pv-ci-lbl">&#9650; Prev High</div>
                    <div class="pv-ci-val">&#8377;{pp.get('prev_high','N/A')}</div>
                </div>
                <div class="pv-ci pv-ci-l">
                    <div class="pv-ci-lbl">&#9660; Prev Low</div>
                    <div class="pv-ci-val">&#8377;{pp.get('prev_low','N/A')}</div>
                </div>
                <div class="pv-ci pv-ci-c">
                    <div class="pv-ci-lbl">&#9679; Prev Close</div>
                    <div class="pv-ci-val">&#8377;{pp.get('prev_close','N/A')}</div>
                </div>
            </div>

            <!-- Column Headers -->
            <div class="pv-col-hdr">
                <div class="pv-hdr-res">
                    <span>Level</span><span>Price</span><span style="text-align:right">Dist</span>
                </div>
                <div class="pv-hdr-pp"><span>Pivot</span></div>
                <div class="pv-hdr-sup">
                    <span>Level</span><span>Price</span><span style="text-align:right">Dist</span>
                </div>
            </div>

            <!-- Split Panel -->
            <div class="pv-split">
                <!-- Resistance side -->
                <div class="pv-col-res">{res_rows_html}</div>

                <!-- Pivot center -->
                <div class="pv-col-pp">
                    <div class="pv-pp-tag">Pivot Point</div>
                    <div class="pv-pp-val">&#8377;{pp.get('pivot','N/A')}</div>
                    <div class="pv-pp-dist">{dist(pp.get('pivot'))} from LTP</div>
                    <div class="pv-pp-line"></div>
                    <div class="pv-ltp-block">
                        <div class="pv-ltp-tag">LTP</div>
                        <div class="pv-ltp-val">&#8377;{current_price}</div>
                        <div class="pv-ltp-dot"></div>
                    </div>
                </div>

                <!-- Support side -->
                <div class="pv-col-sup">{sup_rows_html}</div>
            </div>

            <!-- Footer -->
            <div class="pv-footer">
                <span class="pv-f-mthd">Traditional &middot; 30 Min Candle</span>
                <span class="pv-f-ltp">LTP &#8377;{current_price}</span>
            </div>
        </div>
        <!-- ═══ END PIVOT WIDGET ═══ -->
        '''
        return widget_html

    def _nearest_level_name(self, pivot_points, value):
        """Map a numeric pivot level back to its label string."""
        mapping = {
            pivot_points.get('r1'): 'R1', pivot_points.get('r2'): 'R2',
            pivot_points.get('r3'): 'R3', pivot_points.get('s1'): 'S1',
            pivot_points.get('s2'): 'S2', pivot_points.get('s3'): 'S3',
            pivot_points.get('pivot'): 'PP',
        }
        return mapping.get(value, str(value))

    def create_html_report(self, oc_analysis, tech_analysis, recommendation):
        """Create professional HTML report"""
        now_ist = self.format_ist_time()
        
        colors = self.config['report'].get('colors', {})
        rec    = recommendation['recommendation']
        
        if 'STRONG BUY' in rec:
            rec_color = colors.get('strong_buy', '#28a745')
        elif 'BUY' in rec:
            rec_color = colors.get('buy', '#5cb85c')
        elif 'STRONG SELL' in rec:
            rec_color = colors.get('strong_sell', '#dc3545')
        elif 'SELL' in rec:
            rec_color = colors.get('sell', '#f0ad4e')
        else:
            rec_color = colors.get('neutral', '#ffc107')
        
        title      = self.config['report'].get('title', 'NIFTY DAY TRADING ANALYSIS (1H)')
        strategies = self.get_options_strategies(recommendation, oc_analysis, tech_analysis)
        
        strike_recommendations = self.get_detailed_strike_recommendations(
            oc_analysis, tech_analysis, recommendation
        )
        
        pivot_points   = tech_analysis.get('pivot_points', {})
        current_price  = tech_analysis.get('current_price', 0)
        nearest_levels = self.find_nearest_levels(current_price, pivot_points)

        # Build the Terminal Split Panel pivot widget
        pivot_widget_html = self._build_pivot_widget(pivot_points, current_price, nearest_levels)
        
        momentum_1h_pct    = tech_analysis.get('price_change_pct_1h', 0)
        momentum_1h_signal = tech_analysis.get('momentum_1h_signal', 'Sideways')
        momentum_1h_colors = tech_analysis.get('momentum_1h_colors', {
            'bg': '#6c757d', 'bg_dark': '#5a6268', 'text': '#ffffff', 'border': '#495057'
        })
        
        momentum_5h_pct    = tech_analysis.get('momentum_5h_pct', 0)
        momentum_5h_signal = tech_analysis.get('momentum_5h_signal', 'Sideways')
        momentum_5h_colors = tech_analysis.get('momentum_5h_colors', {
            'bg': '#6c757d', 'bg_dark': '#5a6268', 'text': '#ffffff', 'border': '#495057'
        })
        
        # ---- Build CE rows ----
        top_ce_strikes = oc_analysis.get('top_ce_strikes', [])
        top_pe_strikes = oc_analysis.get('top_pe_strikes', [])
        
        ce_rows_html = ''
        for idx, strike in enumerate(top_ce_strikes, 1):
            badge_class = f"badge-{strike['type'].lower()}"
            ce_rows_html += f"""
                    <tr>
                        <td>{idx}</td>
                        <td><strong>&#8377;{strike['strike']}</strong></td>
                        <td><span class="{badge_class}">{strike['type']}</span></td>
                        <td>{strike['oi']:,}</td>
                        <td>{strike['chng_oi']:,}</td>
                        <td>&#8377;{strike['ltp']:.2f}</td>
                        <td>{strike['iv']:.2f}%</td>
                        <td>{strike['volume']:,}</td>
                    </tr>"""
        
        # ---- Build PE rows ----
        pe_rows_html = ''
        for idx, strike in enumerate(top_pe_strikes, 1):
            badge_class = f"badge-{strike['type'].lower()}"
            pe_rows_html += f"""
                    <tr>
                        <td>{idx}</td>
                        <td><strong>&#8377;{strike['strike']}</strong></td>
                        <td><span class="{badge_class}">{strike['type']}</span></td>
                        <td>{strike['oi']:,}</td>
                        <td>{strike['chng_oi']:,}</td>
                        <td>&#8377;{strike['ltp']:.2f}</td>
                        <td>{strike['iv']:.2f}%</td>
                        <td>{strike['volume']:,}</td>
                    </tr>"""
        
        # ---- Build strategies HTML ----
        strategies_html = ''
        for strategy in strategies:
            strategies_html += f"""
                <div class="strategy-card">
                    <div class="strategy-header">
                        <h4>{strategy['name']}</h4>
                        <span class="strategy-type">{strategy['type']}</span>
                    </div>
                    <div class="strategy-body">
                        <p><strong>Setup:</strong> {strategy['setup']}</p>
                        <p><strong>Profit Potential:</strong> {strategy['profit']}</p>
                        <p><strong>Risk:</strong> {strategy['risk']}</p>
                        <p><strong>Best When:</strong> {strategy['best_when']}</p>
                        <p class="recommendation-stars"><strong>Recommended:</strong> {strategy['recommended']}</p>
                    </div>
                </div>"""
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1e1e2e 0%, #27273f 100%);
            color: #e4e4e7; padding: 15px; line-height: 1.6;
        }}
        .container {{
            max-width: 1400px; margin: 0 auto; background: #1a1a2e;
            border-radius: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            padding: 30px; border: 1px solid #2d2d44;
        }}
        .header {{
            text-align: center; border-bottom: 3px solid #6366f1; padding-bottom: 20px;
            margin-bottom: 30px; background: linear-gradient(135deg, #2d2d44 0%, #1f1f35 100%);
            padding: 25px; border-radius: 12px;
        }}
        .header h1 {{ color: #818cf8; font-size: 28px; font-weight: 700; margin-bottom: 12px; }}
        .timestamp {{ color: #a1a1aa; font-size: 13px; font-weight: 600; margin-top: 10px; }}
        .timeframe-badge {{
            display: inline-block;
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white; padding: 6px 16px; border-radius: 20px;
            font-size: 12px; font-weight: 700; margin-top: 10px;
        }}
        .momentum-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px; }}
        .momentum-box {{
            background: linear-gradient(135deg, var(--momentum-bg) 0%, var(--momentum-bg-dark) 100%);
            color: var(--momentum-text); padding: 20px; border-radius: 12px; text-align: center;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4); border: 2px solid var(--momentum-border);
        }}
        .momentum-box h3 {{ font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; opacity: 0.9; }}
        .momentum-box .value {{ font-size: 36px; font-weight: 900; margin: 10px 0; }}
        .momentum-box .signal {{ font-size: 13px; font-weight: 600; opacity: 0.95; }}
        .recommendation-box {{
            background: linear-gradient(135deg, {rec_color} 0%, {rec_color}dd 100%);
            color: white; padding: 20px; border-radius: 12px; text-align: center;
            margin-bottom: 25px; box-shadow: 0 8px 24px rgba(0,0,0,0.4); border: 2px solid {rec_color};
        }}
        .recommendation-box h2 {{ font-size: 30px; font-weight: 800; margin-bottom: 8px; }}
        .recommendation-box .subtitle {{ font-size: 15px; opacity: 0.95; font-weight: 500; }}
        .signal-badge {{
            display: inline-block; padding: 6px 14px; border-radius: 20px;
            font-size: 12px; font-weight: 700; margin: 8px 4px 0 4px;
        }}
        .bullish {{ background: #22c55e; color: #fff; }}
        .bearish {{ background: #ef4444; color: #fff; }}
        .section {{ margin-bottom: 25px; }}
        .section-title {{
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: white; padding: 12px 20px; border-radius: 10px;
            font-size: 16px; font-weight: 700; margin-bottom: 15px;
        }}
        .data-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; }}
        .data-item {{
            background: linear-gradient(135deg, #2d2d44 0%, #252538 100%);
            padding: 16px; border-radius: 10px; border-left: 4px solid #6366f1;
        }}
        .data-item .label {{ color: #a1a1aa; font-size: 11px; text-transform: uppercase; font-weight: 700; margin-bottom: 6px; }}
        .data-item .value {{ color: #f4f4f5; font-size: 18px; font-weight: 700; }}
        .levels {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .levels-box {{
            flex: 1; min-width: 280px;
            background: linear-gradient(135deg, #2d2d44 0%, #252538 100%);
            padding: 16px; border-radius: 10px;
        }}
        .levels-box.resistance {{ border-left: 4px solid #ef4444; }}
        .levels-box.support {{ border-left: 4px solid #22c55e; }}
        .levels-box h4 {{ font-size: 14px; font-weight: 700; margin-bottom: 10px; color: #f4f4f5; }}
        .levels-box ul {{ list-style: none; padding: 0; }}
        .levels-box li {{ margin: 6px 0; font-size: 14px; color: #d4d4d8; padding-left: 20px; position: relative; }}
        .levels-box li:before {{ content: "▸"; position: absolute; left: 0; color: #6366f1; font-weight: bold; }}
        .oi-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px; }}
        .oi-section {{
            background: linear-gradient(135deg, #2d2d44 0%, #252538 100%);
            padding: 16px; border-radius: 10px;
        }}
        .oi-section h4 {{ font-size: 15px; font-weight: 700; text-align: center; margin-bottom: 12px; color: #f4f4f5; }}
        .oi-section.calls {{ border-top: 4px solid #22c55e; }}
        .oi-section.puts  {{ border-top: 4px solid #ef4444; }}
        .oi-container {{ overflow-x: auto; }}
        .oi-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        .oi-table th {{ background: #6366f1; color: white; padding: 8px 6px; text-align: center; font-weight: 700; font-size: 11px; white-space: nowrap; }}
        .oi-table td {{ padding: 8px 6px; border-bottom: 1px solid #3d3d54; text-align: center; color: #e4e4e7; }}
        .oi-table tbody tr:hover {{ background: #3d3d54; }}
        .badge-itm {{ background: #22c55e; color: white; padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; }}
        .badge-atm {{ background: #f59e0b; color: #000; padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; }}
        .badge-otm {{ background: #71717a; color: white; padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; }}
        .reasons {{
            background: linear-gradient(135deg, #422006 0%, #451a03 100%);
            border-left: 4px solid #f59e0b; padding: 16px; border-radius: 10px;
        }}
        .reasons strong {{ color: #fbbf24; font-size: 15px; }}
        .reasons ul {{ margin: 10px 0 0 0; padding-left: 25px; }}
        .reasons li {{ margin: 6px 0; color: #fde047; font-size: 13px; line-height: 1.6; }}
        .strike-recommendations {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-top: 15px; }}
        .strike-card {{
            background: linear-gradient(135deg, #2d2d44 0%, #252538 100%);
            border: 2px solid #3d3d54; border-radius: 12px; padding: 18px;
        }}
        .strike-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #6366f1; padding-bottom: 10px; margin-bottom: 15px; }}
        .strike-header h4 {{ color: #818cf8; font-size: 17px; font-weight: 700; }}
        .strike-badge {{ padding: 5px 12px; border-radius: 15px; font-size: 11px; font-weight: 700; }}
        .strike-badge.atm {{ background: #f59e0b; color: #000; }}
        .strike-badge.itm {{ background: #22c55e; color: white; }}
        .strike-badge.otm {{ background: #71717a; color: white; }}
        .strike-badge.itm-otm {{ background: linear-gradient(90deg, #22c55e 50%, #71717a 50%); color: white; }}
        .strike-badge.atm-atm {{ background: #ef4444; color: white; }}
        .strike-details {{ background: #3d3d54; padding: 12px; border-radius: 8px; margin-bottom: 15px; }}
        .strike-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px dashed #52525b; }}
        .strike-row:last-child {{ border-bottom: none; }}
        .strike-row .label {{ color: #a1a1aa; font-size: 12px; font-weight: 600; }}
        .strike-row .value {{ color: #f4f4f5; font-size: 13px; font-weight: 700; }}
        .strike-row .premium {{ color: #818cf8; font-size: 15px; font-weight: 800; }}
        .profit-targets {{ background: linear-gradient(135deg, #1e3a8a 0%, #1e293b 100%); padding: 15px; border-radius: 8px; margin-bottom: 12px; }}
        .profit-targets h5 {{ color: #818cf8; font-size: 14px; font-weight: 700; margin-bottom: 12px; }}
        .target-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
        .target-box {{ background: #2d2d44; padding: 10px; border-radius: 8px; text-align: center; border: 2px solid #3d3d54; }}
        .target-box.target-1 {{ border-color: #22c55e; }}
        .target-box.target-2 {{ border-color: #06b6d4; }}
        .target-box.stop-loss-box {{ border-color: #ef4444; }}
        .target-label {{ font-size: 10px; color: #a1a1aa; text-transform: uppercase; font-weight: 700; margin-bottom: 5px; }}
        .target-price {{ font-size: 15px; color: #f4f4f5; font-weight: 800; margin-bottom: 5px; }}
        .target-profit {{ font-size: 11px; color: #86efac; font-weight: 600; }}
        .target-box.stop-loss-box .target-profit {{ color: #fca5a5; }}
        .trade-example {{ background: #422006; border: 1px solid #f59e0b; border-radius: 8px; padding: 12px; font-size: 11px; line-height: 1.6; color: #fde047; }}
        .no-recommendations {{ background: #7f1d1d; border: 1px solid #ef4444; border-radius: 8px; padding: 25px; text-align: center; color: #fca5a5; font-size: 14px; }}
        .strategies-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; margin-top: 15px; }}
        .strategy-card {{ background: linear-gradient(135deg, #2d2d44 0%, #252538 100%); border: 2px solid #3d3d54; border-radius: 10px; padding: 16px; }}
        .strategy-header {{ border-bottom: 2px solid #6366f1; padding-bottom: 8px; margin-bottom: 10px; }}
        .strategy-header h4 {{ color: #818cf8; font-size: 15px; font-weight: 700; }}
        .strategy-type {{ display: inline-block; background: #1e3a8a; color: #93c5fd; padding: 3px 8px; border-radius: 10px; font-size: 10px; margin-top: 5px; font-weight: 600; }}
        .strategy-body p {{ margin: 8px 0; font-size: 13px; line-height: 1.5; color: #d4d4d8; }}
        .recommendation-stars {{ color: #fbbf24; font-size: 14px; font-weight: 700; }}
        .footer {{ text-align: center; margin-top: 30px; padding-top: 20px; border-top: 2px solid #3d3d54; color: #a1a1aa; font-size: 12px; line-height: 1.8; }}
        @media (max-width: 768px) {{
            .momentum-container {{ grid-template-columns: 1fr; }}
            .oi-grid {{ grid-template-columns: 1fr; }}
            .strike-recommendations {{ grid-template-columns: 1fr; }}
            .target-grid {{ grid-template-columns: 1fr; }}
            .levels {{ flex-direction: column; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>&#128202; {title}</h1>
        <div class="timeframe-badge">&#9201;&#65039; 1-HOUR TIMEFRAME</div>
        <div class="timestamp">Generated on: {now_ist}</div>
    </div>

    <!-- DUAL MOMENTUM -->
    <div class="momentum-container">
        <div class="momentum-box" style="--momentum-bg:{momentum_1h_colors['bg']};--momentum-bg-dark:{momentum_1h_colors['bg_dark']};--momentum-text:{momentum_1h_colors['text']};--momentum-border:{momentum_1h_colors['border']};">
            <h3>&#9889; 1H Momentum</h3>
            <div class="value">{momentum_1h_pct:+.2f}%</div>
            <div class="signal">{momentum_1h_signal}</div>
        </div>
        <div class="momentum-box" style="--momentum-bg:{momentum_5h_colors['bg']};--momentum-bg-dark:{momentum_5h_colors['bg_dark']};--momentum-text:{momentum_5h_colors['text']};--momentum-border:{momentum_5h_colors['border']};">
            <h3>&#128202; 5H Momentum</h3>
            <div class="value">{momentum_5h_pct:+.2f}%</div>
            <div class="signal">{momentum_5h_signal}</div>
        </div>
    </div>

    <!-- RECOMMENDATION -->
    <div class="recommendation-box">
        <h2>{recommendation['recommendation']}</h2>
        <div class="subtitle">Market Bias: {recommendation['bias']} | Confidence: {recommendation['confidence']}</div>
        <div style="margin-top:12px;">
            <span class="signal-badge bullish">Bullish: {recommendation['bullish_signals']}</span>
            <span class="signal-badge bearish">Bearish: {recommendation['bearish_signals']}</span>
        </div>
    </div>

    <!-- TECHNICAL ANALYSIS -->
    <div class="section">
        <div class="section-title">&#128200; Technical Analysis (1H)</div>
        <div class="data-grid">
            <div class="data-item"><div class="label">Current Price</div><div class="value">&#8377;{tech_analysis.get('current_price','N/A')}</div></div>
            <div class="data-item"><div class="label">RSI (14)</div><div class="value">{tech_analysis.get('rsi','N/A')}</div></div>
            <div class="data-item"><div class="label">EMA 20</div><div class="value">&#8377;{tech_analysis.get('ema20','N/A')}</div></div>
            <div class="data-item"><div class="label">EMA 50</div><div class="value">&#8377;{tech_analysis.get('ema50','N/A')}</div></div>
            <div class="data-item"><div class="label">Trend</div><div class="value">{tech_analysis.get('trend','N/A')}</div></div>
            <div class="data-item"><div class="label">RSI Signal</div><div class="value">{tech_analysis.get('rsi_signal','N/A')}</div></div>
        </div>
    </div>

    <!-- SUPPORT & RESISTANCE -->
    <div class="section">
        <div class="section-title">&#127919; Support &amp; Resistance (1H)</div>
        <div class="levels">
            <div class="levels-box resistance">
                <h4>&#128308; Resistance</h4>
                <ul>{''.join([f'<li>R{i+1}: &#8377;{r}</li>' for i, r in enumerate(tech_analysis.get('tech_resistances', []))])}</ul>
            </div>
            <div class="levels-box support">
                <h4>&#128994; Support</h4>
                <ul>{''.join([f'<li>S{i+1}: &#8377;{s}</li>' for i, s in enumerate(tech_analysis.get('tech_supports', []))])}</ul>
            </div>
        </div>
    </div>

    <!-- PIVOT POINTS — Terminal Split Panel Widget -->
    <div class="section">
        <div class="section-title">&#128205; Pivot Points (Traditional - 30 Min)</div>
        {pivot_widget_html}
    </div>

    <!-- OPTION CHAIN -->
    <div class="section">
        <div class="section-title">&#128202; Option Chain Analysis</div>
        <div class="data-grid">
            <div class="data-item"><div class="label">Put-Call Ratio</div><div class="value">{oc_analysis.get('pcr','N/A')}</div></div>
            <div class="data-item"><div class="label">Max Pain</div><div class="value">&#8377;{oc_analysis.get('max_pain','N/A')}</div></div>
            <div class="data-item"><div class="label">OI Sentiment</div><div class="value">{oc_analysis.get('oi_sentiment','N/A')}</div></div>
        </div>
        <div style="margin-top:20px;">
            <div class="levels">
                <div class="levels-box resistance">
                    <h4>&#128308; OI Resistance</h4>
                    <ul>{''.join([f'<li>&#8377;{r}</li>' for r in oc_analysis.get('resistances', [])])}</ul>
                </div>
                <div class="levels-box support">
                    <h4>&#128994; OI Support</h4>
                    <ul>{''.join([f'<li>&#8377;{s}</li>' for s in oc_analysis.get('supports', [])])}</ul>
                </div>
            </div>
        </div>
    </div>

    <!-- TOP OI -->
    <div class="section">
        <div class="section-title">&#128293; Top 10 Open Interest (5 CE + 5 PE)</div>
        <div class="oi-grid">
            <div class="oi-section calls">
                <h4>&#128222; Top 5 Call Options (CE)</h4>
                <div class="oi-container">
                    <table class="oi-table">
                        <thead><tr><th>#</th><th>Strike</th><th>Type</th><th>OI</th><th>Chng OI</th><th>LTP</th><th>IV</th><th>Volume</th></tr></thead>
                        <tbody>{ce_rows_html}</tbody>
                    </table>
                </div>
            </div>
            <div class="oi-section puts">
                <h4>&#128201; Top 5 Put Options (PE)</h4>
                <div class="oi-container">
                    <table class="oi-table">
                        <thead><tr><th>#</th><th>Strike</th><th>Type</th><th>OI</th><th>Chng OI</th><th>LTP</th><th>IV</th><th>Volume</th></tr></thead>
                        <tbody>{pe_rows_html}</tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- ANALYSIS SUMMARY -->
    <div class="section">
        <div class="section-title">&#128161; Analysis Summary</div>
        <div class="reasons">
            <strong>Key Factors:</strong>
            <ul>{''.join([f'<li>{reason}</li>' for reason in recommendation.get('reasons', [])])}</ul>
        </div>
    </div>

    <!-- STRIKE RECOMMENDATIONS -->
    <div class="section">
        <div class="section-title">&#128176; Detailed Strike Recommendations with Profit Targets</div>
        <p style="color:#a1a1aa;margin-bottom:15px;font-size:14px;line-height:1.6;">
            <strong style="color:#e4e4e7;">Based on {recommendation['bias']} bias with current Nifty at &#8377;{tech_analysis.get('current_price', 0):.2f}</strong><br>
            These are actionable trades with specific strike prices, LTP, and profit calculations.
        </p>
        <div class="strike-recommendations">"""
        
        # ---- Strike cards ----
        if strike_recommendations:
            for rec_item in strike_recommendations:
                ltp        = rec_item['ltp']
                max_loss   = rec_item['max_loss']
                stop_loss  = rec_item['stop_loss']
                target_1   = rec_item['target_1']
                target_2   = rec_item['target_2']
                p_at_t1    = rec_item['profit_at_target_1']
                p_at_t2    = rec_item['profit_at_target_2']

                p_at_t1_label  = self._fmt_profit_label(p_at_t1)
                p_at_t2_label  = self._fmt_profit_label(p_at_t2)
                example_t1     = self._fmt_profit(p_at_t1, 50)
                example_t2     = self._fmt_profit(p_at_t2, 50)

                if isinstance(p_at_t2, (int, float)):
                    profit_color = '#22c55e' if p_at_t2 > 100 else ('#f59e0b' if p_at_t2 > 50 else '#ef4444')
                else:
                    profit_color = '#6366f1'

                html += f"""
            <div class="strike-card" style="border-left:4px solid {profit_color};">
                <div class="strike-header">
                    <h4>{rec_item['strategy']}</h4>
                    <span class="strike-badge {rec_item['option_type'].lower().replace('/', '-')}">{rec_item['option_type']}</span>
                </div>
                <div class="strike-details">
                    <div class="strike-row"><span class="label">Action:</span><span class="value"><strong>{rec_item['action']}</strong></span></div>
                    <div class="strike-row"><span class="label">Strike Price:</span><span class="value"><strong>&#8377;{rec_item['strike']}</strong></span></div>
                    <div class="strike-row"><span class="label">Current LTP:</span><span class="value premium">&#8377;{ltp:.2f}</span></div>
                    <div class="strike-row"><span class="label">Open Interest:</span><span class="value">{rec_item['oi']}</span></div>
                    <div class="strike-row"><span class="label">Volume:</span><span class="value">{rec_item['volume']}</span></div>
                </div>
                <div class="profit-targets">
                    <h5>&#128202; Profit Targets &amp; Risk</h5>
                    <div class="target-grid">
                        <div class="target-box target-1">
                            <div class="target-label">Target 1</div>
                            <div class="target-price">&#8377;{target_1}</div>
                            <div class="target-profit">{p_at_t1_label}</div>
                        </div>
                        <div class="target-box target-2">
                            <div class="target-label">Target 2</div>
                            <div class="target-price">&#8377;{target_2}</div>
                            <div class="target-profit">{p_at_t2_label}</div>
                        </div>
                        <div class="target-box stop-loss-box">
                            <div class="target-label">Stop Loss</div>
                            <div class="target-price">&#8377;{stop_loss:.2f}</div>
                            <div class="target-profit">Max Loss: &#8377;{max_loss:.2f}</div>
                        </div>
                    </div>
                </div>
                <div class="trade-example">
                    <strong>Example:</strong> Buy 1 lot (50 qty) at LTP &#8377;{ltp:.2f} &rarr; Investment = &#8377;{ltp * 50:.0f}<br>
                    At Target 1: {example_t1} &nbsp;|&nbsp; At Target 2: {example_t2}
                </div>
            </div>"""
        else:
            html += """
            <div class="no-recommendations">
                <p><strong>&#9888;&#65039; No specific strike recommendations available at this time.</strong><br>Check the general strategies below.</p>
            </div>"""
        
        html += f"""
        </div>
    </div>

    <!-- OPTIONS STRATEGIES -->
    <div class="section">
        <div class="section-title">&#127919; Options Strategies</div>
        <p style="color:#a1a1aa;margin-bottom:15px;font-size:13px;">Based on {recommendation['bias']} bias:</p>
        <div class="strategies-grid">{strategies_html}</div>
    </div>

    <div class="footer">
        <p><strong>Disclaimer:</strong> This analysis is for educational purposes only. Trading involves risk. Past performance is not indicative of future results.</p>
        <p>&copy; 2025 Nifty Trading Analyzer | Dual Momentum Analysis (1H + 5H) | Professional Edition</p>
    </div>
</div>
</body>
</html>"""
        
        return html
    
    def send_email(self, html_content):
        """Send email with HTML report"""
        email_config   = self.config['email']
        recipient_email = email_config['recipient']
        sender_email    = email_config['sender']
        sender_password = email_config['app_password']
        subject_prefix  = email_config.get('subject_prefix', 'Nifty 1H Analysis')
        
        ist_time     = self.get_ist_time()
        subject_time = ist_time.strftime('%Y-%m-%d %H:%M IST')
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"{subject_prefix} - {subject_time}"
            msg['From']    = sender_email
            msg['To']      = recipient_email
            
            msg.attach(MIMEText(html_content, 'html'))
            
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
        """Run complete analysis with DUAL MOMENTUM DETECTION"""
        self.logger.info("🚀 Starting Nifty 1-HOUR Analysis with Dual Momentum...")
        self.logger.info("=" * 60)
        
        oc_df, spot_price = self.fetch_option_chain()
        
        if oc_df is not None and spot_price is not None:
            oc_analysis = self.analyze_option_chain(oc_df, spot_price)
        else:
            oc_analysis = self.get_sample_oc_analysis()
        
        tech_df = self.fetch_technical_data()
        
        if tech_df is not None and not tech_df.empty:
            tech_analysis = self.technical_analysis(tech_df)
        else:
            tech_analysis = self.get_sample_tech_analysis()
        
        self.logger.info("🎯 Generating Trading Recommendation with Dual Momentum...")
        recommendation = self.generate_recommendation(oc_analysis, tech_analysis)
        
        self.logger.info("=" * 60)
        self.logger.info(f"📊 RECOMMENDATION: {recommendation['recommendation']}")
        self.logger.info(f"📈 Bias: {recommendation['bias']} | Confidence: {recommendation['confidence']}")
        self.logger.info(f"🎯 RSI (1H): {tech_analysis.get('rsi', 'N/A')}")
        self.logger.info(f"⚡ 1H Momentum: {tech_analysis.get('price_change_pct_1h', 0):+.2f}% - {tech_analysis.get('momentum_1h_signal')}")
        self.logger.info(f"📊 5H Momentum: {tech_analysis.get('momentum_5h_pct', 0):+.2f}% - {tech_analysis.get('momentum_5h_signal')}")
        self.logger.info(f"📍 Pivot Point: ₹{tech_analysis.get('pivot_points', {}).get('pivot', 'N/A')}")
        self.logger.info("=" * 60)
        
        html_report = self.create_html_report(oc_analysis, tech_analysis, recommendation)
        
        if self.config['report']['save_local']:
            report_dir = self.config['report']['local_dir']
            os.makedirs(report_dir, exist_ok=True)
            ist_time        = self.get_ist_time()
            filename_format = self.config['report']['filename_format']
            report_filename = os.path.join(report_dir, ist_time.strftime(filename_format))
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(html_report)
            self.logger.info(f"💾 Report saved as: {report_filename}")
        
        self.logger.info(f"📧 Sending email to {self.config['email']['recipient']}...")
        self.send_email(html_report)
        
        self.logger.info("✅ Dual Momentum Analysis Complete!")
        
        return {
            'oc_analysis':   oc_analysis,
            'tech_analysis': tech_analysis,
            'recommendation': recommendation,
            'html_report':   html_report
        }


if __name__ == "__main__":
    analyzer = NiftyAnalyzer(config_path='config.yml')
    result   = analyzer.run_analysis()
    
    print(f"\n✅ Analysis Complete!")
    print(f"Recommendation: {result['recommendation']['recommendation']}")
    print(f"RSI (1H):       {result['tech_analysis']['rsi']}")
    print(f"1H Momentum:    {result['tech_analysis']['price_change_pct_1h']:+.2f}% - {result['tech_analysis']['momentum_1h_signal']}")
    print(f"5H Momentum:    {result['tech_analysis']['momentum_5h_pct']:+.2f}% - {result['tech_analysis']['momentum_5h_signal']}")
    print(f"Check your email for the detailed report!")
