"""
Nifty Option Chain & Technical Analysis for Day Trading
PROFESSIONAL VERSION - Enhanced Design with Better Contrast
1-HOUR TIMEFRAME with WILDER'S RSI (matches TradingView)
Enhanced with Pivot Points + Dual Momentum Analysis + Top 10 OI Display + OI CHANGE ANALYSIS
EXPIRY: Weekly TUESDAY expiry with 3:30 PM IST cutoff logic
FIXED: Using curl-cffi for NSE API to bypass anti-scraping
UPDATED: Professional grey theme with improved readability
NEW: OI Change Analysis using NSE API changeinOpenInterest field
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
        # Using correct v3 API endpoint
        self.option_chain_base_url = "https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol=NIFTY&expiry="
        
        # Headers that work with NSE (from working script)
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
            # Formula: (1 - current_day) % 7 where 1 is Tuesday
            # This gives: Wed(2)→6 days, Thu(3)→5 days, Fri(4)→4 days, Sat(5)→3 days, Sun(6)→2 days
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
                'use_pivot_points': True,
                'display_top_oi_strikes': 10
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        
        handlers = []
        
        # File handler
        if log_config.get('log_to_file', True):
            log_file = log_config.get('log_file', './logs/nifty_analyzer.log')
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            handlers.append(logging.FileHandler(log_file))
        
        # Console handler
        handlers.append(logging.StreamHandler())
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=handlers,
            force=True
        )
        
        self.logger = logging.getLogger(__name__)
    
    def fetch_option_chain(self):
        """Fetch option chain data from NSE"""
        expiry_date = self.get_next_expiry_date()
        
        base_url = "https://www.nseindia.com"
        api_url = self.option_chain_base_url + expiry_date
        
        max_retries = self.config['data_source']['max_retries']
        retry_delay = self.config['data_source']['retry_delay']
        timeout = self.config['data_source']['timeout']
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Fetching option chain data for expiry {expiry_date} (attempt {attempt + 1}/{max_retries})...")
                
                # Create session with curl-cffi
                session = requests.Session()
                
                # First visit the main page to get cookies (impersonate Chrome)
                session.get(base_url, headers=self.headers, impersonate="chrome", timeout=15)
                
                # Small delay to mimic human behavior
                time.sleep(1)
                
                # Now fetch the option chain data (impersonate Chrome)
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
    
    def analyze_oi_change(self, oc_df):
        """
        Analyze OI changes from NSE API's changeinOpenInterest field
        Returns market direction based on Call and Put OI changes
        """
        if oc_df is None or oc_df.empty:
            return {
                'total_call_oi_change': 0,
                'total_put_oi_change': 0,
                'direction': 'No Data',
                'signal': 'Unable to fetch option chain data',
                'confidence': 'N/A'
            }
        
        try:
            # Sum up all Call and Put OI changes
            total_call_chng_oi = oc_df['Call_Chng_OI'].sum()
            total_put_chng_oi = oc_df['Put_Chng_OI'].sum()
            
            # Calculate net change
            net_oi_change = total_call_chng_oi + total_put_chng_oi
            
            # Determine direction based on OI changes
            # Professional interpretation:
            # Call OI ↑ + Put OI ↓ = Bullish (Long buildup)
            # Call OI ↓ + Put OI ↑ = Bearish (Short covering/Put buildup)
            # Both ↑ = Volatile/Sideways (High activity)
            # Both ↓ = Unwinding (Exit positions)
            
            threshold_strong = 500000  # Strong signal threshold
            threshold_moderate = 200000  # Moderate signal threshold
            
            if total_call_chng_oi > threshold_strong and total_put_chng_oi < -threshold_moderate:
                direction = "Strong Bullish"
                signal = "Heavy Call writing + Put unwinding - Bulls adding positions"
                confidence = "Very High"
            elif total_call_chng_oi > threshold_moderate and total_put_chng_oi < 0:
                direction = "Bullish"
                signal = "Call buildup with Put reduction - Bullish sentiment"
                confidence = "High"
            elif total_call_chng_oi < -threshold_strong and total_put_chng_oi > threshold_moderate:
                direction = "Strong Bearish"
                signal = "Heavy Put writing + Call unwinding - Bears adding positions"
                confidence = "Very High"
            elif total_call_chng_oi < 0 and total_put_chng_oi > threshold_moderate:
                direction = "Bearish"
                signal = "Put buildup with Call reduction - Bearish sentiment"
                confidence = "High"
            elif total_call_chng_oi > threshold_moderate and total_put_chng_oi > threshold_moderate:
                direction = "Neutral - High Volatility Expected"
                signal = "Both Call & Put OI increasing - Big move expected"
                confidence = "Medium"
            elif total_call_chng_oi < -threshold_moderate and total_put_chng_oi < -threshold_moderate:
                direction = "Neutral - Unwinding"
                signal = "Both Call & Put OI decreasing - Position squaring/Profit booking"
                confidence = "Low"
            elif abs(total_call_chng_oi) < threshold_moderate and abs(total_put_chng_oi) < threshold_moderate:
                direction = "Neutral"
                signal = "Minimal OI changes - Low conviction/Consolidation"
                confidence = "Low"
            else:
                # Determine based on which is stronger
                if abs(total_call_chng_oi) > abs(total_put_chng_oi):
                    if total_call_chng_oi > 0:
                        direction = "Moderately Bullish"
                        signal = "Call OI buildup dominant"
                        confidence = "Medium"
                    else:
                        direction = "Moderately Bearish"
                        signal = "Call OI unwinding dominant"
                        confidence = "Medium"
                else:
                    if total_put_chng_oi > 0:
                        direction = "Moderately Bearish"
                        signal = "Put OI buildup dominant"
                        confidence = "Medium"
                    else:
                        direction = "Moderately Bullish"
                        signal = "Put OI unwinding dominant"
                        confidence = "Medium"
            
            self.logger.info(f"📊 OI Change Analysis:")
            self.logger.info(f"   Call OI Change: {total_call_chng_oi:,.0f} | Put OI Change: {total_put_chng_oi:,.0f}")
            self.logger.info(f"   Direction: {direction} ({confidence} confidence)")
            self.logger.info(f"   Signal: {signal}")
            
            return {
                'total_call_oi_change': int(total_call_chng_oi),
                'total_put_oi_change': int(total_put_chng_oi),
                'direction': direction,
                'signal': signal,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing OI change: {e}")
            return {
                'total_call_oi_change': 0,
                'total_put_oi_change': 0,
                'direction': 'Error',
                'signal': str(e),
                'confidence': 'N/A'
            }
    
    def analyze_option_chain(self, oc_df, spot_price):
        """Analyze option chain for trading signals"""
        if oc_df is None or oc_df.empty:
            return self.get_sample_oc_analysis()
        
        try:
            # Calculate PCR
            total_put_oi = oc_df['Put_OI'].sum()
            total_call_oi = oc_df['Call_OI'].sum()
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            
            # Determine PCR signal
            pcr_very_bullish_threshold = self.config['option_chain']['pcr_very_bullish']
            pcr_bullish_threshold = self.config['option_chain']['pcr_bullish']
            pcr_bearish_threshold = self.config['option_chain']['pcr_bearish']
            pcr_very_bearish_threshold = self.config['option_chain']['pcr_very_bearish']
            
            if pcr >= pcr_very_bullish_threshold:
                oi_sentiment = "Very Bullish"
            elif pcr >= pcr_bullish_threshold:
                oi_sentiment = "Bullish"
            elif pcr <= pcr_very_bearish_threshold:
                oi_sentiment = "Very Bearish"
            elif pcr <= pcr_bearish_threshold:
                oi_sentiment = "Bearish"
            else:
                oi_sentiment = "Neutral"
            
            # Calculate max pain (strike with highest total OI)
            oc_df['Total_OI'] = oc_df['Call_OI'] + oc_df['Put_OI']
            max_pain_strike = oc_df.loc[oc_df['Total_OI'].idxmax(), 'Strike']
            
            # Find support and resistance based on Put and Call OI
            atm_strike = round(spot_price / 50) * 50
            strike_range = self.config['option_chain']['strike_range']
            
            nearby_strikes = oc_df[
                (oc_df['Strike'] >= atm_strike - strike_range) & 
                (oc_df['Strike'] <= atm_strike + strike_range)
            ]
            
            # Resistance = strikes with high Call OI above spot
            resistance_strikes = nearby_strikes[nearby_strikes['Strike'] > spot_price].nlargest(2, 'Call_OI')
            resistances = resistance_strikes['Strike'].tolist()
            
            # Support = strikes with high Put OI below spot
            support_strikes = nearby_strikes[nearby_strikes['Strike'] < spot_price].nlargest(2, 'Put_OI')
            supports = support_strikes['Strike'].tolist()
            
            self.logger.info(f"📊 PCR: {pcr:.2f} | Sentiment: {oi_sentiment}")
            self.logger.info(f"📍 Max Pain: ₹{max_pain_strike}")
            
            return {
                'pcr': round(pcr, 2),
                'oi_sentiment': oi_sentiment,
                'max_pain': max_pain_strike,
                'resistances': resistances,
                'supports': supports
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in option chain analysis: {e}")
            return self.get_sample_oc_analysis()
    
    def get_sample_oc_analysis(self):
        """Return sample OC analysis when data fetch fails"""
        return {
            'pcr': 1.15,
            'oi_sentiment': 'Bullish',
            'max_pain': 25800,
            'resistances': [25900, 26000],
            'supports': [25700, 25600]
        }
    
    def fetch_technical_data(self):
        """Fetch technical data from yfinance"""
        try:
            timeframe = self.config['technical']['timeframe']
            period = self.config['technical']['period']
            
            self.logger.info(f"📊 Fetching {timeframe} technical data for period {period}...")
            
            ticker = yf.Ticker(self.nifty_symbol)
            df = ticker.history(period=period, interval=timeframe)
            
            if df.empty:
                self.logger.warning("No technical data retrieved from yfinance")
                return None
            
            self.logger.info(f"✅ Technical data fetched: {len(df)} candles")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching technical data: {e}")
            return None
    
    def calculate_traditional_pivot_points(self, df):
        """
        Calculate Traditional Pivot Points (30-minute timeframe)
        Uses yesterday's High, Low, Close from 30-min data
        """
        try:
            if df is None or len(df) < 2:
                return None
            
            # Get previous day's data (last full day)
            yesterday_high = df['High'].iloc[-2]
            yesterday_low = df['Low'].iloc[-2]
            yesterday_close = df['Close'].iloc[-2]
            
            # Calculate Traditional Pivot Points
            pivot = (yesterday_high + yesterday_low + yesterday_close) / 3
            
            r1 = (2 * pivot) - yesterday_low
            s1 = (2 * pivot) - yesterday_high
            
            r2 = pivot + (yesterday_high - yesterday_low)
            s2 = pivot - (yesterday_high - yesterday_low)
            
            r3 = yesterday_high + 2 * (pivot - yesterday_low)
            s3 = yesterday_low - 2 * (yesterday_high - pivot)
            
            self.logger.info(f"📍 Pivot Points (30m) calculated | PP: ₹{pivot:.2f}")
            
            return {
                'pivot': round(pivot, 2),
                'r1': round(r1, 2),
                'r2': round(r2, 2),
                'r3': round(r3, 2),
                's1': round(s1, 2),
                's2': round(s2, 2),
                's3': round(s3, 2)
            }
        except Exception as e:
            self.logger.error(f"❌ Error calculating pivot points: {e}")
            return None
    
    def technical_analysis(self, df):
        """Perform technical analysis on 1H data"""
        if df is None or df.empty:
            return self.get_sample_tech_analysis()
        
        try:
            # Current price
            current_price = df['Close'].iloc[-1]
            
            # RSI (14-period)
            rsi_period = self.config['technical']['rsi_period']
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = round(rsi.iloc[-1], 2)
            
            # RSI Signal
            rsi_overbought = self.config['technical']['rsi_overbought']
            rsi_oversold = self.config['technical']['rsi_oversold']
            
            if rsi_value > rsi_overbought:
                rsi_signal = "Overbought"
            elif rsi_value < rsi_oversold:
                rsi_signal = "Oversold"
            else:
                rsi_signal = "Neutral"
            
            # EMA
            ema_short_period = self.config['technical']['ema_short']
            ema_long_period = self.config['technical']['ema_long']
            
            ema20 = df['Close'].ewm(span=ema_short_period, adjust=False).mean().iloc[-1]
            ema50 = df['Close'].ewm(span=ema_long_period, adjust=False).mean().iloc[-1]
            
            # Trend based on EMA
            if current_price > ema20 > ema50:
                trend = "Bullish"
            elif current_price < ema20 < ema50:
                trend = "Bearish"
            else:
                trend = "Sideways"
            
            # Support & Resistance from recent price action
            recent_candles = df.tail(20)
            tech_resistances = sorted(recent_candles['High'].nlargest(2).tolist(), reverse=True)
            tech_supports = sorted(recent_candles['Low'].nsmallest(2).tolist())
            
            # Pivot Points (Traditional - 30 Min)
            pivot_points = self.calculate_traditional_pivot_points(df)
            
            # DUAL MOMENTUM CALCULATION
            # 1-HOUR MOMENTUM
            if len(df) > 1:
                price_1h_ago = df['Close'].iloc[-2]
                price_change_1h = current_price - price_1h_ago
                price_change_pct_1h = (price_change_1h / price_1h_ago * 100)
            else:
                price_change_1h = 0
                price_change_pct_1h = 0
            
            momentum_1h_signal, momentum_1h_bias, momentum_1h_colors = self.get_momentum_signal(price_change_pct_1h)
            
            # 5-HOUR MOMENTUM
            if len(df) >= 5:
                price_5h_ago = df['Close'].iloc[-5]
                momentum_5h = current_price - price_5h_ago
                momentum_5h_pct = (momentum_5h / price_5h_ago * 100)
            else:
                momentum_5h = 0
                momentum_5h_pct = 0
            
            momentum_5h_signal, momentum_5h_bias, momentum_5h_colors = self.get_momentum_signal(momentum_5h_pct)
            
            self.logger.info(f"📊 Tech Analysis | Price: ₹{current_price:.2f} | RSI: {rsi_value} | Trend: {trend}")
            self.logger.info(f"⚡ 1H Momentum: {price_change_pct_1h:+.2f}% ({momentum_1h_signal})")
            self.logger.info(f"📊 5H Momentum: {momentum_5h_pct:+.2f}% ({momentum_5h_signal})")
            
            return {
                'current_price': round(current_price, 2),
                'rsi': rsi_value,
                'rsi_signal': rsi_signal,
                'ema20': round(ema20, 2),
                'ema50': round(ema50, 2),
                'trend': trend,
                'tech_resistances': [round(r, 2) for r in tech_resistances],
                'tech_supports': [round(s, 2) for s in tech_supports],
                'pivot_points': pivot_points if pivot_points else {},
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
            
        except Exception as e:
            self.logger.error(f"❌ Error in technical analysis: {e}")
            return self.get_sample_tech_analysis()
    
    def get_momentum_signal(self, momentum_pct):
        """Determine momentum signal and colors"""
        strong_threshold = self.config['technical']['momentum_threshold_strong']
        moderate_threshold = self.config['technical']['momentum_threshold_moderate']
        
        if momentum_pct >= strong_threshold:
            return 'Strong Upward', 'Bullish', {
                'bg': '#1e7e34', 'bg_dark': '#155724', 'text': '#ffffff', 'border': '#28a745'
            }
        elif momentum_pct >= moderate_threshold:
            return 'Moderate Upward', 'Bullish', {
                'bg': '#28a745', 'bg_dark': '#1e7e34', 'text': '#ffffff', 'border': '#28a745'
            }
        elif momentum_pct <= -strong_threshold:
            return 'Strong Downward', 'Bearish', {
                'bg': '#bd2130', 'bg_dark': '#721c24', 'text': '#ffffff', 'border': '#dc3545'
            }
        elif momentum_pct <= -moderate_threshold:
            return 'Moderate Downward', 'Bearish', {
                'bg': '#dc3545', 'bg_dark': '#bd2130', 'text': '#ffffff', 'border': '#dc3545'
            }
        else:
            return 'Sideways/Weak', 'Neutral', {
                'bg': '#6c757d', 'bg_dark': '#5a6268', 'text': '#ffffff', 'border': '#495057'
            }
    
    def get_sample_tech_analysis(self):
        """Return sample technical analysis"""
        return {
            'current_price': 25800,
            'rsi': 58.5,
            'rsi_signal': 'Neutral',
            'ema20': 25750,
            'ema50': 25680,
            'trend': 'Bullish',
            'tech_resistances': [25900, 26000],
            'tech_supports': [25700, 25600],
            'pivot_points': {
                'pivot': 25750,
                'r1': 25820,
                'r2': 25890,
                'r3': 25960,
                's1': 25680,
                's2': 25610,
                's3': 25540
            },
            'price_change_1h': 45.50,
            'price_change_pct_1h': 0.18,
            'momentum_1h_signal': 'Sideways/Weak',
            'momentum_1h_bias': 'Neutral',
            'momentum_1h_colors': {'bg': '#6c757d', 'bg_dark': '#5a6268', 'text': '#ffffff', 'border': '#495057'},
            'momentum_5h': 125.50,
            'momentum_5h_pct': 0.49,
            'momentum_5h_signal': 'Moderate Upward',
            'momentum_5h_bias': 'Bullish',
            'momentum_5h_colors': {'bg': '#28a745', 'bg_dark': '#1e7e34', 'text': '#ffffff', 'border': '#28a745'}
        }
    
    def generate_recommendation(self, oc_analysis, tech_analysis, oi_change_analysis):
        """Generate trading recommendation with OI Change Analysis"""
        config = self.config['recommendation']
        tech_config = self.config['technical']
        
        strong_threshold = tech_config['momentum_threshold_strong']
        moderate_threshold = tech_config['momentum_threshold_moderate']
        
        bullish_signals = 0
        bearish_signals = 0
        reasons = []
        
        # RSI Signals
        rsi = tech_analysis.get('rsi', 50)
        rsi_oversold = tech_config['rsi_oversold']
        rsi_overbought = tech_config['rsi_overbought']
        
        if rsi < rsi_oversold:
            bullish_signals += 2
            reasons.append(f"RSI oversold at {rsi}")
        elif rsi > rsi_overbought:
            bearish_signals += 2
            reasons.append(f"RSI overbought at {rsi}")
        
        # PCR Signals
        pcr = oc_analysis.get('pcr', 1.0)
        oi_sentiment = oc_analysis.get('oi_sentiment', 'Neutral')
        
        if oi_sentiment == 'Very Bullish':
            bullish_signals += 2
            reasons.append(f"PCR at {pcr} indicates strong bullish sentiment")
        elif oi_sentiment == 'Bullish':
            bullish_signals += 1
            reasons.append(f"PCR at {pcr} shows bullish bias")
        elif oi_sentiment == 'Very Bearish':
            bearish_signals += 2
            reasons.append(f"PCR at {pcr} indicates strong bearish sentiment")
        elif oi_sentiment == 'Bearish':
            bearish_signals += 1
            reasons.append(f"PCR at {pcr} shows bearish bias")
        
        # OI CHANGE SIGNALS (NEW - HIGH PRIORITY)
        oi_direction = oi_change_analysis.get('direction', 'Neutral')
        oi_confidence = oi_change_analysis.get('confidence', 'Low')
        
        if 'Strong Bullish' in oi_direction:
            bullish_signals += 3
            reasons.append(f"🔥 OI Change: {oi_direction} ({oi_confidence} confidence)")
        elif 'Bullish' in oi_direction or 'Moderately Bullish' in oi_direction:
            bullish_signals += 2
            reasons.append(f"📊 OI Change: {oi_direction} ({oi_confidence} confidence)")
        elif 'Strong Bearish' in oi_direction:
            bearish_signals += 3
            reasons.append(f"🔥 OI Change: {oi_direction} ({oi_confidence} confidence)")
        elif 'Bearish' in oi_direction or 'Moderately Bearish' in oi_direction:
            bearish_signals += 2
            reasons.append(f"📊 OI Change: {oi_direction} ({oi_confidence} confidence)")
        
        # Momentum Signals
        momentum_1h_pct = tech_analysis.get('price_change_pct_1h', 0)
        weight_1h = config.get('momentum_1h_weight', 1)
        
        if momentum_1h_pct > strong_threshold:
            bullish_signals += weight_1h
            reasons.append(f"1H strong upward momentum: {momentum_1h_pct:+.2f}%")
        elif momentum_1h_pct > moderate_threshold:
            bullish_signals += 1
            reasons.append(f"1H positive momentum: {momentum_1h_pct:+.2f}%")
        elif momentum_1h_pct < -strong_threshold:
            bearish_signals += weight_1h
            reasons.append(f"1H strong downward momentum: {momentum_1h_pct:+.2f}%")
        elif momentum_1h_pct < -moderate_threshold:
            bearish_signals += 1
            reasons.append(f"1H negative momentum: {momentum_1h_pct:+.2f}%")
        
        momentum_5h_pct = tech_analysis.get('momentum_5h_pct', 0)
        weight_5h = config.get('momentum_5h_weight', 2)
        
        if momentum_5h_pct > strong_threshold:
            bullish_signals += weight_5h
            reasons.append(f"5H strong upward trend: {momentum_5h_pct:+.2f}%")
        elif momentum_5h_pct > moderate_threshold:
            bullish_signals += 1
            reasons.append(f"5H positive trend: {momentum_5h_pct:+.2f}%")
        elif momentum_5h_pct < -strong_threshold:
            bearish_signals += weight_5h
            reasons.append(f"5H strong downward trend: {momentum_5h_pct:+.2f}%")
        elif momentum_5h_pct < -moderate_threshold:
            bearish_signals += 1
            reasons.append(f"5H negative trend: {momentum_5h_pct:+.2f}%")
        
        # EMA Trend Signal
        trend = tech_analysis.get('trend', 'Sideways')
        if trend == 'Bullish':
            bullish_signals += 1
            reasons.append("Price above both EMAs (bullish trend)")
        elif trend == 'Bearish':
            bearish_signals += 1
            reasons.append("Price below both EMAs (bearish trend)")
        
        # Final recommendation
        net_score = bullish_signals - bearish_signals
        
        strong_buy_threshold = config['strong_buy_threshold']
        buy_threshold = config['buy_threshold']
        sell_threshold = config['sell_threshold']
        strong_sell_threshold = config['strong_sell_threshold']
        
        if net_score >= strong_buy_threshold:
            recommendation = 'STRONG BUY'
            bias = 'Bullish'
            confidence = 'High'
        elif net_score >= buy_threshold:
            recommendation = 'BUY'
            bias = 'Bullish'
            confidence = 'Medium'
        elif net_score <= strong_sell_threshold:
            recommendation = 'STRONG SELL'
            bias = 'Bearish'
            confidence = 'High'
        elif net_score <= sell_threshold:
            recommendation = 'SELL'
            bias = 'Bearish'
            confidence = 'Medium'
        else:
            recommendation = 'HOLD / NEUTRAL'
            bias = 'Neutral'
            confidence = 'Low'
        
        self.logger.info(f"🎯 Recommendation: {recommendation} | Bias: {bias} | Confidence: {confidence}")
        self.logger.info(f"📊 Signals - Bullish: {bullish_signals} | Bearish: {bearish_signals} | Net: {net_score:+d}")
        
        return {
            'recommendation': recommendation,
            'bias': bias,
            'confidence': confidence,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'net_score': net_score,
            'reasons': reasons
        }
    
    def create_html_report(self, oc_analysis, tech_analysis, recommendation, top_strikes, oi_change_analysis):
        """Create HTML report"""
        
        title = self.config['report']['title']
        now_ist = self.format_ist_time()
        
        # Get momentum colors
        momentum_1h_colors = tech_analysis.get('momentum_1h_colors', {
            'bg': '#6c757d', 'bg_dark': '#5a6268', 'text': '#ffffff', 'border': '#495057'
        })
        momentum_5h_colors = tech_analysis.get('momentum_5h_colors', {
            'bg': '#6c757d', 'bg_dark': '#5a6268', 'text': '#ffffff', 'border': '#495057'
        })
        
        momentum_1h_pct = tech_analysis.get('price_change_pct_1h', 0)
        momentum_1h_signal = tech_analysis.get('momentum_1h_signal', 'Sideways')
        momentum_5h_pct = tech_analysis.get('momentum_5h_pct', 0)
        momentum_5h_signal = tech_analysis.get('momentum_5h_signal', 'Sideways')
        
        # Build Top 5 CE table rows
        ce_rows_html = ""
        for i, strike in enumerate(top_strikes.get('top_ce_strikes', []), 1):
            ce_rows_html += f"""
                                <tr>
                                    <td>{i}</td>
                                    <td class="strike-price">₹{strike['strike']}</td>
                                    <td><span class="type-badge {strike['type'].lower()}">{strike['type']}</span></td>
                                    <td>{strike['oi']:,}</td>
                                    <td class="{'positive' if strike['chng_oi'] > 0 else 'negative'}">{strike['chng_oi']:+,}</td>
                                    <td>₹{strike['ltp']:.2f}</td>
                                    <td>{strike['iv']:.2f}%</td>
                                    <td>{strike['volume']:,}</td>
                                </tr>
            """
        
        # Build Top 5 PE table rows
        pe_rows_html = ""
        for i, strike in enumerate(top_strikes.get('top_pe_strikes', []), 1):
            pe_rows_html += f"""
                                <tr>
                                    <td>{i}</td>
                                    <td class="strike-price">₹{strike['strike']}</td>
                                    <td><span class="type-badge {strike['type'].lower()}">{strike['type']}</span></td>
                                    <td>{strike['oi']:,}</td>
                                    <td class="{'positive' if strike['chng_oi'] > 0 else 'negative'}">{strike['chng_oi']:+,}</td>
                                    <td>₹{strike['ltp']:.2f}</td>
                                    <td>{strike['iv']:.2f}%</td>
                                    <td>{strike['volume']:,}</td>
                                </tr>
            """
        
        # OI Change Analysis colors
        oi_direction = oi_change_analysis.get('direction', 'Neutral')
        if 'Strong Bullish' in oi_direction or 'Bullish' in oi_direction:
            oi_change_color = '#28a745'
        elif 'Strong Bearish' in oi_direction or 'Bearish' in oi_direction:
            oi_change_color = '#dc3545'
        else:
            oi_change_color = '#6c757d'
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
            color: #e0e0e0;
            padding: 20px;
            line-height: 1.5;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: #2a2a2a;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #3d3d3d 0%, #4a4a4a 100%);
            padding: 25px 30px;
            border-bottom: 2px solid #555;
            position: relative;
        }}
        
        .header h1 {{
            font-size: 26px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 8px;
        }}
        
        .timeframe-badge {{
            position: absolute;
            top: 25px;
            right: 30px;
            background: #4a9eff;
            color: white;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .timestamp {{
            font-size: 13px;
            color: #b0b0b0;
            font-weight: 500;
        }}
        
        .momentum-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 25px 30px;
            background: #2a2a2a;
        }}
        
        .momentum-box {{
            background: linear-gradient(135deg, var(--momentum-bg) 0%, var(--momentum-bg-dark) 100%);
            padding: 22px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid var(--momentum-border);
        }}
        
        .momentum-box h3 {{
            font-size: 13px;
            color: var(--momentum-text);
            margin-bottom: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .momentum-box .value {{
            font-size: 36px;
            font-weight: 700;
            color: var(--momentum-text);
            margin-bottom: 8px;
        }}
        
        .momentum-box .signal {{
            font-size: 13px;
            color: var(--momentum-text);
            font-weight: 500;
            opacity: 0.9;
        }}
        
        .recommendation-box {{
            margin: 25px 30px;
            padding: 25px;
            background: linear-gradient(135deg, #3d3d3d 0%, #4a4a4a 100%);
            border-radius: 10px;
            border-left: 5px solid #4a9eff;
        }}
        
        .recommendation-box h2 {{
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            font-size: 15px;
            color: #b0b0b0;
            margin-bottom: 15px;
        }}
        
        .signal-badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            margin-right: 10px;
        }}
        
        .signal-badge.bullish {{
            background: rgba(40, 167, 69, 0.2);
            color: #28a745;
            border: 1px solid #28a745;
        }}
        
        .signal-badge.bearish {{
            background: rgba(220, 53, 69, 0.2);
            color: #dc3545;
            border: 1px solid #dc3545;
        }}
        
        .section {{
            margin: 25px 30px;
            background: #333;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #444;
        }}
        
        .section-title {{
            font-size: 18px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 18px;
            padding: 12px 18px;
            background: linear-gradient(135deg, #3d3d3d 0%, #4a4a4a 100%);
            border-radius: 6px;
            border-left: 4px solid #4a9eff;
        }}
        
        .data-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .data-item {{
            background: #3d3d3d;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #4a4a4a;
        }}
        
        .data-item .label {{
            font-size: 12px;
            color: #9ca3af;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .data-item .value {{
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
        }}
        
        .levels {{
            display: flex;
            gap: 20px;
            margin-top: 15px;
        }}
        
        .levels-box {{
            flex: 1;
            padding: 15px;
            border-radius: 6px;
            min-width: 200px;
        }}
        
        .levels-box.resistance {{
            background: rgba(220, 53, 69, 0.1);
            border: 1px solid rgba(220, 53, 69, 0.3);
        }}
        
        .levels-box.support {{
            background: rgba(40, 167, 69, 0.1);
            border: 1px solid rgba(40, 167, 69, 0.3);
        }}
        
        .levels-box h4 {{
            font-size: 14px;
            margin-bottom: 10px;
            color: #ffffff;
        }}
        
        .levels-box ul {{
            list-style: none;
            padding: 0;
        }}
        
        .levels-box li {{
            padding: 6px 0;
            color: #e0e0e0;
            font-size: 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .levels-box li:last-child {{
            border-bottom: none;
        }}
        
        .oi-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        .oi-section {{
            background: #3d3d3d;
            padding: 18px;
            border-radius: 8px;
            border: 1px solid #4a4a4a;
        }}
        
        .oi-section h4 {{
            font-size: 15px;
            margin-bottom: 15px;
            color: #ffffff;
            font-weight: 600;
        }}
        
        .oi-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}
        
        .oi-table th {{
            background: #2a2a2a;
            padding: 10px 6px;
            text-align: left;
            color: #9ca3af;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 10px;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #4a4a4a;
        }}
        
        .oi-table td {{
            padding: 10px 6px;
            border-bottom: 1px solid #3d3d3d;
            color: #e0e0e0;
        }}
        
        .oi-table tr:hover {{
            background: #404040;
        }}
        
        .strike-price {{
            font-weight: 700;
            color: #4a9eff;
        }}
        
        .type-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 600;
        }}
        
        .type-badge.itm {{
            background: rgba(40, 167, 69, 0.2);
            color: #28a745;
        }}
        
        .type-badge.atm {{
            background: rgba(255, 193, 7, 0.2);
            color: #ffc107;
        }}
        
        .type-badge.otm {{
            background: rgba(108, 117, 125, 0.2);
            color: #9ca3af;
        }}
        
        .positive {{
            color: #28a745;
            font-weight: 600;
        }}
        
        .negative {{
            color: #dc3545;
            font-weight: 600;
        }}
        
        .reasons-box {{
            background: #3d3d3d;
            padding: 18px;
            border-radius: 8px;
            border-left: 4px solid #4a9eff;
            margin-top: 20px;
        }}
        
        .reasons-box ul {{
            list-style: none;
            padding: 0;
        }}
        
        .reasons-box li {{
            padding: 8px 0;
            color: #e0e0e0;
            font-size: 14px;
            border-bottom: 1px solid #4a4a4a;
        }}
        
        .reasons-box li:last-child {{
            border-bottom: none;
        }}
        
        /* OI Change Analysis Specific Styles */
        .oi-change-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .oi-change-main {{
            grid-column: 1 / -1;
            background: linear-gradient(135deg, {oi_change_color} 0%, {oi_change_color}dd 100%);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 2px solid {oi_change_color};
        }}
        
        .oi-change-main h3 {{
            font-size: 16px;
            color: #ffffff;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        
        .oi-change-main .direction {{
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
            margin: 10px 0;
        }}
        
        .oi-change-main .confidence {{
            font-size: 14px;
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 15px;
        }}
        
        .oi-change-main .signal-text {{
            font-size: 13px;
            color: rgba(255, 255, 255, 0.95);
            background: rgba(0, 0, 0, 0.2);
            padding: 12px;
            border-radius: 6px;
            line-height: 1.6;
        }}
        
        @media (max-width: 768px) {{
            .container {{ padding: 15px; border-radius: 8px; }}
            .header h1 {{ font-size: 22px; }}
            .momentum-container {{ grid-template-columns: 1fr; gap: 12px; }}
            .momentum-box .value {{ font-size: 28px; }}
            .recommendation-box h2 {{ font-size: 24px; }}
            .section-title {{ font-size: 15px; padding: 10px 15px; }}
            .data-grid {{ grid-template-columns: repeat(2, 1fr); gap: 10px; }}
            .data-item .value {{ font-size: 16px; }}
            .levels {{ flex-direction: column; }}
            .levels-box {{ min-width: 100%; }}
            .oi-grid {{ grid-template-columns: 1fr; }}
            .strike-recommendations {{ grid-template-columns: 1fr; }}
            .target-grid {{ grid-template-columns: 1fr; }}
        }}
        
        @media (max-width: 480px) {{
            body {{ padding: 8px; }}
            .container {{ padding: 12px; border-radius: 12px; }}
            .header h1 {{ font-size: 20px; }}
            .header {{ padding: 15px; }}
            .timeframe-badge {{ font-size: 10px; padding: 4px 10px; }}
            .momentum-box h3 {{ font-size: 12px; }}
            .momentum-box .value {{ font-size: 24px; }}
            .recommendation-box {{ padding: 15px; }}
            .recommendation-box h2 {{ font-size: 22px; }}
            .data-grid {{ grid-template-columns: 1fr; }}
            .oi-table {{ font-size: 10px; }}
            .oi-table th {{ font-size: 9px; padding: 6px 3px; }}
            .oi-table td {{ font-size: 9px; padding: 6px 3px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {title}</h1>
            <div class="timeframe-badge">⏱️ 1-HOUR TIMEFRAME</div>
            <div class="timestamp">Generated on: {now_ist}</div>
        </div>
        
        <!-- DUAL MOMENTUM DISPLAY -->
        <div class="momentum-container">
            <div class="momentum-box" style="--momentum-bg: {momentum_1h_colors['bg']}; --momentum-bg-dark: {momentum_1h_colors['bg_dark']}; --momentum-text: {momentum_1h_colors['text']}; --momentum-border: {momentum_1h_colors['border']};">
                <h3>⚡ 1H Momentum</h3>
                <div class="value">{momentum_1h_pct:+.2f}%</div>
                <div class="signal">{momentum_1h_signal}</div>
            </div>
            <div class="momentum-box" style="--momentum-bg: {momentum_5h_colors['bg']}; --momentum-bg-dark: {momentum_5h_colors['bg_dark']}; --momentum-text: {momentum_5h_colors['text']}; --momentum-border: {momentum_5h_colors['border']};">
                <h3>📊 5H Momentum</h3>
                <div class="value">{momentum_5h_pct:+.2f}%</div>
                <div class="signal">{momentum_5h_signal}</div>
            </div>
        </div>
        
        <div class="recommendation-box">
            <h2>{recommendation['recommendation']}</h2>
            <div class="subtitle">Market Bias: {recommendation['bias']} | Confidence: {recommendation['confidence']}</div>
            <div style="margin-top: 12px;">
                <span class="signal-badge bullish">Bullish: {recommendation['bullish_signals']}</span>
                <span class="signal-badge bearish">Bearish: {recommendation['bearish_signals']}</span>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📈 Technical Analysis (1H)</div>
            <div class="data-grid">
                <div class="data-item">
                    <div class="label">Current Price</div>
                    <div class="value">₹{tech_analysis.get('current_price', 'N/A')}</div>
                </div>
                <div class="data-item">
                    <div class="label">RSI (14)</div>
                    <div class="value">{tech_analysis.get('rsi', 'N/A')}</div>
                </div>
                <div class="data-item">
                    <div class="label">EMA 20</div>
                    <div class="value">₹{tech_analysis.get('ema20', 'N/A')}</div>
                </div>
                <div class="data-item">
                    <div class="label">EMA 50</div>
                    <div class="value">₹{tech_analysis.get('ema50', 'N/A')}</div>
                </div>
                <div class="data-item">
                    <div class="label">Trend</div>
                    <div class="value">{tech_analysis.get('trend', 'N/A')}</div>
                </div>
                <div class="data-item">
                    <div class="label">RSI Signal</div>
                    <div class="value">{tech_analysis.get('rsi_signal', 'N/A')}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">🎯 Support & Resistance (1H)</div>
            <div class="levels">
                <div class="levels-box resistance">
                    <h4>🔴 Resistance</h4>
                    <ul>{''.join([f'<li>R{i+1}: ₹{r}</li>' for i, r in enumerate(tech_analysis.get('tech_resistances', []))])}</ul>
                </div>
                <div class="levels-box support">
                    <h4>🟢 Support</h4>
                    <ul>{''.join([f'<li>S{i+1}: ₹{s}</li>' for i, s in enumerate(tech_analysis.get('tech_supports', []))])}</ul>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📍 Pivot Points (Traditional - 30 Min)</div>
            <div class="data-grid">
                <div class="data-item">
                    <div class="label">Pivot Point</div>
                    <div class="value">₹{tech_analysis.get('pivot_points', {}).get('pivot', 'N/A')}</div>
                </div>
                <div class="data-item">
                    <div class="label">R1</div>
                    <div class="value">₹{tech_analysis.get('pivot_points', {}).get('r1', 'N/A')}</div>
                </div>
                <div class="data-item">
                    <div class="label">R2</div>
                    <div class="value">₹{tech_analysis.get('pivot_points', {}).get('r2', 'N/A')}</div>
                </div>
                <div class="data-item">
                    <div class="label">R3</div>
                    <div class="value">₹{tech_analysis.get('pivot_points', {}).get('r3', 'N/A')}</div>
                </div>
                <div class="data-item">
                    <div class="label">S1</div>
                    <div class="value">₹{tech_analysis.get('pivot_points', {}).get('s1', 'N/A')}</div>
                </div>
                <div class="data-item">
                    <div class="label">S2</div>
                    <div class="value">₹{tech_analysis.get('pivot_points', {}).get('s2', 'N/A')}</div>
                </div>
                <div class="data-item">
                    <div class="label">S3</div>
                    <div class="value">₹{tech_analysis.get('pivot_points', {}).get('s3', 'N/A')}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📊 Option Chain Analysis</div>
            <div class="data-grid">
                <div class="data-item">
                    <div class="label">Put-Call Ratio</div>
                    <div class="value">{oc_analysis.get('pcr', 'N/A')}</div>
                </div>
                <div class="data-item">
                    <div class="label">Max Pain</div>
                    <div class="value">₹{oc_analysis.get('max_pain', 'N/A')}</div>
                </div>
                <div class="data-item">
                    <div class="label">OI Sentiment</div>
                    <div class="value">{oc_analysis.get('oi_sentiment', 'N/A')}</div>
                </div>
            </div>
            
            <div style="margin-top: 20px;">
                <div class="levels">
                    <div class="levels-box resistance">
                        <h4>🔴 OI Resistance</h4>
                        <ul>{''.join([f'<li>₹{r}</li>' for r in oc_analysis.get('resistances', [])])}</ul>
                    </div>
                    <div class="levels-box support">
                        <h4>🟢 OI Support</h4>
                        <ul>{''.join([f'<li>₹{s}</li>' for s in oc_analysis.get('supports', [])])}</ul>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- TOP 10 OPEN INTEREST SECTION -->
        <div class="section">
            <div class="section-title">🔥 Top 10 Open Interest (5 CE + 5 PE)</div>
            <div class="oi-grid">
                <!-- Call Options (CE) -->
                <div class="oi-section calls">
                    <h4>📞 Top 5 Call Options (CE)</h4>
                    <div class="oi-container">
                        <table class="oi-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Strike</th>
                                    <th>Type</th>
                                    <th>OI</th>
                                    <th>Chng OI</th>
                                    <th>LTP</th>
                                    <th>IV</th>
                                    <th>Volume</th>
                                </tr>
                            </thead>
                            <tbody>
{ce_rows_html}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <!-- Put Options (PE) -->
                <div class="oi-section puts">
                    <h4>📉 Top 5 Put Options (PE)</h4>
                    <div class="oi-container">
                        <table class="oi-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Strike</th>
                                    <th>Type</th>
                                    <th>OI</th>
                                    <th>Chng OI</th>
                                    <th>LTP</th>
                                    <th>IV</th>
                                    <th>Volume</th>
                                </tr>
                            </thead>
                            <tbody>
{pe_rows_html}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- OI CHANGE ANALYSIS SECTION (NEW) -->
        <div class="section">
            <div class="section-title">🔥 OI Change Analysis (Market Direction Prediction)</div>
            <div class="oi-change-grid">
                <div class="data-item">
                    <div class="label">Total Call OI Change</div>
                    <div class="value {'positive' if oi_change_analysis.get('total_call_oi_change', 0) > 0 else 'negative'}">{oi_change_analysis.get('total_call_oi_change', 0):+,}</div>
                </div>
                <div class="data-item">
                    <div class="label">Total Put OI Change</div>
                    <div class="value {'positive' if oi_change_analysis.get('total_put_oi_change', 0) > 0 else 'negative'}">{oi_change_analysis.get('total_put_oi_change', 0):+,}</div>
                </div>
                
                <div class="oi-change-main">
                    <h3>📊 Market Direction Based on OI Changes</h3>
                    <div class="direction">{oi_change_analysis.get('direction', 'N/A')}</div>
                    <div class="confidence">Confidence: {oi_change_analysis.get('confidence', 'N/A')}</div>
                    <div class="signal-text">{oi_change_analysis.get('signal', 'N/A')}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📋 Analysis Summary</div>
            <div class="reasons-box">
                <ul>
                    {''.join([f'<li>{reason}</li>' for reason in recommendation.get('reasons', [])])}
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
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
        """Run complete analysis with OI CHANGE"""
        self.logger.info("🚀 Starting Nifty 1-HOUR Analysis with OI Change Detection...")
        self.logger.info("=" * 60)
        
        # Fetch option chain
        oc_df, spot_price = self.fetch_option_chain()
        
        if oc_df is not None and spot_price is not None:
            oc_analysis = self.analyze_option_chain(oc_df, spot_price)
            top_strikes = self.get_top_strikes_by_oi(oc_df, spot_price)
            oi_change_analysis = self.analyze_oi_change(oc_df)  # NEW - Analyze OI changes
        else:
            spot_price = 25800
            oc_analysis = self.get_sample_oc_analysis()
            top_strikes = {'top_ce_strikes': [], 'top_pe_strikes': []}
            oi_change_analysis = {
                'total_call_oi_change': 0,
                'total_put_oi_change': 0,
                'direction': 'No Data',
                'signal': 'Unable to fetch option chain data',
                'confidence': 'N/A'
            }
        
        # Fetch technical data
        tech_df = self.fetch_technical_data()
        
        if tech_df is not None and not tech_df.empty:
            tech_analysis = self.technical_analysis(tech_df)
        else:
            tech_analysis = self.get_sample_tech_analysis()
        
        # Generate recommendation with OI Change
        self.logger.info("🎯 Generating Trading Recommendation with OI Change Analysis...")
        recommendation = self.generate_recommendation(oc_analysis, tech_analysis, oi_change_analysis)
        
        self.logger.info("=" * 60)
        self.logger.info(f"📊 RECOMMENDATION: {recommendation['recommendation']}")
        self.logger.info(f"📈 Bias: {recommendation['bias']} | Confidence: {recommendation['confidence']}")
        self.logger.info(f"🔥 OI Direction: {oi_change_analysis.get('direction')} ({oi_change_analysis.get('confidence')})")
        self.logger.info(f"🎯 RSI (1H): {tech_analysis.get('rsi', 'N/A')}")
        self.logger.info(f"⚡ 1H Momentum: {tech_analysis.get('price_change_pct_1h', 0):+.2f}% - {tech_analysis.get('momentum_1h_signal')}")
        self.logger.info(f"📊 5H Momentum: {tech_analysis.get('momentum_5h_pct', 0):+.2f}% - {tech_analysis.get('momentum_5h_signal')}")
        self.logger.info("=" * 60)
        
        # Create HTML report
        html_report = self.create_html_report(oc_analysis, tech_analysis, recommendation, top_strikes, oi_change_analysis)
        
        # Save report locally
        if self.config['report']['save_local']:
            report_dir = self.config['report']['local_dir']
            os.makedirs(report_dir, exist_ok=True)
            
            ist_time = self.get_ist_time()
            filename_format = self.config['report']['filename_format']
            report_filename = os.path.join(report_dir, ist_time.strftime(filename_format))
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(html_report)
            self.logger.info(f"💾 Report saved as: {report_filename}")
        
        # Send email
        self.logger.info(f"📧 Sending email to {self.config['email']['recipient']}...")
        self.send_email(html_report)
        
        self.logger.info("✅ Analysis Complete with OI Change Detection!")
        
        return {
            'oc_analysis': oc_analysis,
            'tech_analysis': tech_analysis,
            'recommendation': recommendation,
            'oi_change_analysis': oi_change_analysis,
            'html_report': html_report
        }


if __name__ == "__main__":
    analyzer = NiftyAnalyzer(config_path='config.yml')
    result = analyzer.run_analysis()
    
    print(f"\n✅ Analysis Complete!")
    print(f"Recommendation: {result['recommendation']['recommendation']}")
    print(f"OI Change Direction: {result['oi_change_analysis']['direction']} (Confidence: {result['oi_change_analysis']['confidence']})")
    print(f"RSI (1H): {result['tech_analysis']['rsi']}")
    print(f"1H Momentum: {result['tech_analysis']['price_change_pct_1h']:+.2f}% - {result['tech_analysis']['momentum_1h_signal']}")
    print(f"5H Momentum: {result['tech_analysis']['momentum_5h_pct']:+.2f}% - {result['tech_analysis']['momentum_5h_signal']}")
    print(f"Check your email for the detailed report!")
