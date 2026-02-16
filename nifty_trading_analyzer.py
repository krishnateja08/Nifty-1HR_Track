"""
Nifty Option Chain & Technical Analysis for Day Trading
PROFESSIONAL VERSION - Enhanced Design with Better Contrast
1-HOUR TIMEFRAME with WILDER'S RSI (matches TradingView)
Enhanced with Pivot Points + Dual Momentum Analysis + Top 10 OI Display + OI CHANGE ANALYSIS
EXPIRY: Weekly TUESDAY expiry with 3:30 PM IST cutoff logic
FIXED: Using curl-cffi for NSE API to bypass anti-scraping
UPDATED: Professional grey theme with improved readability
NEW: OI Change Analysis for Market Direction Prediction
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
import json

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
        
        # OI Change tracking
        self.oi_history_file = './data/oi_history.json'
        self.ensure_data_dir()
    
    def ensure_data_dir(self):
        """Ensure data directory exists for OI history"""
        os.makedirs('./data', exist_ok=True)
    
    def save_oi_snapshot(self, oc_df, spot_price):
        """Save current OI snapshot for future comparison"""
        try:
            timestamp = self.get_ist_time().isoformat()
            
            # Aggregate OI data
            total_call_oi = int(oc_df['CE_OI'].sum())
            total_put_oi = int(oc_df['PE_OI'].sum())
            
            # Get ATM and nearby strikes OI
            atm_strike = round(spot_price / 50) * 50
            atm_range = oc_df[
                (oc_df['Strike'] >= atm_strike - 200) & 
                (oc_df['Strike'] <= atm_strike + 200)
            ]
            
            snapshot = {
                'timestamp': timestamp,
                'spot_price': float(spot_price),
                'total_call_oi': total_call_oi,
                'total_put_oi': total_put_oi,
                'atm_call_oi': int(atm_range['CE_OI'].sum()),
                'atm_put_oi': int(atm_range['PE_OI'].sum()),
                'strikes': {}
            }
            
            # Store top strikes
            for _, row in oc_df.head(20).iterrows():
                snapshot['strikes'][int(row['Strike'])] = {
                    'ce_oi': int(row['CE_OI']),
                    'pe_oi': int(row['PE_OI'])
                }
            
            # Load existing history
            history = []
            if os.path.exists(self.oi_history_file):
                with open(self.oi_history_file, 'r') as f:
                    history = json.load(f)
            
            # Keep only last 10 snapshots
            history.append(snapshot)
            history = history[-10:]
            
            # Save updated history
            with open(self.oi_history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            self.logger.info(f"💾 OI snapshot saved ({len(history)} snapshots in history)")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving OI snapshot: {e}")
    
    def analyze_oi_change(self, oc_df, spot_price):
        """
        Analyze OI changes to predict market direction
        
        Trading Logic:
        - Call OI Increase + Put OI Decrease = Bullish (Long buildup)
        - Call OI Decrease + Put OI Increase = Bearish (Short covering)
        - Both Increase = High activity (check which is stronger)
        - Both Decrease = Unwinding (neutral/exit positions)
        """
        try:
            # Load previous snapshot
            if not os.path.exists(self.oi_history_file):
                return self.get_default_oi_change_analysis()
            
            with open(self.oi_history_file, 'r') as f:
                history = json.load(f)
            
            if len(history) < 2:
                return self.get_default_oi_change_analysis()
            
            # Compare current vs previous
            prev_snapshot = history[-2]
            curr_total_call_oi = int(oc_df['CE_OI'].sum())
            curr_total_put_oi = int(oc_df['PE_OI'].sum())
            
            prev_total_call_oi = prev_snapshot['total_call_oi']
            prev_total_put_oi = prev_snapshot['total_put_oi']
            
            # Calculate changes
            call_oi_change = curr_total_call_oi - prev_total_call_oi
            put_oi_change = curr_total_put_oi - prev_total_put_oi
            
            call_oi_change_pct = (call_oi_change / prev_total_call_oi * 100) if prev_total_call_oi > 0 else 0
            put_oi_change_pct = (put_oi_change / prev_total_put_oi * 100) if prev_total_put_oi > 0 else 0
            
            # ATM analysis (more sensitive to direction)
            atm_strike = round(spot_price / 50) * 50
            atm_range = oc_df[
                (oc_df['Strike'] >= atm_strike - 200) & 
                (oc_df['Strike'] <= atm_strike + 200)
            ]
            
            curr_atm_call_oi = int(atm_range['CE_OI'].sum())
            curr_atm_put_oi = int(atm_range['PE_OI'].sum())
            
            prev_atm_call_oi = prev_snapshot.get('atm_call_oi', curr_atm_call_oi)
            prev_atm_put_oi = prev_snapshot.get('atm_put_oi', curr_atm_put_oi)
            
            atm_call_change = curr_atm_call_oi - prev_atm_call_oi
            atm_put_change = curr_atm_put_oi - prev_atm_put_oi
            
            atm_call_change_pct = (atm_call_change / prev_atm_call_oi * 100) if prev_atm_call_oi > 0 else 0
            atm_put_change_pct = (atm_put_change / prev_atm_put_oi * 100) if prev_atm_put_oi > 0 else 0
            
            # Determine market direction based on OI changes
            direction, confidence, signal_strength, explanation = self.interpret_oi_changes(
                call_oi_change_pct, put_oi_change_pct,
                atm_call_change_pct, atm_put_change_pct
            )
            
            self.logger.info(f"📊 OI Change Analysis:")
            self.logger.info(f"   Total Call OI: {call_oi_change_pct:+.2f}% | Put OI: {put_oi_change_pct:+.2f}%")
            self.logger.info(f"   ATM Call OI: {atm_call_change_pct:+.2f}% | ATM Put OI: {atm_put_change_pct:+.2f}%")
            self.logger.info(f"   Direction: {direction} | Confidence: {confidence} | Signal: {signal_strength}")
            
            return {
                'total_call_oi_change': call_oi_change,
                'total_put_oi_change': put_oi_change,
                'total_call_oi_change_pct': round(call_oi_change_pct, 2),
                'total_put_oi_change_pct': round(put_oi_change_pct, 2),
                'atm_call_oi_change': atm_call_change,
                'atm_put_oi_change': atm_put_change,
                'atm_call_oi_change_pct': round(atm_call_change_pct, 2),
                'atm_put_oi_change_pct': round(atm_put_change_pct, 2),
                'direction': direction,
                'confidence': confidence,
                'signal_strength': signal_strength,
                'explanation': explanation,
                'prev_timestamp': prev_snapshot['timestamp']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing OI change: {e}")
            return self.get_default_oi_change_analysis()
    
    def interpret_oi_changes(self, call_pct, put_pct, atm_call_pct, atm_put_pct):
        """
        Interpret OI changes to determine market direction
        
        Returns: (direction, confidence, signal_strength, explanation)
        """
        
        # Thresholds
        strong_threshold = 2.0  # 2% change is significant
        moderate_threshold = 0.5  # 0.5% change is moderate
        
        # Weight ATM more heavily (it's more predictive)
        weighted_call = (call_pct * 0.4) + (atm_call_pct * 0.6)
        weighted_put = (put_pct * 0.4) + (atm_put_pct * 0.6)
        
        explanations = []
        
        # SCENARIO 1: Call OI increasing, Put OI decreasing = STRONG BULLISH
        if weighted_call > moderate_threshold and weighted_put < -moderate_threshold:
            if abs(weighted_call) > strong_threshold and abs(weighted_put) > strong_threshold:
                direction = "Strong Bullish"
                confidence = "Very High"
                signal_strength = 5
                explanations.append("📈 Call writers adding positions (resistance)")
                explanations.append("📉 Put writers covering (support weakening)")
                explanations.append("💪 Strong long buildup expected")
            else:
                direction = "Bullish"
                confidence = "High"
                signal_strength = 4
                explanations.append("📈 Moderate call buildup")
                explanations.append("📉 Put unwinding")
        
        # SCENARIO 2: Call OI decreasing, Put OI increasing = STRONG BEARISH
        elif weighted_call < -moderate_threshold and weighted_put > moderate_threshold:
            if abs(weighted_call) > strong_threshold and abs(weighted_put) > strong_threshold:
                direction = "Strong Bearish"
                confidence = "Very High"
                signal_strength = -5
                explanations.append("📉 Call covering (resistance breaking)")
                explanations.append("📈 Put buildup (strong support)")
                explanations.append("💪 Strong short buildup expected")
            else:
                direction = "Bearish"
                confidence = "High"
                signal_strength = -4
                explanations.append("📉 Call unwinding")
                explanations.append("📈 Moderate put buildup")
        
        # SCENARIO 3: Both increasing = High volatility, check which is stronger
        elif weighted_call > moderate_threshold and weighted_put > moderate_threshold:
            if weighted_call > weighted_put * 1.5:
                direction = "Moderately Bullish"
                confidence = "Medium"
                signal_strength = 2
                explanations.append("📊 Both OI increasing (high activity)")
                explanations.append("📈 Call buildup stronger")
                explanations.append("⚡ Expect volatility with upward bias")
            elif weighted_put > weighted_call * 1.5:
                direction = "Moderately Bearish"
                confidence = "Medium"
                signal_strength = -2
                explanations.append("📊 Both OI increasing (high activity)")
                explanations.append("📉 Put buildup stronger")
                explanations.append("⚡ Expect volatility with downward bias")
            else:
                direction = "Neutral - High Activity"
                confidence = "Low"
                signal_strength = 0
                explanations.append("📊 Both Call & Put OI increasing equally")
                explanations.append("⚖️ Market preparing for big move")
                explanations.append("⚠️ Direction unclear - wait for confirmation")
        
        # SCENARIO 4: Both decreasing = Unwinding (usually neutral to slightly bearish)
        elif weighted_call < -moderate_threshold and weighted_put < -moderate_threshold:
            direction = "Neutral - Unwinding"
            confidence = "Low"
            signal_strength = -1
            explanations.append("📉 Both Call & Put OI decreasing")
            explanations.append("🔄 Position unwinding/profit booking")
            explanations.append("⚠️ Low conviction - avoid new positions")
        
        # SCENARIO 5: Minimal changes
        else:
            direction = "Neutral"
            confidence = "Low"
            signal_strength = 0
            explanations.append("➡️ Minimal OI changes")
            explanations.append("😴 Low activity - market consolidating")
            explanations.append("⏳ Wait for clearer signals")
        
        explanation = " | ".join(explanations)
        
        return direction, confidence, signal_strength, explanation
    
    def get_default_oi_change_analysis(self):
        """Return default OI change analysis when no history available"""
        return {
            'total_call_oi_change': 0,
            'total_put_oi_change': 0,
            'total_call_oi_change_pct': 0.0,
            'total_put_oi_change_pct': 0.0,
            'atm_call_oi_change': 0,
            'atm_put_oi_change': 0,
            'atm_call_oi_change_pct': 0.0,
            'atm_put_oi_change_pct': 0.0,
            'direction': 'No Historical Data',
            'confidence': 'N/A',
            'signal_strength': 0,
            'explanation': 'First run - building OI history. Check back on next analysis.',
            'prev_timestamp': 'N/A'
        }
    
    def get_next_expiry_date(self):
        """
        Calculate the next NIFTY expiry date (Weekly Tuesday)
        If today is Tuesday after 3:30 PM, return next week's Tuesday
        Logic: Every Tuesday is expiry. After 3:30 PM on Tuesday, switch to next Tuesday.
        """
        now_ist = self.get_ist_time()
        current_day = now_ist.weekday()
        
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
                'top_strikes_count': 5,
                'oi_change_weight': 2  # Weight for OI change in recommendation
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
        """Setup logging based on config"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        
        if log_config.get('log_to_file', True):
            log_file = log_config.get('log_file', './logs/nifty_analyzer.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            logging.basicConfig(
                level=log_level,
                format=log_format,
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(level=log_level, format=log_format)
        
        self.logger = logging.getLogger(__name__)
    
    def fetch_option_chain(self):
        """Fetch option chain from NSE using curl-cffi"""
        expiry_date = self.get_next_expiry_date()
        url = self.option_chain_base_url + expiry_date
        
        max_retries = self.config['data_source']['max_retries']
        retry_delay = self.config['data_source']['retry_delay']
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"📡 Fetching option chain (Attempt {attempt + 1}/{max_retries})...")
                
                session = requests.Session(impersonate="chrome120")
                session.get("https://www.nseindia.com", headers=self.headers, timeout=10)
                time.sleep(1)
                
                response = session.get(url, headers=self.headers, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'records' in data and 'data' in data['records']:
                        spot_price = data['records']['underlyingValue']
                        option_data = data['records']['data']
                        
                        records = []
                        for item in option_data:
                            ce = item.get('CE', {})
                            pe = item.get('PE', {})
                            strike = item.get('strikePrice')
                            
                            if ce and pe:
                                records.append({
                                    'Strike': strike,
                                    'CE_OI': ce.get('openInterest', 0),
                                    'CE_Volume': ce.get('totalTradedVolume', 0),
                                    'CE_LTP': ce.get('lastPrice', 0),
                                    'CE_IV': ce.get('impliedVolatility', 0),
                                    'PE_OI': pe.get('openInterest', 0),
                                    'PE_Volume': pe.get('totalTradedVolume', 0),
                                    'PE_LTP': pe.get('lastPrice', 0),
                                    'PE_IV': pe.get('impliedVolatility', 0)
                                })
                        
                        df = pd.DataFrame(records)
                        self.logger.info(f"✅ Successfully fetched {len(df)} strikes | Spot Price: ₹{spot_price:.2f}")
                        
                        # Save OI snapshot for tracking changes
                        self.save_oi_snapshot(df, spot_price)
                        
                        return df, spot_price
                    else:
                        self.logger.warning(f"⚠️ Empty response from NSE")
                else:
                    self.logger.warning(f"⚠️ HTTP {response.status_code}")
                
            except Exception as e:
                self.logger.error(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        self.logger.warning("⚠️ All attempts failed. Using fallback data...")
        if self.config['data_source'].get('fallback_to_sample', True):
            return None, None
        else:
            raise Exception("Failed to fetch option chain data")
    
    def get_sample_oc_analysis(self):
        """Return sample option chain analysis"""
        return {
            'spot_price': 25796,
            'pcr': 1.35,
            'pcr_signal': 'Bullish',
            'max_call_oi_strike': 26000,
            'max_put_oi_strike': 25500,
            'top_10_call_oi': [
                {'strike': 26000, 'oi': 1250000, 'ltp': 125.50},
                {'strike': 26100, 'oi': 980000, 'ltp': 98.20},
                {'strike': 25900, 'oi': 875000, 'ltp': 165.30},
                {'strike': 26200, 'oi': 720000, 'ltp': 75.80},
                {'strike': 25800, 'oi': 685000, 'ltp': 210.40},
                {'strike': 26300, 'oi': 550000, 'ltp': 58.20},
                {'strike': 25700, 'oi': 520000, 'ltp': 265.50},
                {'strike': 26400, 'oi': 485000, 'ltp': 42.30},
                {'strike': 25600, 'oi': 450000, 'ltp': 325.80},
                {'strike': 26500, 'oi': 420000, 'ltp': 30.50}
            ],
            'top_10_put_oi': [
                {'strike': 25500, 'oi': 1450000, 'ltp': 145.25},
                {'strike': 25400, 'oi': 1120000, 'ltp': 98.50},
                {'strike': 25600, 'oi': 950000, 'ltp': 185.30},
                {'strike': 25300, 'oi': 820000, 'ltp': 72.80},
                {'strike': 25700, 'oi': 780000, 'ltp': 235.60},
                {'strike': 25200, 'oi': 650000, 'ltp': 52.40},
                {'strike': 25800, 'oi': 590000, 'ltp': 295.20},
                {'strike': 25100, 'oi': 520000, 'ltp': 38.50},
                {'strike': 25900, 'oi': 480000, 'ltp': 355.80},
                {'strike': 25000, 'oi': 450000, 'ltp': 28.30}
            ]
        }
    
    def analyze_option_chain(self, df, spot_price):
        """Analyze option chain data"""
        total_call_oi = df['CE_OI'].sum()
        total_put_oi = df['PE_OI'].sum()
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        pcr_very_bullish = self.config['option_chain']['pcr_very_bullish']
        pcr_bullish = self.config['option_chain']['pcr_bullish']
        pcr_bearish = self.config['option_chain']['pcr_bearish']
        pcr_very_bearish = self.config['option_chain']['pcr_very_bearish']
        
        if pcr >= pcr_very_bullish:
            pcr_signal = 'Very Bullish'
        elif pcr >= pcr_bullish:
            pcr_signal = 'Bullish'
        elif pcr <= pcr_very_bearish:
            pcr_signal = 'Very Bearish'
        elif pcr <= pcr_bearish:
            pcr_signal = 'Bearish'
        else:
            pcr_signal = 'Neutral'
        
        max_call_oi_strike = df.loc[df['CE_OI'].idxmax(), 'Strike']
        max_put_oi_strike = df.loc[df['PE_OI'].idxmax(), 'Strike']
        
        top_call_oi = df.nlargest(10, 'CE_OI')[['Strike', 'CE_OI', 'CE_LTP']].to_dict('records')
        top_call_oi = [{'strike': int(r['Strike']), 'oi': int(r['CE_OI']), 'ltp': float(r['CE_LTP'])} for r in top_call_oi]
        
        top_put_oi = df.nlargest(10, 'PE_OI')[['Strike', 'PE_OI', 'PE_LTP']].to_dict('records')
        top_put_oi = [{'strike': int(r['Strike']), 'oi': int(r['PE_OI']), 'ltp': float(r['PE_LTP'])} for r in top_put_oi]
        
        self.logger.info(f"📊 PCR: {pcr:.2f} ({pcr_signal})")
        self.logger.info(f"📍 Max Call OI: {max_call_oi_strike} | Max Put OI: {max_put_oi_strike}")
        
        return {
            'spot_price': spot_price,
            'pcr': round(pcr, 2),
            'pcr_signal': pcr_signal,
            'max_call_oi_strike': max_call_oi_strike,
            'max_put_oi_strike': max_put_oi_strike,
            'top_10_call_oi': top_call_oi,
            'top_10_put_oi': top_put_oi
        }
    
    def fetch_technical_data(self):
        """Fetch 1-hour data for technical analysis"""
        try:
            timeframe = self.config['technical']['timeframe']
            period = self.config['technical']['period']
            
            self.logger.info(f"📊 Fetching {timeframe} technical data for {period}...")
            
            ticker = yf.Ticker(self.nifty_symbol)
            df = ticker.history(period=period, interval=timeframe)
            
            if df.empty:
                self.logger.warning("⚠️ No technical data retrieved")
                return None
            
            self.logger.info(f"✅ Fetched {len(df)} {timeframe} candles")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching technical data: {e}")
            return None
    
    def calculate_pivot_points(self, high, low, close):
        """Calculate Pivot Points (Traditional)"""
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': round(pivot, 2),
            'r1': round(r1, 2),
            'r2': round(r2, 2),
            'r3': round(r3, 2),
            's1': round(s1, 2),
            's2': round(s2, 2),
            's3': round(s3, 2)
        }
    
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
    
    def technical_analysis(self, df):
        """Perform technical analysis with DUAL MOMENTUM"""
        rsi_period = self.config['technical']['rsi_period']
        ema_short = self.config['technical']['ema_short']
        ema_long = self.config['technical']['ema_long']
        
        # RSI (WILDER'S METHOD)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = round(rsi.iloc[-1], 2)
        
        # EMA
        ema_20 = df['Close'].ewm(span=ema_short, adjust=False).mean().iloc[-1]
        ema_50 = df['Close'].ewm(span=ema_long, adjust=False).mean().iloc[-1]
        
        # SUPPORT & RESISTANCE
        recent_data = df.tail(20)
        support_levels = sorted(recent_data['Low'].nsmallest(self.config['technical']['num_support_levels']).tolist())
        resistance_levels = sorted(recent_data['High'].nlargest(self.config['technical']['num_resistance_levels']).tolist(), reverse=True)
        
        # PIVOT POINTS
        yesterday_high = df['High'].iloc[-2]
        yesterday_low = df['Low'].iloc[-2]
        yesterday_close = df['Close'].iloc[-2]
        pivot_points = self.calculate_pivot_points(yesterday_high, yesterday_low, yesterday_close)
        
        current_price = df['Close'].iloc[-1]
        
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
        
        return {
            'current_price': round(current_price, 2),
            'rsi': rsi_value,
            'ema_20': round(ema_20, 2),
            'ema_50': round(ema_50, 2),
            'support_levels': [round(s, 2) for s in support_levels],
            'resistance_levels': [round(r, 2) for r in resistance_levels],
            'pivot_points': pivot_points,
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
        """Return sample technical analysis data"""
        return {
            'current_price': 25796.00,
            'rsi': 58.25,
            'ema_20': 25750.50,
            'ema_50': 25680.25,
            'support_levels': [25650.00, 25580.00],
            'resistance_levels': [25850.00, 25920.00],
            'pivot_points': {
                'pivot': 25750.00,
                'r1': 25820.00,
                'r2': 25890.00,
                'r3': 25960.00,
                's1': 25680.00,
                's2': 25610.00,
                's3': 25540.00
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
        """Generate trading recommendation with OI CHANGE ANALYSIS"""
        config = self.config['recommendation']
        tech_config = self.config['technical']
        oc_config = self.config['option_chain']
        
        strong_threshold = tech_config['momentum_threshold_strong']
        moderate_threshold = tech_config['momentum_threshold_moderate']
        
        bullish_signals = 0
        bearish_signals = 0
        reasons = []
        
        # ==================== OI CHANGE SIGNALS (NEW - HIGHEST PRIORITY) ====================
        oi_signal_strength = oi_change_analysis.get('signal_strength', 0)
        oi_direction = oi_change_analysis.get('direction', 'Neutral')
        oi_weight = oc_config.get('oi_change_weight', 2)
        
        if oi_signal_strength >= 4:
            bullish_signals += oi_weight * 2
            reasons.append(f"🔥 OI Change: {oi_direction} (Very Strong)")
        elif oi_signal_strength >= 2:
            bullish_signals += oi_weight
            reasons.append(f"📊 OI Change: {oi_direction} (Moderate)")
        elif oi_signal_strength <= -4:
            bearish_signals += oi_weight * 2
            reasons.append(f"🔥 OI Change: {oi_direction} (Very Strong)")
        elif oi_signal_strength <= -2:
            bearish_signals += oi_weight
            reasons.append(f"📊 OI Change: {oi_direction} (Moderate)")
        elif oi_signal_strength == 0 and 'No Historical Data' not in oi_direction:
            reasons.append(f"⚠️ OI Change: {oi_direction}")
        # =================================================================================
        
        # RSI SIGNALS
        rsi = tech_analysis.get('rsi', 50)
        rsi_oversold = tech_config['rsi_oversold']
        rsi_overbought = tech_config['rsi_overbought']
        
        if rsi < rsi_oversold:
            bullish_signals += 2
            reasons.append(f"📊 RSI oversold at {rsi:.2f}")
        elif rsi > rsi_overbought:
            bearish_signals += 2
            reasons.append(f"📊 RSI overbought at {rsi:.2f}")
        
        # PCR SIGNALS
        pcr = oc_analysis.get('pcr', 1.0)
        pcr_signal = oc_analysis.get('pcr_signal', 'Neutral')
        
        if 'Very Bullish' in pcr_signal:
            bullish_signals += 2
            reasons.append(f"📈 PCR Very Bullish: {pcr:.2f}")
        elif 'Bullish' in pcr_signal:
            bullish_signals += 1
            reasons.append(f"📈 PCR Bullish: {pcr:.2f}")
        elif 'Very Bearish' in pcr_signal:
            bearish_signals += 2
            reasons.append(f"📉 PCR Very Bearish: {pcr:.2f}")
        elif 'Bearish' in pcr_signal:
            bearish_signals += 1
            reasons.append(f"📉 PCR Bearish: {pcr:.2f}")
        
        # MOMENTUM SIGNALS
        momentum_1h_pct = tech_analysis.get('price_change_pct_1h', 0)
        weight_1h = config.get('momentum_1h_weight', 1)
        
        if momentum_1h_pct > strong_threshold:
            bullish_signals += weight_1h
            reasons.append(f"⚡ 1H Strong upward momentum: {momentum_1h_pct:+.2f}%")
        elif momentum_1h_pct > moderate_threshold:
            bullish_signals += 1
            reasons.append(f"⚡ 1H Positive momentum: {momentum_1h_pct:+.2f}%")
        elif momentum_1h_pct < -strong_threshold:
            bearish_signals += weight_1h
            reasons.append(f"⚡ 1H Strong downward momentum: {momentum_1h_pct:+.2f}%")
        elif momentum_1h_pct < -moderate_threshold:
            bearish_signals += 1
            reasons.append(f"⚡ 1H Negative momentum: {momentum_1h_pct:+.2f}%")
        
        momentum_5h_pct = tech_analysis.get('momentum_5h_pct', 0)
        weight_5h = config.get('momentum_5h_weight', 2)
        
        if momentum_5h_pct > strong_threshold:
            bullish_signals += weight_5h
            reasons.append(f"📊 5H Strong upward trend: {momentum_5h_pct:+.2f}%")
        elif momentum_5h_pct > moderate_threshold:
            bullish_signals += 1
            reasons.append(f"📊 5H Positive trend: {momentum_5h_pct:+.2f}%")
        elif momentum_5h_pct < -strong_threshold:
            bearish_signals += weight_5h
            reasons.append(f"📊 5H Strong downward trend: {momentum_5h_pct:+.2f}%")
        elif momentum_5h_pct < -moderate_threshold:
            bearish_signals += 1
            reasons.append(f"📊 5H Negative trend: {momentum_5h_pct:+.2f}%")
        
        # EMA SIGNALS
        current_price = tech_analysis.get('current_price', 0)
        ema_20 = tech_analysis.get('ema_20', 0)
        ema_50 = tech_analysis.get('ema_50', 0)
        
        if current_price > ema_20 > ema_50:
            bullish_signals += 1
            reasons.append(f"📈 Price above both EMAs (bullish alignment)")
        elif current_price < ema_20 < ema_50:
            bearish_signals += 1
            reasons.append(f"📉 Price below both EMAs (bearish alignment)")
        
        # FINAL DECISION
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
        
        # Strike Recommendations
        spot_price = oc_analysis.get('spot_price', tech_analysis.get('current_price', 25000))
        strike_recommendations = self.get_strike_recommendations(
            recommendation, spot_price, oc_analysis
        )
        
        return {
            'recommendation': recommendation,
            'bias': bias,
            'confidence': confidence,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'net_score': net_score,
            'reasons': reasons,
            'strike_recommendations': strike_recommendations
        }
    
    def get_strike_recommendations(self, recommendation, spot_price, oc_analysis):
        """Get specific strike price recommendations"""
        recommendations = []
        
        atm_strike = round(spot_price / 50) * 50
        
        if 'BUY' in recommendation:
            call_strike = atm_strike
            call_data = self.find_strike_data(oc_analysis, call_strike, 'CE')
            
            if call_data:
                recommendations.append({
                    'action': 'BUY CALL',
                    'strike': call_strike,
                    'ltp': call_data['ltp'],
                    'oi': call_data['oi'],
                    'volume': call_data.get('volume', 'N/A'),
                    'target_1': call_data['ltp'] * 1.20,
                    'target_2': call_data['ltp'] * 1.50,
                    'stop_loss': call_data['ltp'] * 0.80,
                    'profit_at_target_1': call_data['ltp'] * 0.20,
                    'profit_at_target_2': call_data['ltp'] * 0.50,
                    'max_loss': call_data['ltp'] * 0.20
                })
            
            put_strike = atm_strike - 100
            put_data = self.find_strike_data(oc_analysis, put_strike, 'PE')
            
            if put_data:
                recommendations.append({
                    'action': 'SELL PUT',
                    'strike': put_strike,
                    'ltp': put_data['ltp'],
                    'oi': put_data['oi'],
                    'volume': put_data.get('volume', 'N/A'),
                    'target_1': put_data['ltp'] * 0.70,
                    'target_2': put_data['ltp'] * 0.40,
                    'stop_loss': put_data['ltp'] * 1.50,
                    'profit_at_target_1': put_data['ltp'] * 0.30,
                    'profit_at_target_2': put_data['ltp'] * 0.60,
                    'max_loss': put_data['ltp'] * 0.50
                })
        
        elif 'SELL' in recommendation:
            put_strike = atm_strike
            put_data = self.find_strike_data(oc_analysis, put_strike, 'PE')
            
            if put_data:
                recommendations.append({
                    'action': 'BUY PUT',
                    'strike': put_strike,
                    'ltp': put_data['ltp'],
                    'oi': put_data['oi'],
                    'volume': put_data.get('volume', 'N/A'),
                    'target_1': put_data['ltp'] * 1.20,
                    'target_2': put_data['ltp'] * 1.50,
                    'stop_loss': put_data['ltp'] * 0.80,
                    'profit_at_target_1': put_data['ltp'] * 0.20,
                    'profit_at_target_2': put_data['ltp'] * 0.50,
                    'max_loss': put_data['ltp'] * 0.20
                })
            
            call_strike = atm_strike + 100
            call_data = self.find_strike_data(oc_analysis, call_strike, 'CE')
            
            if call_data:
                recommendations.append({
                    'action': 'SELL CALL',
                    'strike': call_strike,
                    'ltp': call_data['ltp'],
                    'oi': call_data['oi'],
                    'volume': call_data.get('volume', 'N/A'),
                    'target_1': call_data['ltp'] * 0.70,
                    'target_2': call_data['ltp'] * 0.40,
                    'stop_loss': call_data['ltp'] * 1.50,
                    'profit_at_target_1': call_data['ltp'] * 0.30,
                    'profit_at_target_2': call_data['ltp'] * 0.60,
                    'max_loss': call_data['ltp'] * 0.50
                })
        
        return recommendations
    
    def find_strike_data(self, oc_analysis, strike, option_type):
        """Find data for a specific strike"""
        if option_type == 'CE':
            data_list = oc_analysis.get('top_10_call_oi', [])
        else:
            data_list = oc_analysis.get('top_10_put_oi', [])
        
        for item in data_list:
            if item['strike'] == strike:
                return item
        
        return None
    
    def create_html_report(self, oc_analysis, tech_analysis, recommendation, oi_change_analysis):
        """Create comprehensive HTML report with OI CHANGE ANALYSIS"""
        report_config = self.config['report']
        title = report_config.get('title', 'NIFTY ANALYSIS')
        
        ist_time = self.get_ist_time()
        timestamp = ist_time.strftime('%Y-%m-%d %H:%M:%S IST')
        
        spot_price = oc_analysis.get('spot_price', tech_analysis.get('current_price', 0))
        
        # Get momentum values and colors
        momentum_1h_pct = tech_analysis.get('price_change_pct_1h', 0)
        momentum_1h_signal = tech_analysis.get('momentum_1h_signal', 'Sideways')
        momentum_1h_colors = tech_analysis.get('momentum_1h_colors', {
            'bg': '#6c757d', 'bg_dark': '#5a6268', 'text': '#ffffff', 'border': '#495057'
        })
        
        momentum_5h_pct = tech_analysis.get('momentum_5h_pct', 0)
        momentum_5h_signal = tech_analysis.get('momentum_5h_signal', 'Sideways')
        momentum_5h_colors = tech_analysis.get('momentum_5h_colors', {
            'bg': '#6c757d', 'bg_dark': '#5a6268', 'text': '#ffffff', 'border': '#495057'
        })
        
        # OI Change colors
        oi_direction = oi_change_analysis.get('direction', 'Neutral')
        if 'Strong Bullish' in oi_direction:
            oi_color = '#1e7e34'
        elif 'Bullish' in oi_direction or 'Moderately Bullish' in oi_direction:
            oi_color = '#28a745'
        elif 'Strong Bearish' in oi_direction:
            oi_color = '#bd2130'
        elif 'Bearish' in oi_direction or 'Moderately Bearish' in oi_direction:
            oi_color = '#dc3545'
        else:
            oi_color = '#6c757d'
        
        rec_color_map = {
            'STRONG BUY': '#1e7e34',
            'BUY': '#28a745',
            'HOLD / NEUTRAL': '#6c757d',
            'SELL': '#dc3545',
            'STRONG SELL': '#bd2130'
        }
        rec_color = rec_color_map.get(recommendation['recommendation'], '#6c757d')
        
        pcr_color_map = {
            'Very Bullish': '#1e7e34',
            'Bullish': '#28a745',
            'Neutral': '#6c757d',
            'Bearish': '#dc3545',
            'Very Bearish': '#bd2130'
        }
        pcr_color = pcr_color_map.get(oc_analysis.get('pcr_signal', 'Neutral'), '#6c757d')
        
        rsi = tech_analysis.get('rsi', 50)
        rsi_color = '#bd2130' if rsi > 70 else ('#1e7e34' if rsi < 30 else '#6c757d')
        
        # Top 10 OI tables HTML
        top_call_oi_html = ""
        for idx, strike in enumerate(oc_analysis.get('top_10_call_oi', [])[:10], 1):
            top_call_oi_html += f"""
                <tr>
                    <td>{idx}</td>
                    <td><strong>₹{strike['strike']}</strong></td>
                    <td>{strike['oi']:,}</td>
                    <td class="premium">₹{strike['ltp']:.2f}</td>
                </tr>
            """
        
        top_put_oi_html = ""
        for idx, strike in enumerate(oc_analysis.get('top_10_put_oi', [])[:10], 1):
            top_put_oi_html += f"""
                <tr>
                    <td>{idx}</td>
                    <td><strong>₹{strike['strike']}</strong></td>
                    <td>{strike['oi']:,}</td>
                    <td class="premium">₹{strike['ltp']:.2f}</td>
                </tr>
            """
        
        # Strategies HTML
        if 'Bullish' in recommendation['bias']:
            strategies_html = """
                <div class="strategy-box">
                    <h4>📈 Bull Call Spread</h4>
                    <p>Buy ATM Call + Sell OTM Call</p>
                </div>
                <div class="strategy-box">
                    <h4>🎯 Long Call</h4>
                    <p>Buy ATM or slightly OTM Call</p>
                </div>
                <div class="strategy-box">
                    <h4>💰 Cash Secured Put</h4>
                    <p>Sell OTM Put (collect premium)</p>
                </div>
            """
        elif 'Bearish' in recommendation['bias']:
            strategies_html = """
                <div class="strategy-box">
                    <h4>📉 Bear Put Spread</h4>
                    <p>Buy ATM Put + Sell OTM Put</p>
                </div>
                <div class="strategy-box">
                    <h4>🎯 Long Put</h4>
                    <p>Buy ATM or slightly OTM Put</p>
                </div>
                <div class="strategy-box">
                    <h4>💰 Covered Call</h4>
                    <p>Sell OTM Call (collect premium)</p>
                </div>
            """
        else:
            strategies_html = """
                <div class="strategy-box">
                    <h4>⚖️ Iron Condor</h4>
                    <p>Sell OTM Call + Put, Buy further OTM protection</p>
                </div>
                <div class="strategy-box">
                    <h4>🦋 Butterfly Spread</h4>
                    <p>Profit from low volatility</p>
                </div>
                <div class="strategy-box">
                    <h4>🔄 Straddle/Strangle</h4>
                    <p>Profit from high volatility (either direction)</p>
                </div>
            """
        
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
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            color: #e0e0e0;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: #27272a;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #3f3f46 0%, #52525b 100%);
            padding: 30px 40px;
            border-bottom: 3px solid #71717a;
        }}
        
        .header h1 {{
            font-size: 32px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 8px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}
        
        .timestamp {{
            font-size: 14px;
            color: #d4d4d8;
            font-weight: 500;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            background: #3f3f46;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid #52525b;
        }}
        
        .section-title {{
            font-size: 22px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid #71717a;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}
        
        .stat-box {{
            background: linear-gradient(135deg, #52525b 0%, #71717a 100%);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #a1a1aa;
            text-align: center;
        }}
        
        .stat-label {{
            font-size: 13px;
            color: #d4d4d8;
            margin-bottom: 8px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .stat-value {{
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
        }}
        
        .oi-sentiment-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }}
        
        @media (max-width: 768px) {{
            .oi-sentiment-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .oi-change-box {{
            background: linear-gradient(135deg, {oi_color} 0%, {oi_color}dd 100%);
            padding: 25px;
            border-radius: 12px;
            border: 2px solid {oi_color};
        }}
        
        .oi-change-box h3 {{
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 15px;
        }}
        
        .oi-change-stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
        }}
        
        .oi-stat {{
            background: rgba(0, 0, 0, 0.2);
            padding: 12px;
            border-radius: 8px;
        }}
        
        .oi-stat-label {{
            font-size: 12px;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 5px;
        }}
        
        .oi-stat-value {{
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
        }}
        
        .oi-explanation {{
            background: rgba(0, 0, 0, 0.2);
            padding: 12px;
            border-radius: 8px;
            font-size: 13px;
            color: #ffffff;
            line-height: 1.5;
        }}
        
        .momentum-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }}
        
        .momentum-box {{
            background: linear-gradient(135deg, var(--momentum-bg) 0%, var(--momentum-bg-dark) 100%);
            padding: 20px;
            border-radius: 10px;
            border: 2px solid var(--momentum-border);
            text-align: center;
        }}
        
        .momentum-box h3 {{
            font-size: 14px;
            color: var(--momentum-text);
            margin-bottom: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .momentum-box .value {{
            font-size: 32px;
            font-weight: 700;
            color: var(--momentum-text);
            margin-bottom: 8px;
        }}
        
        .momentum-box .signal {{
            font-size: 13px;
            color: var(--momentum-text);
            font-weight: 600;
            opacity: 0.95;
        }}
        
        .recommendation-box {{
            background: linear-gradient(135deg, {rec_color} 0%, {rec_color}dd 100%);
            padding: 30px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 25px;
            border: 2px solid {rec_color};
        }}
        
        .recommendation-box h2 {{
            font-size: 36px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}
        
        .recommendation-details {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        
        .recommendation-details div {{
            font-size: 15px;
            color: #ffffff;
            font-weight: 600;
        }}
        
        .reasons-list {{
            background: #52525b;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #71717a;
        }}
        
        .reasons-list ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .reasons-list li {{
            padding: 10px 0;
            border-bottom: 1px solid #71717a;
            color: #e0e0e0;
            font-size: 15px;
        }}
        
        .reasons-list li:last-child {{
            border-bottom: none;
        }}
        
        .oi-tables {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }}
        
        @media (max-width: 768px) {{
            .oi-tables {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .oi-table-container {{
            background: #52525b;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #71717a;
        }}
        
        .oi-table-container h3 {{
            font-size: 18px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #71717a;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            background: #71717a;
            padding: 12px 10px;
            text-align: left;
            font-weight: 700;
            color: #ffffff;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        td {{
            padding: 10px;
            border-bottom: 1px solid #71717a;
            color: #e0e0e0;
            font-size: 14px;
        }}
        
        tr:last-child td {{
            border-bottom: none;
        }}
        
        tr:hover {{
            background: #63636b;
        }}
        
        .premium {{
            color: #fbbf24;
            font-weight: 700;
        }}
        
        .strike-recommendations {{
            margin-top: 20px;
        }}
        
        .strike-card {{
            background: #52525b;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            border: 2px solid #71717a;
        }}
        
        .strike-card h4 {{
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 15px;
        }}
        
        .strike-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .strike-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #71717a;
        }}
        
        .strike-row:last-child {{
            border-bottom: none;
        }}
        
        .strike-row .label {{
            color: #d4d4d8;
            font-weight: 600;
            font-size: 14px;
        }}
        
        .strike-row .value {{
            color: #ffffff;
            font-weight: 700;
            font-size: 14px;
        }}
        
        .profit-targets {{
            background: #63636b;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }}
        
        .profit-targets h5 {{
            font-size: 16px;
            color: #ffffff;
            margin-bottom: 12px;
            font-weight: 700;
        }}
        
        .target-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }}
        
        @media (max-width: 600px) {{
            .target-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .target-box {{
            background: #71717a;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }}
        
        .target-1 {{
            border: 2px solid #fbbf24;
        }}
        
        .target-2 {{
            border: 2px solid #10b981;
        }}
        
        .stop-loss-box {{
            border: 2px solid #ef4444;
        }}
        
        .target-label {{
            font-size: 12px;
            color: #d4d4d8;
            font-weight: 600;
            margin-bottom: 6px;
        }}
        
        .target-price {{
            font-size: 18px;
            color: #ffffff;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        
        .target-profit {{
            font-size: 12px;
            color: #d4d4d8;
            font-weight: 600;
        }}
        
        .trade-example {{
            background: #52525b;
            border-radius: 6px;
            padding: 12px;
            margin-top: 12px;
            font-size: 13px;
            color: #e0e0e0;
            border-left: 4px solid #3b82f6;
        }}
        
        .no-recommendations {{
            background: #52525b;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            color: #fbbf24;
            border: 2px dashed #71717a;
        }}
        
        .strategies-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }}
        
        .strategy-box {{
            background: linear-gradient(135deg, #52525b 0%, #71717a 100%);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #a1a1aa;
            text-align: center;
        }}
        
        .strategy-box h4 {{
            font-size: 18px;
            color: #ffffff;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .strategy-box p {{
            font-size: 14px;
            color: #d4d4d8;
        }}
        
        .footer {{
            background: #3f3f46;
            padding: 25px 40px;
            text-align: center;
            border-top: 3px solid #71717a;
            color: #a1a1aa;
            font-size: 13px;
        }}
        
        .footer p {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {title} + OI CHANGE ANALYSIS</h1>
            <div class="timestamp">Generated on: {timestamp}</div>
        </div>
        
        <div class="content">
            <!-- MARKET OVERVIEW -->
            <div class="section">
                <div class="section-title">📈 Market Overview</div>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-label">Nifty Spot</div>
                        <div class="stat-value">₹{spot_price:,.2f}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">RSI (1H)</div>
                        <div class="stat-value" style="color: {rsi_color};">{tech_analysis.get('rsi', 'N/A')}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">PCR Ratio</div>
                        <div class="stat-value" style="color: {pcr_color};">{oc_analysis.get('pcr', 'N/A')}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">PCR Signal</div>
                        <div class="stat-value" style="color: {pcr_color}; font-size: 20px;">{oc_analysis.get('pcr_signal', 'N/A')}</div>
                    </div>
                </div>
            </div>
            
            <!-- OI SENTIMENT & OI CHANGE ANALYSIS -->
            <div class="section">
                <div class="section-title">🔥 Option Chain Sentiment Analysis</div>
                <div class="oi-sentiment-grid">
                    <!-- PCR SENTIMENT BOX -->
                    <div class="stat-box" style="background: linear-gradient(135deg, {pcr_color} 0%, {pcr_color}dd 100%); border: 2px solid {pcr_color};">
                        <h3 style="color: #ffffff; font-size: 18px; margin-bottom: 15px;">📊 PCR Sentiment</h3>
                        <div class="stat-value" style="font-size: 36px; margin-bottom: 10px;">{oc_analysis.get('pcr', 'N/A')}</div>
                        <div style="font-size: 16px; color: #ffffff; font-weight: 600;">{oc_analysis.get('pcr_signal', 'N/A')}</div>
                    </div>
                    
                    <!-- OI CHANGE ANALYSIS BOX -->
                    <div class="oi-change-box">
                        <h3>📊 OI Change Analysis (Market Direction)</h3>
                        <div class="oi-change-stats">
                            <div class="oi-stat">
                                <div class="oi-stat-label">Call OI Change</div>
                                <div class="oi-stat-value">{oi_change_analysis.get('total_call_oi_change_pct', 0):+.2f}%</div>
                            </div>
                            <div class="oi-stat">
                                <div class="oi-stat-label">Put OI Change</div>
                                <div class="oi-stat-value">{oi_change_analysis.get('total_put_oi_change_pct', 0):+.2f}%</div>
                            </div>
                            <div class="oi-stat">
                                <div class="oi-stat-label">ATM Call Change</div>
                                <div class="oi-stat-value">{oi_change_analysis.get('atm_call_oi_change_pct', 0):+.2f}%</div>
                            </div>
                            <div class="oi-stat">
                                <div class="oi-stat-label">ATM Put Change</div>
                                <div class="oi-stat-value">{oi_change_analysis.get('atm_put_oi_change_pct', 0):+.2f}%</div>
                            </div>
                        </div>
                        <div style="background: rgba(0, 0, 0, 0.2); padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                            <div style="font-size: 14px; color: rgba(255, 255, 255, 0.8); margin-bottom: 5px;">Direction Prediction:</div>
                            <div style="font-size: 24px; font-weight: 700; color: #ffffff;">{oi_change_analysis.get('direction', 'N/A')}</div>
                            <div style="font-size: 14px; color: rgba(255, 255, 255, 0.9); margin-top: 5px;">Confidence: {oi_change_analysis.get('confidence', 'N/A')}</div>
                        </div>
                        <div class="oi-explanation">
                            {oi_change_analysis.get('explanation', 'N/A')}
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- DUAL MOMENTUM DISPLAY -->
            <div class="section">
                <div class="section-title">⚡ Dual Momentum Analysis (1H + 5H)</div>
                <div class="momentum-container">
                    <div class="momentum-box" style="--momentum-bg: {momentum_1h_colors['bg']}; --momentum-bg-dark: {momentum_1h_colors['bg_dark']}; --momentum-text: {momentum_1h_colors['text']}; --momentum-border: {momentum_1h_colors['border']};">
                        <h3>⚡ 1 HOUR MOMENTUM</h3>
                        <div class="value">{momentum_1h_pct:+.2f}%</div>
                        <div class="signal">{momentum_1h_signal}</div>
                    </div>
                    <div class="momentum-box" style="--momentum-bg: {momentum_5h_colors['bg']}; --momentum-bg-dark: {momentum_5h_colors['bg_dark']}; --momentum-text: {momentum_5h_colors['text']}; --momentum-border: {momentum_5h_colors['border']};">
                        <h3>📊 5 HOUR MOMENTUM</h3>
                        <div class="value">{momentum_5h_pct:+.2f}%</div>
                        <div class="signal">{momentum_5h_signal}</div>
                    </div>
                </div>
            </div>
            
            <!-- PIVOT POINTS -->
            <div class="section">
                <div class="section-title">📍 Pivot Points (Traditional)</div>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-label">R3</div>
                        <div class="stat-value">₹{tech_analysis.get('pivot_points', {}).get('r3', 'N/A')}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">R2</div>
                        <div class="stat-value">₹{tech_analysis.get('pivot_points', {}).get('r2', 'N/A')}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">R1</div>
                        <div class="stat-value">₹{tech_analysis.get('pivot_points', {}).get('r1', 'N/A')}</div>
                    </div>
                    <div class="stat-box" style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);">
                        <div class="stat-label">PIVOT</div>
                        <div class="stat-value">₹{tech_analysis.get('pivot_points', {}).get('pivot', 'N/A')}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">S1</div>
                        <div class="stat-value">₹{tech_analysis.get('pivot_points', {}).get('s1', 'N/A')}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">S2</div>
                        <div class="stat-value">₹{tech_analysis.get('pivot_points', {}).get('s2', 'N/A')}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">S3</div>
                        <div class="stat-value">₹{tech_analysis.get('pivot_points', {}).get('s3', 'N/A')}</div>
                    </div>
                </div>
            </div>
            
            <!-- RECOMMENDATION -->
            <div class="section">
                <div class="section-title">🎯 Trading Recommendation (with OI Change Weight)</div>
                <div class="recommendation-box">
                    <h2>{recommendation['recommendation']}</h2>
                    <div class="recommendation-details">
                        <div>📊 Bias: {recommendation['bias']}</div>
                        <div>🎯 Confidence: {recommendation['confidence']}</div>
                        <div>📈 Score: {recommendation['bullish_signals']} Bullish | {recommendation['bearish_signals']} Bearish</div>
                    </div>
                </div>
                
                <div class="reasons-list">
                    <h4 style="margin-bottom: 15px; color: #ffffff; font-size: 18px;">📋 Analysis Reasons (OI Change Weighted):</h4>
                    <ul>
        """
        
        for reason in recommendation.get('reasons', []):
            html += f"<li>{reason}</li>\n"
        
        html += """
                    </ul>
                </div>
            </div>
            
            <!-- TOP 10 OI STRIKES -->
            <div class="section">
                <div class="section-title">🔥 Top 10 Strikes by Open Interest</div>
                <div class="oi-tables">
                    <div class="oi-table-container">
                        <h3>📞 Top 10 CALL OI</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Strike</th>
                                    <th>OI</th>
                                    <th>LTP</th>
                                </tr>
                            </thead>
                            <tbody>
        """
        html += top_call_oi_html
        html += """
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="oi-table-container">
                        <h3>📉 Top 10 PUT OI</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Strike</th>
                                    <th>OI</th>
                                    <th>LTP</th>
                                </tr>
                            </thead>
                            <tbody>
        """
        html += top_put_oi_html
        html += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <!-- STRIKE RECOMMENDATIONS -->
            <div class="section">
                <div class="section-title">💡 Specific Strike Recommendations</div>
        """
        
        if recommendation.get('strike_recommendations'):
            for rec in recommendation['strike_recommendations']:
                html += f"""
                <div class="strike-card">
                    <h4 style="color: {'#28a745' if 'BUY' in rec['action'] else '#dc3545'};">{rec['action']}</h4>
                    
                    <div class="strike-details">
                        <div class="strike-row">
                            <span class="label">Action:</span>
                            <span class="value"><strong>{rec['action']}</strong></span>
                        </div>
                        <div class="strike-row">
                            <span class="label">Strike Price:</span>
                            <span class="value"><strong>₹{rec['strike']}</strong></span>
                        </div>
                        <div class="strike-row">
                            <span class="label">Current LTP:</span>
                            <span class="value premium">₹{rec['ltp']:.2f}</span>
                        </div>
                        <div class="strike-row">
                            <span class="label">Open Interest:</span>
                            <span class="value">{rec['oi']}</span>
                        </div>
                        <div class="strike-row">
                            <span class="label">Volume:</span>
                            <span class="value">{rec['volume']}</span>
                        </div>
                    </div>
                    
                    <div class="profit-targets">
                        <h5>📊 Profit Targets & Risk</h5>
                        <div class="target-grid">
                            <div class="target-box target-1">
                                <div class="target-label">Target 1</div>
                                <div class="target-price">₹{rec['target_1']}</div>
                                <div class="target-profit">Profit: ₹{rec['profit_at_target_1']:.2f}</div>
                            </div>
                            <div class="target-box target-2">
                                <div class="target-label">Target 2</div>
                                <div class="target-price">₹{rec['target_2']}</div>
                                <div class="target-profit">{f"Profit: ₹{rec['profit_at_target_2']:.2f}" if isinstance(rec['profit_at_target_2'], (int, float)) else rec['profit_at_target_2']}</div>
                            </div>
                            <div class="target-box stop-loss-box">
                                <div class="target-label">Stop Loss</div>
                                <div class="target-price">₹{rec['stop_loss']:.2f}</div>
                                <div class="target-profit">Max Loss: ₹{rec['max_loss']:.2f}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="trade-example">
                        <strong>Example:</strong> If you buy 1 lot (50 qty) at LTP ₹{rec['ltp']:.2f}, your investment = ₹{rec['ltp'] * 50:.0f}<br>
                        At Target 1: Profit = ₹{rec['profit_at_target_1'] * 50 if isinstance(rec['profit_at_target_1'], (int, float)) else 'Variable':.0f} | At Target 2: Profit = ₹{rec['profit_at_target_2'] * 50 if isinstance(rec['profit_at_target_2'], (int, float)) else 'Variable':.0f}
                    </div>
                </div>
                """
        else:
            html += """
                <div class="no-recommendations">
                    <p><strong>⚠️ No specific strike recommendations available at this time.</strong><br>Check the general strategies below.</p>
                </div>
            """
        
        html += f"""
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">🎯 Options Strategies</div>
            <p style="color: #a1a1aa; margin-bottom: 15px; font-size: 13px;">Based on {recommendation['bias']} bias:</p>
            <div class="strategies-grid">{strategies_html}</div>
        </div>
        
        <div class="footer">
            <p><strong>Disclaimer:</strong> This analysis is for educational purposes only. Trading involves risk. Past performance is not indicative of future results.</p>
            <p>© 2025 Nifty Trading Analyzer | Dual Momentum + OI Change Analysis | Professional Edition</p>
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
            msg['Subject'] = f"{subject_prefix} + OI Change - {subject_time}"
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
        """Run complete analysis with OI CHANGE DETECTION"""
        self.logger.info("🚀 Starting Nifty 1-HOUR Analysis with OI Change Detection...")
        self.logger.info("=" * 60)
        
        oc_df, spot_price = self.fetch_option_chain()
        
        if oc_df is not None and spot_price is not None:
            oc_analysis = self.analyze_option_chain(oc_df, spot_price)
            oi_change_analysis = self.analyze_oi_change(oc_df, spot_price)
        else:
            spot_price = 25796
            oc_analysis = self.get_sample_oc_analysis()
            oi_change_analysis = self.get_default_oi_change_analysis()
        
        tech_df = self.fetch_technical_data()
        
        if tech_df is not None and not tech_df.empty:
            tech_analysis = self.technical_analysis(tech_df)
        else:
            tech_analysis = self.get_sample_tech_analysis()
        
        self.logger.info("🎯 Generating Trading Recommendation with OI Change...")
        recommendation = self.generate_recommendation(oc_analysis, tech_analysis, oi_change_analysis)
        
        self.logger.info("=" * 60)
        self.logger.info(f"📊 RECOMMENDATION: {recommendation['recommendation']}")
        self.logger.info(f"📈 Bias: {recommendation['bias']} | Confidence: {recommendation['confidence']}")
        self.logger.info(f"🔥 OI Direction: {oi_change_analysis.get('direction')} ({oi_change_analysis.get('confidence')})")
        self.logger.info(f"🎯 RSI (1H): {tech_analysis.get('rsi', 'N/A')}")
        self.logger.info(f"⚡ 1H Momentum: {tech_analysis.get('price_change_pct_1h', 0):+.2f}% - {tech_analysis.get('momentum_1h_signal')}")
        self.logger.info(f"📊 5H Momentum: {tech_analysis.get('momentum_5h_pct', 0):+.2f}% - {tech_analysis.get('momentum_5h_signal')}")
        self.logger.info(f"📍 Pivot Point: ₹{tech_analysis.get('pivot_points', {}).get('pivot', 'N/A')}")
        self.logger.info("=" * 60)
        
        html_report = self.create_html_report(oc_analysis, tech_analysis, recommendation, oi_change_analysis)
        
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
        
        self.logger.info("✅ OI Change Analysis Complete!")
        
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
    print(f"OI Direction: {result['oi_change_analysis']['direction']} (Confidence: {result['oi_change_analysis']['confidence']})")
    print(f"RSI (1H): {result['tech_analysis']['rsi']}")
    print(f"1H Momentum: {result['tech_analysis']['price_change_pct_1h']:+.2f}% - {result['tech_analysis']['momentum_1h_signal']}")
    print(f"5H Momentum: {result['tech_analysis']['momentum_5h_pct']:+.2f}% - {result['tech_analysis']['momentum_5h_signal']}")
    print(f"Check your email for the detailed report!")
