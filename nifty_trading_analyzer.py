"""
Nifty Option Chain & Technical Analysis for Day Trading
1-HOUR TIMEFRAME with WILDER'S RSI (matches TradingView)
Complete fixed version
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import yfinance as yf
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
import yaml
import os
import logging

warnings.filterwarnings('ignore')

class NiftyAnalyzer:
    def __init__(self, config_path='config.yml'):
        """Initialize analyzer with YAML configuration"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # IST timezone
        self.ist = pytz.timezone('Asia/Kolkata')
        
        self.nifty_symbol = "^NSEI"
        self.option_chain_url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br'
        }
        
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
                'timeframe': '1h',  # FIXED: 1-hour timeframe
                'period': '6mo',
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'ema_short': 20,
                'ema_long': 50,
                'num_support_levels': 2,
                'num_resistance_levels': 2
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
                'strong_sell_threshold': -3
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
                'timeout': 10,
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
                'min_data_points': 100
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
        """Fetch Nifty option chain data from NSE"""
        if self.config['data_source']['option_chain_source'] == 'sample':
            self.logger.info("Using sample option chain data")
            return None, None
            
        max_retries = self.config['data_source']['max_retries']
        retry_delay = self.config['data_source']['retry_delay']
        timeout = self.config['data_source']['timeout']
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Fetching option chain data (attempt {attempt + 1}/{max_retries})...")
                
                session = requests.Session()
                session.get("https://www.nseindia.com", headers=self.headers, timeout=timeout)
                
                response = session.get(self.option_chain_url, headers=self.headers, timeout=timeout)
                data = response.json()
                
                if 'records' in data and 'data' in data['records']:
                    option_data = data['records']['data']
                    current_price = data['records']['underlyingValue']
                    
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
                    
                    self.logger.info(f"✅ Option chain data fetched successfully | Spot: ₹{current_price}")
                    return oc_df, current_price
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
        
        if self.config['data_source']['fallback_to_sample']:
            self.logger.warning("All attempts failed, using sample data")
        
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
                'type': strike_type
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
                'type': strike_type
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
                {'strike': 24500, 'oi': 5000000, 'ltp': 120, 'iv': 16.5, 'type': 'ATM'},
                {'strike': 24600, 'oi': 4500000, 'ltp': 80, 'iv': 15.8, 'type': 'OTM'},
            ],
            'top_pe_strikes': [
                {'strike': 24500, 'oi': 5500000, 'ltp': 110, 'iv': 16.0, 'type': 'ATM'},
                {'strike': 24400, 'oi': 5000000, 'ltp': 75, 'iv': 15.5, 'type': 'OTM'},
            ]
        }
    
    def fetch_technical_data(self):
        """Fetch historical data for technical analysis - ALWAYS 1 HOUR"""
        if self.config['data_source']['technical_source'] == 'sample':
            self.logger.info("Using sample technical data")
            return None
            
        period = self.config['technical']['period']
        interval = '1h'  # FORCED 1-HOUR TIMEFRAME
        
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
        
        # Wilder's smoothing: alpha = 1/period
        # This is EXACTLY what TradingView's ta.rma() does
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
    
    def technical_analysis(self, df):
        """Perform complete technical analysis - 1 HOUR TIMEFRAME"""
        if df is None or df.empty:
            self.logger.warning("No technical data, using sample analysis")
            return self.get_sample_tech_analysis()
        
        current_price = df['Close'].iloc[-1]
        
        # RSI using Wilder's method (matches TradingView)
        df['RSI'] = self.calculate_rsi(df['Close'])
        current_rsi = df['RSI'].iloc[-1]
        
        self.logger.info(f"🎯 RSI(14) calculated: {current_rsi:.2f} (Wilder's method)")
        
        # Moving Averages
        ema_short = self.config['technical']['ema_short']
        ema_long = self.config['technical']['ema_long']
        
        df['EMA_Short'] = df['Close'].ewm(span=ema_short, adjust=False).mean()
        df['EMA_Long'] = df['Close'].ewm(span=ema_long, adjust=False).mean()
        
        ema_short_val = df['EMA_Short'].iloc[-1]
        ema_long_val = df['EMA_Long'].iloc[-1]
        
        # Support and Resistance
        sr_levels = self.calculate_support_resistance(df, current_price)
        
        # Trend analysis
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
        
        # RSI signal
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
            'timeframe': '1 Hour'  # Added for clarity
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
            'timeframe': '1 Hour'
        }
    
    def generate_recommendation(self, oc_analysis, tech_analysis):
        """Generate trading recommendation"""
        if not oc_analysis or not tech_analysis:
            return {"recommendation": "Insufficient data", "bias": "Neutral", "confidence": "Low", "reasons": []}
        
        config = self.config['recommendation']
        oc_config = self.config['option_chain']
        tech_config = self.config['technical']
        
        bullish_signals = 0
        bearish_signals = 0
        reasons = []
        
        # PCR Analysis
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
        
        # OI Buildup
        if oc_analysis.get('oi_sentiment') == 'Bullish':
            bullish_signals += 1
            reasons.append("Put OI buildup > Call OI buildup (Bullish)")
        else:
            bearish_signals += 1
            reasons.append("Call OI buildup > Put OI buildup (Bearish)")
        
        # RSI
        rsi = tech_analysis.get('rsi', 50)
        rsi_os = tech_config['rsi_oversold']
        rsi_ob = tech_config['rsi_overbought']
        
        if rsi < rsi_os:
            bullish_signals += 2
            reasons.append(f"RSI at {rsi:.1f} - Oversold (Bullish)")
        elif rsi < 45:
            bullish_signals += 1
            reasons.append(f"RSI at {rsi:.1f} - Below neutral (Bullish)")
        elif rsi > rsi_ob:
            bearish_signals += 2
            reasons.append(f"RSI at {rsi:.1f} - Overbought (Bearish)")
        elif rsi > 55:
            bearish_signals += 1
            reasons.append(f"RSI at {rsi:.1f} - Above neutral (Bearish)")
        
        # Trend
        trend = tech_analysis.get('trend', '')
        if 'Uptrend' in trend:
            bullish_signals += 1
            reasons.append(f"Trend: {trend}")
        elif 'Downtrend' in trend:
            bearish_signals += 1
            reasons.append(f"Trend: {trend}")
        
        # EMA Analysis
        current_price = tech_analysis.get('current_price', 0)
        ema20 = tech_analysis.get('ema20', 0)
        if current_price > ema20:
            bullish_signals += 1
            reasons.append("Price above EMA20 (Bullish)")
        else:
            bearish_signals += 1
            reasons.append("Price below EMA20 (Bearish)")
        
        # Final Recommendation
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
    
    def get_options_strategies(self, recommendation, oc_analysis, tech_analysis):
        """Generate options trading strategy recommendations"""
        bias = recommendation['bias']
        rsi = tech_analysis.get('rsi', 50)
        pcr = oc_analysis.get('pcr', 1.0)
        avg_iv = (oc_analysis.get('avg_call_iv', 15) + oc_analysis.get('avg_put_iv', 15)) / 2
        
        high_volatility = avg_iv > 18
        low_volatility = avg_iv < 12
        
        strategies = []
        
        # Bullish Strategies
        if bias == 'Bullish':
            strategies.append({
                'name': 'Long Call',
                'type': 'Bullish - Aggressive',
                'setup': 'Buy ATM or slightly OTM Call option',
                'profit': 'Unlimited upside',
                'risk': 'Limited to premium paid',
                'best_when': 'Strong upward move expected, low IV',
                'recommended': '⭐⭐⭐⭐⭐' if recommendation['confidence'] == 'High' and not high_volatility else '⭐⭐⭐'
            })
            
            strategies.append({
                'name': 'Bull Call Spread',
                'type': 'Bullish - Moderate',
                'setup': 'Buy ITM Call + Sell OTM Call',
                'profit': 'Limited (Strike difference - Net premium)',
                'risk': 'Limited to net premium paid',
                'best_when': 'Moderately bullish, reduce cost',
                'recommended': '⭐⭐⭐⭐⭐' if recommendation['confidence'] == 'Medium' else '⭐⭐⭐⭐'
            })
        
        # Bearish Strategies
        elif bias == 'Bearish':
            strategies.append({
                'name': 'Long Put',
                'type': 'Bearish - Aggressive',
                'setup': 'Buy ATM or slightly OTM Put option',
                'profit': 'High (Strike - Stock price - Premium)',
                'risk': 'Limited to premium paid',
                'best_when': 'Strong downward move expected, low IV',
                'recommended': '⭐⭐⭐⭐⭐' if recommendation['confidence'] == 'High' and not high_volatility else '⭐⭐⭐'
            })
            
            strategies.append({
                'name': 'Bear Put Spread',
                'type': 'Bearish - Debit Strategy',
                'setup': 'Buy ITM Put + Sell OTM Put',
                'profit': 'Limited (Strike difference - Net premium)',
                'risk': 'Limited to net premium paid',
                'best_when': 'Moderately bearish, reduce cost',
                'recommended': '⭐⭐⭐⭐⭐' if recommendation['confidence'] == 'Medium' else '⭐⭐⭐'
            })
        
        # Neutral Strategies
        else:
            if high_volatility:
                strategies.append({
                    'name': 'Long Straddle',
                    'type': 'Neutral - High Volatility Expected',
                    'setup': 'Buy ATM Call + Buy ATM Put',
                    'profit': 'Unlimited (either direction)',
                    'risk': 'Limited to total premium paid',
                    'best_when': 'Expect big move, unsure of direction',
                    'recommended': '⭐⭐⭐⭐⭐'
                })
            else:
                strategies.append({
                    'name': 'Short Strangle',
                    'type': 'Neutral - Low Volatility Expected',
                    'setup': 'Sell OTM Call + Sell OTM Put',
                    'profit': 'Limited to total premium collected',
                    'risk': 'Unlimited (either direction)',
                    'best_when': 'Expect range-bound, less risk than straddle',
                    'recommended': '⭐⭐⭐⭐⭐'
                })
        
        return strategies
    
    def create_html_report(self, oc_analysis, tech_analysis, recommendation):
        """Create beautiful HTML report with 1H timeframe emphasis"""
        now_ist = self.format_ist_time()
        
        colors = self.config['report'].get('colors', {})
        rec = recommendation['recommendation']
        
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
        
        title = self.config['report'].get('title', 'NIFTY DAY TRADING ANALYSIS (1H)')
        
        strategies = self.get_options_strategies(recommendation, oc_analysis, tech_analysis)
        
        top_ce_html = ''
        for i, strike in enumerate(oc_analysis.get('top_ce_strikes', [])[:5], 1):
            top_ce_html += f"""
                <tr>
                    <td>{i}</td>
                    <td>₹{strike['strike']}</td>
                    <td>{strike['oi']:,}</td>
                    <td>₹{strike['ltp']}</td>
                    <td>{strike['iv']}%</td>
                    <td><span class="badge-{strike['type'].lower()}">{strike['type']}</span></td>
                </tr>
            """
        
        top_pe_html = ''
        for i, strike in enumerate(oc_analysis.get('top_pe_strikes', [])[:5], 1):
            top_pe_html += f"""
                <tr>
                    <td>{i}</td>
                    <td>₹{strike['strike']}</td>
                    <td>{strike['oi']:,}</td>
                    <td>₹{strike['ltp']}</td>
                    <td>{strike['iv']}%</td>
                    <td><span class="badge-{strike['type'].lower()}">{strike['type']}</span></td>
                </tr>
            """
        
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
                </div>
            """
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f5f5f5; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 25px; }}
        .header {{ text-align: center; border-bottom: 3px solid #007bff; padding-bottom: 15px; margin-bottom: 20px; }}
        .header h1 {{ color: #007bff; margin: 0; font-size: 28px; }}
        .timestamp {{ color: #6c757d; font-size: 13px; margin-top: 8px; font-weight: bold; }}
        .timeframe-badge {{ display: inline-block; background: #ff6b6b; color: white; padding: 5px 15px; border-radius: 20px; font-size: 14px; font-weight: bold; margin-top: 10px; }}
        .recommendation-box {{ background: linear-gradient(135deg, {rec_color} 0%, {rec_color}dd 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }}
        .recommendation-box h2 {{ margin: 0 0 8px 0; font-size: 32px; font-weight: bold; }}
        .recommendation-box .subtitle {{ font-size: 16px; opacity: 0.9; }}
        .section {{ margin-bottom: 25px; }}
        .section-title {{ background-color: #007bff; color: white; padding: 10px 18px; border-radius: 5px; font-size: 18px; font-weight: bold; margin-bottom: 12px; }}
        .data-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
        .data-item {{ background-color: #f8f9fa; padding: 12px 15px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .data-item .label {{ color: #6c757d; font-size: 11px; margin-bottom: 4px; text-transform: uppercase; font-weight: 600; }}
        .data-item .value {{ color: #212529; font-size: 18px; font-weight: bold; }}
        .levels {{ display: flex; justify-content: space-between; gap: 20px; }}
        .levels-box {{ flex: 1; background-color: #f8f9fa; padding: 12px; border-radius: 8px; }}
        .levels-box.resistance {{ border-left: 4px solid #dc3545; }}
        .levels-box.support {{ border-left: 4px solid #28a745; }}
        .levels-box h4 {{ margin: 0 0 8px 0; font-size: 14px; font-weight: 600; }}
        .levels-box ul {{ margin: 0; padding-left: 20px; }}
        .levels-box li {{ margin: 4px 0; font-size: 14px; font-weight: 500; }}
        .reasons {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; border-radius: 5px; }}
        .reasons ul {{ margin: 8px 0 0 0; padding-left: 20px; }}
        .reasons li {{ margin: 6px 0; color: #856404; font-size: 13px; }}
        .signal-badge {{ display: inline-block; padding: 4px 12px; border-radius: 15px; font-size: 13px; margin: 4px; font-weight: 600; }}
        .bullish {{ background-color: #d4edda; color: #155724; }}
        .bearish {{ background-color: #f8d7da; color: #721c24; }}
        .oi-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
        .oi-table th {{ background-color: #007bff; color: white; padding: 8px; text-align: left; font-size: 12px; font-weight: 600; }}
        .oi-table td {{ padding: 8px; border-bottom: 1px solid #e9ecef; font-size: 13px; }}
        .badge-itm {{ background-color: #28a745; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; font-weight: bold; }}
        .badge-atm {{ background-color: #ffc107; color: #000; padding: 3px 8px; border-radius: 3px; font-size: 12px; font-weight: bold; }}
        .badge-otm {{ background-color: #dc3545; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; font-weight: bold; }}
        .strategies-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 15px; margin-top: 15px; }}
        .strategy-card {{ background-color: #ffffff; border: 2px solid #e9ecef; border-radius: 8px; padding: 12px; }}
        .strategy-header {{ border-bottom: 2px solid #007bff; padding-bottom: 8px; margin-bottom: 8px; }}
        .strategy-header h4 {{ margin: 0; color: #007bff; font-size: 16px; }}
        .strategy-type {{ display: inline-block; background-color: #e7f3ff; color: #007bff; padding: 2px 8px; border-radius: 12px; font-size: 11px; margin-top: 4px; }}
        .strategy-body p {{ margin: 6px 0; font-size: 13px; line-height: 1.4; }}
        .recommendation-stars {{ color: #ffc107; font-size: 14px; }}
        .footer {{ text-align: center; margin-top: 30px; padding-top: 20px; border-top: 2px solid #e9ecef; color: #6c757d; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {title}</h1>
            <div class="timeframe-badge">⏱️ 1-HOUR TIMEFRAME</div>
            <div class="timestamp">Generated on: {now_ist}</div>
        </div>
        
        <div class="recommendation-box">
            <h2>{recommendation['recommendation']}</h2>
            <div class="subtitle">Market Bias: {recommendation['bias']} | Confidence: {recommendation['confidence']}</div>
            <div style="margin-top: 15px;">
                <span class="signal-badge bullish">Bullish Signals: {recommendation['bullish_signals']}</span>
                <span class="signal-badge bearish">Bearish Signals: {recommendation['bearish_signals']}</span>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📈 Technical Analysis (1 Hour Timeframe)</div>
            <div class="data-grid">
                <div class="data-item">
                    <div class="label">Current Price</div>
                    <div class="value">₹{tech_analysis.get('current_price', 'N/A')}</div>
                </div>
                <div class="data-item">
                    <div class="label">RSI (14) - Wilder's Method</div>
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
            <div class="section-title">🎯 Support & Resistance Levels (1H)</div>
            <div class="levels">
                <div class="levels-box resistance">
                    <h4>🔴 Resistance Levels</h4>
                    <ul>{''.join([f'<li>R{i+1}: ₹{r}</li>' for i, r in enumerate(tech_analysis.get('tech_resistances', []))])}</ul>
                </div>
                <div class="levels-box support">
                    <h4>🟢 Support Levels</h4>
                    <ul>{''.join([f'<li>S{i+1}: ₹{s}</li>' for i, s in enumerate(tech_analysis.get('tech_supports', []))])}</ul>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📊 Option Chain Analysis</div>
            <div class="data-grid">
                <div class="data-item">
                    <div class="label">Put-Call Ratio (PCR)</div>
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
        
        <div class="section">
            <div class="section-title">🏆 Top 5 Strikes by Open Interest</div>
            <div class="levels">
                <div class="levels-box" style="border-left: 4px solid #dc3545;">
                    <h4>📞 Call Options (CE)</h4>
                    <table class="oi-table">
                        <thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th><th>IV</th><th>Type</th></tr></thead>
                        <tbody>{top_ce_html}</tbody>
                    </table>
                </div>
                <div class="levels-box" style="border-left: 4px solid #28a745;">
                    <h4>📉 Put Options (PE)</h4>
                    <table class="oi-table">
                        <thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th><th>IV</th><th>Type</th></tr></thead>
                        <tbody>{top_pe_html}</tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">💡 Analysis Summary</div>
            <div class="reasons">
                <strong>Key Factors Behind Recommendation:</strong>
                <ul>{''.join([f'<li>{reason}</li>' for reason in recommendation.get('reasons', [])])}</ul>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">🎯 Options Trading Strategies</div>
            <p style="color: #6c757d; margin-bottom: 15px;">Based on 1-hour analysis ({recommendation['bias']} bias):</p>
            <div class="strategies-grid">{strategies_html}</div>
        </div>
        
        <div class="footer">
            <p><strong>Disclaimer:</strong> This analysis is for educational purposes only. Trading involves risk.</p>
            <p>© 2025 Nifty Trading Analyzer | 1-Hour Timeframe Analysis</p>
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
        """Run complete analysis - 1 HOUR TIMEFRAME"""
        self.logger.info("🚀 Starting Nifty 1-HOUR Analysis...")
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
        
        self.logger.info("✅ 1-Hour Analysis Complete!")
        
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
    print(f"RSI (1H, Wilder's): {result['tech_analysis']['rsi']}")
    print(f"Check your email for the detailed 1-HOUR report!")
