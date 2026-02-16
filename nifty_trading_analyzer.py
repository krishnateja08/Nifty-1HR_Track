#!/usr/bin/env python3
"""
Nifty Option Chain & Technical Analysis for Day Trading
1-HOUR TIMEFRAME with WILDER'S RSI (matches TradingView)
Enhanced with Pivot Points + Dual Momentum Analysis + Top OI Display
EXPIRY: Weekly TUESDAY expiry with 3:30 PM IST cutoff logic
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
        self.option_chain_base_url = (
            "https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol=NIFTY&expiry="
        )

        self.headers = {
            "authority": "www.nseindia.com",
            "accept": "application/json, text/plain, */*",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/121.0.0.0 Safari/537.36",
            "referer": "https://www.nseindia.com/option-chain",
            "accept-language": "en-US,en;q=0.9",
        }

        # placeholders
        self.tech_analysis = {}
        self.oc_analysis = {}
        self.recommendation = {}
        self.pivot_points = {}
        self.support_resistance = {}
        self.current_price = None

    # ---------- Time helpers ----------

    def get_ist_time(self):
        return datetime.now(self.ist)

    def format_ist_time(self, dt=None):
        if dt is None:
            dt = self.get_ist_time()
        elif dt.tzinfo is None:
            dt = self.ist.localize(dt)
        else:
            dt = dt.astimezone(self.ist)
        return dt.strftime("%Y-%m-%d %H:%M:%S IST")

    def get_next_expiry_date(self):
        """
        Weekly Tuesday expiry with 3:30 PM cutoff.
        """
        now_ist = self.get_ist_time()
        current_day = now_ist.weekday()  # 0=Mon,1=Tue,...

        if current_day == 1:
            h, m = now_ist.hour, now_ist.minute
            if h < 15 or (h == 15 and m < 30):
                days_until_tuesday = 0
                self.logger.info("Today is Tuesday before 3:30 PM - Using today as expiry")
            else:
                days_until_tuesday = 7
                self.logger.info("Tuesday after 3:30 PM - Moving to next Tuesday")
        elif current_day == 0:
            days_until_tuesday = 1
        else:
            days_until_tuesday = (1 - current_day) % 7
            if days_until_tuesday == 0:
                days_until_tuesday = 7

        expiry_date = now_ist + timedelta(days=days_until_tuesday)
        expiry_str = expiry_date.strftime('%d-%b-%Y')
        self.logger.info(f"Next NIFTY Expiry: {expiry_str} ({expiry_date.strftime('%A')})")
        return expiry_str

    # ---------- Config & logging ----------

    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            print("Using default configuration...")
            return self.get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.get_default_config()

    def get_default_config(self):
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
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))

        self.logger = logging.getLogger('NiftyAnalyzer')
        self.logger.setLevel(level)

        if not self.logger.handlers:
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

    # ---------- Option chain ----------

    def fetch_option_chain(self):
        if self.config['data_source']['option_chain_source'] == 'sample':
            self.logger.info("Using sample option chain data")
            return None, None

        expiry_date = self.get_next_expiry_date()
        symbol = "NIFTY"

        api_url = f"{self.option_chain_base_url}{expiry_date}"
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

                response = session.get(api_url, headers=self.headers,
                                       impersonate="chrome", timeout=timeout)

                if response.status_code == 200:
                    data = response.json()
                    if 'records' in data and 'data' in data['records']:
                        option_data = data['records']['data']
                        current_price = data['records']['underlyingValue']

                        if not option_data:
                            self.logger.warning(f"No option data for expiry {expiry_date}")
                            continue

                        calls_data, puts_data = [], []

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
                        oc_df = oc_df.fillna(0).sort_values('Strike')

                        self.logger.info(
                            f"Option chain data fetched successfully | "
                            f"Spot: ₹{current_price} | Expiry: {expiry_date}"
                        )
                        self.logger.info(f"Total strikes fetched: {len(oc_df)}")
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
        if oc_df is None or oc_df.empty:
            return {'top_ce_strikes': [], 'top_pe_strikes': []}

        top_count = self.config['option_chain'].get('top_strikes_count', 5)

        ce_data = oc_df[oc_df['Call_OI'] > 0].copy()
        ce_data = ce_data.sort_values('Call_OI', ascending=False).head(top_count)
        top_ce_strikes = []
        for _, row in ce_data.iterrows():
            strike_type = 'ITM' if row['Strike'] < spot_price else (
                'ATM' if row['Strike'] == spot_price else 'OTM'
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
            strike_type = 'ITM' if row['Strike'] > spot_price else (
                'ATM' if row['Strike'] == spot_price else 'OTM'
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
            'top_ce_strikes': [],
            'top_pe_strikes': []
        }

    # ---------- Technical data ----------

    def fetch_technical_data(self):
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

            self.logger.info(
                f"1-HOUR data fetched | {len(df)} bars | "
                f"Price: ₹{df['Close'].iloc[-1]:.2f}"
            )
            return df
        except Exception as e:
            self.logger.error(f"Error fetching technical data: {e}")
            return None

    def calculate_pivot_points(self, df, current_price):
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

            self.logger.info(f"Pivot Points (30m) calculated | PP: ₹{pivot:.2f}")

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
                'pivot': current_price,
                'r1': current_price + 50,
                'r2': current_price + 100,
                'r3': current_price + 150,
                's1': current_price - 50,
                's2': current_price - 100,
                's3': current_price - 150,
                'prev_high': current_price,
                'prev_low': current_price,
                'prev_close': current_price
            }

    def calculate_rsi(self, data, period=None):
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
            'resistances': resistances[:num_resistance] if len(resistances) >= num_resistance else resistances,
            'supports': supports[:num_support] if len(supports) >= num_support else supports
        }

    def get_momentum_signal(self, momentum_pct):
        strong_threshold = self.config['technical'].get('momentum_threshold_strong', 0.5)
        moderate_threshold = self.config['technical'].get('momentum_threshold_moderate', 0.2)

        if momentum_pct > strong_threshold:
            return "Strong Upward", "Bullish"
        elif momentum_pct > moderate_threshold:
            return "Moderate Upward", "Bullish"
        elif momentum_pct < -strong_threshold:
            return "Strong Downward", "Bearish"
        elif momentum_pct < -moderate_threshold:
            return "Moderate Downward", "Bearish"
        else:
            return "Sideways/Weak", "Neutral"

    def technical_analysis(self, df):
        if df is None or df.empty:
            self.logger.warning("No technical data, using sample analysis")
            price = 24500
            return {
                'price': price,
                'rsi': 55.0,
                'trend': 'Sideways',
                'ema_short': price - 20,
                'ema_long': price - 40,
                'momentum_1h': 0.0,
                'momentum_5h': 0.0
            }

        current_price = df['Close'].iloc[-1]
        self.current_price = current_price

        # 1H momentum
        if len(df) > 1:
            price_1h_ago = df['Close'].iloc[-2]
            price_change_1h = current_price - price_1h_ago
            price_change_pct_1h = (price_change_1h / price_1h_ago * 100)
        else:
            price_change_pct_1h = 0.0

        # 5H momentum
        if len(df) >= 5:
            price_5h_ago = df['Close'].iloc[-5]
            momentum_5h = current_price - price_5h_ago
            momentum_5h_pct = (momentum_5h / price_5h_ago * 100)
        else:
            momentum_5h_pct = 0.0

        self.logger.info(f"1H Momentum: {price_change_pct_1h:+.2f}%")
        self.logger.info(f"5H Momentum: {momentum_5h_pct:+.2f}%")

        df['RSI'] = self.calculate_rsi(df['Close'])
        current_rsi = df['RSI'].iloc[-1]
        self.logger.info(f"RSI(14): {current_rsi:.2f}")

        ema_short = self.config['technical']['ema_short']
        ema_long = self.config['technical']['ema_long']

        df['EMA_Short'] = df['Close'].ewm(span=ema_short, adjust=False).mean()
        df['EMA_Long'] = df['Close'].ewm(span=ema_long, adjust=False).mean()

        ema_short_val = df['EMA_Short'].iloc[-1]
        ema_long_val = df['EMA_Long'].iloc[-1]

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

        tech = {
            'price': round(current_price, 2),
            'rsi': round(current_rsi, 2),
            'trend': trend,
            'ema_short': round(ema_short_val, 2),
            'ema_long': round(ema_long_val, 2),
            'momentum_1h': round(price_change_pct_1h, 2),
            'momentum_5h': round(momentum_5h_pct, 2)
        }
        self.tech_analysis = tech
        return tech

    # ---------- Recommendation ----------

    def build_recommendation(self, tech, oc):
        score = 0
        rec_cfg = self.config['recommendation']

        # Momentum weighting
        score += rec_cfg['momentum_1h_weight'] * np.sign(tech['momentum_1h'])
        score += rec_cfg['momentum_5h_weight'] * np.sign(tech['momentum_5h'])

        # OI sentiment
        if oc['oi_sentiment'] == 'Bullish':
            score += 1
        else:
            score -= 1

        # PCR
        pcr = oc['pcr']
        if pcr >= self.config['option_chain']['pcr_very_bullish']:
            score += 2
        elif pcr >= self.config['option_chain']['pcr_bullish']:
            score += 1
        elif pcr <= self.config['option_chain']['pcr_very_bearish']:
            score -= 2
        elif pcr <= self.config['option_chain']['pcr_bearish']:
            score -= 1

        # RSI
        rsi = tech['rsi']
        rsi_ob = self.config['technical']['rsi_overbought']
        rsi_os = self.config['technical']['rsi_oversold']
        if rsi > rsi_ob:
            score -= 1
        elif rsi < rsi_os:
            score += 1

        if score >= rec_cfg['strong_buy_threshold']:
            signal = "STRONG BUY"
        elif score >= rec_cfg['buy_threshold']:
            signal = "BUY"
        elif score <= rec_cfg['strong_sell_threshold']:
            signal = "STRONG SELL"
        elif score <= rec_cfg['sell_threshold']:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        reason = (
            f"Score: {score} | "
            f"Momentum 1H: {tech['momentum_1h']}%, "
            f"Momentum 5H: {tech['momentum_5h']}%, "
            f"PCR: {oc['pcr']}, OI Sentiment: {oc['oi_sentiment']}, "
            f"RSI: {tech['rsi']}"
        )

        reco = {'signal': signal, 'reason': reason, 'score': score}
        self.recommendation = reco
        return reco

    # ---------- HTML report (dark neon theme) ----------

    def generate_html_report(self):
        tech = self.tech_analysis
        oc = self.oc_analysis
        reco = self.recommendation
        piv = self.pivot_points
        sr = self.support_resistance

        timestamp = self.format_ist_time()

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NIFTY Day Trading Dashboard</title>

<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">

<style>
    :root {{
        --primary-bg: #0a0e27;
        --secondary-bg: #141b3d;
        --accent-bg: #1a2347;
        --text-primary: #e8edf5;
        --text-secondary: #a8b2d1;
        --accent-blue: #4a9eff;
        --accent-green: #00ff88;
        --accent-red: #ff4757;
        --accent-yellow: #ffd93d;
        --accent-purple: #a78bfa;
        --accent-pink: #f093fb;
        --accent-cyan: #4facfe;
        --border-color: #2a3a5f;
        --card-shadow: rgba(0, 0, 0, 0.5);
    }}

    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}

    body {{
        font-family: 'IBM Plex Sans', sans-serif;
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1629 100%);
        color: var(--text-primary);
        padding: 20px;
        position: relative;
        overflow-x: hidden;
    }}

    body::before {{
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(74, 158, 255, 0.1) 1px, transparent 1px);
        background-size: 50px 50px;
        animation: moveGrid 20s linear infinite;
        z-index: 0;
    }}

    @keyframes moveGrid {{
        0% {{ transform: translate(0, 0); }}
        100% {{ transform: translate(50px, 50px); }}
    }}

    .container {{
        max-width: 1500px;
        margin: auto;
        position: relative;
        z-index: 1;
    }}

    header {{
        text-align: center;
        padding: 25px;
        background: rgba(26, 35, 71, 0.5);
        border-radius: 15px;
        border: 1px solid var(--border-color);
        box-shadow: 0 15px 40px var(--card-shadow);
        margin-bottom: 40px;
        position: relative;
        overflow: hidden;
    }}

    header::after {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(74, 158, 255, 0.2), transparent);
        animation: shimmer 3s infinite;
    }}

    @keyframes shimmer {{
        0% {{ left: -100%; }}
        100% {{ left: 100%; }}
    }}

    h1 {{
        font-family: 'Playfair Display', serif;
        font-size: 2.4em;
        background: linear-gradient(135deg, #4a9eff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .timestamp {{
        margin-top: 10px;
        font-family: 'Space Mono', monospace;
        color: var(--accent-blue);
        opacity: 0.8;
    }}

    .section-title {{
        font-family: 'Playfair Display', serif;
        font-size: 2em;
        margin: 40px 0 20px;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .card {{
        background: rgba(26, 35, 71, 0.6);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid var(--border-color);
        box-shadow: 0 10px 30px var(--card-shadow);
        margin-bottom: 25px;
        position: relative;
        overflow: hidden;
    }}

    .card::before {{
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at top left, rgba(74,158,255,0.15), transparent 60%);
        opacity: 0.7;
        pointer-events: none;
    }}

    .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 20px;
    }}

    .metric {{
        font-family: 'Space Mono', monospace;
        font-size: 0.9em;
        color: var(--text-secondary);
        margin-bottom: 5px;
    }}

    .value {{
        font-size: 1.4em;
        font-weight: 700;
        color: var(--text-primary);
    }}

    .positive {{ color: var(--accent-green); }}
    .negative {{ color: var(--accent-red); }}
    .neutral {{ color: var(--text-secondary); }}

    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        font-size: 0.9em;
    }}

    th, td {{
        padding: 10px;
        border-bottom: 1px solid var(--border-color);
    }}

    th {{
        color: var(--accent-blue);
        font-weight: 600;
        text-align: left;
        background: rgba(10, 14, 39, 0.6);
    }}

    tr:hover td {{
        background: rgba(10, 14, 39, 0.5);
    }}

    h3 {{
        margin-top: 20px;
        margin-bottom: 5px;
        font-size: 1.1em;
    }}

    @media (max-width: 768px) {{
        h1 {{ font-size: 1.8em; }}
        .section-title {{ font-size: 1.6em; }}
    }}
</style>
</head>

<body>
<div class="container">

<header>
    <h1>🌐 NIFTY Day Trading Dashboard</h1>
    <div class="timestamp">📅 Last Updated: {timestamp}</div>
</header>

<!-- TECHNICAL ANALYSIS -->
<h2 class="section-title">📊 Technical Analysis (1H)</h2>
<div class="card">
    <div class="grid">
        <div>
            <div class="metric">Price</div>
            <div class="value">₹{tech.get('price', 'N/A')}</div>
        </div>
        <div>
            <div class="metric">RSI (14)</div>
            <div class="value">{tech.get('rsi', 'N/A')}</div>
        </div>
        <div>
            <div class="metric">Trend</div>
            <div class="value">{tech.get('trend', 'N/A')}</div>
        </div>
        <div>
            <div class="metric">EMA Short</div>
            <div class="value">₹{tech.get('ema_short', 'N/A')}</div>
        </div>
        <div>
            <div class="metric">EMA Long</div>
            <div class="value">₹{tech.get('ema_long', 'N/A')}</div>
        </div>
    </div>
</div>

<!-- MOMENTUM -->
<h2 class="section-title">⚡ Dual Momentum (1H & 5H)</h2>
<div class="card">
    <div class="grid">
        <div>
            <div class="metric">1H Momentum</div>
            <div class="value">{tech.get('momentum_1h', 0)}%</div>
        </div>
        <div>
            <div class="metric">5H Momentum</div>
            <div class="value">{tech.get('momentum_5h', 0)}%</div>
        </div>
    </div>
</div>

<!-- OPTION CHAIN -->
<h2 class="section-title">📈 Option Chain Analysis</h2>
<div class="card">
    <div class="grid">
        <div>
            <div class="metric">PCR</div>
            <div class="value">{oc.get('pcr', 'N/A')}</div>
        </div>
        <div>
            <div class="metric">Max Pain</div>
            <div class="value">₹{oc.get('max_pain', 'N/A')}</div>
        </div>
        <div>
            <div class="metric">OI Sentiment</div>
            <div class="value">{oc.get('oi_sentiment', 'N/A')}</div>
        </div>
    </div>

    <h3 style="color: var(--accent-yellow);">Top CE Strikes</h3>
    <table>
        <tr><th>Strike</th><th>OI</th><th>LTP</th><th>IV</th></tr>
        {''.join([f"<tr><td>{x['strike']}</td><td>{x['oi']}</td><td>{x['ltp']}</td><td>{x['iv']}</td></tr>" for x in oc.get('top_ce_strikes', [])])}
    </table>

    <h3 style="color: var(--accent-cyan);">Top PE Strikes</h3>
    <table>
        <tr><th>Strike</th><th>OI</th><th>LTP</th><th>IV</th></tr>
        {''.join([f"<tr><td>{x['strike']}</td><td>{x['oi']}</td><td>{x['ltp']}</td><td>{x['iv']}</td></tr>" for x in oc.get('top_pe_strikes', [])])}
    </table>
</div>

<!-- PIVOT POINTS -->
<h2 class="section-title">🎯 Pivot Points (30m)</h2>
<div class="card">
    <div class="grid">
        <div><div class="metric">Pivot</div><div class="value">₹{piv.get('pivot', 'N/A')}</div></div>
        <div><div class="metric">R1</div><div class="value">₹{piv.get('r1', 'N/A')}</div></div>
        <div><div class="metric">R2</div><div class="value">₹{piv.get('r2', 'N/A')}</div></div>
        <div><div class="metric">R3</div><div class="value">₹{piv.get('r3', 'N/A')}</div></div>
        <div><div class="metric">S1</div><div class="value">₹{piv.get('s1', 'N/A')}</div></div>
        <div><div class="metric">S2</div><div class="value">₹{piv.get('s2', 'N/A')}</div></div>
        <div><div class="metric">S3</div><div class="value">₹{piv.get('s3', 'N/A')}</div></div>
    </div>
</div>

<!-- SUPPORT / RESISTANCE -->
<h2 class="section-title">🧱 Support & Resistance</h2>
<div class="card">
    <div class="grid">
        <div>
            <div class="metric">Supports</div>
            <div class="value">{', '.join(map(str, sr.get('supports', [])))}</div>
        </div>
        <div>
            <div class="metric">Resistances</div>
            <div class="value">{', '.join(map(str, sr.get('resistances', [])))}</div>
        </div>
    </div>
</div>

<!-- RECOMMENDATION -->
<h2 class="section-title">🟢 Final Recommendation</h2>
<div class="card">
    <div class="value" style="font-size:2em;color:var(--accent-green);">
        {reco.get('signal', 'N/A')}
    </div>
    <div class="metric" style="margin-top:10px;">
        Reason: {reco.get('reason', '')}
    </div>
</div>

</div>
</body>
</html>
"""
        return html

    # ---------- Save & run ----------

    def save_report(self, html):
        report_cfg = self.config['report']
        if not report_cfg.get('save_local', True):
            return None

        os.makedirs(report_cfg.get('local_dir', './reports'), exist_ok=True)
        filename = datetime.now().strftime(report_cfg.get('filename_format',
                                                          'nifty_analysis_%Y%m%d_%H%M%S.html'))
        path = os.path.join(report_cfg.get('local_dir', './reports'), filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        self.logger.info(f"Report saved to {path}")
        return path

    def run(self):
        self.logger.info("Starting NiftyAnalyzer run...")

        df = self.fetch_technical_data()
        tech = self.technical_analysis(df)

        oc_df, spot = self.fetch_option_chain()
        if spot is None:
            spot = tech['price']
        self.current_price = spot

        oc = self.analyze_option_chain(oc_df, spot)
        self.oc_analysis = oc

        self.pivot_points = self.calculate_pivot_points(df, spot) if df is not None else {}
        self.support_resistance = self.calculate_support_resistance(df, spot) if df is not None else {
            'supports': [], 'resistances': []
        }

        reco = self.build_recommendation(tech, oc)

        html = self.generate_html_report()
        path = self.save_report(html)

        self.logger.info("Run completed.")
        return path


if __name__ == "__main__":
    analyzer = NiftyAnalyzer()
    analyzer.run()
