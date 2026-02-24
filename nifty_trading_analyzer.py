"""
Nifty Option Chain & Technical Analysis for Day Trading
THEME:  PHANTOM SLATE — Charcoal Blue-Grey · Neon Mint · Graphite Steel
PIVOT:  WIDGET 01 — NEON RUNWAY  |  Phantom Slate edition
S/R:    WIDGET 04 — BLOOMBERG TABLE  |  Phantom Slate edition
OI:     WIDGET 01 — NEON LEDGER  |  Phantom Slate edition
STRIKE: DARK TICKER CARD  |  Phantom Slate edition
1-HOUR TIMEFRAME with WILDER'S RSI (matches TradingView)
Enhanced with Pivot Points + Dual Momentum Analysis + Top 10 OI Display
EXPIRY: Weekly TUESDAY expiry with 3:30 PM IST cutoff logic
        If Tuesday is NSE holiday → expiry shifts to Monday
FIXED:  Using curl-cffi for NSE API to bypass anti-scraping
BUGFIX: ValueError: Unknown format code 'f' for object of type 'str'
BUGFIX: Live LTP now fetched via yfinance fast_info to show real-time price
OI FIX: Top 5 CE picked from 10 strikes ABOVE ATM only
        Top 5 PE picked from 10 strikes BELOW ATM only
RESPONSIVE: Auto-adjusts for Mobile / iPad / Desktop / Ultra-wide
AUTO-REFRESH: Silent 30-second background refresh with scroll position preservation
              Displays Last Refresh IST time + Live Current IST time (updates every second)
HOLIDAY FIX: If Tuesday expiry falls on NSE holiday, moves to Monday
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

# ─────────────────────────────────────────────────────────────────────────────
# PHANTOM SLATE COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
PS = {
    'bg':          '#080c12',
    'bg2':         '#0c1018',
    'bg3':         '#0e1420',
    'bg4':         '#141c28',
    'bg5':         '#141820',
    'border':      '#1e2a38',
    'border2':     '#2a3a4a',
    'accent':      '#44eecc',          # neon mint
    'accent_dim':  '#22aa88',
    'accent_glow': 'rgba(68,238,204,0.5)',
    'text':        '#ddeeff',
    'text_dim':    '#88a0b8',
    'text_mute':   '#2a3a4a',
    'ce_color':    '#ff3a5c',          # CE stays red
    'pe_color':    '#00e676',          # PE stays green
    'gold':        '#44eecc',          # replace gold with mint in headings
    'gold_dim':    '#22aa88',
    'gold_bright': '#66ffdd',
}


class NiftyAnalyzer:
    def __init__(self, config_path='config.yml'):
        self.config = self.load_config(config_path)
        self.setup_logging()
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

    # =========================================================================
    # CONFIG / LOGGING
    # =========================================================================
    def load_config(self, config_path):
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
        return {
            'email': {
                'recipient': 'your_email@gmail.com',
                'sender': 'your_email@gmail.com',
                'app_password': 'your_app_password',
                'subject_prefix': 'Nifty Day Trading Report',
                'send_on_failure': False
            },
            'technical': {
                'timeframe': '1h', 'period': '6mo', 'rsi_period': 14,
                'rsi_overbought': 70, 'rsi_oversold': 30,
                'ema_short': 20, 'ema_long': 50,
                'num_support_levels': 2, 'num_resistance_levels': 2,
                'momentum_threshold_strong': 0.5, 'momentum_threshold_moderate': 0.2
            },
            'option_chain': {
                'pcr_bullish': 1.0, 'pcr_very_bullish': 1.2,
                'pcr_bearish': 1.0, 'pcr_very_bearish': 0.8,
                'strike_range': 500, 'min_oi': 100000, 'top_strikes_count': 5
            },
            'recommendation': {
                'strong_buy_threshold': 3, 'buy_threshold': 1,
                'sell_threshold': -1, 'strong_sell_threshold': -3,
                'momentum_5h_weight': 2, 'momentum_1h_weight': 1
            },
            'report': {
                'title': 'NIFTY DAY TRADING ANALYSIS (1H)',
                'save_local': True, 'local_dir': './reports',
                'filename_format': 'nifty_analysis_%Y%m%d_%H%M%S.html'
            },
            'data_source': {
                'option_chain_source': 'nse', 'technical_source': 'yahoo',
                'max_retries': 3, 'retry_delay': 2, 'timeout': 30, 'fallback_to_sample': True
            },
            'logging': {
                'level': 'INFO', 'log_to_file': True,
                'log_file': './logs/nifty_analyzer.log',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            },
            'advanced': {
                'verbose': True, 'debug': False, 'validate_data': True,
                'min_data_points': 100, 'use_momentum_filter': True
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

    # =========================================================================
    # HELPERS
    # =========================================================================
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
        # ── NSE market holidays 2026 ──────────────────────────────────────
        nse_holidays = {
            datetime(2026, 1, 15).date(),   # Municipal Corporation Election - Maharashtra
            datetime(2026, 1, 26).date(),   # Republic Day
            datetime(2026, 3, 3).date(),    # Holi                          ← TUESDAY → Monday 02-Mar
            datetime(2026, 3, 26).date(),   # Shri Ram Navami
            datetime(2026, 3, 31).date(),   # Shri Mahavir Jayanti          ← TUESDAY → Monday 30-Mar
            datetime(2026, 4, 3).date(),    # Good Friday
            datetime(2026, 4, 14).date(),   # Dr. Baba Saheb Ambedkar Jayanti ← TUESDAY → Monday 13-Apr
            datetime(2026, 5, 1).date(),    # Maharashtra Day
            datetime(2026, 5, 28).date(),   # Bakri Id
            datetime(2026, 6, 26).date(),   # Muharram
            datetime(2026, 9, 14).date(),   # Ganesh Chaturthi
            datetime(2026, 10, 2).date(),   # Mahatma Gandhi Jayanti
            datetime(2026, 10, 20).date(),  # Dussehra                      ← TUESDAY → Monday 19-Oct
            datetime(2026, 11, 10).date(),  # Diwali-Balipratipada          ← TUESDAY → Monday 09-Nov
            datetime(2026, 11, 24).date(),  # Prakash Gurpurb Sri Guru Nanak Dev ← TUESDAY → Monday 23-Nov
            datetime(2026, 12, 25).date(),  # Christmas
        }

        now_ist      = self.get_ist_time()
        current_day  = now_ist.weekday()   # Monday=0 ... Sunday=6

        # ── Find the candidate Tuesday ────────────────────────────────────
        if current_day == 1:  # today IS Tuesday
            current_hour   = now_ist.hour
            current_minute = now_ist.minute
            if current_hour < 15 or (current_hour == 15 and current_minute < 30):
                days_until_tuesday = 0
                self.logger.info("📅 Today is Tuesday before 3:30 PM - Using today as expiry")
            else:
                days_until_tuesday = 7
                self.logger.info("📅 Tuesday after 3:30 PM - Moving to next Tuesday")
        elif current_day == 0:   # Monday
            days_until_tuesday = 1
        else:
            days_until_tuesday = (1 - current_day) % 7
            if days_until_tuesday == 0:
                days_until_tuesday = 7

        expiry_date = (now_ist + timedelta(days=days_until_tuesday)).date()

        # ── Holiday check: if Tuesday is a holiday, shift to Monday ──────
        if expiry_date in nse_holidays:
            self.logger.info(
                f"📅 Expiry {expiry_date.strftime('%d-%b-%Y')} is a NSE holiday "
                f"({expiry_date.strftime('%A')}) — shifting expiry to Monday"
            )
            expiry_date = expiry_date - timedelta(days=1)

        expiry_str = expiry_date.strftime('%d-%b-%Y')
        self.logger.info(f"📅 Next NIFTY Expiry: {expiry_str} ({expiry_date.strftime('%A')})")
        return expiry_str

    @staticmethod
    def _fmt_profit(value, multiplier=1):
        if isinstance(value, (int, float)):
            return f"₹{value * multiplier:.0f}"
        return str(value)

    @staticmethod
    def _fmt_profit_label(value, multiplier=1):
        if isinstance(value, (int, float)):
            return f"Profit: ₹{value * multiplier:.0f}"
        return str(value)

    # =========================================================================
    # DATA FETCHING
    # =========================================================================
    def fetch_live_ltp(self):
        try:
            ticker = yf.Ticker(self.nifty_symbol)
            live_price = ticker.fast_info.get('last_price', None)
            if live_price and live_price > 0:
                self.logger.info(f"✅ Live LTP (fast_info): ₹{live_price:.2f}")
                return round(float(live_price), 2)
        except Exception as e:
            self.logger.warning(f"fast_info LTP fetch failed: {e}")
        try:
            ticker = yf.Ticker(self.nifty_symbol)
            df_1m = ticker.history(period='1d', interval='1m')
            if not df_1m.empty:
                live_price = float(df_1m['Close'].iloc[-1])
                self.logger.info(f"✅ Live LTP (1m bar fallback): ₹{live_price:.2f}")
                return round(live_price, 2)
        except Exception as e:
            self.logger.warning(f"1m bar LTP fallback failed: {e}")
        self.logger.warning("⚠️ Could not fetch live LTP — will use last 1h candle close")
        return None

    def fetch_option_chain(self):
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
                        nse_spot_price = data['records']['underlyingValue']
                        if not option_data:
                            self.logger.warning(f"No option data for expiry {expiry_date}")
                            continue
                        calls_data, puts_data = [], []
                        for item in option_data:
                            strike = item.get('strikePrice', 0)
                            if 'CE' in item:
                                ce = item['CE']
                                calls_data.append({'Strike': strike, 'Call_OI': ce.get('openInterest', 0),
                                    'Call_Chng_OI': ce.get('changeinOpenInterest', 0),
                                    'Call_Volume': ce.get('totalTradedVolume', 0),
                                    'Call_IV': ce.get('impliedVolatility', 0),
                                    'Call_LTP': ce.get('lastPrice', 0)})
                            if 'PE' in item:
                                pe = item['PE']
                                puts_data.append({'Strike': strike, 'Put_OI': pe.get('openInterest', 0),
                                    'Put_Chng_OI': pe.get('changeinOpenInterest', 0),
                                    'Put_Volume': pe.get('totalTradedVolume', 0),
                                    'Put_IV': pe.get('impliedVolatility', 0),
                                    'Put_LTP': pe.get('lastPrice', 0)})
                        calls_df = pd.DataFrame(calls_data)
                        puts_df = pd.DataFrame(puts_data)
                        oc_df = pd.merge(calls_df, puts_df, on='Strike', how='outer').fillna(0).sort_values('Strike')
                        self.logger.info(f"✅ Option chain fetched | NSE Spot: ₹{nse_spot_price} | Expiry: {expiry_date}")
                        self.logger.info(f"✅ Total strikes fetched: {len(oc_df)}")
                        return oc_df, nse_spot_price
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
            candle_close = df['Close'].iloc[-1]
            self.logger.info(f"✅ 1-HOUR data fetched | {len(df)} bars | Last candle close: ₹{candle_close:.2f}")
            live_price = self.fetch_live_ltp()
            df.attrs['live_price'] = live_price
            return df
        except Exception as e:
            self.logger.error(f"Error fetching technical data: {e}")
            return None

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    def get_top_strikes_by_oi(self, oc_df, spot_price):
        if oc_df is None or oc_df.empty:
            return {'top_ce_strikes': [], 'top_pe_strikes': []}
        top_count = self.config['option_chain'].get('top_strikes_count', 5)
        window_size = 10
        all_strikes_sorted = sorted(oc_df['Strike'].unique())
        atm_strike = min(all_strikes_sorted, key=lambda x: abs(x - spot_price))
        strikes_above_atm = [s for s in all_strikes_sorted if s > atm_strike]
        ce_window_strikes = strikes_above_atm[:window_size]
        strikes_below_atm = [s for s in reversed(all_strikes_sorted) if s < atm_strike]
        pe_window_strikes = strikes_below_atm[:window_size]
        ce_window_strikes = [atm_strike] + ce_window_strikes
        pe_window_strikes = [atm_strike] + pe_window_strikes
        self.logger.info(f"📊 ATM Strike: ₹{atm_strike} | CE window: ₹{ce_window_strikes[0]}–₹{ce_window_strikes[-1]} | PE window: ₹{pe_window_strikes[-1]}–₹{pe_window_strikes[0]}")
        ce_data = oc_df[oc_df['Strike'].isin(ce_window_strikes) & (oc_df['Call_OI'] > 0)].copy()
        ce_data = ce_data.sort_values('Call_OI', ascending=False).head(top_count)
        top_ce_strikes = []
        for _, row in ce_data.iterrows():
            strike_type = 'ATM' if row['Strike'] == atm_strike else ('ITM' if row['Strike'] < spot_price else 'OTM')
            top_ce_strikes.append({'strike': row['Strike'], 'oi': int(row['Call_OI']), 'ltp': row['Call_LTP'],
                'iv': row['Call_IV'], 'type': strike_type, 'chng_oi': int(row['Call_Chng_OI']), 'volume': int(row['Call_Volume'])})
        pe_data = oc_df[oc_df['Strike'].isin(pe_window_strikes) & (oc_df['Put_OI'] > 0)].copy()
        pe_data = pe_data.sort_values('Put_OI', ascending=False).head(top_count)
        top_pe_strikes = []
        for _, row in pe_data.iterrows():
            strike_type = 'ATM' if row['Strike'] == atm_strike else ('ITM' if row['Strike'] > spot_price else 'OTM')
            top_pe_strikes.append({'strike': row['Strike'], 'oi': int(row['Put_OI']), 'ltp': row['Put_LTP'],
                'iv': row['Put_IV'], 'type': strike_type, 'chng_oi': int(row['Put_Chng_OI']), 'volume': int(row['Put_Volume'])})
        self.logger.info(f"✅ Top CE strikes (±10 ATM window): {[s['strike'] for s in top_ce_strikes]}")
        self.logger.info(f"✅ Top PE strikes (±10 ATM window): {[s['strike'] for s in top_pe_strikes]}")
        return {'top_ce_strikes': top_ce_strikes, 'top_pe_strikes': top_pe_strikes,
                'atm_strike': atm_strike, 'ce_window': ce_window_strikes, 'pe_window': pe_window_strikes}

    def calculate_max_pain(self, oc_df):
        if oc_df is None or oc_df.empty:
            return None
        strikes = sorted(oc_df['Strike'].unique())
        pain_records = []
        for candidate_strike in strikes:
            call_pain = sum(row['Call_OI'] * max(0, candidate_strike - row['Strike']) for _, row in oc_df.iterrows())
            put_pain  = sum(row['Put_OI']  * max(0, row['Strike'] - candidate_strike) for _, row in oc_df.iterrows())
            pain_records.append({'Strike': candidate_strike, 'Call_Pain': call_pain, 'Put_Pain': put_pain, 'Total_Pain': call_pain + put_pain})
        pain_df = pd.DataFrame(pain_records)
        max_pain_strike = pain_df.loc[pain_df['Total_Pain'].idxmin(), 'Strike']
        self.logger.info(f"✅ Max Pain (CORRECTED): ₹{max_pain_strike:,}  [min total buyer pain = {pain_df['Total_Pain'].min():,.0f}]")
        return int(max_pain_strike)

    def analyze_option_chain(self, oc_df, spot_price):
        if oc_df is None or oc_df.empty:
            self.logger.warning("No option chain data, using sample analysis")
            return self.get_sample_oc_analysis()
        config = self.config['option_chain']
        all_strikes_sorted = sorted(oc_df['Strike'].unique())
        atm_strike = min(all_strikes_sorted, key=lambda x: abs(x - spot_price))
        window_size = 10
        strikes_above = [s for s in all_strikes_sorted if s > atm_strike]
        strikes_below = [s for s in reversed(all_strikes_sorted) if s < atm_strike]
        ce_window  = [atm_strike] + strikes_above[:window_size]
        pe_window  = [atm_strike] + strikes_below[:window_size]
        atm_window = sorted(set(ce_window + pe_window))
        atm_df = oc_df[oc_df['Strike'].isin(atm_window)].copy()
        window_call_oi = atm_df['Call_OI'].sum()
        window_put_oi  = atm_df['Put_OI'].sum()
        pcr = window_put_oi / window_call_oi if window_call_oi > 0 else 0
        max_pain_strike = self.calculate_max_pain(oc_df)
        if max_pain_strike is None:
            max_pain_strike = int(atm_strike)
        num_resistance = self.config['technical']['num_resistance_levels']
        num_support    = self.config['technical']['num_support_levels']
        above_atm_df   = atm_df[atm_df['Strike'] > spot_price]
        below_atm_df   = atm_df[atm_df['Strike'] < spot_price]
        resistance_df  = above_atm_df.nlargest(num_resistance, 'Call_OI')
        resistances    = resistance_df['Strike'].tolist()
        support_df     = below_atm_df.nlargest(num_support, 'Put_OI')
        supports       = support_df['Strike'].tolist()
        total_call_buildup = atm_df['Call_Chng_OI'].sum()
        total_put_buildup  = atm_df['Put_Chng_OI'].sum()
        avg_call_iv = atm_df[atm_df['Call_IV'] > 0]['Call_IV'].mean() if (atm_df['Call_IV'] > 0).any() else 0
        avg_put_iv  = atm_df[atm_df['Put_IV']  > 0]['Put_IV'].mean()  if (atm_df['Put_IV']  > 0).any() else 0
        top_strikes = self.get_top_strikes_by_oi(oc_df, spot_price)
        return {
            'pcr': round(pcr, 2), 'max_pain': max_pain_strike, 'atm_strike': atm_strike,
            'resistances': sorted(resistances, reverse=True), 'supports': sorted(supports, reverse=True),
            'call_buildup': total_call_buildup, 'put_buildup': total_put_buildup,
            'avg_call_iv': round(avg_call_iv, 2), 'avg_put_iv': round(avg_put_iv, 2),
            'oi_sentiment': 'Bullish' if total_put_buildup > total_call_buildup else 'Bearish',
            'top_ce_strikes': top_strikes['top_ce_strikes'], 'top_pe_strikes': top_strikes['top_pe_strikes'],
            'ce_window': top_strikes.get('ce_window', []), 'pe_window': top_strikes.get('pe_window', []),
        }

    def get_sample_oc_analysis(self):
        return {
            'pcr': 1.15, 'max_pain': 24500, 'atm_strike': 24500,
            'resistances': [24600, 24650], 'supports': [24400, 24350],
            'call_buildup': 5000000, 'put_buildup': 6000000,
            'avg_call_iv': 15.5, 'avg_put_iv': 16.2, 'oi_sentiment': 'Bullish',
            'ce_window': [24500, 24550, 24600, 24650, 24700, 24750, 24800, 24850, 24900, 24950, 25000],
            'pe_window': [24500, 24450, 24400, 24350, 24300, 24250, 24200, 24150, 24100, 24050, 24000],
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

    def calculate_rsi(self, data, period=None):
        if period is None:
            period = self.config['technical']['rsi_period']
        delta    = data.diff()
        gain     = delta.where(delta > 0, 0)
        loss     = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs  = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_support_resistance(self, df, current_price):
        recent_data = df.tail(300)
        pivots_high, pivots_low = [], []
        for i in range(5, len(recent_data) - 5):
            high = recent_data['High'].iloc[i]
            low  = recent_data['Low'].iloc[i]
            if high == max(recent_data['High'].iloc[i-5:i+6]):
                pivots_high.append(high)
            if low == min(recent_data['Low'].iloc[i-5:i+6]):
                pivots_low.append(low)
        resistances = sorted(list(dict.fromkeys([p for p in pivots_high if p > current_price])))
        supports    = sorted(list(dict.fromkeys([p for p in pivots_low if p < current_price])), reverse=True)
        num_resistance = self.config['technical']['num_resistance_levels']
        num_support    = self.config['technical']['num_support_levels']
        return {'resistances': resistances[:num_resistance], 'supports': supports[:num_support]}

    def calculate_pivot_points(self, df, current_price):
        try:
            ticker     = yf.Ticker(self.nifty_symbol)
            min_30_df  = ticker.history(period='5d', interval='30m')
            if len(min_30_df) >= 2:
                prev_high  = min_30_df['High'].iloc[-2]
                prev_low   = min_30_df['Low'].iloc[-2]
                prev_close = min_30_df['Close'].iloc[-2]
            else:
                prev_high  = df['High'].max()
                prev_low   = df['Low'].min()
                prev_close = df['Close'].iloc[-1]
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = (2 * pivot) - prev_low;  r2 = pivot + (prev_high - prev_low);  r3 = prev_high + 2 * (pivot - prev_low)
            s1 = (2 * pivot) - prev_high; s2 = pivot - (prev_high - prev_low);  s3 = prev_low - 2 * (prev_high - pivot)
            self.logger.info(f"📍 Pivot Points (30m) calculated | PP: ₹{pivot:.2f}")
            return {'pivot': round(pivot, 2), 'r1': round(r1, 2), 'r2': round(r2, 2), 'r3': round(r3, 2),
                    's1': round(s1, 2), 's2': round(s2, 2), 's3': round(s3, 2),
                    'prev_high': round(prev_high, 2), 'prev_low': round(prev_low, 2), 'prev_close': round(prev_close, 2)}
        except Exception as e:
            self.logger.error(f"Error calculating pivot points: {e}")
            return {'pivot': 24520.00, 'r1': 24590.00, 'r2': 24650.00, 'r3': 24720.00,
                    's1': 24450.00, 's2': 24390.00, 's3': 24320.00,
                    'prev_high': 24580.00, 'prev_low': 24420.00, 'prev_close': 24500.00}

    def get_momentum_signal(self, momentum_pct):
        strong_threshold   = self.config['technical'].get('momentum_threshold_strong', 0.5)
        moderate_threshold = self.config['technical'].get('momentum_threshold_moderate', 0.2)
        if momentum_pct > strong_threshold:
            return "Strong Upward",    "Bullish", {'bg': '#004d2e', 'bg_dark': '#003320', 'text': '#00ff8c', 'border': '#00aa55'}
        elif momentum_pct > moderate_threshold:
            return "Moderate Upward",  "Bullish", {'bg': '#00334a', 'bg_dark': '#00223a', 'text': '#00c8ff', 'border': '#0088bb'}
        elif momentum_pct < -strong_threshold:
            return "Strong Downward",  "Bearish", {'bg': '#4a0010', 'bg_dark': '#380008', 'text': '#ff6070', 'border': '#cc2233'}
        elif momentum_pct < -moderate_threshold:
            return "Moderate Downward","Bearish", {'bg': '#3a1500', 'bg_dark': '#280d00', 'text': '#ff8855', 'border': '#cc4400'}
        else:
            return "Sideways/Weak",    "Neutral", {'bg': '#0e1420', 'bg_dark': '#0a1018', 'text': '#44eecc', 'border': '#1e2a38'}

    def technical_analysis(self, df):
        if df is None or df.empty:
            self.logger.warning("No technical data, using sample analysis")
            return self.get_sample_tech_analysis()
        candle_close_price = float(df['Close'].iloc[-1])
        live_price = df.attrs.get('live_price', None)
        if live_price and live_price > 0:
            current_price = live_price
            self.logger.info(f"💹 Using LIVE LTP: ₹{current_price:.2f}  (candle close was ₹{candle_close_price:.2f})")
        else:
            current_price = candle_close_price
            self.logger.warning(f"⚠️ Live LTP unavailable — using candle close: ₹{current_price:.2f}")
        if len(df) > 1:
            price_1h_ago        = df['Close'].iloc[-2]
            price_change_1h     = current_price - float(price_1h_ago)
            price_change_pct_1h = (price_change_1h / float(price_1h_ago) * 100)
        else:
            price_change_1h     = 0
            price_change_pct_1h = 0
        momentum_1h_signal, momentum_1h_bias, momentum_1h_colors = self.get_momentum_signal(price_change_pct_1h)
        if len(df) >= 5:
            price_5h_ago    = df['Close'].iloc[-5]
            momentum_5h     = current_price - float(price_5h_ago)
            momentum_5h_pct = (momentum_5h / float(price_5h_ago) * 100)
        else:
            momentum_5h     = 0
            momentum_5h_pct = 0
        momentum_5h_signal, momentum_5h_bias, momentum_5h_colors = self.get_momentum_signal(momentum_5h_pct)
        bars_2d = 13
        if len(df) >= bars_2d:
            price_2d_ago    = df['Close'].iloc[-bars_2d]
            momentum_2d     = current_price - float(price_2d_ago)
            momentum_2d_pct = (momentum_2d / float(price_2d_ago) * 100)
        else:
            momentum_2d     = 0
            momentum_2d_pct = 0
        momentum_2d_signal, momentum_2d_bias, momentum_2d_colors = self.get_momentum_signal(momentum_2d_pct)
        self.logger.info(f"📊 1H Momentum: {price_change_pct_1h:+.2f}% - {momentum_1h_signal}")
        self.logger.info(f"📊 5H Momentum: {momentum_5h_pct:+.2f}% - {momentum_5h_signal}")
        self.logger.info(f"📊 2D Momentum: {momentum_2d_pct:+.2f}% - {momentum_2d_signal}")
        df['RSI']      = self.calculate_rsi(df['Close'])
        current_rsi    = df['RSI'].iloc[-1]
        self.logger.info(f"🎯 RSI(14) calculated: {current_rsi:.2f} (Wilder's method)")
        ema_short = self.config['technical']['ema_short']
        ema_long  = self.config['technical']['ema_long']
        df['EMA_Short'] = df['Close'].ewm(span=ema_short, adjust=False).mean()
        df['EMA_Long']  = df['Close'].ewm(span=ema_long,  adjust=False).mean()
        ema_short_val = df['EMA_Short'].iloc[-1]
        ema_long_val  = df['EMA_Long'].iloc[-1]
        sr_levels    = self.calculate_support_resistance(df, current_price)
        pivot_points = self.calculate_pivot_points(df, current_price)
        if current_price > ema_short_val > ema_long_val:       trend = "Strong Uptrend"
        elif current_price > ema_short_val:                    trend = "Uptrend"
        elif current_price < ema_short_val < ema_long_val:     trend = "Strong Downtrend"
        elif current_price < ema_short_val:                    trend = "Downtrend"
        else:                                                   trend = "Sideways"
        rsi_ob = self.config['technical']['rsi_overbought']
        rsi_os = self.config['technical']['rsi_oversold']
        if current_rsi > rsi_ob:     rsi_signal = "Overbought - Bearish"
        elif current_rsi < rsi_os:   rsi_signal = "Oversold - Bullish"
        elif current_rsi > 50:       rsi_signal = "Bullish"
        else:                         rsi_signal = "Bearish"
        return {
            'current_price': round(current_price, 2), 'candle_close_price': round(candle_close_price, 2),
            'rsi': round(current_rsi, 2), 'rsi_signal': rsi_signal,
            'ema20': round(ema_short_val, 2), 'ema50': round(ema_long_val, 2), 'trend': trend,
            'tech_resistances': [round(r, 2) for r in sr_levels['resistances']],
            'tech_supports':    [round(s, 2) for s in sr_levels['supports']],
            'pivot_points': pivot_points, 'timeframe': '1 Hour',
            'price_change_1h': round(price_change_1h, 2), 'price_change_pct_1h': round(price_change_pct_1h, 2),
            'momentum_1h_signal': momentum_1h_signal, 'momentum_1h_bias': momentum_1h_bias, 'momentum_1h_colors': momentum_1h_colors,
            'momentum_5h': round(momentum_5h, 2), 'momentum_5h_pct': round(momentum_5h_pct, 2),
            'momentum_5h_signal': momentum_5h_signal, 'momentum_5h_bias': momentum_5h_bias, 'momentum_5h_colors': momentum_5h_colors,
            'momentum_2d': round(momentum_2d, 2), 'momentum_2d_pct': round(momentum_2d_pct, 2),
            'momentum_2d_signal': momentum_2d_signal, 'momentum_2d_bias': momentum_2d_bias, 'momentum_2d_colors': momentum_2d_colors
        }

    def get_sample_tech_analysis(self):
        return {
            'current_price': 24520.50, 'candle_close_price': 24516.45,
            'rsi': 42.82, 'rsi_signal': 'Bearish',
            'ema20': 24480.00, 'ema50': 24450.00, 'trend': 'Uptrend',
            'tech_resistances': [24580.00, 24650.00], 'tech_supports': [24420.00, 24380.00],
            'pivot_points': {'pivot': 24520.00, 'r1': 24590.00, 'r2': 24650.00, 'r3': 24720.00,
                             's1': 24450.00, 's2': 24390.00, 's3': 24320.00,
                             'prev_high': 24580.00, 'prev_low': 24420.00, 'prev_close': 24500.00},
            'timeframe': '1 Hour',
            'price_change_1h': -15.50, 'price_change_pct_1h': -0.06,
            'momentum_1h_signal': 'Sideways/Weak', 'momentum_1h_bias': 'Neutral',
            'momentum_1h_colors': {'bg': '#0e1420', 'bg_dark': '#0a1018', 'text': '#44eecc', 'border': '#1e2a38'},
            'momentum_5h': -35.50, 'momentum_5h_pct': -0.14,
            'momentum_5h_signal': 'Moderate Downward', 'momentum_5h_bias': 'Bearish',
            'momentum_5h_colors': {'bg': '#3a1500', 'bg_dark': '#280d00', 'text': '#ff8855', 'border': '#cc4400'},
            'momentum_2d': 305.50, 'momentum_2d_pct': 1.24,
            'momentum_2d_signal': 'Strong Upward', 'momentum_2d_bias': 'Bullish',
            'momentum_2d_colors': {'bg': '#004d2e', 'bg_dark': '#003320', 'text': '#00ff8c', 'border': '#00aa55'}
        }

    def generate_recommendation(self, oc_analysis, tech_analysis):
        if not oc_analysis or not tech_analysis:
            return {"recommendation": "Insufficient data", "bias": "Neutral", "confidence": "Low",
                    "reasons": [], "bullish_signals": 0, "bearish_signals": 0}
        config      = self.config['recommendation']
        oc_config   = self.config['option_chain']
        tech_config = self.config['technical']
        bullish_signals, bearish_signals, reasons = 0, 0, []
        use_momentum = self.config['advanced'].get('use_momentum_filter', True)
        if use_momentum:
            momentum_5h_pct    = tech_analysis.get('momentum_5h_pct', 0)
            weight_5h          = config.get('momentum_5h_weight', 2)
            strong_threshold   = tech_config.get('momentum_threshold_strong', 0.5)
            moderate_threshold = tech_config.get('momentum_threshold_moderate', 0.2)
            if momentum_5h_pct > strong_threshold:
                bullish_signals += weight_5h; reasons.append(f"🚀 5H Strong upward momentum: {momentum_5h_pct:+.2f}%")
            elif momentum_5h_pct > moderate_threshold:
                bullish_signals += 1; reasons.append(f"📈 5H Positive momentum: {momentum_5h_pct:+.2f}%")
            elif momentum_5h_pct < -strong_threshold:
                bearish_signals += weight_5h; reasons.append(f"🔻 5H Strong downward momentum: {momentum_5h_pct:+.2f}%")
            elif momentum_5h_pct < -moderate_threshold:
                bearish_signals += 1; reasons.append(f"📉 5H Negative momentum: {momentum_5h_pct:+.2f}%")
            momentum_1h_pct = tech_analysis.get('price_change_pct_1h', 0)
            weight_1h = config.get('momentum_1h_weight', 1)
            if momentum_1h_pct > strong_threshold:
                bullish_signals += weight_1h; reasons.append(f"⚡ 1H Strong upward move: {momentum_1h_pct:+.2f}%")
            elif momentum_1h_pct < -strong_threshold:
                bearish_signals += weight_1h; reasons.append(f"⚡ 1H Strong downward move: {momentum_1h_pct:+.2f}%")
        pcr = oc_analysis.get('pcr', 0)
        if pcr >= oc_config['pcr_very_bullish']:
            bullish_signals += 2; reasons.append(f"PCR at {pcr} indicates strong bullish sentiment")
        elif pcr >= oc_config['pcr_bullish']:
            bullish_signals += 1; reasons.append(f"PCR at {pcr} shows bullish bias")
        elif pcr <= oc_config['pcr_very_bearish']:
            bearish_signals += 2; reasons.append(f"PCR at {pcr} indicates strong bearish sentiment")
        elif pcr < oc_config['pcr_bearish']:
            bearish_signals += 1; reasons.append(f"PCR at {pcr} shows bearish bias")
        if oc_analysis.get('oi_sentiment') == 'Bullish':
            bullish_signals += 1; reasons.append("Put OI buildup > Call OI buildup (Bullish)")
        else:
            bearish_signals += 1; reasons.append("Call OI buildup > Put OI buildup (Bearish)")
        rsi    = tech_analysis.get('rsi', 50)
        rsi_os = tech_config['rsi_oversold']
        rsi_ob = tech_config['rsi_overbought']
        if rsi < rsi_os:
            bullish_signals += 2; reasons.append(f"RSI at {rsi:.1f} - Oversold (Bullish reversal)")
        elif rsi < 45:
            bullish_signals += 1; reasons.append(f"RSI at {rsi:.1f} - Below neutral")
        elif rsi > rsi_ob:
            bearish_signals += 2; reasons.append(f"RSI at {rsi:.1f} - Overbought (Bearish)")
        elif rsi > 55:
            bearish_signals += 1; reasons.append(f"RSI at {rsi:.1f} - Above neutral")
        trend = tech_analysis.get('trend', '')
        if 'Uptrend' in trend:
            bullish_signals += 1; reasons.append(f"Trend: {trend}")
        elif 'Downtrend' in trend:
            bearish_signals += 1; reasons.append(f"Trend: {trend}")
        current_price = tech_analysis.get('current_price', 0)
        ema20         = tech_analysis.get('ema20', 0)
        if current_price > ema20:
            bullish_signals += 1; reasons.append("Price above EMA20 (Bullish)")
        else:
            bearish_signals += 1; reasons.append("Price below EMA20 (Bearish)")
        signal_diff   = bullish_signals - bearish_signals
        strong_buy_t  = config['strong_buy_threshold'];  buy_t   = config['buy_threshold']
        sell_t        = config['sell_threshold'];         strong_sell_t = config['strong_sell_threshold']
        if signal_diff >= strong_buy_t:
            recommendation, bias, confidence = "STRONG BUY",    "Bullish", "High"
        elif signal_diff >= buy_t:
            recommendation, bias, confidence = "BUY",           "Bullish", "Medium"
        elif signal_diff <= strong_sell_t:
            recommendation, bias, confidence = "STRONG SELL",   "Bearish", "High"
        elif signal_diff <= sell_t:
            recommendation, bias, confidence = "SELL",          "Bearish", "Medium"
        else:
            recommendation, bias, confidence = "NEUTRAL / WAIT","Neutral",  "Low"
        return {'recommendation': recommendation, 'bias': bias, 'confidence': confidence,
                'bullish_signals': bullish_signals, 'bearish_signals': bearish_signals, 'reasons': reasons}

    def get_options_strategies(self, recommendation, oc_analysis, tech_analysis):
        bias           = recommendation['bias']
        avg_iv         = (oc_analysis.get('avg_call_iv', 15) + oc_analysis.get('avg_put_iv', 15)) / 2
        high_volatility = avg_iv > 18
        strategies = []
        if bias == 'Bullish':
            strategies.append({'name': 'Long Call', 'type': 'Bullish - Aggressive',
                'setup': 'Buy ATM or slightly OTM Call option', 'profit': 'Unlimited upside', 'risk': 'Limited to premium paid',
                'best_when': 'Strong upward move expected, low IV',
                'recommended': '⭐⭐⭐⭐⭐' if recommendation['confidence'] == 'High' and not high_volatility else '⭐⭐⭐'})
            strategies.append({'name': 'Bull Call Spread', 'type': 'Bullish - Moderate',
                'setup': 'Buy ITM Call + Sell OTM Call', 'profit': 'Limited (Strike difference - Net premium)', 'risk': 'Limited to net premium paid',
                'best_when': 'Moderately bullish, reduce cost',
                'recommended': '⭐⭐⭐⭐⭐' if recommendation['confidence'] == 'Medium' else '⭐⭐⭐⭐'})
        elif bias == 'Bearish':
            strategies.append({'name': 'Long Put', 'type': 'Bearish - Aggressive',
                'setup': 'Buy ATM or slightly OTM Put option', 'profit': 'High (Strike - Stock price - Premium)', 'risk': 'Limited to premium paid',
                'best_when': 'Strong downward move expected, low IV',
                'recommended': '⭐⭐⭐⭐⭐' if recommendation['confidence'] == 'High' and not high_volatility else '⭐⭐⭐'})
            strategies.append({'name': 'Bear Put Spread', 'type': 'Bearish - Debit Strategy',
                'setup': 'Buy ITM Put + Sell OTM Put', 'profit': 'Limited (Strike difference - Net premium)', 'risk': 'Limited to net premium paid',
                'best_when': 'Moderately bearish, reduce cost',
                'recommended': '⭐⭐⭐⭐⭐' if recommendation['confidence'] == 'Medium' else '⭐⭐⭐⭐'})
        else:
            if high_volatility:
                strategies.append({'name': 'Long Straddle', 'type': 'Neutral - High Volatility Expected',
                    'setup': 'Buy ATM Call + Buy ATM Put', 'profit': 'Unlimited (either direction)', 'risk': 'Limited to total premium paid',
                    'best_when': 'Expect big move, unsure of direction', 'recommended': '⭐⭐⭐⭐⭐'})
            else:
                strategies.append({'name': 'Short Strangle', 'type': 'Neutral - Low Volatility Expected',
                    'setup': 'Sell OTM Call + Sell OTM Put', 'profit': 'Limited to total premium collected', 'risk': 'Unlimited (either direction)',
                    'best_when': 'Expect range-bound, less risk than straddle', 'recommended': '⭐⭐⭐⭐⭐'})
        return strategies

    def get_detailed_strike_recommendations(self, oc_analysis, tech_analysis, recommendation):
        current_price  = tech_analysis.get('current_price', 0)
        bias           = recommendation['bias']
        atm_strike     = round(current_price / 50) * 50
        top_ce_strikes = oc_analysis.get('top_ce_strikes', [])
        top_pe_strikes = oc_analysis.get('top_pe_strikes', [])
        def find_closest_strike(target_strike, strike_list):
            if not strike_list: return None
            return min(strike_list, key=lambda x: abs(x['strike'] - target_strike))
        strike_recommendations = []
        if bias == 'Bullish':
            atm_ce = find_closest_strike(atm_strike, top_ce_strikes)
            if atm_ce:
                actual_strike = atm_ce['strike']
                strike_recommendations.append({'strategy': 'Long Call (ATM)', 'action': 'BUY', 'strike': actual_strike, 'type': 'CE',
                    'ltp': atm_ce['ltp'], 'option_type': 'ATM', 'target_1': actual_strike + 100, 'target_2': actual_strike + 200,
                    'stop_loss': atm_ce['ltp'] * 0.3, 'max_loss': atm_ce['ltp'],
                    'profit_at_target_1': 100 - atm_ce['ltp'], 'profit_at_target_2': 200 - atm_ce['ltp'],
                    'oi': atm_ce['oi'], 'volume': atm_ce['volume']})
            otm_target = atm_strike + 50
            otm_ce = find_closest_strike(otm_target, top_ce_strikes)
            if otm_ce and otm_ce['strike'] != (atm_ce['strike'] if atm_ce else None):
                actual_strike = otm_ce['strike']
                strike_recommendations.append({'strategy': 'Long Call (OTM)', 'action': 'BUY', 'strike': actual_strike, 'type': 'CE',
                    'ltp': otm_ce['ltp'], 'option_type': 'OTM', 'target_1': actual_strike + 100, 'target_2': actual_strike + 150,
                    'stop_loss': otm_ce['ltp'] * 0.3, 'max_loss': otm_ce['ltp'],
                    'profit_at_target_1': 100 - otm_ce['ltp'], 'profit_at_target_2': 150 - otm_ce['ltp'],
                    'oi': otm_ce['oi'], 'volume': otm_ce['volume']})
            itm_target = atm_strike - 50
            itm_ce = find_closest_strike(itm_target, top_ce_strikes)
            if itm_ce and otm_ce and len(strike_recommendations) >= 1:
                itm_k = itm_ce['strike']; otm_k = otm_ce['strike']
                net_premium = itm_ce['ltp'] - otm_ce['ltp']; max_profit = (otm_k - itm_k) - net_premium
                strike_recommendations.append({'strategy': 'Bull Call Spread', 'action': f"BUY {itm_k} CE + SELL {otm_k} CE",
                    'strike': f"{itm_k}/{otm_k}", 'type': 'Spread', 'ltp': net_premium, 'option_type': 'ITM/OTM',
                    'target_1': itm_k + 25, 'target_2': otm_k, 'stop_loss': net_premium * 0.4, 'max_loss': net_premium,
                    'profit_at_target_1': 25 - net_premium, 'profit_at_target_2': max_profit,
                    'oi': f"{itm_ce['oi']:,} / {otm_ce['oi']:,}", 'volume': f"{itm_ce['volume']:,} / {otm_ce['volume']:,}"})
        elif bias == 'Bearish':
            atm_pe = find_closest_strike(atm_strike, top_pe_strikes)
            if atm_pe:
                actual_strike = atm_pe['strike']
                strike_recommendations.append({'strategy': 'Long Put (ATM)', 'action': 'BUY', 'strike': actual_strike, 'type': 'PE',
                    'ltp': atm_pe['ltp'], 'option_type': 'ATM', 'target_1': actual_strike - 100, 'target_2': actual_strike - 200,
                    'stop_loss': atm_pe['ltp'] * 0.3, 'max_loss': atm_pe['ltp'],
                    'profit_at_target_1': 100 - atm_pe['ltp'], 'profit_at_target_2': 200 - atm_pe['ltp'],
                    'oi': atm_pe['oi'], 'volume': atm_pe['volume']})
            otm_target = atm_strike - 50
            otm_pe = find_closest_strike(otm_target, top_pe_strikes)
            if otm_pe and otm_pe['strike'] != (atm_pe['strike'] if atm_pe else None):
                actual_strike = otm_pe['strike']
                strike_recommendations.append({'strategy': 'Long Put (OTM)', 'action': 'BUY', 'strike': actual_strike, 'type': 'PE',
                    'ltp': otm_pe['ltp'], 'option_type': 'OTM', 'target_1': actual_strike - 100, 'target_2': actual_strike - 150,
                    'stop_loss': otm_pe['ltp'] * 0.3, 'max_loss': otm_pe['ltp'],
                    'profit_at_target_1': 100 - otm_pe['ltp'], 'profit_at_target_2': 150 - otm_pe['ltp'],
                    'oi': otm_pe['oi'], 'volume': otm_pe['volume']})
            itm_target = atm_strike + 50
            itm_pe = find_closest_strike(itm_target, top_pe_strikes)
            if itm_pe and otm_pe and len(strike_recommendations) >= 1:
                itm_k = itm_pe['strike']; otm_k = otm_pe['strike']
                net_premium = itm_pe['ltp'] - otm_pe['ltp']; max_profit = (itm_k - otm_k) - net_premium
                strike_recommendations.append({'strategy': 'Bear Put Spread', 'action': f"BUY {itm_k} PE + SELL {otm_k} PE",
                    'strike': f"{itm_k}/{otm_k}", 'type': 'Spread', 'ltp': net_premium, 'option_type': 'ITM/OTM',
                    'target_1': itm_k - 25, 'target_2': otm_k, 'stop_loss': net_premium * 0.4, 'max_loss': net_premium,
                    'profit_at_target_1': 25 - net_premium, 'profit_at_target_2': max_profit,
                    'oi': f"{itm_pe['oi']:,} / {otm_pe['oi']:,}", 'volume': f"{itm_pe['volume']:,} / {otm_pe['volume']:,}"})
        else:
            atm_ce = find_closest_strike(atm_strike, top_ce_strikes)
            atm_pe = find_closest_strike(atm_strike, top_pe_strikes)
            if atm_ce and atm_pe:
                actual_strike = atm_ce['strike']; total_premium = atm_ce['ltp'] + atm_pe['ltp']
                strike_recommendations.append({'strategy': 'Long Straddle (ATM)', 'action': f"BUY {actual_strike} CE + BUY {actual_strike} PE",
                    'strike': actual_strike, 'type': 'Straddle', 'ltp': total_premium, 'option_type': 'ATM/ATM',
                    'target_1': actual_strike + total_premium, 'target_2': actual_strike - total_premium,
                    'stop_loss': total_premium * 0.5, 'max_loss': total_premium,
                    'profit_at_target_1': f"Profit if moves ±{total_premium:.0f} points", 'profit_at_target_2': 'Unlimited both sides',
                    'oi': f"{atm_ce['oi']:,} / {atm_pe['oi']:,}", 'volume': f"{atm_ce['volume']:,} / {atm_pe['volume']:,}"})
        if strike_recommendations:
            self.logger.info(f"✅ Generated {len(strike_recommendations)} strike recommendations")
        else:
            self.logger.warning(f"⚠️ No strike recommendations generated. ATM={atm_strike}, Available CE={[s['strike'] for s in top_ce_strikes[:3]]}, Available PE={[s['strike'] for s in top_pe_strikes[:3]]}")
        return strike_recommendations

    def find_nearest_levels(self, current_price, pivot_points):
        all_resistances = [pivot_points['r1'], pivot_points['r2'], pivot_points['r3']]
        all_supports    = [pivot_points['s1'], pivot_points['s2'], pivot_points['s3']]
        resistances_above  = [r for r in all_resistances if r > current_price]
        nearest_resistance = min(resistances_above) if resistances_above else None
        supports_below  = [s for s in all_supports if s < current_price]
        nearest_support = max(supports_below) if supports_below else None
        return {'nearest_resistance': nearest_resistance, 'nearest_support': nearest_support}

    # =========================================================================
    # WIDGET 01 — NEON LEDGER | PHANTOM SLATE EDITION
    # =========================================================================
    def _build_oi_neon_ledger_widget(self, top_ce_strikes, top_pe_strikes, atm_strike=None):
        all_oi = [s['oi'] for s in top_ce_strikes] + [s['oi'] for s in top_pe_strikes]
        max_oi = max(all_oi) if all_oi else 1

        def type_badge(t):
            if t == 'ITM': return '<span class="nl-tbadge nl-itm">ITM</span>'
            elif t == 'ATM': return '<span class="nl-tbadge nl-atm">ATM</span>'
            else: return '<span class="nl-tbadge nl-otm">OTM</span>'

        def fmt_oi(val):
            if val >= 1_000_000: return f"{val/1_000_000:.2f}M"
            elif val >= 1_000: return f"{val/1_000:.1f}K"
            return str(val)

        def fmt_vol(val):
            if val >= 1_000_000: return f"{val/1_000_000:.1f}M"
            elif val >= 1_000: return f"{val/1_000:.0f}K"
            return str(val)

        def chng_oi_cell(val):
            if val > 0:   return f'<span class="nl-chng nl-chng-up">+{fmt_oi(val)}</span>'
            elif val < 0: return f'<span class="nl-chng nl-chng-dn">{fmt_oi(val)}</span>'
            return        f'<span class="nl-chng nl-chng-flat">{fmt_oi(val)}</span>'

        def rank_badge(rank, side):
            glow_col = '#ff3a5c' if side == 'ce' else '#44eecc'
            bg_col   = 'rgba(255,58,92,0.15)'   if side == 'ce' else 'rgba(68,238,204,0.12)'
            brd_col  = 'rgba(255,58,92,0.5)'    if side == 'ce' else 'rgba(68,238,204,0.45)'
            txt_col  = '#ff6680'                 if side == 'ce' else '#44eecc'
            return (f'<div class="nl-rank" style="background:{bg_col};border:1px solid {brd_col};'
                    f'color:{txt_col};box-shadow:0 0 10px {glow_col}33;">{rank}</div>')

        def build_ce_rows(strikes):
            rows = ''
            for idx, s in enumerate(strikes, 1):
                bar_w = int((s['oi'] / max_oi) * 100)
                rows += f'''
                <tr class="nl-row">
                    <td class="nl-td-rank">{rank_badge(idx, "ce")}</td>
                    <td class="nl-td-strike"><span class="nl-strike-val">&#8377;{int(s["strike"]):,}</span></td>
                    <td class="nl-td-type">{type_badge(s["type"])}</td>
                    <td class="nl-td-oi">
                        <div class="nl-oi-wrap">
                            <span class="nl-oi-val nl-oi-ce">{fmt_oi(s["oi"])}</span>
                            <div class="nl-bar-track"><div class="nl-bar-fill nl-bar-ce" style="width:{bar_w}%;"></div></div>
                        </div>
                    </td>
                    <td class="nl-td-chng">{chng_oi_cell(s["chng_oi"])}</td>
                    <td class="nl-td-ltp"><span class="nl-ltp nl-ltp-ce">&#8377;{s["ltp"]:.2f}</span></td>
                    <td class="nl-td-vol"><span class="nl-vol">{fmt_vol(s["volume"])}</span></td>
                </tr>'''
            return rows

        def build_pe_rows(strikes):
            rows = ''
            for idx, s in enumerate(strikes, 1):
                bar_w = int((s['oi'] / max_oi) * 100)
                rows += f'''
                <tr class="nl-row">
                    <td class="nl-td-rank">{rank_badge(idx, "pe")}</td>
                    <td class="nl-td-strike"><span class="nl-strike-val">&#8377;{int(s["strike"]):,}</span></td>
                    <td class="nl-td-type">{type_badge(s["type"])}</td>
                    <td class="nl-td-oi">
                        <div class="nl-oi-wrap">
                            <span class="nl-oi-val nl-oi-pe">{fmt_oi(s["oi"])}</span>
                            <div class="nl-bar-track"><div class="nl-bar-fill nl-bar-pe" style="width:{bar_w}%;"></div></div>
                        </div>
                    </td>
                    <td class="nl-td-chng">{chng_oi_cell(s["chng_oi"])}</td>
                    <td class="nl-td-ltp"><span class="nl-ltp nl-ltp-pe">&#8377;{s["ltp"]:.2f}</span></td>
                    <td class="nl-td-vol"><span class="nl-vol">{fmt_vol(s["volume"])}</span></td>
                </tr>'''
            return rows

        ce_rows_html = build_ce_rows(top_ce_strikes)
        pe_rows_html = build_pe_rows(top_pe_strikes)

        col_heads = '''
            <th class="nl-th">#</th>
            <th class="nl-th">STRIKE</th>
            <th class="nl-th">TYPE</th>
            <th class="nl-th">OPEN INTEREST</th>
            <th class="nl-th">CHG OI</th>
            <th class="nl-th">LTP</th>
            <th class="nl-th">VOLUME</th>'''

        return f'''
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Mono:wght@400;600;700&display=swap');
            .nl-wrap {{
                font-family: 'Chakra Petch', 'Segoe UI', sans-serif;
                background: #0c1018; border: 1px solid #1e2a38; border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 0 0 1px #141c28, 0 0 60px rgba(68,238,204,.04), 0 24px 60px rgba(0,0,0,.9);
            }}
            .nl-master-hdr {{
                background: linear-gradient(135deg, #0e1420 0%, #141c28 100%);
                border-bottom: 1px solid #1e2a38; padding: 16px 24px;
                display: flex; align-items: center; justify-content: space-between; position: relative;
                flex-wrap: wrap; gap: 10px;
            }}
            .nl-master-hdr::after {{
                content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
                background: linear-gradient(90deg, transparent 0%, #ff3a5c 20%, #ff3a5c 49%, #44eecc 51%, #44eecc 80%, transparent 100%);
                opacity: 0.7;
            }}
            .nl-master-title {{ display: flex; align-items: center; gap: 12px; }}
            .nl-master-icon {{
                width: 40px; height: 40px; border-radius: 10px;
                background: linear-gradient(135deg, #141c28, #1e2a38); border: 1px solid #2a3a4a;
                display: flex; align-items: center; justify-content: center; font-size: 18px;
                box-shadow: 0 0 16px rgba(68,238,204,.15); flex-shrink: 0;
            }}
            .nl-master-text h2 {{
                font-size: clamp(13px, 2.5vw, 16px); font-weight: 700; color: #ddeeff;
                letter-spacing: 3px; text-transform: uppercase; text-shadow: 0 0 20px rgba(68,238,204,.35);
            }}
            .nl-master-text p {{ font-size: 10px; color: #667788; margin-top: 3px; letter-spacing: 2px; font-weight: 600; text-transform: uppercase; }}
            .nl-master-badges {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
            .nl-ce-badge {{
                background: rgba(255,58,92,.12); border: 1px solid rgba(255,58,92,.5); color: #ff3a5c;
                padding: 6px 18px; border-radius: 20px; font-size: 11px; font-weight: 800; letter-spacing: 2px;
                text-shadow: 0 0 10px rgba(255,58,92,.6); box-shadow: 0 0 12px rgba(255,58,92,.15);
            }}
            .nl-atm-badge {{
                background: rgba(68,238,204,.12); border: 1px solid rgba(68,238,204,.5); color: #44eecc;
                padding: 6px 18px; border-radius: 20px; font-size: 11px; font-weight: 800; letter-spacing: 2px;
                text-shadow: 0 0 10px rgba(68,238,204,.6); box-shadow: 0 0 12px rgba(68,238,204,.15);
            }}
            .nl-pe-badge {{
                background: rgba(0,230,118,.12); border: 1px solid rgba(0,230,118,.5); color: #00e676;
                padding: 6px 18px; border-radius: 20px; font-size: 11px; font-weight: 800; letter-spacing: 2px;
                text-shadow: 0 0 10px rgba(0,230,118,.6); box-shadow: 0 0 12px rgba(0,230,118,.15);
            }}
            .nl-live-dot {{
                width: 8px; height: 8px; border-radius: 50%; background: #44eecc;
                box-shadow: 0 0 10px #44eecc; animation: nl-pulse 1.5s ease-in-out infinite; flex-shrink: 0;
            }}
            @keyframes nl-pulse {{ 0%,100% {{ opacity:1;transform:scale(1); }} 50% {{ opacity:0.5;transform:scale(0.8); }} }}
            .nl-panels {{ display: grid; grid-template-columns: 1fr 1fr; }}
            .nl-panel {{ overflow-x: auto; }}
            .nl-panel.nl-panel-ce {{ border-right: 2px solid #1e2a38; }}
            .nl-panel-hdr {{ display: flex; align-items: center; gap: 10px; padding: 14px 20px; position: relative; flex-wrap: wrap; }}
            .nl-panel-ce .nl-panel-hdr {{ background: linear-gradient(135deg, #1a0610, #110310); border-bottom: 2px solid #ff3a5c; }}
            .nl-panel-pe .nl-panel-hdr {{ background: linear-gradient(135deg, #041410, #030e0a); border-bottom: 2px solid #44eecc; }}
            .nl-panel-hdr-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
            .nl-ce-dot {{ background: #ff3a5c; box-shadow: 0 0 10px #ff3a5c; }}
            .nl-pe-dot {{ background: #44eecc; box-shadow: 0 0 10px #44eecc; }}
            .nl-panel-hdr-title {{ font-size: clamp(10px, 2vw, 13px); font-weight: 800; letter-spacing: 2.5px; text-transform: uppercase; }}
            .nl-ce-title {{ color: #ff3a5c; text-shadow: 0 0 12px rgba(255,58,92,.5); }}
            .nl-pe-title {{ color: #44eecc; text-shadow: 0 0 12px rgba(68,238,204,.5); }}
            .nl-panel-hdr-sub {{ margin-left: auto; font-size: 9px; font-weight: 700; letter-spacing: 1.5px; }}
            .nl-ce-sub {{ color: rgba(255,58,92,.45); }} .nl-pe-sub {{ color: rgba(68,238,204,.45); }}
            .nl-col-hdr-row {{ background: #0a1018; border-bottom: 1px solid #1e2a38; }}
            .nl-th {{ font-size: 9px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #667788; padding: 8px 10px; text-align: left; white-space: nowrap; }}
            .nl-th:first-child {{ padding-left: 16px; }} .nl-th:last-child {{ padding-right: 16px; }}
            .nl-table {{ width: 100%; border-collapse: collapse; min-width: 380px; }}
            .nl-row {{ border-bottom: 1px solid #0c1418; transition: background .15s; cursor: default; }}
            .nl-row:last-child {{ border-bottom: none; }}
            .nl-panel-ce .nl-row:hover {{ background: rgba(255,58,92,.03); }}
            .nl-panel-pe .nl-row:hover {{ background: rgba(68,238,204,.03); }}
            .nl-td-rank {{ padding: 12px 8px 12px 14px; width: 42px; vertical-align: middle; }}
            .nl-td-strike,.nl-td-type,.nl-td-oi,.nl-td-chng,.nl-td-ltp {{ padding: 12px 8px; vertical-align: middle; }}
            .nl-td-oi {{ min-width: 100px; }}
            .nl-td-vol {{ padding: 12px 14px 12px 8px; vertical-align: middle; }}
            .nl-rank {{ width: 30px; height: 30px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 800; font-family: 'IBM Plex Mono', monospace; flex-shrink: 0; }}
            .nl-strike-val {{ font-family: 'IBM Plex Mono', monospace; font-size: clamp(12px, 2vw, 15px); font-weight: 700; color: #ddeeff; letter-spacing: -0.3px; }}
            .nl-tbadge {{ display: inline-block; padding: 3px 9px; border-radius: 6px; font-size: 10px; font-weight: 800; letter-spacing: 1px; }}
            .nl-itm {{ background: rgba(68,238,204,.1); color: #44eecc; border: 1px solid rgba(68,238,204,.4); }}
            .nl-atm {{ background: rgba(68,238,204,.08); color: #66ffee; border: 1px solid rgba(68,238,204,.35); }}
            .nl-otm {{ background: rgba(80,140,200,.1); color: #7ab4d8; border: 1px solid rgba(80,140,200,.35); }}
            .nl-oi-wrap {{ display: flex; flex-direction: column; gap: 5px; }}
            .nl-oi-val {{ font-family: 'IBM Plex Mono', monospace; font-size: clamp(12px, 2vw, 14px); font-weight: 700; }}
            .nl-oi-ce {{ color: #ff6680; text-shadow: 0 0 8px rgba(255,58,92,.3); }}
            .nl-oi-pe {{ color: #44eecc; text-shadow: 0 0 8px rgba(68,238,204,.3); }}
            .nl-bar-track {{ height: 5px; background: #0e1420; border-radius: 3px; overflow: hidden; width: 100%; max-width: 100px; }}
            .nl-bar-fill {{ height: 100%; border-radius: 3px; min-width: 3px; }}
            .nl-bar-ce {{ background: linear-gradient(90deg, #ff3a5c44, #ff3a5c); box-shadow: 0 0 5px #ff3a5c55; }}
            .nl-bar-pe {{ background: linear-gradient(90deg, #44eecc44, #44eecc); box-shadow: 0 0 5px #44eecc55; }}
            .nl-chng {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 700; }}
            .nl-chng-up {{ color: #44eecc; }} .nl-chng-dn {{ color: #ff4d6d; }} .nl-chng-flat {{ color: #556070; }}
            .nl-ltp {{ font-family: 'IBM Plex Mono', monospace; font-size: clamp(12px, 2vw, 14px); font-weight: 800; }}
            .nl-ltp-ce {{ color: #ffaacc; }} .nl-ltp-pe {{ color: #88ffee; }}
            .nl-vol {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600; color: #667788; }}
            .nl-footer {{
                background: #080c12; border-top: 1px solid #1e2a38; padding: 10px 24px;
                display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;
            }}
            .nl-footer-l {{ font-size: 9px; color: #556070; letter-spacing: 2px; font-weight: 700; text-transform: uppercase; }}
            .nl-footer-r {{ display: flex; align-items: center; gap: 7px; font-size: 9px; color: #44eecc; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }}
            @media (max-width: 700px) {{
                .nl-panels {{ grid-template-columns: 1fr; }}
                .nl-panel.nl-panel-ce {{ border-right: none; border-bottom: 2px solid #1e2a38; }}
                .nl-td-vol, .nl-th:last-child {{ display: none; }}
            }}
            @media (max-width: 420px) {{ .nl-td-chng, .nl-th:nth-child(5) {{ display: none; }} }}
        </style>
        <div class="nl-wrap">
            <div class="nl-master-hdr">
                <div class="nl-master-title">
                    <div class="nl-master-icon">&#9651;</div>
                    <div class="nl-master-text">
                        <h2>Top 10 Open Interest</h2>
                        <p>NIFTY &middot; ±10 Strikes from ATM &middot; Highest OI in Window</p>
                    </div>
                </div>
                <div class="nl-master-badges">
                    <span class="nl-ce-badge">5 CE</span>
                    <span class="nl-atm-badge">ATM &#8377;{atm_strike:,}</span>
                    <span class="nl-pe-badge">5 PE</span>
                    <div class="nl-live-dot"></div>
                </div>
            </div>
            <div class="nl-panels">
                <div class="nl-panel nl-panel-ce">
                    <div class="nl-panel-hdr">
                        <div class="nl-panel-hdr-dot nl-ce-dot"></div>
                        <span class="nl-panel-hdr-title nl-ce-title">Top 5 Call Options (CE)</span>
                        <span class="nl-panel-hdr-sub nl-ce-sub">10 STRIKES ABOVE ATM</span>
                    </div>
                    <table class="nl-table">
                        <thead class="nl-col-hdr-row"><tr>{col_heads}</tr></thead>
                        <tbody>{ce_rows_html}</tbody>
                    </table>
                </div>
                <div class="nl-panel nl-panel-pe">
                    <div class="nl-panel-hdr">
                        <div class="nl-panel-hdr-dot nl-pe-dot"></div>
                        <span class="nl-panel-hdr-title nl-pe-title">Top 5 Put Options (PE)</span>
                        <span class="nl-panel-hdr-sub nl-pe-sub">10 STRIKES BELOW ATM</span>
                    </div>
                    <table class="nl-table">
                        <thead class="nl-col-hdr-row"><tr>{col_heads}</tr></thead>
                        <tbody>{pe_rows_html}</tbody>
                    </table>
                </div>
            </div>
            <div class="nl-footer">
                <span class="nl-footer-l">NEON LEDGER &middot; TOP OI &middot; ±10 ATM STRIKES WINDOW</span>
                <span class="nl-footer-r"><div class="nl-live-dot"></div>LIVE</span>
            </div>
        </div>'''

    # =========================================================================
    # WIDGET 02 — PLASMA RADIAL | PHANTOM SLATE EDITION
    # =========================================================================
    def _build_oc_plasma_widget(self, oc_analysis):
        pcr          = oc_analysis.get('pcr', 0)
        max_pain     = oc_analysis.get('max_pain', 'N/A')
        oi_sentiment = oc_analysis.get('oi_sentiment', 'N/A')
        call_buildup = oc_analysis.get('call_buildup', 0)
        put_buildup  = oc_analysis.get('put_buildup', 0)
        avg_call_iv  = oc_analysis.get('avg_call_iv', 0)
        avg_put_iv   = oc_analysis.get('avg_put_iv', 0)
        resistances  = oc_analysis.get('resistances', [])
        supports     = oc_analysis.get('supports', [])

        if oi_sentiment == 'Bullish':
            sent_col = '#44eecc'; sent_bg = 'rgba(68,238,204,.12)'; sent_brd = '#22aa8866'; sent_icon = '&#8679;'
        else:
            sent_col = '#ff6070'; sent_bg = 'rgba(255,60,80,.12)';  sent_brd = '#cc223366'; sent_icon = '&#8681;'

        circ      = 408.4
        pcr_ratio = min(pcr / 2.0, 1.0)
        arc_len   = pcr_ratio * (circ * 0.69)
        arc_offset = -(circ * 0.155)

        if pcr >= 1.2:   arc_col1, arc_col2 = '#22aa88', '#44eecc'
        elif pcr >= 1.0: arc_col1, arc_col2 = '#1e6688', '#44eecc'
        elif pcr >= 0.8: arc_col1, arc_col2 = '#886600', '#ccaa00'
        else:            arc_col1, arc_col2 = '#cc2233', '#ff6070'

        r_bars_html = ''
        for idx, level in enumerate(resistances):
            bar_w = max(15, 85 - idx * 20)
            r_bars_html += f'''
            <div class="w2oc-level-row">
                <span class="w2oc-level-tag w2oc-r-tag">R{idx+1}</span>
                <div class="w2oc-level-track"><div class="w2oc-level-fill w2oc-fill-r" style="width:{bar_w}%;"></div></div>
                <span class="w2oc-level-price w2oc-r-price">&#8377;{level:,.0f}</span>
            </div>'''

        s_bars_html = ''
        for idx, level in enumerate(supports):
            bar_w = max(15, 85 - idx * 20)
            s_bars_html += f'''
            <div class="w2oc-level-row">
                <span class="w2oc-level-tag w2oc-s-tag">S{idx+1}</span>
                <div class="w2oc-level-track"><div class="w2oc-level-fill w2oc-fill-s" style="width:{bar_w}%;"></div></div>
                <span class="w2oc-level-price w2oc-s-price">&#8377;{level:,.0f}</span>
            </div>'''

        def fmt_millions(val):
            if abs(val) >= 1_000_000: return f"{val/1_000_000:.1f}M"
            elif abs(val) >= 1_000:   return f"{val/1_000:.0f}K"
            return str(int(val))

        call_b_str = fmt_millions(call_buildup)
        put_b_str  = fmt_millions(put_buildup)

        return f'''
        <style>
            .w2oc-wrap {{
                font-family: 'Chakra Petch', sans-serif;
                background: linear-gradient(135deg, #0c1018, #0e1420);
                border: 1px solid #1e2a38; border-radius: 8px; overflow: hidden;
                box-shadow: 0 0 50px rgba(68,238,204,.04), 0 20px 60px rgba(0,0,0,.9);
            }}
            .w2oc-hdr {{
                background: linear-gradient(135deg, #0e1420, #141c28);
                border-bottom: 1px solid #1e2a38; padding: 14px 22px;
                display: flex; align-items: center; gap: 12px; position: relative; flex-wrap: wrap;
            }}
            .w2oc-hdr::after {{
                content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
                background: linear-gradient(90deg, transparent, #1e6688, #44eeccaa, #1e6688, transparent);
            }}
            .w2oc-hdr-icon {{
                width: 36px; height: 36px; border-radius: 10px;
                background: linear-gradient(135deg, #1a2840, #1e3050);
                display: flex; align-items: center; justify-content: center; font-size: 17px; flex-shrink: 0;
                box-shadow: 0 0 16px rgba(68,238,204,.25), inset 0 1px 0 rgba(255,255,255,.05);
            }}
            .w2oc-hdr-text h3 {{
                font-size: clamp(12px, 2.5vw, 14px); font-weight: 700; color: #ddeeff;
                letter-spacing: 2.5px; text-transform: uppercase; text-shadow: 0 0 18px rgba(68,238,204,.4);
            }}
            .w2oc-hdr-text p {{ font-size: 10px; color: #667788; margin-top: 3px; letter-spacing: 1.5px; font-weight: 600; }}
            .w2oc-hdr-badge {{
                margin-left: auto; background: rgba(68,238,204,.1); border: 1px solid #22aa8877; color: #44eecc;
                padding: 5px 16px; border-radius: 20px; font-size: 10px; font-weight: 700; letter-spacing: 2px;
                text-shadow: 0 0 10px rgba(68,238,204,.6); animation: w2oc-pulse 2s ease-in-out infinite;
            }}
            @keyframes w2oc-pulse {{ 0%,100% {{ box-shadow: 0 0 0 0 rgba(68,238,204,.2); }} 50% {{ box-shadow: 0 0 0 6px rgba(68,238,204,0); }} }}
            .w2oc-body {{ display: grid; grid-template-columns: 240px 1fr; }}
            .w2oc-gauge-col {{
                padding: 24px 16px; border-right: 1px solid #1e2a38; background: #0a1018;
                display: flex; flex-direction: column; align-items: center; gap: 16px;
            }}
            .w2oc-gauge-wrap {{ position: relative; width: 170px; height: 170px; }}
            .w2oc-gauge-wrap svg {{ position: absolute; inset: 0; width: 100%; height: 100%; filter: drop-shadow(0 0 8px {arc_col2}44); }}
            .w2oc-gauge-center {{ position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 2px; }}
            .w2oc-gauge-lbl {{ font-size: 9px; color: #667788; letter-spacing: 3px; text-transform: uppercase; }}
            .w2oc-gauge-val {{ font-family: 'IBM Plex Mono', monospace; font-size: 32px; font-weight: 700; color: #ddeeff; letter-spacing: -1px; text-shadow: 0 0 24px {arc_col2}; }}
            .w2oc-gauge-sub {{ font-size: 9px; color: #667788; letter-spacing: 2px; }}
            .w2oc-gauge-title {{ font-size: 11px; font-weight: 700; color: {arc_col2}; letter-spacing: 2px; text-transform: uppercase; text-shadow: 0 0 12px {arc_col2}88; }}
            .w2oc-sent-pill {{ background: {sent_bg}; border: 1px solid {sent_brd}; border-radius: 10px; padding: 10px 16px; text-align: center; width: 100%; }}
            .w2oc-sent-lbl {{ font-size: 9px; color: #44eecc; letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 5px; font-weight: 700; }}
            .w2oc-sent-val {{ font-size: 20px; font-weight: 700; color: {sent_col}; text-shadow: 0 0 16px {sent_col}88; letter-spacing: 1px; }}
            .w2oc-maxpain {{ background: rgba(68,238,204,.06); border: 1px solid #1e3a3a; border-radius: 10px; padding: 10px 16px; text-align: center; width: 100%; }}
            .w2oc-maxpain-lbl {{ font-size: 9px; color: #44eecc; letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 5px; font-weight: 700; }}
            .w2oc-maxpain-val {{ font-family: 'IBM Plex Mono', monospace; font-size: 20px; font-weight: 700; color: #44eecc; text-shadow: 0 0 16px rgba(68,238,204,.6); }}
            .w2oc-maxpain-note {{ font-size: 9px; color: #667788; margin-top: 4px; letter-spacing: 1px; }}
            .w2oc-right {{ padding: 20px 22px; display: flex; flex-direction: column; gap: 18px; }}
            .w2oc-stats-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
            .w2oc-stat-card {{ background: rgba(68,238,204,.04); border: 1px solid #1e2a38; border-radius: 10px; padding: 14px 16px; position: relative; overflow: hidden; }}
            .w2oc-stat-card::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background:var(--sc-accent); box-shadow:0 0 8px var(--sc-accent); }}
            .w2oc-stat-card-lbl {{ font-size: 10px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; color: #88a0b8; }}
            .w2oc-stat-card-val {{ font-family: 'IBM Plex Mono', monospace; font-size: clamp(18px, 3vw, 22px); font-weight: 700; color: var(--sc-col); text-shadow: 0 0 14px var(--sc-accent); }}
            .w2oc-stat-card-sub {{ font-size: 10px; margin-top: 5px; color: var(--sc-col); font-family: 'IBM Plex Mono', monospace; opacity: 0.55; }}
            .w2oc-div {{ height: 1px; background: linear-gradient(90deg, transparent, #1e2a38, transparent); }}
            .w2oc-levels-section {{ display: flex; flex-direction: column; gap: 10px; }}
            .w2oc-levels-title {{ font-size: 10px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 2px; display: flex; align-items: center; gap: 8px; }}
            .w2oc-level-row {{ display: flex; align-items: center; gap: 12px; }}
            .w2oc-level-tag {{ font-size: 10px; font-weight: 800; width: 28px; height: 28px; border-radius: 6px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
            .w2oc-r-tag {{ background:rgba(255,64,80,.12);border:1px solid rgba(255,64,80,.45);color:#ff5060; }}
            .w2oc-s-tag {{ background:rgba(68,238,204,.12);border:1px solid rgba(68,238,204,.45);color:#44eecc; }}
            .w2oc-level-track {{ flex: 1; height: 8px; background: #0e1420; border-radius: 4px; overflow: hidden; }}
            .w2oc-level-fill {{ height: 100%; border-radius: 4px; }}
            .w2oc-fill-r {{ background: linear-gradient(90deg,#ff405033,#ff4050cc); box-shadow:0 0 6px #ff405055; }}
            .w2oc-fill-s {{ background: linear-gradient(90deg,#44eecc33,#44eecccc); box-shadow:0 0 6px #44eecc55; }}
            .w2oc-level-price {{ font-family: 'IBM Plex Mono', monospace; font-size: clamp(13px, 2vw, 16px); font-weight: 700; min-width: 80px; text-align: right; flex-shrink: 0; }}
            .w2oc-r-price {{ color: #ff8090; text-shadow: 0 0 8px #ff405055; }}
            .w2oc-s-price {{ color: #44eecc; text-shadow: 0 0 8px #44eecc55; }}
            .w2oc-footer {{ background: #080c12; border-top: 1px solid #1e2a38; padding: 10px 22px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
            .w2oc-footer-l {{ font-size: 9px; color: #556070; letter-spacing: 1.5px; font-weight: 600; }}
            .w2oc-footer-r {{ display: flex; align-items: center; gap: 6px; font-size: 9px; color: #44eecc; letter-spacing: 2px; font-weight: 700; }}
            .w2oc-footer-dot {{ width: 6px; height: 6px; border-radius: 50%; background: #44eecc; box-shadow: 0 0 8px #44eecc; animation: w2oc-pulse 1.5s ease-in-out infinite; }}
            @media (max-width: 700px) {{
                .w2oc-body {{ grid-template-columns: 1fr; }}
                .w2oc-gauge-col {{ border-right: none; border-bottom: 1px solid #1e2a38; padding: 20px; }}
                .w2oc-gauge-wrap {{ width: 140px; height: 140px; }}
                .w2oc-gauge-val {{ font-size: 26px; }}
                .w2oc-stats-row {{ grid-template-columns: 1fr; gap: 8px; }}
            }}
        </style>
        <div class="w2oc-wrap">
            <div class="w2oc-hdr">
                <div class="w2oc-hdr-icon">&#128202;</div>
                <div class="w2oc-hdr-text">
                    <h3>Option Chain Analysis</h3>
                    <p>NIFTY &middot; WEEKLY EXPIRY &middot; LIVE DATA</p>
                </div>
                <div class="w2oc-hdr-badge">&#9679; LIVE</div>
            </div>
            <div class="w2oc-body">
                <div class="w2oc-gauge-col">
                    <div class="w2oc-gauge-wrap">
                        <svg viewBox="0 0 170 170">
                            <defs>
                                <linearGradient id="pcr-arc-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                                    <stop offset="0%" stop-color="{arc_col1}"/>
                                    <stop offset="100%" stop-color="{arc_col2}"/>
                                </linearGradient>
                                <filter id="arc-glow">
                                    <feGaussianBlur stdDeviation="3" result="blur"/>
                                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                                </filter>
                            </defs>
                            <circle cx="85" cy="85" r="80" fill="none" stroke="#141c28" stroke-width="1"/>
                            <circle cx="85" cy="85" r="65" fill="none" stroke="#1e2a38" stroke-width="11"
                                stroke-dasharray="{circ * 0.69:.1f} {circ:.1f}" stroke-dashoffset="{arc_offset:.1f}" stroke-linecap="round"/>
                            <circle cx="85" cy="85" r="65" fill="none" stroke="{arc_col2}" stroke-width="18"
                                stroke-dasharray="{arc_len:.1f} {circ:.1f}" stroke-dashoffset="{arc_offset:.1f}" stroke-linecap="round" opacity="0.08"/>
                            <circle cx="85" cy="85" r="65" fill="none" stroke="url(#pcr-arc-grad)" stroke-width="11"
                                stroke-dasharray="{arc_len:.1f} {circ:.1f}" stroke-dashoffset="{arc_offset:.1f}"
                                stroke-linecap="round" filter="url(#arc-glow)"/>
                            <circle cx="85" cy="85" r="52" fill="none" stroke="#141c28" stroke-width="1"/>
                        </svg>
                        <div class="w2oc-gauge-center">
                            <span class="w2oc-gauge-lbl">PUT / CALL</span>
                            <span class="w2oc-gauge-val">{pcr:.2f}</span>
                            <span class="w2oc-gauge-sub">PCR RATIO</span>
                        </div>
                    </div>
                    <div class="w2oc-gauge-title">&#9679; PCR GAUGE</div>
                    <div class="w2oc-sent-pill">
                        <div class="w2oc-sent-lbl">OI Sentiment</div>
                        <div class="w2oc-sent-val">{sent_icon} {oi_sentiment.upper()}</div>
                    </div>
                    <div class="w2oc-maxpain">
                        <div class="w2oc-maxpain-lbl">Max Pain Strike</div>
                        <div class="w2oc-maxpain-val">&#8377;{max_pain:,}</div>
                        <div class="w2oc-maxpain-note">Price magnet at expiry</div>
                    </div>
                </div>
                <div class="w2oc-right">
                    <div class="w2oc-stats-row">
                        <div class="w2oc-stat-card" style="--sc-accent:#ff5060;--sc-col:#ff8090;">
                            <div class="w2oc-stat-card-lbl">Call Buildup (OI)</div>
                            <div class="w2oc-stat-card-val">&#8679; {call_b_str}</div>
                            <div class="w2oc-stat-card-sub">Avg IV: {avg_call_iv:.1f}%</div>
                        </div>
                        <div class="w2oc-stat-card" style="--sc-accent:#44eecc;--sc-col:#66ffee;">
                            <div class="w2oc-stat-card-lbl">Put Buildup (OI)</div>
                            <div class="w2oc-stat-card-val">&#8679; {put_b_str}</div>
                            <div class="w2oc-stat-card-sub">Avg IV: {avg_put_iv:.1f}%</div>
                        </div>
                    </div>
                    <div class="w2oc-div"></div>
                    <div class="w2oc-levels-section">
                        <div class="w2oc-levels-title" style="color:#ff5060;">
                            <span style="width:8px;height:8px;border-radius:50%;background:#ff5060;display:inline-block;box-shadow:0 0 8px #ff5060;flex-shrink:0;"></span>
                            OI RESISTANCE WALLS
                        </div>
                        {r_bars_html if r_bars_html else '<div style="color:#667788;font-size:12px;padding:6px 0;">No resistance data</div>'}
                    </div>
                    <div class="w2oc-div"></div>
                    <div class="w2oc-levels-section">
                        <div class="w2oc-levels-title" style="color:#44eecc;">
                            <span style="width:8px;height:8px;border-radius:50%;background:#44eecc;display:inline-block;box-shadow:0 0 8px #44eecc;flex-shrink:0;"></span>
                            OI SUPPORT FLOORS
                        </div>
                        {s_bars_html if s_bars_html else '<div style="color:#667788;font-size:12px;padding:6px 0;">No support data</div>'}
                    </div>
                </div>
            </div>
            <div class="w2oc-footer">
                <span class="w2oc-footer-l">PLASMA RADIAL &middot; MAX PAIN = MIN BUYER PAIN (CORRECTED)</span>
                <span class="w2oc-footer-r"><div class="w2oc-footer-dot"></div>NIFTY &middot; WEEKLY EXPIRY</span>
            </div>
        </div>'''

    # =========================================================================
    # PIVOT POINTS WIDGET — PHANTOM SLATE EDITION
    # =========================================================================
    def _build_pivot_widget(self, pivot_points, current_price, nearest_levels):
        pp = pivot_points

        def dist(val):
            if val is None: return 'N/A'
            return f"{val - current_price:+.2f}"

        def is_nearest_r(val): return val == nearest_levels.get('nearest_resistance')
        def is_nearest_s(val): return val == nearest_levels.get('nearest_support')

        nr = nearest_levels.get('nearest_resistance')
        ns = nearest_levels.get('nearest_support')
        if nr and ns:
            zone_text   = f"Between {self._nearest_level_name(pp, ns)} and {self._nearest_level_name(pp, nr)}"
            above_dist  = current_price - pp.get('pivot', current_price)
            zone_detail = f"+{above_dist:.2f} above PP" if above_dist >= 0 else f"{above_dist:.2f} below PP"
        elif nr:
            zone_text   = f"Below {self._nearest_level_name(pp, nr)}"
            zone_detail = f"Next R: &#8377;{nr}"
        elif ns:
            zone_text   = f"Above {self._nearest_level_name(pp, ns)}"
            zone_detail = f"Next S: &#8377;{ns}"
        else:
            zone_text   = "At Pivot Zone"
            zone_detail = f"PP: &#8377;{pp.get('pivot','N/A')}"

        s1_val      = pp.get('s1', current_price - 100)
        r1_val      = pp.get('r1', current_price + 100)
        total_range = r1_val - s1_val
        if total_range > 0:
            dot_pct = max(5, min(95, ((current_price - s1_val) / total_range) * 100))
        else:
            dot_pct = 50

        def res_row(lbl, val, key):
            is_near = is_nearest_r(val)
            if key == 'r1':
                nc = '#ff6070'; pc = '#ffcccc'; ps = '18px'
                rb = 'background:rgba(255,60,80,0.05);border-left:3px solid rgba(255,96,112,0.5);'
            elif key == 'r2':
                nc = 'rgba(255,96,112,0.75)'; pc = 'rgba(255,180,180,0.75)'; ps = '16px'; rb = ''
            else:
                nc = 'rgba(255,96,112,0.45)'; pc = 'rgba(255,180,180,0.45)'; ps = '15px'; rb = ''
            near_html = '<span class="w1-near-tag w1-near-r">NEAREST&nbsp;R</span>' if is_near else ''
            icon_html = '<span class="w1-icon w1-icon-r">&#9650;</span>' if lbl == 'R1' else ''
            return f'''
                <div class="w1-level-row" style="{rb}">
                    <span class="w1-lv-name" style="color:{nc};">{lbl}</span>
                    <span style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                        <span class="w1-lv-price" style="color:{pc};font-size:{ps};">&#8377;{val}</span>
                        {near_html}
                    </span>
                    {icon_html}
                </div>'''

        def sup_row(lbl, val, key):
            is_near = is_nearest_s(val)
            if key == 's1':
                nc = '#44eecc'; pc = '#ccffee'; ps = '18px'
                rb = 'background:rgba(68,238,204,0.04);border-right:3px solid rgba(68,238,204,0.5);'
            elif key == 's2':
                nc = 'rgba(68,238,204,0.75)'; pc = 'rgba(180,255,230,0.75)'; ps = '16px'; rb = ''
            else:
                nc = 'rgba(68,238,204,0.45)'; pc = 'rgba(180,255,230,0.45)'; ps = '15px'; rb = ''
            near_html = '<span class="w1-near-tag w1-near-s">NEAREST&nbsp;S</span>' if is_near else ''
            icon_html = '<span class="w1-icon w1-icon-s">&#9660;</span>' if lbl == 'S1' else ''
            return f'''
                <div class="w1-level-row w1-sup-row" style="{rb}">
                    {icon_html}
                    <span style="display:flex;align-items:center;justify-content:flex-end;gap:8px;flex-wrap:wrap;">
                        {near_html}
                        <span class="w1-lv-price" style="color:{pc};font-size:{ps};">&#8377;{val}</span>
                    </span>
                    <span class="w1-lv-name" style="color:{nc};text-align:right;">{lbl}</span>
                </div>'''

        res_rows_html = (res_row('R3', pp.get('r3','N/A'), 'r3') +
                         res_row('R2', pp.get('r2','N/A'), 'r2') +
                         res_row('R1', pp.get('r1','N/A'), 'r1'))
        sup_rows_html = (sup_row('S1', pp.get('s1','N/A'), 's1') +
                         sup_row('S2', pp.get('s2','N/A'), 's2') +
                         sup_row('S3', pp.get('s3','N/A'), 's3'))

        return f'''
        <style>
            .w1-pv {{
                background: #0c1018; border: 1px solid #1e2a38; border-radius: 8px; overflow: hidden;
                font-family: 'Chakra Petch', 'Segoe UI', sans-serif;
                box-shadow: 0 0 0 1px #141c28, 0 16px 50px rgba(0,0,0,.9); width: 100%;
            }}
            .w1-hdr {{
                background: linear-gradient(135deg, #0e1420, #141c28); padding: 14px 20px;
                display: flex; align-items: center; justify-content: space-between;
                border-bottom: 2px solid #44eecc; flex-wrap: wrap; gap: 10px;
            }}
            .w1-hdr-title {{ font-size: clamp(13px, 3vw, 15px); font-weight: 700; color: #ddeeff; letter-spacing: 2px; }}
            .w1-hdr-sub   {{ font-size: 11px; color: #667788; margin-top: 3px; letter-spacing: .5px; }}
            .w1-hdr-badge {{ background: #44eecc; color: #080c12; font-size: 10px; font-weight: 800; padding: 4px 14px; border-radius: 20px; letter-spacing: 2px; }}
            .w1-gauge {{ padding: 16px 20px 4px; background: #0a1018; border-bottom: 1px solid #1e2a38; }}
            .w1-gauge-track {{
                height: 10px; border-radius: 20px; position: relative; overflow: visible;
                background: linear-gradient(90deg, rgba(68,238,204,.1) 0%, rgba(68,238,204,.3) 30%, rgba(255,255,255,.04) 50%, rgba(255,96,112,.3) 70%, rgba(255,96,112,.1) 100%);
                border: 1px solid #1e2a38;
            }}
            .w1-gdot {{
                position: absolute; left: {dot_pct:.1f}%; top: 50%; transform: translate(-50%, -50%);
                width: 18px; height: 18px; background: #44eecc; border-radius: 50%;
                border: 3px solid #0a1018; box-shadow: 0 0 0 2px #44eecc, 0 0 18px rgba(68,238,204,.8);
                animation: w1-pulse 2s ease-in-out infinite; z-index: 2;
            }}
            @keyframes w1-pulse {{
                0%,100% {{ box-shadow: 0 0 0 2px #44eecc, 0 0 18px rgba(68,238,204,.8); }}
                50%      {{ box-shadow: 0 0 0 3px #44eecc, 0 0 28px rgba(68,238,204,1); }}
            }}
            .w1-gauge-labels {{
                display: flex; justify-content: space-between; margin-top: 10px; padding-bottom: 10px;
                font-family: 'IBM Plex Mono', monospace; font-size: clamp(11px, 2vw, 13px); font-weight: 700; flex-wrap: wrap; gap: 4px;
            }}
            .w1-gl-s {{ color: #44eecc; }} .w1-gl-ltp {{ color: #ddeeff; font-size: 14px; }} .w1-gl-r {{ color: #ff6070; }}
            .w1-zone {{
                margin: 10px 16px; padding: 10px 16px; background: rgba(68,238,204,.05);
                border: 1px solid #1e3a3a; border-radius: 8px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
            }}
            .w1-zone-dot {{ width:9px;height:9px;border-radius:50%;background:#44eecc;flex-shrink:0;box-shadow:0 0 10px #44eecc;animation:w1-pulse 2s ease-in-out infinite; }}
            .w1-zone-text {{ font-size: clamp(12px, 2.5vw, 14px); font-weight: 700; color: #ddeeff; letter-spacing: .5px; }}
            .w1-zone-val  {{ margin-left: auto; font-size: 12px; color: #44eecc; font-family: 'IBM Plex Mono'; white-space: nowrap; }}
            .w1-candle {{ display: flex; margin: 0 16px 14px; border: 1px solid #1e2a38; border-radius: 8px; overflow: hidden; }}
            .w1-ci {{ flex: 1; padding: 10px 14px; border-right: 1px solid #1e2a38; }}
            .w1-ci:last-child {{ border-right: none; }}
            .w1-ci-lbl {{ font-size: 10px; color: #667788; letter-spacing: 1.5px; margin-bottom: 5px; text-transform: uppercase; }}
            .w1-ci-val {{ font-size: clamp(13px, 2vw, 15px); font-weight: 700; font-family: 'IBM Plex Mono'; }}
            .w1-ci-h .w1-ci-val {{ color: #ff8090; }} .w1-ci-l .w1-ci-val {{ color: #44eecc; }} .w1-ci-c .w1-ci-val {{ color: #88a0b8; }}
            .w1-grid {{ display: grid; grid-template-columns: 1fr auto 1fr; border-top: 1px solid #1e2a38; }}
            .w1-col-res {{ border-right: 1px solid #1e2a38; }}
            .w1-level-row {{
                display: flex; align-items: center; justify-content: space-between; padding: 13px 18px;
                border-bottom: 1px solid #0c1418; gap: 8px; min-height: 54px; transition: background .15s; cursor: default;
            }}
            .w1-level-row:last-child {{ border-bottom: none; }}
            .w1-level-row:hover {{ background: rgba(68,238,204,.03) !important; }}
            .w1-sup-row {{ flex-direction: row-reverse; }}
            .w1-lv-name  {{ font-size: clamp(12px, 2vw, 14px); font-weight: 700; letter-spacing: 1px; min-width: 26px; flex-shrink: 0; }}
            .w1-lv-price {{ font-family: 'IBM Plex Mono', monospace; font-weight: 700; letter-spacing: .5px; }}
            .w1-icon {{ width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;flex-shrink:0; }}
            .w1-icon-r {{ background:rgba(255,96,112,.12);color:#ff6070;border:1px solid rgba(255,96,112,.45); }}
            .w1-icon-s {{ background:rgba(68,238,204,.12);color:#44eecc;border:1px solid rgba(68,238,204,.45); }}
            .w1-near-tag {{ font-size:10px;padding:2px 8px;border-radius:6px;font-weight:800;letter-spacing:.5px;white-space:nowrap; }}
            .w1-near-r {{ background:rgba(255,96,112,.15);color:#ff6070;border:1px solid rgba(255,96,112,.5); }}
            .w1-near-s {{ background:rgba(68,238,204,.12);color:#44eecc;border:1px solid rgba(68,238,204,.45); }}
            .w1-pivot-col {{
                display: flex; flex-direction: column; align-items: center; justify-content: center;
                padding: 18px 16px; gap: 6px; background: rgba(68,238,204,.03);
                border-left: 1px solid #1e2a38; border-right: 1px solid #1e2a38; min-width: 130px;
            }}
            .w1-pp-tag  {{ font-size: 10px; color: #667788; letter-spacing: 2px; text-transform: uppercase; }}
            .w1-pp-val  {{ font-size: clamp(16px, 3vw, 20px); font-weight: 700; color: #44eecc; font-family: 'IBM Plex Mono', monospace; text-shadow: 0 0 12px rgba(68,238,204,.5); text-align: center; }}
            .w1-pp-dist {{ font-size: 12px; color: #667788; font-family: 'IBM Plex Mono'; }}
            .w1-pp-sep  {{ width: 36px; height: 1px; background: #1e2a38; margin: 3px 0; }}
            .w1-ltp-chip {{ background: #44eecc; color: #080c12; border-radius: 8px; padding: 7px 18px; text-align: center; margin-top: 4px; }}
            .w1-ltp-chip-lbl {{ font-size: 9px; font-weight: 800; letter-spacing: 2px; }}
            .w1-ltp-chip-val {{ font-size: clamp(13px, 2.5vw, 15px); font-weight: 800; font-family: 'IBM Plex Mono'; }}
            .w1-footer {{ display:flex;justify-content:space-between;align-items:center;padding:10px 20px;background:#080c12;border-top:1px solid #1e2a38;font-family:'IBM Plex Mono',monospace;font-size:12px;flex-wrap:wrap;gap:8px; }}
            .w1-footer-l {{ color: #556070; letter-spacing: 1px; }}
            .w1-footer-r {{ color: #44eecc; font-weight: 700; text-shadow: 0 0 8px rgba(68,238,204,.4); }}
            @media (max-width: 600px) {{
                .w1-grid {{ grid-template-columns: 1fr; }}
                .w1-pivot-col {{ border: none; border-top: 1px solid #1e2a38; border-bottom: 1px solid #1e2a38; flex-direction: row; justify-content: space-around; padding: 12px 16px; }}
                .w1-col-res {{ border-right: none; }}
                .w1-candle {{ flex-direction: column; }}
                .w1-ci {{ border-right: none; border-bottom: 1px solid #1e2a38; }}
                .w1-ci:last-child {{ border-bottom: none; }}
            }}
        </style>
        <div class="w1-pv">
            <div class="w1-hdr">
                <div>
                    <div class="w1-hdr-title">&#128205; PIVOT POINTS</div>
                    <div class="w1-hdr-sub">Traditional Method &middot; 30 Min &middot; Auto-calculated</div>
                </div>
                <div class="w1-hdr-badge">30 MIN</div>
            </div>
            <div class="w1-gauge">
                <div class="w1-gauge-track"><div class="w1-gdot"></div></div>
                <div class="w1-gauge-labels">
                    <span class="w1-gl-s">S1 &#8377;{pp.get('s1','N/A')}</span>
                    <span class="w1-gl-ltp">&#9650; &#8377;{current_price} LTP</span>
                    <span class="w1-gl-r">R1 &#8377;{pp.get('r1','N/A')}</span>
                </div>
            </div>
            <div class="w1-zone">
                <div class="w1-zone-dot"></div>
                <span class="w1-zone-text">{zone_text}</span>
                <span class="w1-zone-val">{zone_detail}</span>
            </div>
            <div class="w1-candle">
                <div class="w1-ci w1-ci-h"><div class="w1-ci-lbl">&#9650; PREV HIGH</div><div class="w1-ci-val">&#8377;{pp.get('prev_high','N/A')}</div></div>
                <div class="w1-ci w1-ci-l"><div class="w1-ci-lbl">&#9660; PREV LOW</div><div class="w1-ci-val">&#8377;{pp.get('prev_low','N/A')}</div></div>
                <div class="w1-ci w1-ci-c"><div class="w1-ci-lbl">&#9679; PREV CLOSE</div><div class="w1-ci-val">&#8377;{pp.get('prev_close','N/A')}</div></div>
            </div>
            <div class="w1-grid">
                <div class="w1-col-res">{res_rows_html}</div>
                <div class="w1-pivot-col">
                    <div class="w1-pp-tag">PIVOT POINT</div>
                    <div class="w1-pp-val">&#8377;{pp.get('pivot','N/A')}</div>
                    <div class="w1-pp-dist">{dist(pp.get('pivot'))} from LTP</div>
                    <div class="w1-pp-sep"></div>
                    <div class="w1-ltp-chip">
                        <div class="w1-ltp-chip-lbl">LTP</div>
                        <div class="w1-ltp-chip-val">&#8377;{current_price}</div>
                    </div>
                </div>
                <div class="w1-col-sup">{sup_rows_html}</div>
            </div>
            <div class="w1-footer">
                <span class="w1-footer-l">Traditional &middot; 30 Min Candle</span>
                <span class="w1-footer-r">LTP &#8377;{current_price}</span>
            </div>
        </div>'''

    def _nearest_level_name(self, pivot_points, value):
        mapping = {
            pivot_points.get('r1'): 'R1', pivot_points.get('r2'): 'R2', pivot_points.get('r3'): 'R3',
            pivot_points.get('s1'): 'S1', pivot_points.get('s2'): 'S2', pivot_points.get('s3'): 'S3',
            pivot_points.get('pivot'): 'PP',
        }
        return mapping.get(value, str(value))

    # =========================================================================
    # WIDGET 04 — BLOOMBERG TABLE | PHANTOM SLATE EDITION
    # =========================================================================
    def _build_sr_bloomberg_widget(self, tech_resistances, tech_supports, current_price):
        def strength_dots(distance_pct, level_type):
            abs_dist = abs(distance_pct)
            if abs_dist <= 0.3:   filled = 5
            elif abs_dist <= 0.6: filled = 4
            elif abs_dist <= 1.0: filled = 3
            elif abs_dist <= 1.5: filled = 2
            else:                 filled = 1
            dot_color   = '#ff4d6d' if level_type == 'R' else '#44eecc'
            empty_color = '#1e2a38'
            dots_html = ''
            for i in range(5):
                if i < filled:
                    dots_html += f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{dot_color};margin:0 2px;box-shadow:0 0 5px {dot_color}77;"></span>'
                else:
                    dots_html += f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{empty_color};border:1px solid #2a3a4a;margin:0 2px;"></span>'
            return dots_html

        def build_resistance_rows(resistances):
            rows = ''
            for idx, level in enumerate(resistances):
                label        = f"R{idx + 1}"
                dist         = level - current_price
                dist_pct     = (dist / current_price) * 100
                dist_str     = f"+{dist:.1f}"
                dist_pct_str = f"+{dist_pct:.2f}%"
                dots         = strength_dots(dist_pct, 'R')
                row_opacity  = '1' if idx == 0 else ('0.82' if idx == 1 else '0.65')
                price_size   = '20px' if idx == 0 else ('17px' if idx == 1 else '15px')
                rows += f'''
                <tr class="w4-row w4-row-r" style="opacity:{row_opacity};">
                    <td class="w4-td-label"><span class="w4-badge w4-badge-r">{label}</span></td>
                    <td class="w4-td-price"><span style="font-family:'IBM Plex Mono',monospace;font-size:{price_size};font-weight:800;color:#ddeeff;letter-spacing:-0.5px;">&#8377;{level:,.1f}</span></td>
                    <td class="w4-td-dist">
                        <span class="w4-dist w4-dist-r">{dist_str}</span>
                        <span class="w4-dist-pct w4-dist-pct-r">{dist_pct_str}</span>
                    </td>
                    <td class="w4-td-strength"><div class="w4-dots">{dots}</div></td>
                    <td class="w4-td-bar"><div class="w4-bar-track"><div class="w4-bar-fill w4-bar-r" style="width:{min(100, dist_pct * 30):.0f}%;"></div></div></td>
                </tr>'''
            return rows

        def build_support_rows(supports):
            rows = ''
            for idx, level in enumerate(supports):
                label        = f"S{idx + 1}"
                dist         = current_price - level
                dist_pct     = (dist / current_price) * 100
                dist_str     = f"-{dist:.1f}"
                dist_pct_str = f"-{dist_pct:.2f}%"
                dots         = strength_dots(dist_pct, 'S')
                row_opacity  = '1' if idx == 0 else ('0.82' if idx == 1 else '0.65')
                price_size   = '20px' if idx == 0 else ('17px' if idx == 1 else '15px')
                rows += f'''
                <tr class="w4-row w4-row-s" style="opacity:{row_opacity};">
                    <td class="w4-td-label"><span class="w4-badge w4-badge-s">{label}</span></td>
                    <td class="w4-td-price"><span style="font-family:'IBM Plex Mono',monospace;font-size:{price_size};font-weight:800;color:#ddeeff;letter-spacing:-0.5px;">&#8377;{level:,.1f}</span></td>
                    <td class="w4-td-dist">
                        <span class="w4-dist w4-dist-s">{dist_str}</span>
                        <span class="w4-dist-pct w4-dist-pct-s">{dist_pct_str}</span>
                    </td>
                    <td class="w4-td-strength"><div class="w4-dots">{dots}</div></td>
                    <td class="w4-td-bar"><div class="w4-bar-track"><div class="w4-bar-fill w4-bar-s" style="width:{min(100, dist_pct * 30):.0f}%;"></div></div></td>
                </tr>'''
            return rows

        resistance_rows_html = build_resistance_rows(tech_resistances)
        support_rows_html    = build_support_rows(tech_supports)

        return f'''
        <style>
            .w4-wrap {{
                font-family: 'Chakra Petch', 'Segoe UI', sans-serif; background: #0c1018;
                border: 1px solid #1e2a38; border-radius: 8px; overflow: hidden;
                box-shadow: 0 0 0 1px #141c28, 0 0 30px rgba(68,238,204,.03), 0 20px 50px rgba(0,0,0,.9);
            }}
            .w4-header {{ background: linear-gradient(135deg, #0e1420 0%, #141c28 100%); border-bottom: 1px solid #1e2a38; display: flex; }}
            .w4-hdr-half {{ flex: 1; padding: 14px 20px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
            .w4-hdr-half.resistance-hdr {{ border-right: 1px solid #1e2a38; border-bottom: 2px solid #ff4d6d; }}
            .w4-hdr-half.support-hdr {{ border-bottom: 2px solid #44eecc; }}
            .w4-hdr-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
            .w4-hdr-dot.r-dot {{ background: #ff4d6d; box-shadow: 0 0 8px #ff4d6d; }}
            .w4-hdr-dot.s-dot {{ background: #44eecc; box-shadow: 0 0 8px #44eecc; }}
            .w4-hdr-title {{ font-size: clamp(10px, 2vw, 12px); font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase; }}
            .w4-hdr-title.r-title {{ color: #ff4d6d; }} .w4-hdr-title.s-title {{ color: #44eecc; }}
            .w4-hdr-tf {{ margin-left: auto; font-size: 10px; font-weight: 600; letter-spacing: 1.5px; color: #667788; }}
            .w4-ltp-bar {{ background: #0a1018; border-bottom: 1px solid #1e2a38; display: flex; align-items: center; justify-content: center; padding: 10px 20px; gap: 12px; }}
            .w4-ltp-line {{ flex: 1; height: 1px; background: linear-gradient(90deg, transparent, #1e3a3a, transparent); }}
            .w4-ltp-chip {{ background: linear-gradient(135deg, #0a1820, #0c1c28); border: 1px solid #44eecc44; border-radius: 8px; padding: 8px 20px; display: flex; align-items: center; gap: 10px; }}
            .w4-ltp-label {{ font-size: 9px; font-weight: 700; letter-spacing: 2px; color: #22aa88; text-transform: uppercase; }}
            .w4-ltp-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 18px; font-weight: 800; color: #44eecc; text-shadow: 0 0 14px rgba(68,238,204,.45); }}
            .w4-body {{ display: grid; grid-template-columns: 1fr 1fr; }}
            .w4-col {{ padding: 4px 0; }} .w4-col.r-col {{ border-right: 1px solid #1e2a38; }}
            .w4-col-hdr {{ display: grid; grid-template-columns: 44px 1fr 90px 80px 1fr; padding: 6px 14px; border-bottom: 1px solid #1e2a38; }}
            .w4-col-hdr span {{ font-size: 9px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #667788; }}
            .w4-table {{ width: 100%; border-collapse: collapse; min-width: 220px; }}
            .w4-row {{ transition: background 0.15s; cursor: default; }}
            .w4-row:hover {{ background: rgba(68,238,204,.025); }}
            .w4-row-r, .w4-row-s {{ border-bottom: 1px solid #0c1418; }} .w4-row:last-child {{ border-bottom: none; }}
            .w4-td-label {{ padding: 14px 6px 14px 16px; width: 48px; vertical-align: middle; }}
            .w4-td-price {{ padding: 14px 8px; vertical-align: middle; }}
            .w4-td-dist {{ padding: 14px 8px; vertical-align: middle; white-space: nowrap; text-align: right; }}
            .w4-td-strength {{ padding: 14px 8px; vertical-align: middle; text-align: center; }}
            .w4-td-bar {{ padding: 14px 14px 14px 4px; vertical-align: middle; width: 60px; }}
            .w4-badge {{ display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; border-radius: 7px; font-size: 12px; font-weight: 800; }}
            .w4-badge-r {{ background: rgba(255,77,109,0.1); border: 1px solid rgba(255,77,109,0.4); color: #ff4d6d; }}
            .w4-badge-s {{ background: rgba(68,238,204,0.1); border: 1px solid rgba(68,238,204,0.4); color: #44eecc; }}
            .w4-dist {{ display: block; font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 700; }}
            .w4-dist-pct {{ display: block; font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 600; margin-top: 2px; }}
            .w4-dist-r {{ color: #ff8099; }} .w4-dist-pct-r {{ color: rgba(255,128,153,0.6); }}
            .w4-dist-s {{ color: #44eecc; }} .w4-dist-pct-s {{ color: rgba(68,238,204,0.6); }}
            .w4-dots {{ display: flex; align-items: center; justify-content: center; }}
            .w4-bar-track {{ height: 4px; background: #1e2a38; border-radius: 2px; overflow: hidden; }}
            .w4-bar-fill {{ height: 100%; border-radius: 2px; min-width: 4px; }}
            .w4-bar-r {{ background: linear-gradient(90deg, #ff4d6d44, #ff4d6d); }}
            .w4-bar-s {{ background: linear-gradient(90deg, #44eecc44, #44eecc); }}
            .w4-footer {{ background: #080c12; border-top: 1px solid #1e2a38; display: flex; justify-content: space-between; align-items: center; padding: 10px 20px; flex-wrap: wrap; gap: 8px; }}
            .w4-footer-left {{ font-size: 10px; color: #556070; letter-spacing: 1px; font-weight: 600; }}
            .w4-footer-right {{ font-size: 10px; color: #22aa88; font-family: 'IBM Plex Mono', monospace; font-weight: 700; }}
            .w4-footer-dot {{ width:5px;height:5px;border-radius:50%;background:#44eecc;display:inline-block;margin-right:6px;box-shadow:0 0 6px #44eecc;animation:w4-blink 2s ease-in-out infinite; }}
            @keyframes w4-blink {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.35; }} }}
            @media (max-width: 600px) {{
                .w4-body {{ grid-template-columns: 1fr; }}
                .w4-col.r-col {{ border-right: none; border-bottom: 2px solid #1e2a38; }}
                .w4-td-dist, .w4-td-bar {{ display: none; }}
            }}
        </style>
        <div class="w4-wrap">
            <div class="w4-header">
                <div class="w4-hdr-half resistance-hdr">
                    <div class="w4-hdr-dot r-dot"></div>
                    <span class="w4-hdr-title r-title">RESISTANCE LEVELS</span>
                    <span class="w4-hdr-tf">1H TIMEFRAME</span>
                </div>
                <div class="w4-hdr-half support-hdr">
                    <div class="w4-hdr-dot s-dot"></div>
                    <span class="w4-hdr-title s-title">SUPPORT LEVELS</span>
                    <span class="w4-hdr-tf">1H TIMEFRAME</span>
                </div>
            </div>
            <div class="w4-ltp-bar">
                <div class="w4-ltp-line"></div>
                <div class="w4-ltp-chip">
                    <span class="w4-ltp-label">LTP</span>
                    <span class="w4-ltp-value">&#8377;{current_price:,.2f}</span>
                </div>
                <div class="w4-ltp-line"></div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid #1e2a38;">
                <div class="w4-col-hdr" style="border-right:1px solid #1e2a38;">
                    <span></span><span>PRICE</span><span style="text-align:right;">DISTANCE</span><span style="text-align:center;">STRENGTH</span><span></span>
                </div>
                <div class="w4-col-hdr">
                    <span></span><span>PRICE</span><span style="text-align:right;">DISTANCE</span><span style="text-align:center;">STRENGTH</span><span></span>
                </div>
            </div>
            <div class="w4-body">
                <div class="w4-col r-col"><table class="w4-table"><tbody>{resistance_rows_html}</tbody></table></div>
                <div class="w4-col s-col"><table class="w4-table"><tbody>{support_rows_html}</tbody></table></div>
            </div>
            <div class="w4-footer">
                <span class="w4-footer-left">BLOOMBERG TABLE &middot; PRICE ACTION S/R &middot; 1H</span>
                <span class="w4-footer-right"><span class="w4-footer-dot"></span>LTP &#8377;{current_price:,.2f}</span>
            </div>
        </div>'''

    # =========================================================================
    # DARK TICKER CARD — PHANTOM SLATE EDITION
    # =========================================================================
    def _build_strike_ticker_card_widget(self, strike_recommendations, recommendation, tech_analysis):
        bias          = recommendation['bias']
        current_price = tech_analysis.get('current_price', 0)
        confidence    = recommendation['confidence']
        candle_close  = tech_analysis.get('candle_close_price', current_price)

        if bias == 'Bullish':
            hdr_accent = '#44eecc'; hdr_bg_from = '#041414'; hdr_bg_to = '#020e0e'
            hdr_border = '#22aa88'; bias_color  = '#44eecc'
            bias_bg    = 'rgba(68,238,204,0.1)'; bias_brd = 'rgba(68,238,204,0.45)'
        elif bias == 'Bearish':
            hdr_accent = '#ff4060'; hdr_bg_from = '#200810'; hdr_bg_to = '#14040a'
            hdr_border = '#cc2233'; bias_color  = '#ff4060'
            bias_bg    = 'rgba(255,64,96,0.1)'; bias_brd = 'rgba(255,64,96,0.45)'
        else:
            hdr_accent = '#44eecc'; hdr_bg_from = '#061420'; hdr_bg_to = '#040e18'
            hdr_border = '#1e6688'; bias_color  = '#44eecc'
            bias_bg    = 'rgba(68,238,204,0.1)'; bias_brd = 'rgba(68,238,204,0.45)'

        def get_option_type_style(opt_type):
            t = opt_type.upper()
            if 'ATM' in t and 'OTM' not in t:
                return {'bg':'rgba(68,238,204,0.12)','color':'#44eecc','border':'rgba(68,238,204,0.5)','label':'ATM'}
            elif 'OTM' in t:
                return {'bg':'rgba(80,160,220,0.12)','color':'#55aaff','border':'rgba(80,160,220,0.5)','label':'OTM'}
            elif 'ITM' in t:
                return {'bg':'rgba(68,238,204,0.1)','color':'#33ffcc','border':'rgba(68,238,204,0.4)','label':'ITM'}
            elif 'SPREAD' in t or '/' in t:
                return {'bg':'rgba(180,80,255,0.12)','color':'#cc66ff','border':'rgba(180,80,255,0.5)','label':'SPREAD'}
            else:
                return {'bg':'rgba(68,238,204,0.1)','color':'#44eecc','border':'rgba(68,238,204,0.4)','label':opt_type}

        def get_action_style(action):
            a = str(action).upper()
            if a == 'BUY':
                return {'bg':'rgba(68,238,204,0.15)','color':'#44eecc','border':'#22aa88','glow':'#44eecc33'}
            elif a == 'SELL':
                return {'bg':'rgba(255,64,80,0.15)','color':'#ff4050','border':'#cc2233','glow':'#ff405033'}
            else:
                return {'bg':'rgba(68,238,204,0.1)','color':'#44eecc','border':'#1e6688','glow':'#44eecc22'}

        def fmt_oi_short(val):
            if isinstance(val, str): return val
            if val >= 1_000_000: return f"{val/1_000_000:.2f}M"
            elif val >= 100_000: return f"{val/100_000:.1f}L"
            elif val >= 1_000:   return f"{val/1_000:.1f}K"
            return str(val)

        def fmt_vol_short(val):
            if isinstance(val, str): return val
            if val >= 100_000: return f"{val/100_000:.1f}L"
            elif val >= 1_000:  return f"{val/1_000:.1f}K"
            return str(val)

        cards_html = ''
        for idx, rec_item in enumerate(strike_recommendations):
            strategy    = rec_item['strategy']
            action      = rec_item['action']
            strike      = rec_item['strike']
            ltp         = rec_item['ltp']
            option_type = rec_item['option_type']
            target_1    = rec_item['target_1']
            target_2    = rec_item['target_2']
            stop_loss   = rec_item['stop_loss']
            max_loss    = rec_item['max_loss']
            p_at_t1     = rec_item['profit_at_target_1']
            p_at_t2     = rec_item['profit_at_target_2']
            oi_val      = rec_item['oi']
            vol_val     = rec_item['volume']

            ot_style     = get_option_type_style(option_type)
            action_style = get_action_style(action if action in ['BUY','SELL'] else 'BUY+SELL')

            p_at_t1_label  = self._fmt_profit_label(p_at_t1)
            p_at_t2_label  = self._fmt_profit_label(p_at_t2)
            example_invest = ltp * 50 if isinstance(ltp, (int, float)) else 0
            example_t1     = self._fmt_profit(p_at_t1, 50)
            example_t2     = self._fmt_profit(p_at_t2, 50)

            t1_profit_color = '#44eecc' if isinstance(p_at_t1, (int, float)) and p_at_t1 > 0 else '#ff6070'
            t2_profit_color = '#44eecc' if isinstance(p_at_t2, (int, float)) and p_at_t2 > 0 else ('#88a0b8' if isinstance(p_at_t2, str) else '#ff6070')

            if idx == 0:   card_border_color = '#44eecc'; card_glow = 'rgba(68,238,204,0.07)'
            elif idx == 1: card_border_color = '#44eecc' if bias == 'Bullish' else '#ff4060'; card_glow = 'rgba(68,238,204,0.05)' if bias == 'Bullish' else 'rgba(255,64,96,0.05)'
            else:          card_border_color = '#cc66ff'; card_glow = 'rgba(200,100,255,0.05)'

            strike_display  = f"₹{strike:,}" if isinstance(strike, (int, float)) else f"₹{strike}"
            ltp_display     = f"₹{ltp:,.2f}" if isinstance(ltp, (int, float)) else f"₹{ltp}"
            sl_display      = f"₹{stop_loss:,.2f}" if isinstance(stop_loss, (int, float)) else f"₹{stop_loss}"
            ml_display      = f"₹{max_loss:,.2f}" if isinstance(max_loss, (int, float)) else f"₹{max_loss}"
            t1_display      = f"₹{target_1:,}" if isinstance(target_1, (int, float)) else f"₹{target_1}"
            t2_display      = f"₹{target_2:,}" if isinstance(target_2, (int, float)) else f"₹{target_2}"
            invest_display  = f"₹{example_invest:,.0f}" if example_invest else "N/A"
            oi_display      = fmt_oi_short(oi_val)
            vol_display     = fmt_vol_short(vol_val)
            risk_per_lot    = f"₹{max_loss * 50:.0f}" if isinstance(max_loss, (int, float)) else 'N/A'
            rr_ratio        = abs(p_at_t2 / max_loss) if isinstance(p_at_t2, (int, float)) and isinstance(max_loss, (int, float)) and max_loss != 0 else 0

            action_pill = f'''<span style="display:inline-flex;align-items:center;gap:5px;background:{action_style['bg']};border:1.5px solid {action_style['border']};color:{action_style['color']};padding:5px 14px;border-radius:20px;font-size:11px;font-weight:800;letter-spacing:1.5px;text-shadow:0 0 10px {action_style['glow']};box-shadow:0 0 12px {action_style['glow']};">{action}</span>'''
            ot_badge    = f'''<span style="background:{ot_style['bg']};border:1px solid {ot_style['border']};color:{ot_style['color']};padding:4px 12px;border-radius:16px;font-size:10px;font-weight:800;letter-spacing:1px;">{ot_style['label']}</span>'''

            cards_html += f'''
            <div style="background:linear-gradient(135deg,#0e1420 0%,#0c1018 100%);border:1px solid #1e2a38;border-left:4px solid {card_border_color};border-radius:8px;overflow:hidden;box-shadow:0 0 24px {card_glow},0 10px 30px rgba(0,0,0,.75);font-family:'Chakra Petch',sans-serif;">
                <div style="background:linear-gradient(135deg,#141c28 0%,#0e1420 100%);border-bottom:1px solid #1e2a38;padding:14px 20px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
                    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                        <span style="font-size:clamp(14px,3vw,17px);font-weight:800;color:#ddeeff;letter-spacing:1px;text-shadow:0 0 16px rgba(68,238,204,.2);">{strategy}</span>
                        {action_pill}
                        {ot_badge}
                    </div>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <span style="font-size:10px;color:#88a0b8;letter-spacing:1.5px;font-weight:600;">NIFTY</span>
                        <span style="width:6px;height:6px;border-radius:50%;background:#44eecc;box-shadow:0 0 8px #44eecc;display:inline-block;animation:stc-pulse 1.5s ease-in-out infinite;"></span>
                        <span style="font-size:10px;color:#44eecc;font-weight:700;letter-spacing:1.5px;">LIVE</span>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));border-bottom:1px solid #1e2a38;">
                    <div style="padding:16px;border-right:1px solid #1e2a38;background:rgba(68,238,204,.03);">
                        <div style="font-size:9px;color:#88a0b8;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">LTP</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:clamp(18px,4vw,24px);font-weight:800;color:#44eecc;text-shadow:0 0 18px rgba(68,238,204,.6);">{ltp_display}</div>
                        <div style="font-size:9px;color:#667788;margin-top:4px;letter-spacing:1px;">CURRENT PREMIUM</div>
                    </div>
                    <div style="padding:16px;border-right:1px solid #1e2a38;">
                        <div style="font-size:9px;color:#88a0b8;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">STRIKE</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:clamp(16px,3vw,20px);font-weight:800;color:#ddeeff;">{strike_display}</div>
                        <div style="font-size:9px;color:#667788;margin-top:4px;letter-spacing:1px;">PRICE</div>
                    </div>
                    <div style="padding:16px;border-right:1px solid #1e2a38;background:rgba(68,238,204,.02);">
                        <div style="font-size:9px;color:#88a0b8;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">INVEST</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:clamp(14px,3vw,18px);font-weight:800;color:#66ffdd;text-shadow:0 0 12px rgba(68,238,204,.4);">{invest_display}</div>
                        <div style="font-size:9px;color:#667788;margin-top:4px;letter-spacing:1px;">1 LOT (50 QTY)</div>
                    </div>
                    <div style="padding:16px;border-right:1px solid #1e2a38;">
                        <div style="font-size:9px;color:#88a0b8;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">OI</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:clamp(14px,3vw,18px);font-weight:800;color:#88a0b8;">{oi_display}</div>
                        <div style="font-size:9px;color:#667788;margin-top:4px;letter-spacing:1px;">OPEN INTEREST</div>
                    </div>
                    <div style="padding:16px;">
                        <div style="font-size:9px;color:#88a0b8;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">VOL</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:clamp(14px,3vw,18px);font-weight:800;color:#5a7a8a;">{vol_display}</div>
                        <div style="font-size:9px;color:#667788;margin-top:4px;letter-spacing:1px;">VOLUME</div>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));border-bottom:1px solid #1e2a38;">
                    <div style="padding:16px 18px;border-right:1px solid #1e2a38;background:rgba(68,238,204,.03);border-bottom:2px solid rgba(68,238,204,.2);">
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                            <span style="background:rgba(68,238,204,.15);border:1px solid #22aa88;color:#44eecc;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:800;letter-spacing:1.5px;">TARGET 1</span>
                        </div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:clamp(18px,4vw,22px);font-weight:800;color:#ddeeff;margin-bottom:6px;">{t1_display}</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;font-weight:700;color:{t1_profit_color};">{p_at_t1_label}</div>
                        <div style="font-size:11px;color:#44eecc;margin-top:4px;font-weight:600;">Per lot: {example_t1}</div>
                    </div>
                    <div style="padding:16px 18px;border-right:1px solid #1e2a38;background:rgba(68,238,204,.02);border-bottom:2px solid rgba(68,238,204,.15);">
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                            <span style="background:rgba(68,238,204,.1);border:1px solid #1e6688;color:#44ddbb;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:800;letter-spacing:1.5px;">TARGET 2</span>
                        </div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:clamp(18px,4vw,22px);font-weight:800;color:#ddeeff;margin-bottom:6px;">{t2_display}</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;font-weight:700;color:{t2_profit_color};">{p_at_t2_label}</div>
                        <div style="font-size:11px;color:#1e5a6a;margin-top:4px;font-weight:600;">Per lot: {example_t2}</div>
                    </div>
                    <div style="padding:16px 18px;background:rgba(255,50,80,.03);border-bottom:2px solid rgba(255,50,80,.2);">
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                            <span style="background:rgba(255,50,80,.15);border:1px solid #cc2233;color:#ff4455;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:800;letter-spacing:1.5px;">STOP LOSS</span>
                        </div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:clamp(18px,4vw,22px);font-weight:800;color:#ff6677;margin-bottom:6px;">{sl_display}</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;font-weight:700;color:#ff3344;">Max: -{ml_display}</div>
                        <div style="font-size:11px;color:#6a2233;margin-top:4px;font-weight:600;">Risk per lot: {risk_per_lot}</div>
                    </div>
                </div>
                <div style="padding:10px 18px;background:#0a1018;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
                    <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
                        <span style="font-size:10px;color:#88a0b8;letter-spacing:1.5px;font-weight:600;">RISK:REWARD</span>
                        <span style="font-family:'IBM Plex Mono',monospace;font-size:14px;font-weight:800;color:#44eecc;text-shadow:0 0 10px rgba(68,238,204,.45);">1 : {rr_ratio:.1f}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                        <span style="font-size:10px;color:#88a0b8;letter-spacing:1px;">TYPE: <span style="color:{ot_style['color']};font-weight:700;">{option_type}</span></span>
                        <span style="font-size:10px;color:#88a0b8;letter-spacing:1px;">LOT SIZE: <span style="color:#88a0b8;font-weight:700;">50</span></span>
                    </div>
                </div>
            </div>'''

        if not strike_recommendations:
            cards_html = '''<div style="background:rgba(255,64,80,.04);border:1px solid rgba(255,64,80,.25);border-radius:8px;padding:30px;text-align:center;">
                <div style="font-size:28px;margin-bottom:12px;">⚠️</div>
                <div style="font-size:16px;color:#556677;font-weight:700;letter-spacing:1px;">No Strike Recommendations Available</div>
                <div style="font-size:12px;color:#3a4a5a;margin-top:8px;">Check general strategies section below</div>
            </div>'''

        return f'''
        <style>
            .stc-wrap {{ font-family: 'Chakra Petch', sans-serif; }}
            .stc-master-hdr {{
                background: linear-gradient(135deg, {hdr_bg_from} 0%, {hdr_bg_to} 100%);
                border: 1px solid {hdr_border}33; border-bottom: 2px solid {hdr_accent};
                border-radius: 8px 8px 0 0; padding: 16px 22px;
                display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;
            }}
            .stc-master-title {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
            .stc-master-icon {{
                width: 44px; height: 44px; border-radius: 12px;
                background: linear-gradient(135deg, #0e1420, #141c28); border: 1px solid {hdr_border}44;
                display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink: 0;
                box-shadow: 0 0 16px {hdr_accent}22;
            }}
            .stc-master-text h2 {{
                font-size: clamp(15px, 3.5vw, 18px); font-weight: 800; color: #ddeeff;
                letter-spacing: 3px; text-transform: uppercase; text-shadow: 0 0 20px {hdr_accent}66;
            }}
            .stc-master-text p {{ font-size: 10px; color: #667788; margin-top: 3px; letter-spacing: 2px; font-weight: 600; text-transform: uppercase; }}
            .stc-master-right {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
            .stc-bias-pill {{ background:{bias_bg};border:1.5px solid {bias_brd};color:{bias_color};padding:7px 20px;border-radius:22px;font-size:13px;font-weight:800;letter-spacing:2px;text-shadow:0 0 12px {bias_color}77;box-shadow:0 0 16px {bias_color}18; }}
            .stc-price-chip {{ background:rgba(0,0,0,0.3);border:1px solid #1e2a38;border-radius:10px;padding:7px 16px;display:flex;align-items:center;gap:8px;flex-direction:column; }}
            .stc-price-lbl {{ font-size:9px;color:#88a0b8;letter-spacing:2px;font-weight:700;text-transform:uppercase; }}
            .stc-price-val {{ font-family:'IBM Plex Mono',monospace;font-size:15px;font-weight:800;color:#44eecc;text-shadow:0 0 12px rgba(68,238,204,.5); }}
            .stc-price-sub {{ font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:600;color:#667788; }}
            .stc-cards {{ display:flex;flex-direction:column;gap:16px;padding:16px 0 0 0; }}
            @keyframes stc-pulse {{ 0%,100% {{ opacity:1;transform:scale(1); }} 50% {{ opacity:0.5;transform:scale(0.8); }} }}
            .stc-footer {{ background:#080c12;border:1px solid #1e2a38;border-top:none;border-radius:0 0 8px 8px;padding:10px 22px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px; }}
            .stc-footer-l {{ font-size:9px;color:#556070;letter-spacing:2px;font-weight:700;text-transform:uppercase; }}
            .stc-footer-r {{ display:flex;align-items:center;gap:7px;font-size:9px;color:{hdr_accent};font-weight:700;letter-spacing:2px; }}
            .stc-footer-dot {{ width:6px;height:6px;border-radius:50%;background:{hdr_accent};box-shadow:0 0 8px {hdr_accent};animation:stc-pulse 1.5s ease-in-out infinite; }}
        </style>
        <div class="stc-wrap">
            <div class="stc-master-hdr">
                <div class="stc-master-title">
                    <div class="stc-master-icon">&#9654;</div>
                    <div class="stc-master-text">
                        <h2>&#9650; Strike Recommendations</h2>
                        <p>NIFTY &middot; Weekly Expiry &middot; Actionable Trades</p>
                    </div>
                </div>
                <div class="stc-master-right">
                    <div class="stc-bias-pill">{bias.upper()}</div>
                    <div class="stc-price-chip">
                        <span class="stc-price-lbl">NSE SPOT PRICE</span>
                        <span class="stc-price-val">&#8377;{current_price:,.2f}</span>
                        <span class="stc-price-sub">1H Close: &#8377;{candle_close:,.2f}</span>
                    </div>
                </div>
            </div>
            <div class="stc-cards">{cards_html}</div>
            <div class="stc-footer">
                <span class="stc-footer-l">DARK TICKER CARD &middot; STRIKE RECOMMENDATIONS &middot; {confidence.upper()} CONFIDENCE</span>
                <span class="stc-footer-r"><div class="stc-footer-dot"></div>{len(strike_recommendations)} TRADE{'S' if len(strike_recommendations) != 1 else ''} IDENTIFIED</span>
            </div>
        </div>'''

    # =========================================================================
    # HTML REPORT — PHANTOM SLATE THEME
    # =========================================================================
    def create_html_report(self, oc_analysis, tech_analysis, recommendation, nse_spot_price=None):
        now_ist = self.format_ist_time()
        rec     = recommendation['recommendation']

        if 'STRONG BUY' in rec:
            rec_color='#041818'; rec_text_col='#44eecc'; rec_border='#22aa88'; rec_glow='rgba(68,238,204,0.2)'
        elif 'BUY' in rec:
            rec_color='#041420'; rec_text_col='#44eecc'; rec_border='#1e6688'; rec_glow='rgba(68,238,204,0.15)'
        elif 'STRONG SELL' in rec:
            rec_color='#200810'; rec_text_col='#ff6070'; rec_border='#cc2233'; rec_glow='rgba(220,50,70,0.2)'
        elif 'SELL' in rec:
            rec_color='#180e04'; rec_text_col='#ff8855'; rec_border='#cc4400'; rec_glow='rgba(200,80,0,0.15)'
        else:
            rec_color='#0e1420'; rec_text_col='#88a0b8'; rec_border='#1e2a38'; rec_glow='rgba(68,238,204,0.08)'

        title      = self.config['report'].get('title', 'NIFTY DAY TRADING ANALYSIS (1H)')
        strategies = self.get_options_strategies(recommendation, oc_analysis, tech_analysis)

        strike_recommendations = self.get_detailed_strike_recommendations(oc_analysis, tech_analysis, recommendation)

        pivot_points   = tech_analysis.get('pivot_points', {})
        current_price  = tech_analysis.get('current_price', 0)
        candle_close   = tech_analysis.get('candle_close_price', current_price)
        nearest_levels = self.find_nearest_levels(current_price, pivot_points)

        pivot_widget_html    = self._build_pivot_widget(pivot_points, current_price, nearest_levels)
        tech_resistances     = tech_analysis.get('tech_resistances', [])
        tech_supports        = tech_analysis.get('tech_supports', [])
        sr_widget_html       = self._build_sr_bloomberg_widget(tech_resistances, tech_supports, current_price)
        oc_plasma_widget_html = self._build_oc_plasma_widget(oc_analysis)

        top_ce_strikes  = oc_analysis.get('top_ce_strikes', [])
        top_pe_strikes  = oc_analysis.get('top_pe_strikes', [])
        atm_strike_val  = oc_analysis.get('atm_strike', 0)
        oi_neon_ledger_html   = self._build_oi_neon_ledger_widget(top_ce_strikes, top_pe_strikes, atm_strike=atm_strike_val)
        strike_ticker_card_html = self._build_strike_ticker_card_widget(strike_recommendations, recommendation, tech_analysis)

        momentum_1h_pct    = tech_analysis.get('price_change_pct_1h', 0)
        momentum_1h_signal = tech_analysis.get('momentum_1h_signal', 'Sideways')
        momentum_1h_colors = tech_analysis.get('momentum_1h_colors', {'bg':'#0e1420','bg_dark':'#0a1018','text':'#44eecc','border':'#1e2a38'})
        momentum_5h_pct    = tech_analysis.get('momentum_5h_pct', 0)
        momentum_5h_signal = tech_analysis.get('momentum_5h_signal', 'Sideways')
        momentum_5h_colors = tech_analysis.get('momentum_5h_colors', {'bg':'#0e1420','bg_dark':'#0a1018','text':'#44eecc','border':'#1e2a38'})
        momentum_2d_pct    = tech_analysis.get('momentum_2d_pct', 0)
        momentum_2d_signal = tech_analysis.get('momentum_2d_signal', 'Sideways')
        momentum_2d_colors = tech_analysis.get('momentum_2d_colors', {'bg':'#0e1420','bg_dark':'#0a1018','text':'#44eecc','border':'#1e2a38'})

        bar_1h = min(abs(momentum_1h_pct) * 40, 100)
        bar_5h = min(abs(momentum_5h_pct) * 20, 100)
        bar_2d = min(abs(momentum_2d_pct) * 10, 100)

        spot_display = nse_spot_price if nse_spot_price else current_price

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

        auto_refresh_js = """
<script>
(function () {
    'use strict';

    var REFRESH_INTERVAL_MS = 30000;
    var SCROLL_KEY          = 'nifty_scroll_y';
    var LAST_REFRESH_KEY    = 'nifty_last_refresh';

    var savedScroll = sessionStorage.getItem(SCROLL_KEY);
    if (savedScroll !== null) {
        window.scrollTo(0, parseInt(savedScroll, 10));
        sessionStorage.removeItem(SCROLL_KEY);
    }

    function getISTString() {
        var now = new Date();
        var utcMs = now.getTime() + now.getTimezoneOffset() * 60000;
        var istMs = utcMs + 5.5 * 3600000;
        var ist   = new Date(istMs);
        var pad   = function (n) { return n < 10 ? '0' + n : '' + n; };
        return ist.getFullYear() + '-' +
               pad(ist.getMonth() + 1) + '-' +
               pad(ist.getDate())      + '  ' +
               pad(ist.getHours())     + ':' +
               pad(ist.getMinutes())   + ':' +
               pad(ist.getSeconds())   + ' IST';
    }

    sessionStorage.setItem(LAST_REFRESH_KEY, getISTString());

    var liveClockEl    = document.getElementById('ar-live-clock');
    var lastRefreshEl  = document.getElementById('ar-last-refresh');
    var countdownEl    = document.getElementById('ar-countdown');
    var progressBarEl  = document.getElementById('ar-progress-fill');

    var lastRefreshText = sessionStorage.getItem(LAST_REFRESH_KEY) || getISTString();
    if (lastRefreshEl) { lastRefreshEl.textContent = lastRefreshText; }

    var refreshAt = Date.now() + REFRESH_INTERVAL_MS;

    function tick() {
        if (liveClockEl) { liveClockEl.textContent = getISTString(); }
        var remaining = Math.max(0, Math.ceil((refreshAt - Date.now()) / 1000));
        if (countdownEl) { countdownEl.textContent = remaining + 's'; }
        if (progressBarEl) {
            var elapsed = REFRESH_INTERVAL_MS - (refreshAt - Date.now());
            var pct     = Math.min(100, Math.max(0, (elapsed / REFRESH_INTERVAL_MS) * 100));
            progressBarEl.style.width = pct.toFixed(1) + '%';
        }
    }

    var tickTimer = setInterval(tick, 1000);
    tick();

    var refreshTimer = setTimeout(function () {
        clearInterval(tickTimer);
        sessionStorage.setItem(SCROLL_KEY, window.scrollY.toString());
        window.location.reload();
    }, REFRESH_INTERVAL_MS);

    var pausedAt = null;
    document.addEventListener('visibilitychange', function () {
        if (document.hidden) {
            pausedAt = refreshAt - Date.now();
            clearTimeout(refreshTimer);
            clearInterval(tickTimer);
        } else {
            if (pausedAt !== null && pausedAt > 0) {
                refreshAt = Date.now() + pausedAt;
                pausedAt  = null;
                tickTimer    = setInterval(tick, 1000);
                refreshTimer = setTimeout(function () {
                    clearInterval(tickTimer);
                    sessionStorage.setItem(SCROLL_KEY, window.scrollY.toString());
                    window.location.reload();
                }, refreshAt - Date.now());
                tick();
            }
        }
    });
})();
</script>
"""

        auto_refresh_bar_html = """
<div id="ar-status-bar" style="
    position: sticky; top: 0; z-index: 9999;
    background: linear-gradient(135deg, #0a1018 0%, #0e1828 100%);
    border-bottom: 1px solid #1e2a38;
    border-top: 2px solid #44eecc;
    padding: 0;
    font-family: 'IBM Plex Mono', 'Chakra Petch', monospace;
    box-shadow: 0 4px 24px rgba(0,0,0,0.8);
">
    <div style="height:3px;background:#0c1018;width:100%;overflow:hidden;">
        <div id="ar-progress-fill" style="
            height:100%;width:0%;
            background:linear-gradient(90deg,#22aa88,#44eecc);
            box-shadow:0 0 10px #44eecc;
            transition:width 0.9s linear;
        "></div>
    </div>
    <div style="
        display:flex; align-items:center; justify-content:space-between;
        padding:7px 18px; flex-wrap:wrap; gap:8px;
    ">
        <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:7px;height:7px;border-radius:50%;background:#44eecc;
                        box-shadow:0 0 8px #44eecc;
                        animation:ar-blink 1s ease-in-out infinite;flex-shrink:0;"></div>
            <span style="font-size:10px;color:#88a0b8;letter-spacing:1.5px;font-weight:600;text-transform:uppercase;">IST NOW</span>
            <span id="ar-live-clock" style="font-size:11px;color:#44eecc;font-weight:700;
                                            text-shadow:0 0 10px rgba(68,238,204,0.5);letter-spacing:0.5px;">
                --:--:-- IST
            </span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:10px;color:#556070;letter-spacing:1.5px;font-weight:600;text-transform:uppercase;">LAST REFRESH</span>
            <span id="ar-last-refresh" style="font-size:11px;color:#66ffdd;font-weight:700;letter-spacing:0.5px;">
                --
            </span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:10px;color:#556070;letter-spacing:1.5px;font-weight:600;text-transform:uppercase;">NEXT REFRESH</span>
            <span id="ar-countdown" style="
                font-size:13px;font-weight:800;color:#44eecc;
                background:rgba(68,238,204,0.08);
                border:1px solid rgba(68,238,204,0.3);
                border-radius:6px;padding:2px 10px;
                text-shadow:0 0 10px rgba(68,238,204,0.5);
                min-width:44px;text-align:center;
            ">30s</span>
            <span style="font-size:9px;color:#2a4a3a;letter-spacing:1px;text-transform:uppercase;">AUTO</span>
        </div>
    </div>
</div>
<style>
@keyframes ar-blink {
    0%,100% { opacity:1; transform:scale(1);   }
    50%      { opacity:0.4; transform:scale(0.8); }
}
</style>
"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Chakra Petch', 'Segoe UI', sans-serif;
            background: radial-gradient(ellipse at 40% 0%, #141820 0%, #0c1018 50%, #080c12 100%);
            color: #88a0b8;
            padding: clamp(6px, 2vw, 16px); min-height: 100vh;
        }}
        body::before {{
            content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
            background: repeating-linear-gradient(0deg, transparent, transparent 70px, rgba(100,140,180,0.005) 70px, rgba(100,140,180,0.005) 71px);
        }}
        .container {{
            position: relative; z-index: 1; max-width: 1400px; margin: 0 auto;
            background: linear-gradient(160deg, #0e1420 0%, #0a1018 100%);
            border-radius: 6px;
            box-shadow: 0 0 0 1px #1e2a38, 0 0 80px rgba(68,238,204,.03), 0 32px 80px rgba(0,0,0,.95);
            padding: clamp(12px, 3vw, 28px); border: 1px solid #1e2a38;
        }}
        .header {{
            text-align: center;
            background: linear-gradient(180deg, #0e1420, #141c28 60%, #0e1420);
            padding: clamp(18px, 4vw, 32px) clamp(14px, 3vw, 24px) clamp(16px, 3vw, 28px);
            margin-bottom: clamp(16px, 3vw, 28px);
            border: 1px solid #1e2a38; border-top: 3px solid #44eecc;
            position: relative; overflow: hidden;
        }}
        .header::after {{
            content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
            background: linear-gradient(90deg, transparent, #22aa8877, #44eeccaa, #22aa8877, transparent);
        }}
        .header h1 {{
            font-size: clamp(14px, 4vw, 24px); font-weight: 700; color: #ddeeff;
            letter-spacing: clamp(3px, 1.5vw, 7px); text-transform: uppercase;
            text-shadow: 0 0 40px rgba(68,238,204,.5), 0 0 80px rgba(34,170,136,.2);
            margin-bottom: 14px;
        }}
        .header-sub {{ color: #88a0b8; font-size: 11px; letter-spacing: 5px; text-transform: uppercase; font-weight: 600; }}
        .live-price-banner {{
            display: flex; align-items: center; justify-content: center; gap: clamp(16px, 4vw, 40px);
            background: linear-gradient(135deg, #0e1420, #141c28);
            border: 1px solid #1e2a38; border-top: 2px solid #44eecc;
            padding: clamp(12px, 2vw, 18px) clamp(16px, 3vw, 28px);
            margin: 14px 0; flex-wrap: wrap;
        }}
        .live-price-item {{ text-align: center; }}
        .live-price-label {{ font-size: clamp(9px, 1.5vw, 11px); color: #88a0b8; letter-spacing: 3px; text-transform: uppercase; font-weight: 600; margin-bottom: 6px; }}
        .live-price-value {{ font-family: 'IBM Plex Mono', monospace; font-size: clamp(20px, 4vw, 28px); font-weight: 800; color: #44eecc; text-shadow: 0 0 20px rgba(68,238,204,.45); }}
        .live-price-sub {{ font-size: clamp(9px, 1.5vw, 10px); color: #667788; margin-top: 4px; }}
        .live-price-sep {{ width: 1px; height: 50px; background: linear-gradient(180deg, transparent, #1e2a38, transparent); flex-shrink: 0; }}
        .live-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #44eecc; box-shadow: 0 0 10px #44eecc; display: inline-block; margin-right: 6px; animation: lp 1.5s ease-in-out infinite; }}
        @keyframes lp {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }} }}
        .timestamp {{ color: #667788; font-size: clamp(10px, 2vw, 12px); font-weight: 600; margin-top: 10px; letter-spacing: 2px; }}
        .timeframe-badge {{
            display: inline-flex; align-items: center; gap: 12px;
            color: #88a0b8; padding: 8px 0; font-size: clamp(9px, 1.5vw, 11px);
            font-weight: 600; margin-top: 10px; letter-spacing: 4px; text-transform: uppercase;
        }}
        .timeframe-badge::before,.timeframe-badge::after {{ content: '▸'; color: #44eecc; }}
        .momentum-container {{
            display: grid; grid-template-columns: repeat(3, 1fr);
            gap: clamp(10px, 2vw, 14px); margin-bottom: clamp(14px, 2.5vw, 20px);
        }}
        .momentum-box {{
            background: linear-gradient(135deg, var(--mb-bg,#0e1420), var(--mb-bg2,#0a1018));
            color: var(--mb-text,#44eecc); padding: clamp(14px, 2.5vw, 22px);
            border-radius: 4px; text-align: center;
            border: 1px solid var(--mb-border,#1e2a38); border-top: 2px solid var(--mb-border,#44eecc);
            box-shadow: 0 8px 24px rgba(0,0,0,.7);
        }}
        .momentum-box h3 {{ font-size: clamp(9px, 1.5vw, 10px); font-weight: 700; text-transform: uppercase; letter-spacing: 3px; margin-bottom: 10px; opacity: 0.75; }}
        .momentum-box .mval {{ font-family: 'IBM Plex Mono', monospace; font-size: clamp(22px, 5vw, 30px); font-weight: 800; margin: 10px 0; text-shadow: 0 0 20px currentColor; }}
        .momentum-box .msig {{ font-size: clamp(10px, 1.5vw, 11px); font-weight: 600; opacity: 0.9; letter-spacing: 1px; }}
        .mb-bar-wrap {{ margin-top: 14px; height: 3px; background: rgba(255,255,255,0.04); overflow: hidden; }}
        .mb-bar-fill {{ height: 100%; background: var(--mb-border); box-shadow: 0 0 8px var(--mb-border); }}
        .recommendation-box {{
            background: linear-gradient(135deg, {rec_color}, {rec_color}dd);
            color: {rec_text_col}; padding: clamp(16px, 3vw, 24px);
            border-radius: 4px; text-align: center; margin-bottom: clamp(14px, 2.5vw, 20px);
            box-shadow: 0 8px 28px {rec_glow}; border: 1px solid {rec_border}; border-top: 2px solid {rec_border};
        }}
        .recommendation-box h2 {{ font-size: clamp(18px, 5vw, 28px); font-weight: 800; margin-bottom: 8px; letter-spacing: clamp(2px, 1vw, 4px); text-shadow: 0 0 28px {rec_glow}; }}
        .recommendation-box .subtitle {{ font-size: clamp(11px, 2vw, 14px); opacity: 0.85; font-weight: 600; letter-spacing: .5px; }}
        .signal-badge {{ display: inline-block; padding: 5px 14px; border-radius: 20px; font-size: clamp(10px, 1.5vw, 11px); font-weight: 700; margin: 8px 4px 0; letter-spacing: 1px; }}
        .badge-bull {{ background: rgba(68,238,204,.1); border: 1px solid #22aa88; color: #44eecc; }}
        .badge-bear {{ background: rgba(255,64,80,.1); border: 1px solid #cc2233; color: #ff4455; }}
        .section {{ margin-bottom: clamp(16px, 3vw, 24px); }}
        .section-title {{
            background: linear-gradient(135deg, #141c28, #0e1420);
            color: #44eecc; padding: clamp(10px, 2vw, 14px) clamp(14px, 3vw, 22px);
            font-size: clamp(10px, 1.8vw, 12px); font-weight: 700;
            margin-bottom: clamp(10px, 2vw, 14px); letter-spacing: clamp(2px, 0.8vw, 4px);
            text-transform: uppercase; border-top: 1px solid #44eecc; border-bottom: 1px solid #1e2a38;
            display: flex; align-items: center; justify-content: center; gap: 14px;
            text-shadow: 0 0 16px rgba(68,238,204,.4);
        }}
        .section-title::before,.section-title::after {{ content: '▸'; color: #44eecc; font-size: 12px; flex-shrink: 0; }}
        .data-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap: clamp(8px, 1.5vw, 12px); }}
        .data-item {{
            background: linear-gradient(135deg, #141c28, #0e1420);
            padding: clamp(12px, 2vw, 16px); border: 1px solid #1e2a38;
            border-top: 2px solid #2a3a4a; border-radius: 3px; transition: border-top-color .2s;
        }}
        .data-item:hover {{ border-top-color: #44eecc; }}
        .data-item .dlabel {{ color: #667788; font-size: 9px; text-transform: uppercase; font-weight: 700; letter-spacing: 2px; margin-bottom: 7px; }}
        .data-item .dvalue {{ color: #ddeeff; font-size: clamp(14px, 2.5vw, 18px); font-weight: 700; font-family: 'IBM Plex Mono', monospace; }}
        .reasons {{ background: linear-gradient(135deg, #141c28, #0e1420); border: 1px solid #1e2a38; border-left: 4px solid #22aa88; padding: 16px; border-radius: 3px; }}
        .reasons strong {{ color: #44eecc; font-size: clamp(12px, 2vw, 14px); letter-spacing: 1px; }}
        .reasons ul {{ margin: 10px 0 0 0; padding-left: 22px; }}
        .reasons li {{ margin: 6px 0; color: #88a0b8; font-size: clamp(11px, 1.8vw, 13px); line-height: 1.6; }}
        .strategies-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px,1fr)); gap: clamp(10px, 2vw, 14px); margin-top: 14px; }}
        .strategy-card {{ background: linear-gradient(135deg, #141c28, #0e1420); border: 1px solid #1e2a38; border-top: 2px solid #2a3a4a; padding: 16px; border-radius: 3px; }}
        .strategy-header {{ border-bottom: 1px solid #1e2a38; padding-bottom: 8px; margin-bottom: 10px; }}
        .strategy-header h4 {{ color: #ddeeff; font-size: clamp(12px, 2vw, 14px); font-weight: 700; letter-spacing: 1px; }}
        .strategy-type {{ display: inline-block; background: rgba(68,238,204,.06); color: #88a0b8; padding: 3px 8px; font-size: 10px; margin-top: 4px; font-weight: 600; border: 1px solid #1e2a38; letter-spacing: .5px; border-radius: 3px; }}
        .strategy-body p {{ margin: 7px 0; font-size: clamp(11px, 1.8vw, 13px); line-height: 1.5; color: #88a0b8; }}
        .strategy-body strong {{ color: #44eecc; }}
        .recommendation-stars {{ color: #44eecc; font-size: 13px; font-weight: 700; }}
        .footer {{ text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #1e2a38; color: #556070; font-size: clamp(9px, 1.5vw, 11px); line-height: 1.8; letter-spacing: 1px; }}
        @media (max-width: 900px) {{ .data-grid {{ grid-template-columns: repeat(3,1fr); }} }}
        @media (max-width: 680px) {{
            .momentum-container {{ grid-template-columns: 1fr 1fr; gap: 10px; }}
            .data-grid {{ grid-template-columns: repeat(2,1fr); gap: 8px; }}
            .strategies-grid {{ grid-template-columns: 1fr; }}
            .live-price-sep {{ display: none; }}
        }}
        @media (max-width: 480px) {{
            .momentum-container {{ grid-template-columns: 1fr; gap: 10px; }}
            .data-grid {{ grid-template-columns: 1fr; }}
        }}
        @media (min-width: 1600px) {{ .data-grid {{ grid-template-columns: repeat(6,1fr); }} }}
    </style>
</head>
<body>

{auto_refresh_bar_html}

<div class="container">

    <div class="header">
        <div style="color:#556070;font-size:9px;letter-spacing:6px;margin-bottom:10px;">▸ &nbsp; ALGORITHMIC MARKET INTELLIGENCE &nbsp; ◂</div>
        <h1>&#9679; {title} &#9679;</h1>
        <div class="timeframe-badge">ONE HOUR TIMEFRAME</div>
        <div class="header-sub">LIVE DATA &nbsp;·&nbsp; NSE OPTION CHAIN &nbsp;·&nbsp; PHANTOM SLATE</div>
        <div class="timestamp">Generated on: {now_ist}</div>
    </div>

    <div class="live-price-banner">
        <div class="live-price-item">
            <div class="live-price-label"><span class="live-dot"></span>NSE SPOT PRICE</div>
            <div class="live-price-value">&#8377;{spot_display:,.2f}</div>
            <div class="live-price-sub">via NSE Option Chain API</div>
        </div>
        <div class="live-price-sep"></div>
        <div class="live-price-item">
            <div class="live-price-label">1H CANDLE CLOSE</div>
            <div class="live-price-value" style="color:#44eecc;font-size:clamp(16px,3vw,22px);">&#8377;{candle_close:,.2f}</div>
            <div class="live-price-sub">last completed 1H bar</div>
        </div>
        <div class="live-price-sep"></div>
        <div class="live-price-item">
            <div class="live-price-label">&#128197; EXPIRY DATE</div>
            <div class="live-price-value" style="color:#ff3a5c;font-size:clamp(13px,2.5vw,18px);text-shadow:0 0 16px rgba(255,58,92,.5);">{self.get_next_expiry_date()}</div>
            <div class="live-price-sub">weekly tuesday expiry</div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Option Chain Analysis</div>
        {oc_plasma_widget_html}
    </div>

    <div class="momentum-container">
        <div class="momentum-box" style="--mb-bg:{momentum_1h_colors['bg']};--mb-bg2:{momentum_1h_colors['bg_dark']};--mb-text:{momentum_1h_colors['text']};--mb-border:{momentum_1h_colors['border']};">
            <h3>&#9889; 1H MOMENTUM</h3>
            <div class="mval">{momentum_1h_pct:+.2f}%</div>
            <div class="msig">{momentum_1h_signal}</div>
            <div class="mb-bar-wrap"><div class="mb-bar-fill" style="width:{bar_1h:.0f}%;"></div></div>
        </div>
        <div class="momentum-box" style="--mb-bg:{momentum_5h_colors['bg']};--mb-bg2:{momentum_5h_colors['bg_dark']};--mb-text:{momentum_5h_colors['text']};--mb-border:{momentum_5h_colors['border']};">
            <h3>&#128202; 5H MOMENTUM</h3>
            <div class="mval">{momentum_5h_pct:+.2f}%</div>
            <div class="msig">{momentum_5h_signal}</div>
            <div class="mb-bar-wrap"><div class="mb-bar-fill" style="width:{bar_5h:.0f}%;"></div></div>
        </div>
        <div class="momentum-box" style="--mb-bg:{momentum_2d_colors['bg']};--mb-bg2:{momentum_2d_colors['bg_dark']};--mb-text:{momentum_2d_colors['text']};--mb-border:{momentum_2d_colors['border']};">
            <h3>&#128374; 2D MOMENTUM</h3>
            <div class="mval">{momentum_2d_pct:+.2f}%</div>
            <div class="msig">{momentum_2d_signal}</div>
            <div class="mb-bar-wrap"><div class="mb-bar-fill" style="width:{bar_2d:.0f}%;"></div></div>
        </div>
    </div>

    <div class="recommendation-box">
        <h2>{recommendation['recommendation']}</h2>
        <div class="subtitle">Market Bias: {recommendation['bias']} &nbsp;|&nbsp; Confidence: {recommendation['confidence']}</div>
        <div style="margin-top:12px;">
            <span class="signal-badge badge-bull">&#9650; Bullish: {recommendation['bullish_signals']}</span>
            <span class="signal-badge badge-bear">&#9660; Bearish: {recommendation['bearish_signals']}</span>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Technical Analysis (1H)</div>
        <div class="data-grid">
            <div class="data-item"><div class="dlabel">1H Candle Close</div><div class="dvalue">&#8377;{candle_close:,.2f}</div></div>
            <div class="data-item"><div class="dlabel">RSI (14)</div><div class="dvalue">{tech_analysis.get('rsi','N/A')}</div></div>
            <div class="data-item"><div class="dlabel">EMA 20</div><div class="dvalue">&#8377;{tech_analysis.get('ema20','N/A')}</div></div>
            <div class="data-item"><div class="dlabel">EMA 50</div><div class="dvalue">&#8377;{tech_analysis.get('ema50','N/A')}</div></div>
            <div class="data-item"><div class="dlabel">Trend</div><div class="dvalue">{tech_analysis.get('trend','N/A')}</div></div>
            <div class="data-item"><div class="dlabel">RSI Signal</div><div class="dvalue">{tech_analysis.get('rsi_signal','N/A')}</div></div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Support &amp; Resistance (1H)</div>
        {sr_widget_html}
    </div>

    <div class="section">
        <div class="section-title">Pivot Points (Traditional - 30 Min)</div>
        {pivot_widget_html}
    </div>

    <div class="section">
        <div class="section-title">Top OI — ±10 Strikes from ATM (5 CE + 5 PE)</div>
        {oi_neon_ledger_html}
    </div>

    <div class="section">
        <div class="section-title">Analysis Summary</div>
        <div class="reasons">
            <strong>&#128161; Key Factors:</strong>
            <ul>{''.join([f'<li>{reason}</li>' for reason in recommendation.get('reasons', [])])}</ul>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Strike Recommendations</div>
        <p style="color:#88a0b8;margin-bottom:16px;font-size:clamp(11px,2vw,13px);line-height:1.6;">
            <strong style="color:#44eecc;">Based on {recommendation['bias']} bias &mdash; NSE Spot &#8377;{current_price:,.2f}</strong><br>
            Actionable trades with specific strike prices, LTP, targets &amp; risk management.
        </p>
        {strike_ticker_card_html}
    </div>

    <div class="section">
        <div class="section-title">Options Strategies</div>
        <p style="color:#88a0b8;margin-bottom:14px;font-size:clamp(11px,2vw,13px);letter-spacing:.5px;">
            Based on <strong style="color:#44eecc;">{recommendation['bias']}</strong> bias:
        </p>
        <div class="strategies-grid">{strategies_html}</div>
    </div>

    <div class="footer">
        <p><strong style="color:#44eecc;">Disclaimer:</strong> This analysis is for educational purposes only. Trading involves risk. Past performance is not indicative of future results.</p>
        <p style="margin-top:6px;">&#9679; PHANTOM SLATE THEME &nbsp;&#9679;&nbsp; Charcoal Blue-Grey + Neon Mint + Graphite Steel &nbsp;&#9679;&nbsp; Max Pain = Min Buyer Pain (CORRECTED)</p>
        <p style="margin-top:6px;color:#2a4a3a;">AUTO-REFRESH: 30s &nbsp;&#9679;&nbsp; SCROLL POSITION PRESERVED &nbsp;&#9679;&nbsp; SILENT BACKGROUND RELOAD</p>
        <p style="margin-top:6px;color:#2a4a3a;">HOLIDAY FIX: Tuesday NSE Holidays → Expiry shifts to Monday</p>
    </div>

</div>

{auto_refresh_js}

</body>
</html>"""
        return html

    # =========================================================================
    # EMAIL
    # =========================================================================
    def send_email(self, html_content):
        email_config    = self.config['email']
        recipient_email = email_config['recipient']
        sender_email    = email_config['sender']
        sender_password = email_config['app_password']
        subject_prefix  = email_config.get('subject_prefix', 'Nifty 1H Analysis')
        ist_time        = self.get_ist_time()
        subject_time    = ist_time.strftime('%Y-%m-%d %H:%M IST')
        try:
            msg            = MIMEMultipart('alternative')
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

    # =========================================================================
    # RUN
    # =========================================================================
    def run_analysis(self):
        self.logger.info("🚀 Starting Nifty 1-HOUR Analysis — PHANTOM SLATE THEME")
        self.logger.info("=" * 60)

        oc_df, nse_spot_price = self.fetch_option_chain()
        if oc_df is not None and nse_spot_price is not None:
            oc_analysis = self.analyze_option_chain(oc_df, nse_spot_price)
        else:
            nse_spot_price = None
            oc_analysis    = self.get_sample_oc_analysis()

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
        self.logger.info(f"💹 Live LTP (yfinance):  ₹{tech_analysis.get('current_price', 'N/A')}")
        self.logger.info(f"💹 NSE Spot (API):       ₹{nse_spot_price or 'N/A'}")
        self.logger.info(f"📉 1H Candle Close:      ₹{tech_analysis.get('candle_close_price', 'N/A')}")
        self.logger.info(f"🎯 RSI (1H): {tech_analysis.get('rsi', 'N/A')}")
        self.logger.info(f"⚡ 1H Momentum: {tech_analysis.get('price_change_pct_1h', 0):+.2f}% - {tech_analysis.get('momentum_1h_signal')}")
        self.logger.info(f"📊 5H Momentum: {tech_analysis.get('momentum_5h_pct', 0):+.2f}% - {tech_analysis.get('momentum_5h_signal')}")
        self.logger.info(f"📍 Pivot Point: ₹{tech_analysis.get('pivot_points', {}).get('pivot', 'N/A')}")
        self.logger.info(f"🎯 Max Pain (Corrected): ₹{oc_analysis.get('max_pain', 'N/A'):,}")
        self.logger.info("=" * 60)

        html_report = self.create_html_report(oc_analysis, tech_analysis, recommendation, nse_spot_price=nse_spot_price)

        if self.config['report']['save_local']:
            report_dir      = self.config['report']['local_dir']
            os.makedirs(report_dir, exist_ok=True)
            ist_time        = self.get_ist_time()
            filename_format = self.config['report']['filename_format']
            report_filename = os.path.join(report_dir, ist_time.strftime(filename_format))
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(html_report)
            self.logger.info(f"💾 Report saved as: {report_filename}")

        self.logger.info(f"📧 Sending email to {self.config['email']['recipient']}...")
        self.send_email(html_report)
        self.logger.info("✅ Analysis Complete!")

        return {'oc_analysis': oc_analysis, 'tech_analysis': tech_analysis,
                'recommendation': recommendation, 'html_report': html_report,
                'nse_spot_price': nse_spot_price}


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    analyzer = NiftyAnalyzer(config_path='config.yml')
    result   = analyzer.run_analysis()

    print(f"\n✅ Analysis Complete! — PHANTOM SLATE THEME")
    print(f"Recommendation:    {result['recommendation']['recommendation']}")
    print(f"Live LTP:          ₹{result['tech_analysis']['current_price']:,.2f}")
    print(f"NSE Spot:          ₹{result['nse_spot_price'] or 'N/A'}")
    print(f"1H Candle Close:   ₹{result['tech_analysis']['candle_close_price']:,.2f}")
    print(f"RSI (1H):          {result['tech_analysis']['rsi']}")
    print(f"Max Pain:          ₹{result['oc_analysis']['max_pain']:,}")
    print(f"1H Momentum:       {result['tech_analysis']['price_change_pct_1h']:+.2f}% - {result['tech_analysis']['momentum_1h_signal']}")
    print(f"5H Momentum:       {result['tech_analysis']['momentum_5h_pct']:+.2f}% - {result['tech_analysis']['momentum_5h_signal']}")
    print(f"Check your email or ./reports/ for the detailed HTML report!")
