"""
NIFTY 50 COMPLETE ANALYSIS - DEEP OCEAN THEME
CARD STYLE: Glassmorphism Frosted — Stat Card + Progress Bar (Layout 4)
CHANGE IN OPEN INTEREST: Navy Command Theme (v3)
FIXES:
  1. Expiry: TUESDAY weekly - always picks NEXT Tuesday (never today)
  2. Auto-fallback: if NSE returns empty, fetches real expiry list
  3. SIDEWAYS targets: T1=Resistance, T2=Support (distinct values)
  4. Stop loss fallback text for neutral strategies
  5. Option chain filtered to ATM ±10 strikes only
  6. [NEW] Glassmorphism Stat Card + Progress Bar layout for all indicator cards
  7. [NEW] FII / DII 5-Day Sentiment section below Market Snapshot
  8. [NEW] Navy Command theme for Change in Open Interest section
"""
from curl_cffi import requests
import pandas as pd
import time
from datetime import datetime, timedelta, date
import calendar
import yfinance as yf
import warnings
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import pytz
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  FII / DII HELPER
# ─────────────────────────────────────────────
def fetch_fii_dii_data():
    hardcoded = [
        {'date': 'Feb 11', 'day': 'Tue', 'fii': -1540.20, 'dii': 2103.50},
        {'date': 'Feb 12', 'day': 'Wed', 'fii':   823.60, 'dii':  891.40},
        {'date': 'Feb 13', 'day': 'Thu', 'fii':  -411.80, 'dii': 1478.30},
        {'date': 'Feb 14', 'day': 'Fri', 'fii':    69.45, 'dii': 1174.21},
        {'date': 'Feb 17', 'day': 'Mon', 'fii':  -972.13, 'dii': 1666.98},
    ]
    return hardcoded


def compute_fii_dii_summary(data):
    fii_vals = [d['fii'] for d in data]
    dii_vals = [d['dii'] for d in data]
    fii_avg  = sum(fii_vals) / len(fii_vals)
    dii_avg  = sum(dii_vals) / len(dii_vals)
    net_avg  = fii_avg + dii_avg

    fii_span = f'<span style="color:#ff5252;font-weight:700;">₹{fii_avg:.0f} Cr/day</span>'
    dii_span = f'<span style="color:#40c4ff;font-weight:700;">₹{dii_avg:+.0f} Cr/day</span>'
    net_span = f'<span style="color:#b388ff;font-weight:700;">₹{net_avg:+.0f} Cr/day</span>'

    if fii_avg > 0 and dii_avg > 0:
        label = 'STRONGLY BULLISH'; emoji = '🚀'; color = '#00e676'
        badge_cls = 'fii-bull'
        fii_span = f'<span style="color:#00e676;font-weight:700;">₹{fii_avg:+.0f} Cr/day</span>'
        insight = (f"Both FIIs (avg {fii_span}) and DIIs (avg {dii_span}) are net buyers — "
                   f"strong dual institutional confirmation. Net combined flow: {net_span}.")
    elif fii_avg < 0 and dii_avg > 0 and dii_avg > abs(fii_avg):
        label = 'CAUTIOUSLY BULLISH'; emoji = '📈'; color = '#69f0ae'
        badge_cls = 'fii-cbull'
        insight = (f"FIIs are net sellers (avg {fii_span}) but DIIs are absorbing strongly (avg {dii_span}). "
                   f"DII support is cushioning downside — FII return is key for breakout. "
                   f"Net combined flow: {net_span}.")
    elif fii_avg < 0 and dii_avg > 0:
        label = 'MIXED / NEUTRAL'; emoji = '⚖️'; color = '#ffd740'
        badge_cls = 'fii-neu'
        insight = (f"FII selling (avg {fii_span}) is partly offset by DII buying (avg {dii_span}). "
                   f"Watch for 3+ consecutive days of FII buying for trend confirmation. "
                   f"Net combined flow: {net_span}.")
    elif fii_avg < 0 and dii_avg < 0:
        label = 'BEARISH'; emoji = '📉'; color = '#ff5252'
        badge_cls = 'fii-bear'
        dii_span = f'<span style="color:#ff5252;font-weight:700;">₹{dii_avg:.0f} Cr/day</span>'
        insight = (f"Both FIIs (avg {fii_span}) and DIIs (avg {dii_span}) are net sellers — "
                   f"clear bearish institutional pressure. Exercise caution. "
                   f"Net combined flow: {net_span}.")
    else:
        label = 'NEUTRAL'; emoji = '🔄'; color = '#b0bec5'
        badge_cls = 'fii-neu'
        insight = "Mixed signals from institutional participants. Wait for a clearer trend."

    max_abs = max(abs(v) for row in data for v in (row['fii'], row['dii'])) or 1

    return {
        'fii_avg': fii_avg, 'dii_avg': dii_avg, 'net_avg': net_avg,
        'label': label, 'emoji': emoji, 'color': color,
        'badge_cls': badge_cls, 'insight': insight,
        'max_abs': max_abs,
    }


class NiftyHTMLAnalyzer:
    def __init__(self):
        self.yf_symbol  = "^NSEI"
        self.nse_symbol = "NIFTY"
        self.report_lines = []
        self.html_data    = {}

    def log(self, message):
        print(message)
        self.report_lines.append(message)

    # ─────────────────────────────────────────────
    #  NSE SESSION
    # ─────────────────────────────────────────────
    def _make_nse_session(self):
        headers = {
            "authority":       "www.nseindia.com",
            "accept":          "application/json, text/plain, */*",
            "user-agent":      (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "referer":         "https://www.nseindia.com/option-chain",
            "accept-language": "en-US,en;q=0.9",
        }
        session = requests.Session()
        try:
            session.get("https://www.nseindia.com/",            headers=headers, impersonate="chrome", timeout=15)
            time.sleep(1.5)
            session.get("https://www.nseindia.com/option-chain", headers=headers, impersonate="chrome", timeout=15)
            time.sleep(1)
        except Exception as e:
            print(f"  ⚠️  Session warm-up warning: {e}")
        return session, headers

    # ─────────────────────────────────────────────
    #  EXPIRY DETECTION
    # ─────────────────────────────────────────────
    def get_upcoming_expiry_tuesday(self):
        ist_tz    = pytz.timezone('Asia/Kolkata')
        today_ist = datetime.now(ist_tz).date()
        weekday   = today_ist.weekday()
        days_ahead = 7 if weekday == 1 else (1 - weekday) % 7 or 7
        next_tuesday = today_ist + timedelta(days=days_ahead)
        expiry_str   = next_tuesday.strftime('%d-%b-%Y')
        print(f"  📅 Today (IST): {today_ist.strftime('%A %d-%b-%Y')} | Expiry: {expiry_str}")
        return expiry_str

    def fetch_available_expiries(self, session, headers):
        try:
            url  = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={self.nse_symbol}"
            resp = session.get(url, headers=headers, impersonate="chrome", timeout=20)
            if resp.status_code == 200:
                data     = resp.json()
                expiries = data.get('records', {}).get('expiryDates', [])
                if expiries:
                    print(f"  📅 NSE available expiries: {expiries[:5]}")
                    return expiries[0]
        except Exception as e:
            print(f"  ⚠️  Could not fetch expiry list: {e}")
        return None

    # ─────────────────────────────────────────────
    #  OPTION CHAIN FETCH
    # ─────────────────────────────────────────────
    def fetch_nse_option_chain_silent(self):
        session, headers    = self._make_nse_session()
        selected_expiry     = self.get_upcoming_expiry_tuesday()
        print(f"  🗓️  Fetching option chain for: {selected_expiry}")
        result = self._fetch_chain_for_expiry(session, headers, selected_expiry)
        if result is None:
            print(f"  ⚠️  No data for {selected_expiry}. Trying NSE expiry list...")
            real_expiry = self.fetch_available_expiries(session, headers)
            if real_expiry and real_expiry != selected_expiry:
                print(f"  🔄 Retrying with: {real_expiry}")
                result = self._fetch_chain_for_expiry(session, headers, real_expiry)
        if result is None:
            print("  ❌ Option chain fetch failed after all attempts.")
        return result

    def _fetch_chain_for_expiry(self, session, headers, expiry):
        api_url = (
            f"https://www.nseindia.com/api/option-chain-v3"
            f"?type=Indices&symbol={self.nse_symbol}&expiry={expiry}"
        )
        for attempt in range(1, 3):
            try:
                print(f"    Attempt {attempt}: expiry={expiry}")
                resp = session.get(api_url, headers=headers, impersonate="chrome", timeout=30)
                print(f"    HTTP {resp.status_code}")
                if resp.status_code != 200:
                    time.sleep(2); continue
                json_data  = resp.json()
                data       = json_data.get('records', {}).get('data', [])
                if not data:
                    print(f"    ⚠️  Empty data for expiry={expiry}")
                    return None
                rows = []
                for item in data:
                    strike = item.get('strikePrice')
                    ce = item.get('CE', {})
                    pe = item.get('PE', {})
                    rows.append({
                        'Expiry':       expiry,
                        'Strike':       strike,
                        'CE_LTP':       ce.get('lastPrice', 0),
                        'CE_OI':        ce.get('openInterest', 0),
                        'CE_Vol':       ce.get('totalTradedVolume', 0),
                        'PE_LTP':       pe.get('lastPrice', 0),
                        'PE_OI':        pe.get('openInterest', 0),
                        'PE_Vol':       pe.get('totalTradedVolume', 0),
                        'CE_OI_Change': ce.get('changeinOpenInterest', 0),
                        'PE_OI_Change': pe.get('changeinOpenInterest', 0),
                    })
                df_full    = pd.DataFrame(rows).sort_values('Strike').reset_index(drop=True)
                underlying = json_data.get('records', {}).get('underlyingValue', 0)
                atm_strike = round(underlying / 50) * 50
                all_strikes = sorted(df_full['Strike'].unique())
                if atm_strike in all_strikes:
                    atm_idx = all_strikes.index(atm_strike)
                else:
                    atm_idx    = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - underlying))
                    atm_strike = all_strikes[atm_idx]
                lower_idx        = max(0, atm_idx - 10)
                upper_idx        = min(len(all_strikes) - 1, atm_idx + 10)
                selected_strikes = all_strikes[lower_idx : upper_idx + 1]
                df = df_full[df_full['Strike'].isin(selected_strikes)].reset_index(drop=True)
                print(f"    ✅ Strikes: {len(df_full)} → ATM±10 filtered: {len(df)}")
                print(f"    📍 ATM: {atm_strike} | Range: {selected_strikes[0]}–{selected_strikes[-1]}")
                print(f"    💹 Underlying: {underlying}")
                return {
                    'expiry':     expiry,
                    'df':         df,
                    'raw_data':   data,
                    'underlying': underlying,
                    'atm_strike': atm_strike,
                }
            except Exception as e:
                print(f"    ❌ Attempt {attempt} error: {e}")
                time.sleep(2)
        return None

    # ─────────────────────────────────────────────
    #  OPTION CHAIN ANALYSIS
    # ─────────────────────────────────────────────
    def analyze_option_chain_data(self, oc_data):
        if not oc_data:
            return None
        df = oc_data['df']
        total_ce_oi  = df['CE_OI'].sum()
        total_pe_oi  = df['PE_OI'].sum()
        total_ce_vol = df['CE_Vol'].sum()
        total_pe_vol = df['PE_Vol'].sum()
        pcr_oi  = total_pe_oi  / total_ce_oi  if total_ce_oi  > 0 else 0
        pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
        total_ce_oi_change = int(df['CE_OI_Change'].sum())
        total_pe_oi_change = int(df['PE_OI_Change'].sum())
        net_oi_change      = total_pe_oi_change - total_ce_oi_change
        if   total_ce_oi_change > 0 and total_pe_oi_change < 0:
            oi_direction, oi_signal, oi_icon, oi_class = "Strong Bearish",        "Call Build-up + Put Unwinding",   "🔴", "bearish"
        elif total_ce_oi_change < 0 and total_pe_oi_change > 0:
            oi_direction, oi_signal, oi_icon, oi_class = "Strong Bullish",        "Put Build-up + Call Unwinding",   "🟢", "bullish"
        elif total_ce_oi_change > 0 and total_pe_oi_change > 0:
            if   total_pe_oi_change > total_ce_oi_change * 1.5:
                oi_direction, oi_signal, oi_icon, oi_class = "Bullish",           "Put Build-up Dominant",           "🟢", "bullish"
            elif total_ce_oi_change > total_pe_oi_change * 1.5:
                oi_direction, oi_signal, oi_icon, oi_class = "Bearish",           "Call Build-up Dominant",          "🔴", "bearish"
            else:
                oi_direction, oi_signal, oi_icon, oi_class = "Neutral (High Vol)","Both Calls & Puts Building",      "🟡", "neutral"
        elif total_ce_oi_change < 0 and total_pe_oi_change < 0:
            oi_direction, oi_signal, oi_icon, oi_class = "Neutral (Unwinding)",   "Both Calls & Puts Unwinding",     "🟡", "neutral"
        else:
            if   net_oi_change > 0:
                oi_direction, oi_signal, oi_icon, oi_class = "Moderately Bullish","Net Put Accumulation",            "🟢", "bullish"
            elif net_oi_change < 0:
                oi_direction, oi_signal, oi_icon, oi_class = "Moderately Bearish","Net Call Accumulation",           "🔴", "bearish"
            else:
                oi_direction, oi_signal, oi_icon, oi_class = "Neutral",           "Balanced OI Changes",             "🟡", "neutral"
        max_ce_oi_row = df.loc[df['CE_OI'].idxmax()]
        max_pe_oi_row = df.loc[df['PE_OI'].idxmax()]
        df['pain']    = abs(df['CE_OI'] - df['PE_OI'])
        max_pain_row  = df.loc[df['pain'].idxmin()]
        df['Total_OI'] = df['CE_OI'] + df['PE_OI']
        top_ce_strikes = df.nlargest(5, 'CE_OI')[['Strike','CE_OI','CE_LTP']].to_dict('records')
        top_pe_strikes = df.nlargest(5, 'PE_OI')[['Strike','PE_OI','PE_LTP']].to_dict('records')
        return {
            'expiry':             oc_data['expiry'],
            'underlying_value':   oc_data['underlying'],
            'atm_strike':         oc_data['atm_strike'],
            'pcr_oi':             round(pcr_oi, 3),
            'pcr_volume':         round(pcr_vol, 3),
            'total_ce_oi':        int(total_ce_oi),
            'total_pe_oi':        int(total_pe_oi),
            'max_ce_oi_strike':   int(max_ce_oi_row['Strike']),
            'max_ce_oi_value':    int(max_ce_oi_row['CE_OI']),
            'max_pe_oi_strike':   int(max_pe_oi_row['Strike']),
            'max_pe_oi_value':    int(max_pe_oi_row['PE_OI']),
            'max_pain':           int(max_pain_row['Strike']),
            'top_ce_strikes':     top_ce_strikes,
            'top_pe_strikes':     top_pe_strikes,
            'total_ce_oi_change': total_ce_oi_change,
            'total_pe_oi_change': total_pe_oi_change,
            'net_oi_change':      net_oi_change,
            'oi_direction':       oi_direction,
            'oi_signal':          oi_signal,
            'oi_icon':            oi_icon,
            'oi_class':           oi_class,
            'df':                 df,
        }

    # ─────────────────────────────────────────────
    #  TECHNICAL ANALYSIS
    # ─────────────────────────────────────────────
    def get_technical_data(self):
        try:
            print("Calculating technical indicators...")
            nifty = yf.Ticker(self.yf_symbol)
            df    = nifty.history(period="1y")
            if df.empty:
                print("Warning: Failed to fetch historical data")
                return None
            df['SMA_20']  = df['Close'].rolling(20).mean()
            df['SMA_50']  = df['Close'].rolling(50).mean()
            df['SMA_200'] = df['Close'].rolling(200).mean()
            delta        = df['Close'].diff()
            gain         = delta.where(delta > 0, 0).rolling(14).mean()
            loss         = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['RSI']    = 100 - (100 / (1 + gain / loss))
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD']   = df['EMA_12'] - df['EMA_26']
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            latest = df.iloc[-1]
            recent = df.tail(60)
            technical = {
                'current_price': latest['Close'],
                'sma_20':        latest['SMA_20'],
                'sma_50':        latest['SMA_50'],
                'sma_200':       latest['SMA_200'],
                'rsi':           latest['RSI'],
                'macd':          latest['MACD'],
                'signal':        latest['Signal'],
                'resistance':    recent['High'].quantile(0.90),
                'support':       recent['Low'].quantile(0.10),
            }
            print(f"✓ Technical | Price: {technical['current_price']:.2f} | RSI: {technical['rsi']:.1f}")
            return technical
        except Exception as e:
            print(f"Technical error: {e}")
            return None

    # ─────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────
    def calculate_smart_stop_loss(self, current_price, support, resistance, bias):
        if bias == "BULLISH":
            return round(max(support - 30, current_price - 150), 0)
        elif bias == "BEARISH":
            return round(min(resistance + 30, current_price + 150), 0)
        return None

    def select_best_technical_strategy(self, bias, option_strategies):
        name_map = {"BULLISH": "Bull Call Spread", "BEARISH": "Bear Put Spread", "SIDEWAYS": "Iron Condor"}
        target   = name_map.get(bias, "")
        for s in option_strategies:
            if s['name'] == target:
                return s
        return option_strategies[0]

    def select_best_oi_strategy(self, oi_direction, atm_strike):
        if "Strong Bullish" in oi_direction or oi_direction == "Bullish":
            return {'name':'Long Call','market_bias':'Bullish','type':'Debit','risk':'High',
                    'max_profit':'Unlimited','max_loss':'Premium Paid',
                    'description':f'Buy {atm_strike} CE','best_for':'Put build-up indicates bullish momentum',
                    'signal':'🟢 Institutional buying interest detected','time_horizon':'Intraday to 1-2 days'}
        elif "Strong Bearish" in oi_direction or oi_direction == "Bearish":
            return {'name':'Long Put','market_bias':'Bearish','type':'Debit','risk':'High',
                    'max_profit':'Huge (to ₹0)','max_loss':'Premium Paid',
                    'description':f'Buy {atm_strike} PE','best_for':'Call build-up indicates bearish momentum',
                    'signal':'🔴 Institutional selling interest detected','time_horizon':'Intraday to 1-2 days'}
        elif "High Vol" in oi_direction:
            return {'name':'Long Straddle','market_bias':'Volatile','type':'Debit','risk':'High',
                    'max_profit':'Unlimited','max_loss':'Premium Paid',
                    'description':f'Buy {atm_strike} CE + {atm_strike} PE',
                    'best_for':'Both Calls & Puts building',
                    'signal':'🟡 Big move expected, direction uncertain','time_horizon':'Intraday'}
        elif "Unwinding" in oi_direction:
            return {'name':'Iron Butterfly','market_bias':'Neutral','type':'Credit','risk':'Low',
                    'max_profit':'Premium Received','max_loss':'Capped',
                    'description':f'Sell {atm_strike} CE + PE, Buy {atm_strike+100} CE + {atm_strike-100} PE',
                    'best_for':'Unwinding suggests reduced volatility',
                    'signal':'🟡 Position squaring, range-bound expected','time_horizon':'Intraday'}
        else:
            return {'name':'Vertical Spread','market_bias':'Moderate','type':'Debit','risk':'Moderate',
                    'max_profit':'Capped','max_loss':'Limited',
                    'description':f'Vertical spread near {atm_strike}',
                    'best_for':'Balanced OI changes',
                    'signal':'🟡 Moderate directional move expected','time_horizon':'1-2 days'}

    # ─────────────────────────────────────────────
    #  MAIN ANALYSIS DATA
    # ─────────────────────────────────────────────
    def generate_analysis_data(self, technical, option_analysis):
        if not technical:
            self.log("⚠️  Technical data unavailable")
            return
        current    = technical['current_price']
        support    = technical['support']
        resistance = technical['resistance']
        ist_now    = datetime.now(pytz.timezone('Asia/Kolkata'))
        bullish_score = bearish_score = 0
        for sma in ['sma_20','sma_50','sma_200']:
            if current > technical[sma]: bullish_score += 1
            else:                         bearish_score += 1
        rsi = technical['rsi']
        if   rsi > 70: bearish_score += 1
        elif rsi < 30: bullish_score += 2
        if technical['macd'] > technical['signal']: bullish_score += 1
        else:                                        bearish_score += 1
        if option_analysis:
            pcr      = option_analysis['pcr_oi']
            max_pain = option_analysis['max_pain']
            if   pcr > 1.2:              bullish_score += 2
            elif pcr < 0.7:              bearish_score += 2
            if   current > max_pain+100: bearish_score += 1
            elif current < max_pain-100: bullish_score += 1
        score_diff = bullish_score - bearish_score
        print(f"  📊 Bullish: {bullish_score} | Bearish: {bearish_score} | Diff: {score_diff}")
        if   score_diff >= 3:  bias, bias_icon, bias_class = "BULLISH",  "📈", "bullish"; confidence = "HIGH" if score_diff >= 4 else "MEDIUM"
        elif score_diff <= -3: bias, bias_icon, bias_class = "BEARISH",  "📉", "bearish"; confidence = "HIGH" if score_diff <= -4 else "MEDIUM"
        else:                  bias, bias_icon, bias_class = "SIDEWAYS", "↔️",  "sideways"; confidence = "MEDIUM"
        if   rsi > 70: rsi_status, rsi_badge, rsi_icon = "Overbought", "bearish", "🔴"
        elif rsi < 30: rsi_status, rsi_badge, rsi_icon = "Oversold",   "bullish", "🟢"
        else:          rsi_status, rsi_badge, rsi_icon = "Neutral",    "neutral", "🟡"
        macd_bullish = technical['macd'] > technical['signal']
        if option_analysis:
            pcr = option_analysis['pcr_oi']
            if   pcr > 1.2: pcr_status, pcr_badge, pcr_icon = "Bullish", "bullish", "🟢"
            elif pcr < 0.7: pcr_status, pcr_badge, pcr_icon = "Bearish", "bearish", "🔴"
            else:           pcr_status, pcr_badge, pcr_icon = "Neutral", "neutral", "🟡"
        else:
            pcr_status, pcr_badge, pcr_icon = "N/A", "neutral", "🟡"
        if option_analysis:
            max_ce_strike = option_analysis['max_ce_oi_strike']
            max_pe_strike = option_analysis['max_pe_oi_strike']
            atm_strike    = option_analysis['atm_strike']
        else:
            atm_strike    = int(current/50)*50
            max_ce_strike = atm_strike + 200
            max_pe_strike = atm_strike - 200
        if bias == "BULLISH":
            mid = (support + resistance) / 2
            entry_low  = current - 100 if current > mid else current - 50
            entry_high = current - 50  if current > mid else current
            target_1   = resistance
            target_2   = max_ce_strike
            stop_loss  = self.calculate_smart_stop_loss(current, support, resistance, "BULLISH")
            option_strategies = [
                {'name':'Long Call',       'market_bias':'Bullish','type':'Debit', 'risk':'High',
                 'max_profit':'Unlimited', 'max_loss':'Premium Paid',
                 'description':f'Buy {atm_strike} CE','best_for':'Strong upward momentum expected'},
                {'name':'Bull Call Spread','market_bias':'Bullish','type':'Debit', 'risk':'Moderate',
                 'max_profit':'Capped',    'max_loss':'Premium Paid',
                 'description':f'Buy {atm_strike} CE, Sell {atm_strike+200} CE','best_for':'Moderate upside with limited risk'},
                {'name':'Bull Put Spread', 'market_bias':'Bullish','type':'Credit','risk':'Moderate',
                 'max_profit':'Premium Received','max_loss':'Capped',
                 'description':f'Sell {atm_strike-100} PE, Buy {atm_strike-200} PE','best_for':'Expect market to stay above support'},
            ]
        elif bias == "BEARISH":
            mid = (support + resistance) / 2
            entry_low  = current
            entry_high = current + 100 if current < mid else current + 50
            target_1   = support
            target_2   = max_pe_strike
            stop_loss  = self.calculate_smart_stop_loss(current, support, resistance, "BEARISH")
            option_strategies = [
                {'name':'Long Put',        'market_bias':'Bearish','type':'Debit', 'risk':'High',
                 'max_profit':'Huge (to ₹0)','max_loss':'Premium Paid',
                 'description':f'Buy {atm_strike} PE','best_for':'Strong downward momentum expected'},
                {'name':'Bear Put Spread', 'market_bias':'Bearish','type':'Debit', 'risk':'Moderate',
                 'max_profit':'Capped',    'max_loss':'Premium Paid',
                 'description':f'Buy {atm_strike} PE, Sell {atm_strike-200} PE','best_for':'Moderate downside with limited risk'},
                {'name':'Bear Call Spread','market_bias':'Bearish','type':'Credit','risk':'Moderate',
                 'max_profit':'Premium Received','max_loss':'Capped',
                 'description':f'Sell {atm_strike+100} CE, Buy {atm_strike+200} CE','best_for':'Expect market to stay below resistance'},
            ]
        else:
            entry_low  = support
            entry_high = resistance
            target_1   = resistance
            target_2   = support
            stop_loss  = None
            option_strategies = [
                {'name':'Iron Condor',    'market_bias':'Neutral', 'type':'Credit','risk':'Low',
                 'max_profit':'Premium Received','max_loss':'Capped',
                 'description':f'Sell {atm_strike+100} CE + Buy {atm_strike+200} CE, Sell {atm_strike-100} PE + Buy {atm_strike-200} PE',
                 'best_for':'Expect low volatility, range-bound market'},
                {'name':'Iron Butterfly', 'market_bias':'Neutral', 'type':'Credit','risk':'Low',
                 'max_profit':'Premium Received','max_loss':'Capped',
                 'description':f'Sell {atm_strike} CE + Sell {atm_strike} PE, Buy {atm_strike+100} CE + Buy {atm_strike-100} PE',
                 'best_for':'Expect price to remain near ATM strike'},
                {'name':'Short Straddle', 'market_bias':'Neutral', 'type':'Credit','risk':'Very High',
                 'max_profit':'Premium Received','max_loss':'Unlimited',
                 'description':f'Sell {atm_strike} CE + Sell {atm_strike} PE','best_for':'ONLY for experienced traders!'},
                {'name':'Long Strangle',  'market_bias':'Volatile','type':'Debit', 'risk':'High',
                 'max_profit':'Unlimited','max_loss':'Premium Paid',
                 'description':f'Buy {atm_strike+100} CE + Buy {atm_strike-100} PE','best_for':'Expect big move but unsure of direction'},
            ]
        if stop_loss and bias != "SIDEWAYS":
            risk_points       = abs(current - stop_loss)
            reward_points     = abs(target_1 - current)
            risk_reward_ratio = round(reward_points / risk_points, 2) if risk_points > 0 else 0
        else:
            risk_points = reward_points = risk_reward_ratio = 0

        rsi_pct = min(100, max(0, rsi))
        def sma_bar(sma_val):
            diff = (current - sma_val) / sma_val * 100
            return min(100, max(0, 50 + diff * 10))
        macd_val  = technical['macd']
        macd_pct  = min(100, max(0, 50 + macd_val * 2))
        pcr_pct   = min(100, max(0, (option_analysis['pcr_oi'] / 2 * 100))) if option_analysis else 50
        if option_analysis:
            rng     = resistance - support if resistance != support else 1
            mp_pct  = min(100, max(0, (option_analysis['max_pain'] - support) / rng * 100))
            total_oi     = option_analysis['total_ce_oi'] + option_analysis['total_pe_oi']
            ce_oi_pct    = min(100, max(0, option_analysis['total_ce_oi'] / total_oi * 100)) if total_oi > 0 else 50
            pe_oi_pct    = 100 - ce_oi_pct
        else:
            mp_pct = ce_oi_pct = pe_oi_pct = 50

        fii_dii_raw  = fetch_fii_dii_data()
        fii_dii_summ = compute_fii_dii_summary(fii_dii_raw)

        self.html_data = {
            'timestamp':      ist_now.strftime('%d-%b-%Y %H:%M IST'),
            'current_price':  current,
            'expiry':         option_analysis['expiry'] if option_analysis else 'N/A',
            'atm_strike':     atm_strike,
            'bias':           bias,
            'bias_icon':      bias_icon,
            'bias_class':     bias_class,
            'confidence':     confidence,
            'bullish_score':  bullish_score,
            'bearish_score':  bearish_score,
            'rsi':            rsi,
            'rsi_pct':        rsi_pct,
            'rsi_status':     rsi_status,
            'rsi_badge':      rsi_badge,
            'rsi_icon':       rsi_icon,
            'sma_20':         technical['sma_20'],
            'sma_20_above':   current > technical['sma_20'],
            'sma_20_pct':     sma_bar(technical['sma_20']),
            'sma_50':         technical['sma_50'],
            'sma_50_above':   current > technical['sma_50'],
            'sma_50_pct':     sma_bar(technical['sma_50']),
            'sma_200':        technical['sma_200'],
            'sma_200_above':  current > technical['sma_200'],
            'sma_200_pct':    sma_bar(technical['sma_200']),
            'macd':           technical['macd'],
            'macd_signal':    technical['signal'],
            'macd_bullish':   macd_bullish,
            'macd_pct':       macd_pct,
            'pcr':            option_analysis['pcr_oi'] if option_analysis else 0,
            'pcr_pct':        pcr_pct,
            'pcr_status':     pcr_status,
            'pcr_badge':      pcr_badge,
            'pcr_icon':       pcr_icon,
            'max_pain':       option_analysis['max_pain']           if option_analysis else 0,
            'max_pain_pct':   mp_pct,
            'max_ce_oi':      max_ce_strike,
            'max_pe_oi':      max_pe_strike,
            'ce_oi_pct':      ce_oi_pct,
            'pe_oi_pct':      pe_oi_pct,
            'total_ce_oi_change': option_analysis['total_ce_oi_change'] if option_analysis else 0,
            'total_pe_oi_change': option_analysis['total_pe_oi_change'] if option_analysis else 0,
            'net_oi_change':      option_analysis['net_oi_change']       if option_analysis else 0,
            'oi_direction':       option_analysis['oi_direction']        if option_analysis else 'N/A',
            'oi_signal':          option_analysis['oi_signal']           if option_analysis else 'N/A',
            'oi_icon':            option_analysis['oi_icon']             if option_analysis else '🟡',
            'oi_class':           option_analysis['oi_class']            if option_analysis else 'neutral',
            'support':            support,
            'resistance':         resistance,
            'strong_support':     support    - 100,
            'strong_resistance':  resistance + 100,
            'strategy_type':  bias,
            'entry_low':      entry_low,
            'entry_high':     entry_high,
            'target_1':       target_1,
            'target_2':       target_2,
            'stop_loss':      stop_loss,
            'risk_points':    int(risk_points),
            'reward_points':  int(reward_points),
            'risk_reward_ratio': risk_reward_ratio,
            'option_strategies':               option_strategies,
            'recommended_technical_strategy':  self.select_best_technical_strategy(bias, option_strategies),
            'recommended_oi_strategy':         self.select_best_oi_strategy(
                option_analysis['oi_direction'] if option_analysis else 'Neutral', atm_strike
            ) if option_analysis else None,
            'top_ce_strikes':  option_analysis['top_ce_strikes'] if option_analysis else [],
            'top_pe_strikes':  option_analysis['top_pe_strikes'] if option_analysis else [],
            'has_option_data': option_analysis is not None,
            'fii_dii_data':  fii_dii_raw,
            'fii_dii_summ':  fii_dii_summ,
        }

    # ─────────────────────────────────────────────
    #  HTML HELPERS
    # ─────────────────────────────────────────────
    def _bar_color_class(self, badge):
        return {'bullish':'bar-teal','bearish':'bar-red','neutral':'bar-gold'}.get(badge,'bar-teal')

    def _stat_card(self, icon, label, value, badge_text, badge_class, bar_pct, bar_type, sub_text=""):
        tag_map = {
            'bullish': ('tag-bull', '#00e5ff'),
            'bearish': ('tag-bear', '#ff5252'),
            'neutral': ('tag-neu',  '#ffb74d'),
        }
        tag_cls, _  = tag_map.get(badge_class, tag_map['neutral'])
        hi_cls      = 'g-hi' if badge_class == 'bullish' else ('g-red' if badge_class == 'bearish' else '')
        sub_html    = f'<div class="sub">{sub_text}</div>' if sub_text else ''
        return f"""
            <div class="g {hi_cls}">
                <div class="card-top-row">
                    <span class="card-ico">{icon}</span>
                    <div class="lbl">{label}</div>
                </div>
                <span class="val">{value}</span>
                <div class="bar-wrap"><div class="bar-fill {bar_type}" style="width:{bar_pct:.1f}%"></div></div>
                <div class="card-foot">
                    {sub_html}
                    <span class="tag {tag_cls}">{badge_text}</span>
                </div>
            </div>"""

    # ─────────────────────────────────────────────
    #  KEY LEVELS FIRE ROW
    # ─────────────────────────────────────────────
    def _key_levels_fire_row(self, d):
        if d['stop_loss']:
            sl_html = f"""
                <div class="fire-col-label">STOP LOSS</div>
                <div class="fire-col-value stop-price">₹{d['stop_loss']:,.0f}</div>
                <div class="fire-rr">Risk {d['risk_points']} pts &nbsp;·&nbsp; R:R 1:{d['risk_reward_ratio']}</div>"""
        else:
            sl_html = """
                <div class="fire-col-label">STOP LOSS</div>
                <div class="fire-col-value stop-text">Use option premium<br>as max loss</div>"""
        return f"""
        <div class="fire-row">
            <div class="fire-col">
                <div class="fire-heading">📍 KEY<br>LEVELS</div>
            </div>
            <div class="fire-col">
                <div class="fire-col-label">ENTRY ZONE</div>
                <div class="fire-col-value">₹{d['entry_low']:,.0f}–₹{d['entry_high']:,.0f}</div>
            </div>
            <div class="fire-col">
                <div class="fire-col-label">TARGET 1 / 2</div>
                <div class="fire-col-value">₹{d['target_1']:,.0f} &nbsp;/&nbsp; ₹{d['target_2']:,.0f}</div>
            </div>
            <div class="fire-col">
                {sl_html}
            </div>
        </div>"""

    # ─────────────────────────────────────────────
    #  FII / DII SECTION
    # ─────────────────────────────────────────────
    def _fiidii_section_html(self):
        data  = self.html_data['fii_dii_data']
        summ  = self.html_data['fii_dii_summ']
        badge_map = {
            'fii-bull':  ('#00e676', 'rgba(0,230,118,0.12)',  'rgba(0,230,118,0.3)'),
            'fii-cbull': ('#69f0ae', 'rgba(105,240,174,0.10)', 'rgba(105,240,174,0.28)'),
            'fii-neu':   ('#ffd740', 'rgba(255,215,64,0.10)',  'rgba(255,215,64,0.28)'),
            'fii-bear':  ('#ff5252', 'rgba(255,82,82,0.10)',   'rgba(255,82,82,0.28)'),
        }
        s_color, s_bg, s_border = badge_map.get(summ['badge_cls'], badge_map['fii-neu'])

        def kpi(label, value, color):
            sign = '+' if value >= 0 else ''
            return f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(79,195,247,0.12);
                            border-radius:14px;padding:16px 14px;text-align:center;">
                    <div style="font-size:9px;letter-spacing:2.5px;color:rgba(128,222,234,0.55);
                                text-transform:uppercase;margin-bottom:8px;">{label}</div>
                    <div style="font-family:'Oxanium',sans-serif;font-size:22px;font-weight:700;
                                color:{color};">{sign}{value:.0f}</div>
                    <div style="font-size:9px;color:#37474f;margin-top:3px;">₹ Cr / day</div>
                </div>"""

        fii_col = '#00e676' if summ['fii_avg'] >= 0 else '#ff5252'
        dii_col = '#40c4ff' if summ['dii_avg'] >= 0 else '#ff5252'
        net_col = '#b388ff' if summ['net_avg'] >= 0 else '#ff5252'
        kpi_html = (kpi('FII 5D Avg', summ['fii_avg'], fii_col) +
                    kpi('DII 5D Avg', summ['dii_avg'], dii_col) +
                    kpi('Net Flow',   summ['net_avg'], net_col))
        date_range = f"{data[0]['date']} – {data[-1]['date']}, 2026"

        return f"""
    <div class="section">
        <div class="section-title"><span>🏦</span> FII / DII INSTITUTIONAL FLOW
            <span style="font-size:11px;color:#80deea;font-weight:400;letter-spacing:1px;">5-Day Average</span>
        </div>
        <div class="fii-kpi-grid" style="display:grid;grid-template-columns:auto 1fr 1fr 1fr;gap:12px;
                    align-items:stretch;margin-bottom:18px;">
            <div style="background:{s_bg};border:1px solid {s_border};border-radius:14px;
                        padding:18px 20px;display:flex;flex-direction:column;
                        align-items:center;justify-content:center;min-width:130px;">
                <div style="font-size:28px;margin-bottom:6px;">{summ['emoji']}</div>
                <div style="font-size:9px;font-weight:800;letter-spacing:1.5px;
                            color:{s_color};text-align:center;line-height:1.4;">{summ['label']}</div>
                <div style="font-size:8px;color:rgba(176,190,197,0.4);
                            margin-top:5px;letter-spacing:1px;">{date_range}</div>
            </div>
            {kpi_html}
        </div>
        <div style="background:{s_bg};border:1px solid {s_border};
                    border-radius:10px;padding:14px 16px;">
            <div style="font-size:9px;color:{s_color};letter-spacing:2px;
                        font-weight:700;margin-bottom:6px;">5-DAY INSIGHT</div>
            <div style="font-size:13px;color:#cfd8dc;line-height:1.8;font-weight:500;">{summ['insight']}</div>
        </div>
    </div>
"""

    # ─────────────────────────────────────────────
    #  ★ NAVY COMMAND — CHANGE IN OI SECTION ★
    # ─────────────────────────────────────────────
    def _oi_navy_command_section(self, d):
        """
        Renders the 'Change in Open Interest' block in the Navy Command theme
        that exactly matches the user-selected design:
          • Header bar: icon + title + ATM±10 badge
          • Direction box: label, big direction name, signal text, bear/bull strength meter
          • 3 stat cards: CALL OI CHANGE | PUT OI CHANGE | NET OI CHANGE
            each with icon top-right, value, sub-label, full-width badge button
          • Logic/how-to-read box
        """
        oi_cls    = d['oi_class']   # 'bearish' | 'bullish' | 'neutral'
        direction = d['oi_direction']
        signal    = d['oi_signal']

        # ── strength meter: what % of total OI change does the dominant side hold?
        total_abs = abs(d['total_ce_oi_change']) + abs(d['total_pe_oi_change'])
        if total_abs > 0:
            if oi_cls == 'bearish':
                strength_pct = round(abs(d['total_ce_oi_change']) / total_abs * 100)
            elif oi_cls == 'bullish':
                strength_pct = round(abs(d['total_pe_oi_change']) / total_abs * 100)
            else:
                strength_pct = 50
        else:
            strength_pct = 50

        # ── direction box colour scheme
        if oi_cls == 'bearish':
            dir_bg        = 'rgba(30,10,14,0.92)'
            dir_border    = 'rgba(239,68,68,0.35)'
            dir_left_bar  = 'linear-gradient(180deg,#ef4444,#b91c1c)'
            dir_name_col  = '#fb7185'
            dir_desc_col  = 'rgba(251,113,133,0.5)'
            meter_label   = 'BEAR STRENGTH'
            meter_fill    = 'linear-gradient(90deg,#ef4444,#f97316)'
            meter_col     = '#fb7185'
            dir_tag_txt   = 'BEARISH SIGNAL'
            dir_tag_bg    = 'rgba(239,68,68,0.12)'
            dir_tag_bdr   = 'rgba(239,68,68,0.3)'
        elif oi_cls == 'bullish':
            dir_bg        = 'rgba(10,30,20,0.92)'
            dir_border    = 'rgba(16,185,129,0.35)'
            dir_left_bar  = 'linear-gradient(180deg,#10b981,#047857)'
            dir_name_col  = '#34d399'
            dir_desc_col  = 'rgba(52,211,153,0.5)'
            meter_label   = 'BULL STRENGTH'
            meter_fill    = 'linear-gradient(90deg,#10b981,#34d399)'
            meter_col     = '#34d399'
            dir_tag_txt   = 'BULLISH SIGNAL'
            dir_tag_bg    = 'rgba(16,185,129,0.12)'
            dir_tag_bdr   = 'rgba(16,185,129,0.3)'
        else:
            dir_bg        = 'rgba(20,20,10,0.92)'
            dir_border    = 'rgba(251,191,36,0.3)'
            dir_left_bar  = 'linear-gradient(180deg,#f59e0b,#d97706)'
            dir_name_col  = '#fbbf24'
            dir_desc_col  = 'rgba(251,191,36,0.5)'
            meter_label   = 'NEUTRAL ZONE'
            meter_fill    = 'linear-gradient(90deg,#f59e0b,#fbbf24)'
            meter_col     = '#fbbf24'
            dir_tag_txt   = 'NEUTRAL'
            dir_tag_bg    = 'rgba(251,191,36,0.12)'
            dir_tag_bdr   = 'rgba(251,191,36,0.3)'

        # ── 3 stat cards config
        # CALL OI
        ce_val     = d['total_ce_oi_change']
        ce_is_bear = ce_val > 0
        ce_col     = '#fb7185' if ce_is_bear else '#34d399'
        ce_dot_col = '#ef4444' if ce_is_bear else '#10b981'
        ce_lbl     = 'Bearish Signal' if ce_is_bear else 'Bullish Signal'
        ce_btn_col = '#ef4444' if ce_is_bear else '#10b981'
        ce_btn_bg  = 'rgba(239,68,68,0.12)' if ce_is_bear else 'rgba(16,185,129,0.12)'
        ce_btn_bdr = 'rgba(239,68,68,0.4)' if ce_is_bear else 'rgba(16,185,129,0.4)'
        ce_sub     = 'CE open interest Δ'

        # PUT OI
        pe_val     = d['total_pe_oi_change']
        pe_is_bull = pe_val > 0
        pe_col     = '#34d399' if pe_is_bull else '#fb7185'
        pe_dot_col = '#10b981' if pe_is_bull else '#ef4444'
        pe_lbl     = 'Bullish Signal' if pe_is_bull else 'Bearish Signal'
        pe_btn_col = '#10b981' if pe_is_bull else '#ef4444'
        pe_btn_bg  = 'rgba(16,185,129,0.12)' if pe_is_bull else 'rgba(239,68,68,0.12)'
        pe_btn_bdr = 'rgba(16,185,129,0.4)' if pe_is_bull else 'rgba(239,68,68,0.4)'
        pe_sub     = 'PE open interest Δ'

        # NET OI
        net_val    = d['net_oi_change']
        if net_val > 0:
            net_col = '#34d399'; net_dot_col = '#10b981'
            net_lbl = 'Bullish Net'; net_btn_col = '#10b981'
            net_btn_bg = 'rgba(16,185,129,0.12)'; net_btn_bdr = 'rgba(16,185,129,0.4)'
        elif net_val < 0:
            net_col = '#fb7185'; net_dot_col = '#ef4444'
            net_lbl = 'Bearish Net'; net_btn_col = '#ef4444'
            net_btn_bg = 'rgba(239,68,68,0.12)'; net_btn_bdr = 'rgba(239,68,68,0.4)'
        else:
            net_col = '#fbbf24'; net_dot_col = '#f59e0b'
            net_lbl = 'Balanced'; net_btn_col = '#f59e0b'
            net_btn_bg = 'rgba(245,158,11,0.12)'; net_btn_bdr = 'rgba(245,158,11,0.4)'
        net_sub = 'PE Δ − CE Δ'

        def nc_card(label, icon_dot_col, value, val_col, sub, btn_lbl, btn_col, btn_bg, btn_bdr, icon_char):
            return f"""
            <div class="nc-card">
                <div class="nc-card-header">
                    <span class="nc-card-label">{label}</span>
                    <span style="font-size:18px;line-height:1;color:{icon_dot_col};">{icon_char}</span>
                </div>
                <div class="nc-card-value" style="color:{val_col};">{value:+,}</div>
                <div class="nc-card-sub">{sub}</div>
                <div class="nc-card-btn" style="color:{btn_col};background:{btn_bg};border:1px solid {btn_bdr};">
                    {btn_lbl}
                </div>
            </div>"""

        cards_html = (
            nc_card('CALL OI CHANGE', ce_dot_col, ce_val, ce_col, ce_sub,
                    ce_lbl, ce_btn_col, ce_btn_bg, ce_btn_bdr,
                    '🔴' if ce_is_bear else '🟢') +
            nc_card('PUT OI CHANGE',  pe_dot_col, pe_val, pe_col, pe_sub,
                    pe_lbl, pe_btn_col, pe_btn_bg, pe_btn_bdr,
                    '🟢' if pe_is_bull else '🔴') +
            nc_card('NET OI CHANGE',  net_dot_col, net_val, net_col, net_sub,
                    net_lbl, net_btn_col, net_btn_bg, net_btn_bdr, '⚖️')
        )

        return f"""
    <!-- ══════════════════════════════════════════
         CHANGE IN OPEN INTEREST — NAVY COMMAND THEME
         ══════════════════════════════════════════ -->
    <div class="section">

        <!-- ── Section header bar ── -->
        <div class="nc-section-header">
            <div class="nc-header-left">
                <div class="nc-header-icon">📊</div>
                <div>
                    <div class="nc-header-title">Change in Open Interest</div>
                    <div class="nc-header-sub">Today's Direction Analysis</div>
                </div>
            </div>
            <div class="nc-atm-badge">ATM ±10</div>
        </div>

        <!-- ── Direction box ── -->
        <div class="nc-dir-box" style="background:{dir_bg};border:1px solid {dir_border};">
            <div style="display:flex;align-items:stretch;gap:18px;">
                <div class="nc-dir-bar" style="background:{dir_left_bar};"></div>
                <div style="flex:1;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
                    <div>
                        <div class="nc-dir-tag">MARKET DIRECTION</div>
                        <div class="nc-dir-name" style="color:{dir_name_col};">{direction}</div>
                        <div class="nc-dir-signal" style="color:{dir_desc_col};">{signal}</div>
                    </div>
                    <div class="nc-meter-wrap">
                        <div class="nc-meter-label">{meter_label}</div>
                        <div class="nc-meter-track">
                            <div class="nc-meter-fill" style="width:{strength_pct}%;background:{meter_fill};"></div>
                            <div class="nc-meter-head" style="left:{strength_pct}%;background:{meter_col};box-shadow:0 0 8px {meter_col};"></div>
                        </div>
                        <div class="nc-meter-pct" style="color:{meter_col};">{strength_pct}%</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ── 3 stat cards ── -->
        <div class="nc-cards-grid">
            {cards_html}
        </div>

        <!-- ── How to read logic box ── -->
        <div class="logic-box" style="margin-top:16px;">
            <p>
                <strong>📖 How to Read:</strong><br>
                • <strong>Call OI +</strong> = Writers selling calls (Bearish) &nbsp;|&nbsp; <strong>Call OI −</strong> = Unwinding (Bullish)<br>
                • <strong>Put OI +</strong>  = Writers selling puts (Bullish) &nbsp;|&nbsp; <strong>Put OI −</strong> = Unwinding (Bearish)<br>
                • <strong>Net OI</strong> = Put Δ − Call Δ &nbsp;(Positive = Bullish, Negative = Bearish)
            </p>
        </div>
    </div>
"""

    # ─────────────────────────────────────────────
    #  MAIN HTML GENERATION
    # ─────────────────────────────────────────────
    def generate_html_email(self):
        d = self.html_data
        sma20_bar  = 'bar-teal' if d['sma_20_above']  else 'bar-red'
        sma50_bar  = 'bar-teal' if d['sma_50_above']  else 'bar-red'
        sma200_bar = 'bar-teal' if d['sma_200_above'] else 'bar-red'
        macd_bar   = 'bar-teal' if d['macd_bullish']  else 'bar-red'
        pcr_bar    = self._bar_color_class(d['pcr_badge'])
        sma20_badge  = 'bullish' if d['sma_20_above']  else 'bearish'
        sma50_badge  = 'bullish' if d['sma_50_above']  else 'bearish'
        sma200_badge = 'bullish' if d['sma_200_above'] else 'bearish'
        macd_badge   = 'bullish' if d['macd_bullish']  else 'bearish'
        sma20_lbl    = 'Above'   if d['sma_20_above']  else 'Below'
        sma50_lbl    = 'Above'   if d['sma_50_above']  else 'Below'
        sma200_lbl   = 'Above'   if d['sma_200_above'] else 'Below'
        macd_lbl     = 'Bullish' if d['macd_bullish']  else 'Bearish'
        sma20_ico    = '✅'      if d['sma_20_above']  else '❌'
        sma50_ico    = '✅'      if d['sma_50_above']  else '❌'
        sma200_ico   = '✅'      if d['sma_200_above'] else '❌'
        macd_ico     = '🟢'      if d['macd_bullish']  else '🔴'
        tech_cards = (
            self._stat_card(d['rsi_icon'], 'RSI (14)',  f"{d['rsi']:.1f}",
                            d['rsi_status'], d['rsi_badge'], d['rsi_pct'], 'bar-gold', '14-period momentum') +
            self._stat_card(sma20_ico,  'SMA 20',  f"₹{d['sma_20']:,.0f}",
                            sma20_lbl,  sma20_badge,  d['sma_20_pct'],  sma20_bar,  '20-day average') +
            self._stat_card(sma50_ico,  'SMA 50',  f"₹{d['sma_50']:,.0f}",
                            sma50_lbl,  sma50_badge,  d['sma_50_pct'],  sma50_bar,  '50-day average') +
            self._stat_card(sma200_ico, 'SMA 200', f"₹{d['sma_200']:,.0f}",
                            sma200_lbl, sma200_badge, d['sma_200_pct'], sma200_bar, '200-day average') +
            self._stat_card(macd_ico,   'MACD',    f"{d['macd']:.2f}",
                            macd_lbl,   macd_badge,   d['macd_pct'],    macd_bar,   f"Signal: {d['macd_signal']:.2f}")
        )
        if d['has_option_data']:
            ce_badge = 'bearish'; pe_badge = 'bullish'
            oc_cards = (
                self._stat_card(d['pcr_icon'], 'PCR Ratio (OI)', f"{d['pcr']:.3f}",
                                d['pcr_status'], d['pcr_badge'], d['pcr_pct'], pcr_bar, 'Put/Call OI ratio') +
                self._stat_card('🎯', 'Max Pain', f"₹{d['max_pain']:,}",
                                'Expiry Magnet', 'neutral', d['max_pain_pct'], 'bar-gold', 'Price gravity level') +
                self._stat_card('🔴', 'Max Call OI', f"₹{d['max_ce_oi']:,}",
                                'Resistance', ce_badge, d['ce_oi_pct'], 'bar-red', 'CE wall') +
                self._stat_card('🟢', 'Max Put OI', f"₹{d['max_pe_oi']:,}",
                                'Support', pe_badge, d['pe_oi_pct'], 'bar-teal', 'PE floor')
            )
        else:
            oc_cards = '<div style="color:#80deea;padding:20px;">Option chain data unavailable</div>'

        _ss  = d['strong_support']
        _sr  = d['strong_resistance']
        _rng = _sr - _ss if _sr != _ss else 1
        def _pct_real(val):
            return round(max(3, min(97, (val - _ss) / _rng * 100)), 2)
        _pct_cp      = _pct_real(d['current_price'])
        _pts_to_res  = int(d['resistance']    - d['current_price'])
        _pts_to_sup  = int(d['current_price'] - d['support'])
        _mp_node = ""
        if d['has_option_data']:
            _mp_node = f"""
              <div class="rl-node-b" style="left:43%;">
                <div class="rl-dot" style="background:#ffb74d;box-shadow:0 0 8px #ffb74d;margin:0 auto 5px;"></div>
                <div class="rl-lbl" style="color:#ffb74d;">Max Pain</div>
                <div class="rl-val" style="color:#ffb74d;">₹{d['max_pain']:,}</div>
              </div>"""

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nifty 50 Daily Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Oxanium:wght@400;600;700;800&family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        *{{margin:0;padding:0;box-sizing:border-box;}}
        body{{
            font-family:'Rajdhani',sans-serif;
            background:linear-gradient(135deg,#0f2027 0%,#203a43 50%,#2c5364 100%);
            min-height:100vh; padding:20px; color:#b0bec5;
        }}
        .container{{
            max-width:1200px; margin:0 auto;
            background:rgba(15,32,39,0.85);
            backdrop-filter:blur(20px);
            border-radius:20px; overflow:hidden;
            box-shadow:0 20px 60px rgba(0,0,0,0.5);
            border:1px solid rgba(79,195,247,0.18);
        }}
        .header{{
            background:linear-gradient(135deg,#0f2027,#203a43);
            padding:40px 30px; text-align:center;
            border-bottom:2px solid rgba(79,195,247,0.3);
            position:relative; overflow:hidden;
        }}
        .header::before{{
            content:''; position:absolute; inset:0;
            background:radial-gradient(circle at 50% 50%,rgba(79,195,247,0.08) 0%,transparent 70%);
            pointer-events:none;
        }}
        .header h1{{
            font-family:'Oxanium',sans-serif;
            font-size:30px; font-weight:800; color:#4fc3f7;
            text-shadow:0 0 30px rgba(79,195,247,0.5);
            letter-spacing:2px; position:relative; z-index:1;
        }}
        .header p{{color:#80deea;font-size:13px;margin-top:8px;position:relative;z-index:1;}}
        .section{{padding:28px 26px;border-bottom:1px solid rgba(79,195,247,0.08);}}
        .section:last-child{{border-bottom:none;}}
        .section-title{{
            font-family:'Oxanium',sans-serif;
            font-size:13px; font-weight:700; letter-spacing:2.5px; color:#4fc3f7;
            text-transform:uppercase; display:flex; align-items:center; gap:10px;
            margin-bottom:20px; padding-bottom:12px;
            border-bottom:1px solid rgba(79,195,247,0.18);
        }}
        .section-title span{{font-size:18px;}}
        .g{{
            background:rgba(255,255,255,0.04);
            backdrop-filter:blur(20px);
            -webkit-backdrop-filter:blur(20px);
            border:1px solid rgba(79,195,247,0.18);
            border-radius:16px;
            position:relative; overflow:hidden;
            transition:all 0.35s cubic-bezier(0.4,0,0.2,1);
        }}
        .g::before{{
            content:''; position:absolute; top:0; left:0; right:0; height:1px;
            background:linear-gradient(90deg,transparent,rgba(255,255,255,0.25),transparent);
            z-index:1;
        }}
        .g::after{{
            content:''; position:absolute; top:-60%; left:-30%; width:50%; height:200%;
            background:linear-gradient(105deg,transparent,rgba(255,255,255,0.04),transparent);
            transform:skewX(-15deg); transition:left 0.6s ease; z-index:0;
        }}
        .g:hover::after{{left:130%;}}
        .g:hover{{
            background:rgba(79,195,247,0.09);
            border-color:rgba(79,195,247,0.45);
            box-shadow:0 12px 40px rgba(0,0,0,0.35),inset 0 1px 0 rgba(255,255,255,0.1);
            transform:translateY(-4px);
        }}
        .g-hi{{background:rgba(79,195,247,0.09);border-color:rgba(79,195,247,0.35);}}
        .g-red{{background:rgba(244,67,54,0.06);border-color:rgba(244,67,54,0.25);}}
        .g-red:hover{{background:rgba(244,67,54,0.1);border-color:rgba(244,67,54,0.45);}}
        .card-grid{{display:grid;gap:14px;}}
        .grid-5{{grid-template-columns:repeat(5,1fr);}}
        .grid-4{{grid-template-columns:repeat(4,1fr);}}
        .g .card-top-row{{
            display:flex; align-items:center; gap:10px;
            margin-bottom:10px; position:relative; z-index:2; padding:14px 16px 0;
        }}
        .card-ico{{font-size:22px;line-height:1;}}
        .lbl{{
            font-size:9px; letter-spacing:2.5px; color:rgba(128,222,234,0.65);
            text-transform:uppercase; font-weight:600; line-height:1.3;
        }}
        .val{{
            font-family:'Oxanium',sans-serif; font-size:24px; font-weight:700;
            color:#fff; display:block; margin-bottom:10px; position:relative; z-index:2;
            padding:0 16px;
        }}
        .bar-wrap{{
            height:5px; background:rgba(0,0,0,0.35);
            border-radius:3px; margin:0 16px 12px; overflow:hidden;
            position:relative; z-index:2;
        }}
        .bar-fill{{height:100%;border-radius:3px;transition:width 1.2s cubic-bezier(0.4,0,0.2,1);}}
        .bar-teal{{background:linear-gradient(90deg,#00bcd4,#4fc3f7);box-shadow:0 0 8px rgba(79,195,247,0.6);}}
        .bar-red {{background:linear-gradient(90deg,#f44336,#ff5722);box-shadow:0 0 8px rgba(244,67,54,0.5);}}
        .bar-gold{{background:linear-gradient(90deg,#ffb74d,#ffd54f);box-shadow:0 0 8px rgba(255,183,77,0.5);}}
        .bar-grn {{background:linear-gradient(90deg,#00e676,#00bcd4);box-shadow:0 0 8px rgba(0,230,118,0.5);}}
        .card-foot{{
            display:flex; justify-content:space-between; align-items:center;
            padding:0 16px 14px; position:relative; z-index:2;
        }}
        .sub{{font-size:10px;color:#455a64;font-family:'JetBrains Mono',monospace;}}
        .tag{{
            display:inline-flex; align-items:center;
            padding:3px 11px; border-radius:20px; font-size:11px; font-weight:700;
            letter-spacing:0.5px; font-family:'Rajdhani',sans-serif;
        }}
        .tag-neu {{background:rgba(255,183,77,0.15); color:#ffb74d; border:1px solid rgba(255,183,77,0.35);}}
        .tag-bull{{background:rgba(0,229,255,0.12);  color:#00e5ff; border:1px solid rgba(0,229,255,0.35);}}
        .tag-bear{{background:rgba(255,82,82,0.12);  color:#ff5252; border:1px solid rgba(255,82,82,0.35);}}
        .direction-box{{
            padding:28px; border-radius:14px; text-align:center; margin:20px 0;
            border:2px solid #4fc3f7;
            background:rgba(79,195,247,0.06);
            backdrop-filter:blur(10px);
        }}
        .direction-box.bullish {{background:linear-gradient(135deg,#00bcd4,#26c6da);border-color:#00bcd4;box-shadow:0 0 30px rgba(0,188,212,0.35);}}
        .direction-box.bearish {{background:linear-gradient(135deg,#d32f2f,#f44336);border-color:#f44336;box-shadow:0 0 30px rgba(244,67,54,0.35);}}
        .direction-box.sideways{{background:linear-gradient(135deg,#ffa726,#ffb74d);border-color:#ffb74d;box-shadow:0 0 30px rgba(255,183,77,0.35);}}
        .direction-box.neutral {{background:rgba(79,195,247,0.08);border-color:#4fc3f7;}}
        .direction-title{{font-family:'Oxanium',sans-serif;font-size:28px;font-weight:800;margin-bottom:8px;color:#fff;}}
        .direction-box.bullish .direction-title,
        .direction-box.bearish .direction-title,
        .direction-box.sideways .direction-title{{color:#000;}}
        .direction-subtitle{{font-size:13px;opacity:0.9;color:#e0f7fa;font-weight:600;}}
        .direction-box.bullish .direction-subtitle,
        .direction-box.bearish .direction-subtitle,
        .direction-box.sideways .direction-subtitle{{color:#000;}}
        .logic-box{{
            background:rgba(79,195,247,0.07);
            backdrop-filter:blur(8px);
            padding:16px 18px; border-radius:10px; margin-top:18px;
            border-left:4px solid #4fc3f7;
        }}
        .logic-box p{{font-size:13px;line-height:1.9;color:#80deea;}}
        .logic-box strong{{color:#4fc3f7;}}
        .rl-node-a{{position:absolute;bottom:0;transform:translateX(-50%);text-align:center;}}
        .rl-node-b{{position:absolute;top:0;transform:translateX(-50%);text-align:center;}}
        .rl-dot{{width:12px;height:12px;border-radius:50%;border:2px solid rgba(10,20,35,0.9);}}
        .rl-lbl{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;line-height:1.4;white-space:nowrap;color:#b0bec5;}}
        .rl-val{{font-size:13px;font-weight:700;color:#fff;white-space:nowrap;margin-top:2px;}}
        /* ── SCOREBOARD ── */
        .sb-wrap{{background:rgba(6,13,20,0.85);border:1px solid rgba(79,195,247,0.15);border-radius:12px;overflow:hidden;margin-bottom:14px;box-shadow:0 4px 24px rgba(0,0,0,0.3);}}
        .sb-header{{padding:11px 18px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid rgba(79,195,247,0.1);}}
        .sb-header.cyan{{background:rgba(79,195,247,0.08);border-left:4px solid #4fc3f7;}}
        .sb-header.gold{{background:rgba(255,183,77,0.08);border-left:4px solid #ffb74d;}}
        .sb-h-title{{font-family:'Oxanium',sans-serif;font-size:13px;font-weight:700;color:#4fc3f7;letter-spacing:1px;}}
        .sb-h-title.gold{{color:#ffb74d;}}
        .sb-h-badge{{font-size:9px;padding:3px 12px;border-radius:10px;font-weight:700;letter-spacing:1px;}}
        .sb-h-badge.cyan{{background:rgba(79,195,247,0.12);color:#4fc3f7;border:1px solid rgba(79,195,247,0.3);}}
        .sb-h-badge.gold{{background:rgba(255,183,77,0.12);color:#ffb74d;border:1px solid rgba(255,183,77,0.3);}}
        .sb-body{{display:grid;grid-template-columns:repeat(5,1fr);}}
        .sb-cell{{padding:12px 10px;text-align:center;border-right:1px solid rgba(79,195,247,0.06);}}
        .sb-cell:last-child{{border-right:none;}}
        .sb-lbl{{font-size:9px;letter-spacing:1.5px;color:#80a0b8;text-transform:uppercase;margin-bottom:5px;font-weight:600;}}
        .sb-val{{font-family:'Oxanium',sans-serif;font-size:14px;font-weight:700;color:#e0f7fa;}}
        .sb-val.green{{color:#00e676;}} .sb-val.cyan{{color:#00bcd4;}} .sb-val.gold{{color:#ffb74d;}} .sb-val.red{{color:#ff5252;}} .sb-val.small{{font-size:11px;}}
        .sb-signal{{padding:10px 16px;background:rgba(79,195,247,0.05);border-top:1px solid rgba(79,195,247,0.06);border-bottom:1px solid rgba(79,195,247,0.06);font-size:14px;color:#ffffff;font-weight:500;display:flex;align-items:center;gap:10px;}}
        .sb-signal .sig-lbl{{font-size:10px;letter-spacing:2px;color:#4fc3f7;text-transform:uppercase;flex-shrink:0;font-weight:700;}}
        .sb-footer{{padding:11px 16px;background:rgba(0,0,0,0.25);font-family:'JetBrains Mono',monospace;font-size:13px;color:#ffffff;font-weight:500;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;}}
        .sb-footer .sf-lbl{{font-size:10px;letter-spacing:2px;color:#4fc3f7;text-transform:uppercase;flex-shrink:0;font-family:'Rajdhani',sans-serif;font-weight:700;}}
        .sb-footer .sf-why{{font-size:13px;color:#ffffff;font-style:italic;font-family:'Rajdhani',sans-serif;font-weight:500;margin-left:auto;}}
        /* ── FIRE ROW ── */
        .fire-row{{background:rgba(12,6,0,0.75);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border:1px solid rgba(255,140,0,0.25);border-left:4px solid #ff8c00;border-radius:10px;margin-top:18px;padding:18px 22px;display:grid;grid-template-columns:1.4fr 1.6fr 1.6fr 1.8fr;gap:0;align-items:center;box-shadow:0 0 28px rgba(255,140,0,0.08),inset 0 1px 0 rgba(255,140,0,0.08);transition:box-shadow 0.3s ease;}}
        .fire-row:hover{{box-shadow:0 0 40px rgba(255,140,0,0.15),inset 0 1px 0 rgba(255,140,0,0.12);}}
        .fire-col{{padding:4px 18px;border-right:1px solid rgba(255,140,0,0.12);}}
        .fire-col:first-child{{padding-left:6px;}}
        .fire-col:last-child{{border-right:none;}}
        .fire-heading{{display:flex;align-items:center;gap:8px;font-family:'Oxanium',sans-serif;font-size:11px;font-weight:800;color:#ff8c00;letter-spacing:2.5px;text-transform:uppercase;line-height:1.4;}}
        .fire-col-label{{font-size:9px;letter-spacing:2px;color:rgba(255,140,0,0.5);text-transform:uppercase;font-weight:700;margin-bottom:7px;font-family:'Rajdhani',sans-serif;}}
        .fire-col-value{{font-family:'Oxanium',sans-serif;font-size:20px;font-weight:800;color:#ff8c00;letter-spacing:0.5px;line-height:1.2;}}
        .fire-col-value.stop-text{{font-size:13px;font-weight:700;color:#ff8c00;line-height:1.5;opacity:0.85;}}
        .fire-col-value.stop-price{{color:#f44336;font-size:20px;}}
        .fire-rr{{margin-top:6px;font-size:10px;color:rgba(255,140,0,0.45);font-family:'JetBrains Mono',monospace;letter-spacing:1px;}}
        /* ── STRIKES TABLE ── */
        .strikes-table{{width:100%;border-collapse:collapse;margin-top:18px;border-radius:10px;overflow:hidden;}}
        .strikes-table th{{background:linear-gradient(135deg,#4fc3f7,#26c6da);color:#000;padding:14px;text-align:left;font-weight:700;font-size:12px;text-transform:uppercase;}}
        .strikes-table td{{padding:14px;border-bottom:1px solid rgba(79,195,247,0.08);color:#b0bec5;font-size:13px;}}
        .strikes-table tr:hover{{background:rgba(79,195,247,0.06);}}
        .strikes-table tbody tr:last-child td{{border-bottom:none;}}
        /* ── SNAP GRID ── */
        .snap-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}}
        .snap-card{{padding:18px 16px;}}
        .snap-card .card-top-row{{margin-bottom:8px;padding:0;}}
        .snap-card .val{{font-size:26px;padding:0;margin-bottom:0;}}
        .disclaimer{{background:rgba(255,183,77,0.1);backdrop-filter:blur(8px);padding:22px;border-radius:12px;border-left:4px solid #ffb74d;font-size:13px;color:#ffb74d;line-height:1.8;}}
        .footer{{text-align:center;padding:24px;color:#546e7a;font-size:12px;background:rgba(10,20,28,0.4);}}

        /* ══════════════════════════════════════════
           NAVY COMMAND — OI SECTION STYLES
        ══════════════════════════════════════════ */
        .nc-section-header{{
            display:flex; align-items:center; justify-content:space-between;
            margin-bottom:18px; padding-bottom:14px;
            border-bottom:1px solid rgba(79,195,247,0.14);
        }}
        .nc-header-left{{display:flex;align-items:center;gap:14px;}}
        .nc-header-icon{{
            width:44px;height:44px;border-radius:10px;
            background:linear-gradient(135deg,#1e3a5f,#1a3052);
            border:1px solid rgba(79,195,247,0.3);
            display:flex;align-items:center;justify-content:center;
            font-size:20px;flex-shrink:0;
            box-shadow:0 4px 14px rgba(79,195,247,0.15);
        }}
        .nc-header-title{{
            font-family:'Outfit',sans-serif;font-size:17px;font-weight:700;
            color:#93c5fd;letter-spacing:0.3px;
        }}
        .nc-header-sub{{
            font-family:'Outfit',sans-serif;font-size:11px;font-weight:400;
            color:rgba(79,195,247,0.45);margin-top:2px;letter-spacing:0.5px;
        }}
        .nc-atm-badge{{
            background:#1f2a42;color:#60a5fa;
            font-family:'Outfit',sans-serif;font-size:12px;font-weight:700;
            padding:6px 16px;border-radius:20px;letter-spacing:1.5px;
            border:1px solid rgba(96,165,250,0.25);
            box-shadow:0 2px 10px rgba(96,165,250,0.1);
        }}
        /* Direction box */
        .nc-dir-box{{
            border-radius:14px;padding:20px 22px;margin-bottom:18px;
            box-shadow:0 4px 24px rgba(0,0,0,0.3);
            transition:box-shadow 0.3s;
        }}
        .nc-dir-box:hover{{box-shadow:0 8px 36px rgba(0,0,0,0.4);}}
        .nc-dir-bar{{
            width:4px;border-radius:2px;flex-shrink:0;min-height:60px;
        }}
        .nc-dir-tag{{
            font-family:'Outfit',sans-serif;font-size:9px;font-weight:700;
            letter-spacing:2.5px;color:rgba(148,163,184,0.5);
            text-transform:uppercase;margin-bottom:6px;
        }}
        .nc-dir-name{{
            font-family:'Outfit',sans-serif;font-size:28px;font-weight:700;
            line-height:1;margin-bottom:6px;letter-spacing:-0.5px;
        }}
        .nc-dir-signal{{
            font-family:'Outfit',sans-serif;font-size:12px;font-weight:400;
        }}
        /* Strength meter */
        .nc-meter-wrap{{
            text-align:right;min-width:160px;
        }}
        .nc-meter-label{{
            font-family:'Outfit',sans-serif;font-size:9px;font-weight:700;
            letter-spacing:2px;color:rgba(148,163,184,0.45);
            text-transform:uppercase;margin-bottom:8px;
        }}
        .nc-meter-track{{
            position:relative;height:8px;background:rgba(0,0,0,0.4);
            border-radius:4px;overflow:visible;width:180px;margin-left:auto;
        }}
        .nc-meter-fill{{
            height:100%;border-radius:4px;
            transition:width 1.2s cubic-bezier(0.4,0,0.2,1);
        }}
        .nc-meter-head{{
            position:absolute;top:50%;transform:translateY(-50%);
            width:14px;height:14px;border-radius:50%;
            margin-left:-7px;
            border:2px solid rgba(10,18,30,0.8);
            transition:left 1.2s cubic-bezier(0.4,0,0.2,1);
        }}
        .nc-meter-pct{{
            font-family:'Outfit',sans-serif;font-size:15px;font-weight:700;
            margin-top:8px;letter-spacing:0.5px;
        }}
        /* 3 OI cards */
        .nc-cards-grid{{
            display:grid;grid-template-columns:repeat(3,1fr);gap:14px;
        }}
        .nc-card{{
            background:rgba(20,28,45,0.85);
            border:1px solid rgba(79,195,247,0.12);
            border-radius:14px;padding:18px 18px 14px;
            transition:all 0.3s ease;
            position:relative;overflow:hidden;
        }}
        .nc-card::before{{
            content:'';position:absolute;top:0;left:0;right:0;height:1px;
            background:linear-gradient(90deg,transparent,rgba(255,255,255,0.1),transparent);
        }}
        .nc-card:hover{{
            border-color:rgba(79,195,247,0.3);
            background:rgba(25,35,55,0.9);
            transform:translateY(-3px);
            box-shadow:0 10px 30px rgba(0,0,0,0.3);
        }}
        .nc-card-header{{
            display:flex;align-items:center;justify-content:space-between;
            margin-bottom:12px;
        }}
        .nc-card-label{{
            font-family:'Outfit',sans-serif;font-size:10px;font-weight:700;
            letter-spacing:2px;color:rgba(148,163,184,0.6);text-transform:uppercase;
        }}
        .nc-card-value{{
            font-family:'Oxanium',sans-serif;font-size:30px;font-weight:700;
            line-height:1;margin-bottom:6px;letter-spacing:-0.5px;
        }}
        .nc-card-sub{{
            font-family:'JetBrains Mono',monospace;font-size:10px;
            color:rgba(100,116,139,0.7);margin-bottom:14px;
        }}
        .nc-card-btn{{
            display:block;width:100%;padding:9px 14px;border-radius:7px;
            text-align:center;font-family:'Outfit',sans-serif;
            font-size:13px;font-weight:700;letter-spacing:0.5px;
            transition:all 0.2s;cursor:default;
        }}
        .nc-card:hover .nc-card-btn{{
            filter:brightness(1.1);
        }}

        /* ══════════════════════════════════════════
           RESPONSIVE — TABLET (≤ 1024px)
        ══════════════════════════════════════════ */
        @media(max-width:1024px){{
            body{{padding:16px;}}
            .container{{border-radius:16px;}}
            .header{{padding:28px 20px;}}
            .header h1{{font-size:24px;letter-spacing:1.5px;}}
            .section{{padding:22px 18px;}}
            .grid-5{{grid-template-columns:repeat(3,1fr);}}
            .grid-4{{grid-template-columns:repeat(2,1fr);}}
            .snap-grid{{grid-template-columns:repeat(3,1fr);}}
            .fii-kpi-grid{{grid-template-columns:1fr 1fr !important;}}
            .sb-body{{grid-template-columns:repeat(3,1fr);}}
            .sb-cell:nth-child(4),.sb-cell:nth-child(5){{border-top:1px solid rgba(79,195,247,0.06);}}
            .fire-row{{grid-template-columns:1fr 1fr;gap:14px;padding:16px;}}
            .fire-col{{border-right:none;padding:8px 12px;}}
            .fire-col:nth-child(odd){{border-right:1px solid rgba(255,140,0,0.12);}}
            .fire-col-value{{font-size:16px;}}
            .rl-lbl{{font-size:9px;}}
            .rl-val{{font-size:11px;}}
            .strikes-wrap{{grid-template-columns:1fr !important;}}
            .direction-title{{font-size:22px;}}
            .val{{font-size:20px;}}
            /* NC responsive */
            .nc-cards-grid{{grid-template-columns:repeat(3,1fr);}}
            .nc-dir-name{{font-size:22px;}}
            .nc-meter-track{{width:140px;}}
            .nc-card-value{{font-size:24px;}}
        }}

        /* ══════════════════════════════════════════
           RESPONSIVE — MOBILE (≤ 600px)
        ══════════════════════════════════════════ */
        @media(max-width:600px){{
            body{{padding:8px;}}
            .container{{border-radius:12px;}}
            .header{{padding:20px 14px;}}
            .header h1{{font-size:18px;letter-spacing:1px;}}
            .header p{{font-size:11px;}}
            .section{{padding:16px 12px;}}
            .section-title{{font-size:11px;letter-spacing:1.5px;gap:6px;}}
            .section-title span{{font-size:15px;}}
            .grid-5,.grid-4{{grid-template-columns:1fr;}}
            .snap-grid{{grid-template-columns:1fr;}}
            div[style*="grid-template-columns:auto 1fr 1fr 1fr"]{{grid-template-columns:1fr 1fr !important;}}
            .val{{font-size:20px;}}
            .card-ico{{font-size:18px;}}
            .sb-body{{grid-template-columns:1fr 1fr;}}
            .sb-cell{{border-right:none !important;border-bottom:1px solid rgba(79,195,247,0.06);}}
            .sb-cell:last-child{{border-bottom:none;}}
            .sb-h-title{{font-size:11px;letter-spacing:0.5px;}}
            .sb-h-badge{{font-size:8px;padding:2px 8px;}}
            .sb-val{{font-size:13px;}}
            .sb-lbl{{font-size:8px;}}
            .sb-footer{{font-size:11px;flex-direction:column;gap:6px;}}
            .sb-footer .sf-why{{margin-left:0;}}
            .sb-signal{{font-size:12px;flex-wrap:wrap;gap:6px;}}
            .fire-row{{grid-template-columns:1fr;gap:12px;padding:14px 12px;}}
            .fire-col{{border-right:none !important;border-bottom:1px solid rgba(255,140,0,0.08);padding:8px 4px;}}
            .fire-col:last-child{{border-bottom:none;}}
            .fire-col-value{{font-size:15px;}}
            .fire-heading{{font-size:10px;}}
            .direction-box{{padding:18px 14px;}}
            .direction-title{{font-size:20px;}}
            .direction-subtitle{{font-size:11px;}}
            .logic-box p{{font-size:11px;line-height:1.7;}}
            .rl-node-a,.rl-node-b{{font-size:9px;}}
            .rl-val{{font-size:10px;}}
            .rl-lbl{{font-size:8px;letter-spacing:0;}}
            div[style*="grid-template-columns:1fr 1fr"]{{grid-template-columns:1fr !important;}}
            div[style*="font-size:22px"]{{font-size:18px !important;}}
            .strikes-wrap{{grid-template-columns:1fr !important;}}
            .strikes-table th,.strikes-table td{{padding:10px 8px;font-size:11px;}}
            .disclaimer{{padding:14px;font-size:12px;}}
            /* NC mobile */
            .nc-section-header{{flex-direction:column;align-items:flex-start;gap:10px;}}
            .nc-atm-badge{{align-self:flex-end;}}
            .nc-cards-grid{{grid-template-columns:1fr;}}
            .nc-dir-name{{font-size:20px;}}
            .nc-meter-wrap{{text-align:left;min-width:unset;}}
            .nc-meter-track{{width:100%;margin-left:0;}}
            .nc-card-value{{font-size:26px;}}
            .nc-meter-pct{{font-size:13px;}}
            .nc-header-title{{font-size:15px;}}
        }}

        /* ══════════════════════════════════════════
           RESPONSIVE — SMALL MOBILE (≤ 380px)
        ══════════════════════════════════════════ */
        @media(max-width:380px){{
            .header h1{{font-size:15px;}}
            .sb-body{{grid-template-columns:1fr;}}
            .section{{padding:12px 10px;}}
            .val{{font-size:17px;}}
            .sb-val{{font-size:12px;}}
            .fire-col-value{{font-size:13px;}}
            .direction-title{{font-size:17px;}}
            .nc-dir-name{{font-size:17px;}}
            .nc-card-value{{font-size:22px;}}
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>📊 NIFTY 50 DAILY REPORT</h1>
        <p>Generated: {d['timestamp']}</p>
    </div>

    <!-- ── MARKET SNAPSHOT ── -->
    <div class="section">
        <div class="section-title"><span>📈</span> MARKET SNAPSHOT</div>
        <div class="snap-grid">
            <div class="g snap-card g-hi">
                <div class="card-top-row"><span class="card-ico">💹</span><div class="lbl">NIFTY 50 SPOT</div></div>
                <span class="val">₹{d['current_price']:,.2f}</span>
            </div>
            <div class="g snap-card">
                <div class="card-top-row"><span class="card-ico">🎯</span><div class="lbl">ATM STRIKE</div></div>
                <span class="val">₹{d['atm_strike']:,}</span>
            </div>
            <div class="g snap-card">
                <div class="card-top-row"><span class="card-ico">📅</span><div class="lbl">EXPIRY DATE</div></div>
                <span class="val" style="font-size:20px">{d['expiry']}</span>
            </div>
        </div>
    </div>
"""
        # ── CHANGE IN OI — NAVY COMMAND (position 2: right after Market Snapshot)
        if d['has_option_data']:
            html += self._oi_navy_command_section(d)

        html += f"""
    <!-- ── KEY TRADING LEVELS ── -->
    <div class="section">
        <div class="section-title"><span>📍</span> KEY TRADING LEVELS</div>
        {self._key_levels_fire_row(d)}
    </div>

    <!-- ── FII / DII 5-DAY SENTIMENT ── -->
    {self._fiidii_section_html()}

    <!-- ── MARKET DIRECTION ── -->
    <div class="section">
        <div class="section-title"><span>🧭</span> MARKET DIRECTION (Algorithmic)</div>
        <div class="direction-box {d['bias_class']}">
            <div class="direction-title">{d['bias_icon']} {d['bias']}</div>
            <div class="direction-subtitle">Confidence: {d['confidence']} &nbsp;|&nbsp; Bullish: {d['bullish_score']} vs Bearish: {d['bearish_score']}</div>
        </div>
        <div class="logic-box">
            <p>
                <strong>📊 Scoring Logic:</strong><br>
                • <strong>BULLISH</strong>: Diff ≥ +3 · Price above SMAs, oversold RSI, +MACD, PCR &gt; 1.2<br>
                • <strong>BEARISH</strong>: Diff ≤ −3 · Price below SMAs, overbought RSI, −MACD, PCR &lt; 0.7<br>
                • <strong>SIDEWAYS</strong>: Diff −2 to +2 · Mixed signals, consolidation<br>
                • <strong>Confidence</strong>: HIGH when gap ≥ 4 | OI scope: ATM ±10 strikes only
            </p>
        </div>
    </div>

    <!-- ── TECHNICAL INDICATORS ── -->
    <div class="section">
        <div class="section-title"><span>🔍</span> TECHNICAL INDICATORS</div>
        <div class="card-grid grid-5">
            {tech_cards}
        </div>
    </div>
"""
        # ── OPTION CHAIN ANALYSIS
        if d['has_option_data']:
            html += f"""
    <div class="section">
        <div class="section-title">
            <span>🎯</span> OPTION CHAIN ANALYSIS
            <span style="font-size:11px;color:#80deea;font-weight:400;letter-spacing:1px;">(ATM ±10 Strikes Only)</span>
        </div>
        <div class="card-grid grid-4">
            {oc_cards}
        </div>
    </div>
"""

        # ── KEY LEVELS
        html += f"""
    <div class="section">
        <div class="section-title"><span>📊</span> KEY LEVELS</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-size:11px;color:#26c6da;font-weight:700;letter-spacing:1px;">◄ SUPPORT ZONE</span>
            <span style="font-size:11px;color:#f44336;font-weight:700;letter-spacing:1px;">RESISTANCE ZONE ►</span>
        </div>
        <div style="position:relative;height:62px;">
            <div class="rl-node-a" style="left:3%;">
                <div class="rl-lbl" style="color:#26c6da;">Strong<br>Support</div>
                <div class="rl-val" style="color:#26c6da;">₹{d['strong_support']:,.0f}</div>
                <div class="rl-dot" style="background:#26c6da;margin:6px auto 0;"></div>
            </div>
            <div class="rl-node-a" style="left:22%;">
                <div class="rl-lbl" style="color:#00bcd4;">Support</div>
                <div class="rl-val" style="color:#00bcd4;">₹{d['support']:,.0f}</div>
                <div class="rl-dot" style="background:#00bcd4;box-shadow:0 0 8px #00bcd4;margin:6px auto 0;"></div>
            </div>
            <div style="position:absolute;left:{_pct_cp}%;transform:translateX(-50%);
                        bottom:4px;background:#4fc3f7;color:#000;font-size:11px;font-weight:700;
                        padding:4px 13px;border-radius:6px;white-space:nowrap;z-index:10;
                        box-shadow:0 0 16px rgba(79,195,247,0.7);">
                ▼ NOW &nbsp;₹{d['current_price']:,.0f}
            </div>
            <div class="rl-node-a" style="left:75%;">
                <div class="rl-lbl" style="color:#ff7043;">Resistance</div>
                <div class="rl-val" style="color:#ff7043;">₹{d['resistance']:,.0f}</div>
                <div class="rl-dot" style="background:#ff7043;box-shadow:0 0 8px #ff7043;margin:6px auto 0;"></div>
            </div>
            <div class="rl-node-a" style="left:95%;">
                <div class="rl-lbl" style="color:#f44336;">Strong<br>Resistance</div>
                <div class="rl-val" style="color:#f44336;">₹{d['strong_resistance']:,.0f}</div>
                <div class="rl-dot" style="background:#f44336;margin:6px auto 0;"></div>
            </div>
        </div>
        <div style="position:relative;height:8px;border-radius:4px;
                    background:linear-gradient(90deg,#26c6da 0%,#00bcd4 20%,#4fc3f7 40%,#ffb74d 58%,#ff7043 76%,#f44336 100%);
                    box-shadow:0 2px 14px rgba(0,0,0,0.5);">
            <div style="position:absolute;left:{_pct_cp}%;top:50%;transform:translate(-50%,-50%);
                        width:4px;height:22px;background:#fff;border-radius:2px;
                        box-shadow:0 0 16px rgba(255,255,255,1);z-index:10;"></div>
        </div>
        <div style="position:relative;height:58px;">{_mp_node}</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px;">
            <div style="background:rgba(244,67,54,0.08);border:1px solid rgba(244,67,54,0.25);
                        border-radius:8px;padding:10px 16px;display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:12px;color:#b0bec5;">📍 Distance to Resistance</span>
                <span style="font-size:15px;font-weight:700;color:#f44336;">+{_pts_to_res:,} pts</span>
            </div>
            <div style="background:rgba(0,188,212,0.08);border:1px solid rgba(0,188,212,0.25);
                        border-radius:8px;padding:10px 16px;display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:12px;color:#b0bec5;">📍 Distance to Support</span>
                <span style="font-size:15px;font-weight:700;color:#00bcd4;">−{_pts_to_sup:,} pts</span>
            </div>
        </div>
    </div>
"""
        html += self._generate_recommendations_html(d)

        if d['has_option_data'] and (d['top_ce_strikes'] or d['top_pe_strikes']):
            html += """
    <div class="section">
        <div class="section-title"><span>📋</span> TOP 5 STRIKES BY OPEN INTEREST
            <span style="font-size:11px;color:#80deea;font-weight:400;letter-spacing:1px;">(ATM ±10 Only)</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;" class="strikes-wrap">
            <div>
                <h3 style="color:#00bcd4;margin-bottom:14px;font-size:15px;font-family:'Oxanium',sans-serif;">📞 CALL OPTIONS (CE)</h3>
                <table class="strikes-table">
                    <thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th></tr></thead><tbody>
"""
            for i, s in enumerate(d['top_ce_strikes'], 1):
                html += f"<tr><td><strong>{i}</strong></td><td><strong>₹{int(s['Strike']):,}</strong></td><td>{int(s['CE_OI']):,}</td><td style='color:#00bcd4;font-weight:700;font-family:Oxanium,sans-serif;'>₹{s['CE_LTP']:.2f}</td></tr>\n"
            html += """</tbody></table></div>
            <div>
                <h3 style="color:#f44336;margin-bottom:14px;font-size:15px;font-family:'Oxanium',sans-serif;">📉 PUT OPTIONS (PE)</h3>
                <table class="strikes-table">
                    <thead><tr><th>#</th><th>Strike</th><th>OI</th><th>LTP</th></tr></thead><tbody>
"""
            for i, s in enumerate(d['top_pe_strikes'], 1):
                html += f"<tr><td><strong>{i}</strong></td><td><strong>₹{int(s['Strike']):,}</strong></td><td>{int(s['PE_OI']):,}</td><td style='color:#f44336;font-weight:700;font-family:Oxanium,sans-serif;'>₹{s['PE_LTP']:.2f}</td></tr>\n"
            html += """</tbody></table></div></div></div>
"""
        html += """
    <div class="section">
        <div class="disclaimer">
            <strong>⚠️ DISCLAIMER</strong><br><br>
            This report is for <strong>EDUCATIONAL purposes only</strong> — NOT financial advice.<br>
            Always use stop losses and consult a SEBI registered investment advisor.<br>
            Past performance does not guarantee future results.
        </div>
    </div>
    <div class="footer">
        <p>Automated Nifty 50 Option Chain + Technical Analysis Report</p>
        <p style="margin-top:6px;">© 2026 · Glassmorphism UI · Deep Ocean Theme · Navy Command OI · For Educational Purposes Only</p>
    </div>
</div>
</body>
</html>
"""
        return html

    # ─────────────────────────────────────────────
    #  DUAL RECOMMENDATIONS — SCOREBOARD STYLE
    # ─────────────────────────────────────────────
    def _generate_recommendations_html(self, d):
        ts = d['recommended_technical_strategy']

        def bias_color(bias):
            b = bias.lower()
            if 'bull' in b: return 'green'
            if 'bear' in b: return 'red'
            if 'vol'  in b: return 'gold'
            return 'gold'

        def risk_color(risk):
            r = risk.lower()
            if 'low'  in r: return 'cyan'
            if 'high' in r: return 'red'
            if 'very' in r: return 'red'
            return 'gold'

        def profit_color(p):
            p = p.lower()
            if 'unlimited' in p or 'receiv' in p: return 'green'
            return 'cyan'

        def loss_color(l):
            l = l.lower()
            if 'unlimit' in l: return 'red'
            return 'gold'

        ts_bias_cls   = bias_color(ts['market_bias'])
        ts_risk_cls   = risk_color(ts['risk'])
        ts_profit_cls = profit_color(ts['max_profit'])
        ts_loss_cls   = loss_color(ts['max_loss'])

        html = """
    <div class="section">
        <div class="section-title"><span>💡</span> TRADING RECOMMENDATIONS (Dual Strategy)</div>
        <p style="color:#90a4ae;font-size:13px;margin-bottom:16px;letter-spacing:0.5px;">
            Two independent strategy recommendations based on
            <strong style="color:#4fc3f7;">Technical Analysis</strong> and
            <strong style="color:#ffb74d;">OI Momentum</strong>.
        </p>
"""
        html += f"""
        <div class="sb-wrap">
            <div class="sb-header cyan">
                <span class="sb-h-title">1️⃣ TECHNICAL ANALYSIS STRATEGY &nbsp;·&nbsp; {ts['name']}</span>
                <span class="sb-h-badge cyan">Positional · 1–5 Days</span>
            </div>
            <div class="sb-body">
                <div class="sb-cell"><div class="sb-lbl">Market Bias</div><div class="sb-val {ts_bias_cls}">{ts['market_bias']}</div></div>
                <div class="sb-cell"><div class="sb-lbl">Type</div><div class="sb-val cyan">{ts['type']}</div></div>
                <div class="sb-cell"><div class="sb-lbl">Risk</div><div class="sb-val {ts_risk_cls}">{ts['risk']}</div></div>
                <div class="sb-cell"><div class="sb-lbl">Max Profit</div><div class="sb-val {ts_profit_cls} small">{ts['max_profit']}</div></div>
                <div class="sb-cell"><div class="sb-lbl">Max Loss</div><div class="sb-val {ts_loss_cls} small">{ts['max_loss']}</div></div>
            </div>
            <div class="sb-footer">
                <span class="sf-lbl">📋 Setup</span>
                <span>{ts['description']}</span>
                <span class="sf-why">💡 {ts['best_for']}</span>
            </div>
        </div>
"""
        if d['recommended_oi_strategy']:
            oi = d['recommended_oi_strategy']
            oi_bias_cls   = bias_color(oi['market_bias'])
            oi_risk_cls   = risk_color(oi['risk'])
            oi_profit_cls = profit_color(oi['max_profit'])
            oi_loss_cls   = loss_color(oi['max_loss'])
            signal_txt    = oi.get('signal', 'Market signal detected')
            horizon       = oi.get('time_horizon', 'Intraday')
            html += f"""
        <div class="sb-wrap">
            <div class="sb-header gold">
                <span class="sb-h-title gold">2️⃣ OI MOMENTUM STRATEGY &nbsp;·&nbsp; {oi['name']}</span>
                <span class="sb-h-badge gold">{horizon}</span>
            </div>
            <div class="sb-signal">
                <span class="sig-lbl">📊 OI Signal</span>
                <span style="color:#ffffff;font-weight:500;">{signal_txt}</span>
            </div>
            <div class="sb-body">
                <div class="sb-cell"><div class="sb-lbl">Market Bias</div><div class="sb-val {oi_bias_cls}">{oi['market_bias']}</div></div>
                <div class="sb-cell"><div class="sb-lbl">Type</div><div class="sb-val cyan">{oi['type']}</div></div>
                <div class="sb-cell"><div class="sb-lbl">Risk</div><div class="sb-val {oi_risk_cls}">{oi['risk']}</div></div>
                <div class="sb-cell"><div class="sb-lbl">Max Profit</div><div class="sb-val {oi_profit_cls} small">{oi['max_profit']}</div></div>
                <div class="sb-cell"><div class="sb-lbl">Max Loss</div><div class="sb-val {oi_loss_cls} small">{oi['max_loss']}</div></div>
            </div>
            <div class="sb-footer" style="border-top-color:rgba(255,183,77,0.08);">
                <span class="sf-lbl">📋 Setup</span>
                <span>{oi['description']}</span>
                <span class="sf-why">💡 {oi['best_for']}</span>
            </div>
        </div>
"""
        return html

    # ─────────────────────────────────────────────
    #  SAVE & EMAIL
    # ─────────────────────────────────────────────
    def save_html_to_file(self, filename='index.html'):
        try:
            print(f"\n📄 Saving HTML to {filename}...")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.generate_html_email())
            print(f"   ✅ Saved {filename}")
            metadata = {
                'timestamp':         self.html_data['timestamp'],
                'current_price':     float(self.html_data['current_price']),
                'bias':              self.html_data['bias'],
                'confidence':        self.html_data['confidence'],
                'rsi':               float(self.html_data['rsi']),
                'pcr':               float(self.html_data['pcr']) if self.html_data['has_option_data'] else None,
                'stop_loss':         float(self.html_data['stop_loss']) if self.html_data['stop_loss'] else None,
                'risk_reward_ratio': self.html_data.get('risk_reward_ratio', 0),
            }
            with open('latest_report.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            print("   ✅ Saved latest_report.json")
            return True
        except Exception as e:
            print(f"\n❌ Save failed: {e}")
            return False

    def send_html_email_report(self):
        gmail_user     = os.getenv('GMAIL_USER')
        gmail_password = os.getenv('GMAIL_APP_PASSWORD')
        recipient1     = os.getenv('RECIPIENT_EMAIL_1')
        recipient2     = os.getenv('RECIPIENT_EMAIL_2')
        if not all([gmail_user, gmail_password, recipient1, recipient2]):
            print("\n⚠️  Email credentials not set. Skipping.")
            return False
        try:
            ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
            msg = MIMEMultipart('alternative')
            msg['From']    = gmail_user
            msg['To']      = f"{recipient1}, {recipient2}"
            msg['Subject'] = f"📊 Nifty 50 OI & Technical Report — {ist_now.strftime('%d-%b-%Y %H:%M IST')}"
            msg.attach(MIMEText(self.generate_html_email(), 'html'))
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(gmail_user, gmail_password)
                server.send_message(msg)
            print("   ✅ Email sent!")
            return True
        except Exception as e:
            print(f"\n❌ Email failed: {e}")
            return False

    # ─────────────────────────────────────────────
    #  MAIN
    # ─────────────────────────────────────────────
    def generate_full_report(self):
        ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
        print("=" * 70)
        print("NIFTY 50 DAILY REPORT — GLASSMORPHISM UI · NAVY COMMAND OI THEME")
        print(f"Generated: {ist_now.strftime('%d-%b-%Y %H:%M IST')}")
        print("=" * 70)
        oc_data         = self.fetch_nse_option_chain_silent()
        option_analysis = self.analyze_option_chain_data(oc_data) if oc_data else None
        if option_analysis:
            print(f"✅ Option data | Expiry: {option_analysis['expiry']} | Spot: {option_analysis['underlying_value']}")
        else:
            print("⚠️  No option data — technical-only mode")
        print("\nFetching technical data...")
        technical = self.get_technical_data()
        self.generate_analysis_data(technical, option_analysis)
        return option_analysis


def main():
    try:
        print("\n🚀 Starting Nifty 50 Analysis...\n")
        analyzer = NiftyHTMLAnalyzer()
        analyzer.generate_full_report()
        print("\n" + "=" * 70)
        save_ok = analyzer.save_html_to_file('index.html')
        if save_ok:
            analyzer.send_html_email_report()
        else:
            print("\n⚠️  Skipping email due to save failure")
        print("\n✅ Done! Open index.html in your browser.\n")
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
