"""
Nifty Option Chain & Technical Analysis for Day Trading
Fetches option chain data, performs 1-hour technical analysis, and emails HTML report
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')

class NiftyAnalyzer:
    def __init__(self):
        self.nifty_symbol = "^NSEI"
        self.option_chain_url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br'
        }
        
    def fetch_option_chain(self):
        """Fetch Nifty option chain data from NSE"""
        try:
            session = requests.Session()
            # First request to get cookies
            session.get("https://www.nseindia.com", headers=self.headers, timeout=10)
            
            # Fetch option chain
            response = session.get(self.option_chain_url, headers=self.headers, timeout=10)
            data = response.json()
            
            if 'records' in data and 'data' in data['records']:
                option_data = data['records']['data']
                current_price = data['records']['underlyingValue']
                
                # Process option chain
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
                
                # Merge on strike
                oc_df = pd.merge(calls_df, puts_df, on='Strike', how='outer')
                oc_df = oc_df.fillna(0)
                oc_df = oc_df.sort_values('Strike')
                
                return oc_df, current_price
            
            return None, None
            
        except Exception as e:
            print(f"Error fetching option chain: {e}")
            return None, None
    
    def analyze_option_chain(self, oc_df, spot_price):
        """Analyze option chain for trading signals"""
        if oc_df is None or oc_df.empty:
            return {}
        
        # Calculate Put-Call Ratio
        total_call_oi = oc_df['Call_OI'].sum()
        total_put_oi = oc_df['Put_OI'].sum()
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        # Max Pain calculation
        oc_df['Call_Pain'] = oc_df.apply(
            lambda row: row['Call_OI'] * max(0, spot_price - row['Strike']), axis=1
        )
        oc_df['Put_Pain'] = oc_df.apply(
            lambda row: row['Put_OI'] * max(0, row['Strike'] - spot_price), axis=1
        )
        oc_df['Total_Pain'] = oc_df['Call_Pain'] + oc_df['Put_Pain']
        
        max_pain_strike = oc_df.loc[oc_df['Total_Pain'].idxmax(), 'Strike']
        
        # Find support and resistance (top 2 each near spot)
        nearby_strikes = oc_df[
            (oc_df['Strike'] >= spot_price - 500) & 
            (oc_df['Strike'] <= spot_price + 500)
        ].copy()
        
        # Resistance (Call OI above spot)
        resistance_df = nearby_strikes[nearby_strikes['Strike'] > spot_price].nlargest(2, 'Call_OI')
        resistances = resistance_df['Strike'].tolist()
        
        # Support (Put OI below spot)
        support_df = nearby_strikes[nearby_strikes['Strike'] < spot_price].nlargest(2, 'Put_OI')
        supports = support_df['Strike'].tolist()
        
        # Change in OI analysis
        total_call_buildup = oc_df['Call_Chng_OI'].sum()
        total_put_buildup = oc_df['Put_Chng_OI'].sum()
        
        # IV analysis
        avg_call_iv = oc_df['Call_IV'].mean()
        avg_put_iv = oc_df['Put_IV'].mean()
        
        return {
            'pcr': round(pcr, 2),
            'max_pain': max_pain_strike,
            'resistances': sorted(resistances, reverse=True),
            'supports': sorted(supports, reverse=True),
            'call_buildup': total_call_buildup,
            'put_buildup': total_put_buildup,
            'avg_call_iv': round(avg_call_iv, 2),
            'avg_put_iv': round(avg_put_iv, 2),
            'oi_sentiment': 'Bullish' if total_put_buildup > total_call_buildup else 'Bearish'
        }
    
    def fetch_technical_data(self, period='6mo', interval='1h'):
        """Fetch historical data for technical analysis"""
        try:
            ticker = yf.Ticker(self.nifty_symbol)
            df = ticker.history(period=period, interval=interval)
            return df
        except Exception as e:
            print(f"Error fetching technical data: {e}")
            return None
    
    def calculate_rsi(self, data, period=14):
        """Calculate RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_support_resistance(self, df, current_price):
        """Calculate nearest support and resistance levels from price action"""
        # Get recent highs and lows
        recent_data = df.tail(300)  # Last 300 hours (~6 weeks of 1hr data)
        
        # Find pivot highs and lows
        pivots_high = []
        pivots_low = []
        
        for i in range(5, len(recent_data) - 5):
            high = recent_data['High'].iloc[i]
            low = recent_data['Low'].iloc[i]
            
            # Pivot High
            if high == max(recent_data['High'].iloc[i-5:i+6]):
                pivots_high.append(high)
            
            # Pivot Low
            if low == min(recent_data['Low'].iloc[i-5:i+6]):
                pivots_low.append(low)
        
        # Get resistance levels (above current price)
        resistances = sorted([p for p in pivots_high if p > current_price])
        resistances = list(dict.fromkeys(resistances))  # Remove duplicates
        
        # Get support levels (below current price)
        supports = sorted([p for p in pivots_low if p < current_price], reverse=True)
        supports = list(dict.fromkeys(supports))  # Remove duplicates
        
        return {
            'resistances': resistances[:2] if len(resistances) >= 2 else resistances,
            'supports': supports[:2] if len(supports) >= 2 else supports
        }
    
    def technical_analysis(self, df):
        """Perform complete technical analysis"""
        if df is None or df.empty:
            return {}
        
        current_price = df['Close'].iloc[-1]
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        current_rsi = df['RSI'].iloc[-1]
        
        # Moving Averages
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        ema20 = df['EMA20'].iloc[-1]
        ema50 = df['EMA50'].iloc[-1]
        
        # Support and Resistance
        sr_levels = self.calculate_support_resistance(df, current_price)
        
        # Trend analysis
        if current_price > ema20 > ema50:
            trend = "Strong Uptrend"
        elif current_price > ema20:
            trend = "Uptrend"
        elif current_price < ema20 < ema50:
            trend = "Strong Downtrend"
        elif current_price < ema20:
            trend = "Downtrend"
        else:
            trend = "Sideways"
        
        # RSI signal
        if current_rsi > 70:
            rsi_signal = "Overbought - Bearish"
        elif current_rsi < 30:
            rsi_signal = "Oversold - Bullish"
        elif current_rsi > 50:
            rsi_signal = "Bullish"
        else:
            rsi_signal = "Bearish"
        
        return {
            'current_price': round(current_price, 2),
            'rsi': round(current_rsi, 2),
            'rsi_signal': rsi_signal,
            'ema20': round(ema20, 2),
            'ema50': round(ema50, 2),
            'trend': trend,
            'tech_resistances': [round(r, 2) for r in sr_levels['resistances']],
            'tech_supports': [round(s, 2) for s in sr_levels['supports']]
        }
    
    def generate_recommendation(self, oc_analysis, tech_analysis):
        """Generate trading recommendation"""
        if not oc_analysis or not tech_analysis:
            return "Insufficient data for recommendation"
        
        bullish_signals = 0
        bearish_signals = 0
        reasons = []
        
        # PCR Analysis
        pcr = oc_analysis.get('pcr', 0)
        if pcr > 1.2:
            bullish_signals += 2
            reasons.append(f"PCR at {pcr} indicates strong bullish sentiment")
        elif pcr > 1.0:
            bullish_signals += 1
            reasons.append(f"PCR at {pcr} shows bullish bias")
        elif pcr < 0.8:
            bearish_signals += 2
            reasons.append(f"PCR at {pcr} indicates bearish sentiment")
        elif pcr < 1.0:
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
        if rsi < 30:
            bullish_signals += 2
            reasons.append(f"RSI at {rsi:.1f} - Oversold (Bullish)")
        elif rsi < 45:
            bullish_signals += 1
            reasons.append(f"RSI at {rsi:.1f} - Below neutral (Bullish)")
        elif rsi > 70:
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
        
        if signal_diff >= 3:
            recommendation = "STRONG BUY"
            bias = "Bullish"
            confidence = "High"
        elif signal_diff >= 1:
            recommendation = "BUY"
            bias = "Bullish"
            confidence = "Medium"
        elif signal_diff <= -3:
            recommendation = "STRONG SELL"
            bias = "Bearish"
            confidence = "High"
        elif signal_diff <= -1:
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
        """Create beautiful HTML report"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine colors
        if 'BUY' in recommendation['recommendation']:
            rec_color = '#28a745'
        elif 'SELL' in recommendation['recommendation']:
            rec_color = '#dc3545'
        else:
            rec_color = '#ffc107'
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #f5f5f5;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 900px;
                    margin: 0 auto;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    padding: 30px;
                }}
                .header {{
                    text-align: center;
                    border-bottom: 3px solid #007bff;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    color: #007bff;
                    margin: 0;
                    font-size: 32px;
                }}
                .timestamp {{
                    color: #6c757d;
                    font-size: 14px;
                    margin-top: 10px;
                }}
                .recommendation-box {{
                    background: linear-gradient(135deg, {rec_color} 0%, {rec_color}dd 100%);
                    color: white;
                    padding: 25px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                }}
                .recommendation-box h2 {{
                    margin: 0 0 10px 0;
                    font-size: 36px;
                    font-weight: bold;
                }}
                .recommendation-box .subtitle {{
                    font-size: 18px;
                    opacity: 0.9;
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                .section-title {{
                    background-color: #007bff;
                    color: white;
                    padding: 12px 20px;
                    border-radius: 5px;
                    font-size: 20px;
                    font-weight: bold;
                    margin-bottom: 15px;
                }}
                .data-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 15px;
                }}
                .data-item {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #007bff;
                }}
                .data-item .label {{
                    color: #6c757d;
                    font-size: 14px;
                    margin-bottom: 5px;
                }}
                .data-item .value {{
                    color: #212529;
                    font-size: 20px;
                    font-weight: bold;
                }}
                .levels {{
                    display: flex;
                    justify-content: space-between;
                    gap: 20px;
                }}
                .levels-box {{
                    flex: 1;
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                }}
                .levels-box.resistance {{
                    border-left: 4px solid #dc3545;
                }}
                .levels-box.support {{
                    border-left: 4px solid #28a745;
                }}
                .levels-box h4 {{
                    margin: 0 0 10px 0;
                    font-size: 16px;
                }}
                .levels-box ul {{
                    margin: 0;
                    padding-left: 20px;
                }}
                .levels-box li {{
                    margin: 5px 0;
                    font-size: 16px;
                    font-weight: 500;
                }}
                .reasons {{
                    background-color: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 15px;
                    border-radius: 5px;
                }}
                .reasons ul {{
                    margin: 10px 0 0 0;
                    padding-left: 20px;
                }}
                .reasons li {{
                    margin: 8px 0;
                    color: #856404;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 2px solid #e9ecef;
                    color: #6c757d;
                    font-size: 12px;
                }}
                .signal-badge {{
                    display: inline-block;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-size: 14px;
                    margin: 5px;
                }}
                .bullish {{
                    background-color: #d4edda;
                    color: #155724;
                }}
                .bearish {{
                    background-color: #f8d7da;
                    color: #721c24;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 NIFTY DAY TRADING ANALYSIS</h1>
                    <div class="timestamp">Generated on: {now}</div>
                </div>
                
                <div class="recommendation-box">
                    <h2>{recommendation['recommendation']}</h2>
                    <div class="subtitle">
                        Market Bias: {recommendation['bias']} | Confidence: {recommendation['confidence']}
                    </div>
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
                    <div class="section-title">🎯 Support & Resistance Levels</div>
                    <div class="levels">
                        <div class="levels-box resistance">
                            <h4>🔴 Resistance Levels</h4>
                            <ul>
                                {''.join([f'<li>R{i+1}: ₹{r}</li>' for i, r in enumerate(tech_analysis.get('tech_resistances', []))])}
                            </ul>
                        </div>
                        <div class="levels-box support">
                            <h4>🟢 Support Levels</h4>
                            <ul>
                                {''.join([f'<li>S{i+1}: ₹{s}</li>' for i, s in enumerate(tech_analysis.get('tech_supports', []))])}
                            </ul>
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
                        <div class="data-item">
                            <div class="label">Avg Call IV</div>
                            <div class="value">{oc_analysis.get('avg_call_iv', 'N/A')}%</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <div class="levels">
                            <div class="levels-box resistance">
                                <h4>🔴 OI Resistance (Call Buildup)</h4>
                                <ul>
                                    {''.join([f'<li>₹{r}</li>' for r in oc_analysis.get('resistances', [])])}
                                </ul>
                            </div>
                            <div class="levels-box support">
                                <h4>🟢 OI Support (Put Buildup)</h4>
                                <ul>
                                    {''.join([f'<li>₹{s}</li>' for s in oc_analysis.get('supports', [])])}
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">💡 Analysis Summary</div>
                    <div class="reasons">
                        <strong>Key Factors Behind Recommendation:</strong>
                        <ul>
                            {''.join([f'<li>{reason}</li>' for reason in recommendation.get('reasons', [])])}
                        </ul>
                    </div>
                </div>
                
                <div class="footer">
                    <p><strong>Disclaimer:</strong> This analysis is for educational purposes only. Trading in derivatives involves substantial risk. 
                    Always use proper risk management and consult with a financial advisor before making trading decisions.</p>
                    <p>© 2025 Nifty Trading Analyzer | Automated Report</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def send_email(self, html_content, recipient_email, sender_email, sender_password):
        """Send email with HTML report"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"Nifty Trading Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            msg['From'] = sender_email
            msg['To'] = recipient_email
            
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Use Gmail SMTP
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Email sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            print(f"❌ Error sending email: {e}")
            return False
    
    def run_analysis(self, recipient_email=None, sender_email=None, sender_password=None):
        """Run complete analysis"""
        print("🚀 Starting Nifty Analysis...")
        print("=" * 60)
        
        # Fetch Option Chain
        print("\n📊 Fetching Option Chain Data...")
        oc_df, spot_price = self.fetch_option_chain()
        
        if oc_df is not None:
            print(f"✅ Option Chain Data Fetched | Spot Price: ₹{spot_price}")
            oc_analysis = self.analyze_option_chain(oc_df, spot_price)
        else:
            print("⚠️ Using sample option chain data")
            spot_price = 24500  # Sample
            oc_analysis = {
                'pcr': 1.15,
                'max_pain': 24500,
                'resistances': [24600, 24650],
                'supports': [24400, 24350],
                'call_buildup': 5000000,
                'put_buildup': 6000000,
                'avg_call_iv': 15.5,
                'avg_put_iv': 16.2,
                'oi_sentiment': 'Bullish'
            }
        
        # Fetch Technical Data
        print("\n📈 Fetching Technical Data (1 Hour Timeframe)...")
        tech_df = self.fetch_technical_data(period='6mo', interval='1h')
        
        if tech_df is not None and not tech_df.empty:
            print(f"✅ Technical Data Fetched | Total Bars: {len(tech_df)}")
            tech_analysis = self.technical_analysis(tech_df)
        else:
            print("⚠️ Using sample technical data")
            tech_analysis = {
                'current_price': spot_price,
                'rsi': 58.5,
                'rsi_signal': 'Bullish',
                'ema20': 24480,
                'ema50': 24450,
                'trend': 'Uptrend',
                'tech_resistances': [24580, 24650],
                'tech_supports': [24420, 24380]
            }
        
        # Generate Recommendation
        print("\n🎯 Generating Trading Recommendation...")
        recommendation = self.generate_recommendation(oc_analysis, tech_analysis)
        
        print("\n" + "=" * 60)
        print(f"📊 RECOMMENDATION: {recommendation['recommendation']}")
        print(f"📈 Bias: {recommendation['bias']} | Confidence: {recommendation['confidence']}")
        print("=" * 60)
        
        # Create HTML Report
        html_report = self.create_html_report(oc_analysis, tech_analysis, recommendation)
        
        # Save HTML file
        report_filename = f"nifty_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f"\n💾 Report saved as: {report_filename}")
        
        # Send Email if credentials provided
        if recipient_email and sender_email and sender_password:
            print(f"\n📧 Sending email to {recipient_email}...")
            self.send_email(html_report, recipient_email, sender_email, sender_password)
        
        return {
            'oc_analysis': oc_analysis,
            'tech_analysis': tech_analysis,
            'recommendation': recommendation,
            'html_report': html_report
        }


if __name__ == "__main__":
    # Configuration
    RECIPIENT_EMAIL = "your_email@gmail.com"  # Change this
    SENDER_EMAIL = "your_email@gmail.com"     # Change this
    SENDER_PASSWORD = "your_app_password"      # Use App Password for Gmail
    
    # Create analyzer instance
    analyzer = NiftyAnalyzer()
    
    # Run analysis
    # Note: For email to work, you need to:
    # 1. Enable 2-Factor Authentication in Gmail
    # 2. Generate App Password from Google Account settings
    # 3. Use the App Password instead of your regular password
    
    result = analyzer.run_analysis(
        recipient_email=RECIPIENT_EMAIL,
        sender_email=SENDER_EMAIL,
        sender_password=SENDER_PASSWORD
    )
    
    print("\n✅ Analysis Complete!")
    print(f"\nRecommendation: {result['recommendation']['recommendation']}")
    print(f"Check your email for the detailed report!")
