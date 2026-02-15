# This file contains ONLY the improved create_html_report method
# Replace the existing method in your script with this enhanced version

def create_html_report(self, oc_analysis, tech_analysis, recommendation):
    """Create beautiful HTML report with modern news feed style layout"""
    now_ist = self.format_ist_time()
    
    rec = recommendation['recommendation']
    
    # Enhanced color scheme based on recommendation
    if 'STRONG BUY' in rec:
        rec_gradient = 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)'
        rec_badge_color = '#22c55e'
    elif 'BUY' in rec:
        rec_gradient = 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)'
        rec_badge_color = '#3b82f6'
    elif 'STRONG SELL' in rec:
        rec_gradient = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)'
        rec_badge_color = '#ef4444'
    elif 'SELL' in rec:
        rec_gradient = 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)'
        rec_badge_color = '#f97316'
    else:
        rec_gradient = 'linear-gradient(135deg, #eab308 0%, #ca8a04 100%)'
        rec_badge_color = '#eab308'
    
    title = self.config['report'].get('title', 'NIFTY DAY TRADING ANALYSIS')
    
    strategies = self.get_options_strategies(recommendation, oc_analysis, tech_analysis)
    strike_recommendations = self.get_detailed_strike_recommendations(oc_analysis, tech_analysis, recommendation)
    
    pivot_points = tech_analysis.get('pivot_points', {})
    current_price = tech_analysis.get('current_price', 0)
    nearest_levels = self.find_nearest_levels(current_price, pivot_points)
    
    # Momentum values
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
    
    # Build Top OI tables
    top_ce_strikes = oc_analysis.get('top_ce_strikes', [])
    top_pe_strikes = oc_analysis.get('top_pe_strikes', [])
    
    ce_rows_html = ''
    for idx, strike in enumerate(top_ce_strikes, 1):
        badge_color = {'ITM': '#22c55e', 'ATM': '#eab308', 'OTM': '#6b7280'}.get(strike['type'], '#6b7280')
        ce_rows_html += f"""
                        <tr>
                            <td class="rank-cell">{idx}</td>
                            <td class="strike-cell">₹{strike['strike']}</td>
                            <td><span class="type-badge" style="background: {badge_color};">{strike['type']}</span></td>
                            <td class="number-cell">{strike['oi']:,}</td>
                            <td class="number-cell {'positive' if strike['chng_oi'] > 0 else 'negative'}">{strike['chng_oi']:+,}</td>
                            <td class="number-cell">₹{strike['ltp']:.2f}</td>
                            <td class="number-cell">{strike['iv']:.2f}%</td>
                            <td class="number-cell">{strike['volume']:,}</td>
                        </tr>
        """
    
    pe_rows_html = ''
    for idx, strike in enumerate(top_pe_strikes, 1):
        badge_color = {'ITM': '#22c55e', 'ATM': '#eab308', 'OTM': '#6b7280'}.get(strike['type'], '#6b7280')
        pe_rows_html += f"""
                        <tr>
                            <td class="rank-cell">{idx}</td>
                            <td class="strike-cell">₹{strike['strike']}</td>
                            <td><span class="type-badge" style="background: {badge_color};">{strike['type']}</span></td>
                            <td class="number-cell">{strike['oi']:,}</td>
                            <td class="number-cell {'positive' if strike['chng_oi'] > 0 else 'negative'}">{strike['chng_oi']:+,}</td>
                            <td class="number-cell">₹{strike['ltp']:.2f}</td>
                            <td class="number-cell">{strike['iv']:.2f}%</td>
                            <td class="number-cell">{strike['volume']:,}</td>
                        </tr>
        """
    
    # Build strategies HTML
    strategies_html = ''
    for strategy in strategies:
        star_count = strategy['recommended'].count('⭐')
        strategies_html += f"""
            <div class="strategy-card">
                <div class="strategy-card-header">
                    <h4>{strategy['name']}</h4>
                    <span class="strategy-type-badge">{strategy['type']}</span>
                </div>
                <div class="strategy-card-body">
                    <div class="strategy-row">
                        <span class="strategy-label">Setup:</span>
                        <span class="strategy-value">{strategy['setup']}</span>
                    </div>
                    <div class="strategy-row">
                        <span class="strategy-label">Profit:</span>
                        <span class="strategy-value">{strategy['profit']}</span>
                    </div>
                    <div class="strategy-row">
                        <span class="strategy-label">Risk:</span>
                        <span class="strategy-value">{strategy['risk']}</span>
                    </div>
                    <div class="strategy-row">
                        <span class="strategy-label">Best When:</span>
                        <span class="strategy-value">{strategy['best_when']}</span>
                    </div>
                    <div class="rating-stars">{'⭐' * star_count}</div>
                </div>
            </div>
        """
    
    # Build pivot table
    def get_level_class(level_value):
        if level_value == nearest_levels.get('nearest_resistance'):
            return 'nearest-resistance'
        elif level_value == nearest_levels.get('nearest_support'):
            return 'nearest-support'
        return ''
    
    pivot_rows = f"""
                    <tr class="pivot-row resistance-row {get_level_class(pivot_points.get('r3'))}">
                        <td class="level-name">R3</td>
                        <td class="level-value">₹{pivot_points.get('r3', 'N/A')}</td>
                        <td class="distance-value">{f'+{pivot_points.get("r3", 0) - current_price:.2f}' if pivot_points.get('r3') else 'N/A'}</td>
                    </tr>
                    <tr class="pivot-row resistance-row {get_level_class(pivot_points.get('r2'))}">
                        <td class="level-name">R2</td>
                        <td class="level-value">₹{pivot_points.get('r2', 'N/A')}</td>
                        <td class="distance-value">{f'+{pivot_points.get("r2", 0) - current_price:.2f}' if pivot_points.get('r2') else 'N/A'}</td>
                    </tr>
                    <tr class="pivot-row resistance-row {get_level_class(pivot_points.get('r1'))}">
                        <td class="level-name">R1</td>
                        <td class="level-value">₹{pivot_points.get('r1', 'N/A')}</td>
                        <td class="distance-value">{f'+{pivot_points.get("r1", 0) - current_price:.2f}' if pivot_points.get('r1') else 'N/A'}</td>
                    </tr>
                    <tr class="pivot-row pivot-center">
                        <td class="level-name">PP</td>
                        <td class="level-value">₹{pivot_points.get('pivot', 'N/A')}</td>
                        <td class="distance-value">{f'{pivot_points.get("pivot", 0) - current_price:+.2f}' if pivot_points.get('pivot') else 'N/A'}</td>
                    </tr>
                    <tr class="pivot-row support-row {get_level_class(pivot_points.get('s1'))}">
                        <td class="level-name">S1</td>
                        <td class="level-value">₹{pivot_points.get('s1', 'N/A')}</td>
                        <td class="distance-value">{f'{pivot_points.get("s1", 0) - current_price:.2f}' if pivot_points.get('s1') else 'N/A'}</td>
                    </tr>
                    <tr class="pivot-row support-row {get_level_class(pivot_points.get('s2'))}">
                        <td class="level-name">S2</td>
                        <td class="level-value">₹{pivot_points.get('s2', 'N/A')}</td>
                        <td class="distance-value">{f'{pivot_points.get("s2", 0) - current_price:.2f}' if pivot_points.get('s2') else 'N/A'}</td>
                    </tr>
                    <tr class="pivot-row support-row {get_level_class(pivot_points.get('s3'))}">
                        <td class="level-name">S3</td>
                        <td class="level-value">₹{pivot_points.get('s3', 'N/A')}</td>
                        <td class="distance-value">{f'{pivot_points.get("s3", 0) - current_price:.2f}' if pivot_points.get('s3') else 'N/A'}</td>
                    </tr>
    """
    
    html = f"""
<!DOCTYPE html>
<html>
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
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        /* Header Section */
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.1);
        }}
        
        .header h1 {{
            color: #38bdf8;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 0 2px 10px rgba(56, 189, 248, 0.3);
        }}
        
        .header-badges {{
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 15px;
        }}
        
        .header-badge {{
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: #38bdf8;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }}
        
        .timestamp {{
            color: #94a3b8;
            font-size: 14px;
            margin-top: 10px;
        }}
        
        /* Recommendation Hero Card */
        .recommendation-hero {{
            background: {rec_gradient};
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .recommendation-hero h2 {{
            font-size: 36px;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 15px;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }}
        
        .recommendation-meta {{
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }}
        
        .meta-item {{
            background: rgba(255, 255, 255, 0.15);
            padding: 10px 20px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }}
        
        .meta-label {{
            font-size: 12px;
            opacity: 0.9;
            font-weight: 600;
        }}
        
        .meta-value {{
            font-size: 18px;
            font-weight: 700;
            margin-top: 5px;
        }}
        
        .signal-badges {{
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        
        .signal-badge {{
            background: rgba(255, 255, 255, 0.2);
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            backdrop-filter: blur(10px);
        }}
        
        /* Momentum Section */
        .momentum-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .momentum-card {{
            background: linear-gradient(135deg, var(--momentum-bg) 0%, var(--momentum-bg-dark) 100%);
            border-radius: 16px;
            padding: 25px;
            border: 2px solid var(--momentum-border);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}
        
        .momentum-card h3 {{
            font-size: 16px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
            color: var(--momentum-text);
        }}
        
        .momentum-value {{
            font-size: 42px;
            font-weight: 900;
            color: var(--momentum-text);
            margin-bottom: 10px;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }}
        
        .momentum-signal {{
            font-size: 16px;
            font-weight: 600;
            color: var(--momentum-text);
            opacity: 0.95;
        }}
        
        /* News Feed Style Cards */
        .news-feed-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}
        
        .news-category {{
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-radius: 16px;
            padding: 25px;
            border: 1px solid rgba(148, 163, 184, 0.1);
        }}
        
        .category-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(148, 163, 184, 0.2);
        }}
        
        .category-icon {{
            font-size: 24px;
        }}
        
        .category-title {{
            font-size: 20px;
            font-weight: 700;
            color: #38bdf8;
        }}
        
        /* News Item Cards */
        .news-item {{
            background: rgba(30, 41, 59, 0.6);
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 15px;
            border-left: 4px solid #38bdf8;
            transition: all 0.3s ease;
        }}
        
        .news-item:hover {{
            transform: translateX(5px);
            background: rgba(30, 41, 59, 0.8);
            box-shadow: 0 4px 15px rgba(56, 189, 248, 0.2);
        }}
        
        .news-item:last-child {{
            margin-bottom: 0;
        }}
        
        .news-item-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }}
        
        .news-title {{
            font-size: 16px;
            font-weight: 700;
            color: #e2e8f0;
            line-height: 1.4;
            flex: 1;
        }}
        
        .news-source-badge {{
            background: rgba(56, 189, 248, 0.2);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: #38bdf8;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            white-space: nowrap;
            margin-left: 10px;
        }}
        
        .news-description {{
            font-size: 14px;
            color: #cbd5e1;
            line-height: 1.6;
            margin-bottom: 12px;
        }}
        
        .news-meta {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: #94a3b8;
        }}
        
        /* Data Grid */
        .data-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .data-card {{
            background: rgba(30, 41, 59, 0.6);
            border-radius: 12px;
            padding: 18px;
            border-left: 4px solid #38bdf8;
        }}
        
        .data-label {{
            font-size: 12px;
            color: #94a3b8;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        
        .data-value {{
            font-size: 22px;
            font-weight: 700;
            color: #e2e8f0;
        }}
        
        /* Levels Section */
        .levels-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .level-box {{
            background: rgba(30, 41, 59, 0.6);
            border-radius: 12px;
            padding: 20px;
        }}
        
        .level-box.resistance {{
            border-left: 4px solid #ef4444;
        }}
        
        .level-box.support {{
            border-left: 4px solid #22c55e;
        }}
        
        .level-box h4 {{
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 15px;
            color: #e2e8f0;
        }}
        
        .level-box ul {{
            list-style: none;
        }}
        
        .level-box li {{
            padding: 8px 12px;
            background: rgba(15, 23, 42, 0.5);
            border-radius: 8px;
            margin-bottom: 8px;
            font-weight: 600;
            font-size: 15px;
        }}
        
        .level-box.resistance li {{
            color: #fca5a5;
        }}
        
        .level-box.support li {{
            color: #86efac;
        }}
        
        /* Table Styles */
        .table-container {{
            overflow-x: auto;
            background: rgba(30, 41, 59, 0.6);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        thead {{
            background: rgba(56, 189, 248, 0.1);
        }}
        
        th {{
            padding: 12px;
            text-align: left;
            font-size: 12px;
            font-weight: 700;
            color: #38bdf8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid rgba(56, 189, 248, 0.3);
        }}
        
        td {{
            padding: 12px;
            font-size: 14px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        }}
        
        tr:hover {{
            background: rgba(56, 189, 248, 0.05);
        }}
        
        .rank-cell {{
            font-weight: 700;
            color: #38bdf8;
        }}
        
        .strike-cell {{
            font-weight: 700;
            color: #e2e8f0;
        }}
        
        .number-cell {{
            text-align: right;
            font-weight: 600;
        }}
        
        .number-cell.positive {{
            color: #22c55e;
        }}
        
        .number-cell.negative {{
            color: #ef4444;
        }}
        
        .type-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
            color: #ffffff;
        }}
        
        /* Pivot Table Styles */
        .pivot-row {{
            transition: all 0.2s ease;
        }}
        
        .pivot-row.resistance-row {{
            background: rgba(239, 68, 68, 0.1);
        }}
        
        .pivot-row.support-row {{
            background: rgba(34, 197, 94, 0.1);
        }}
        
        .pivot-row.pivot-center {{
            background: rgba(234, 179, 8, 0.15);
            font-weight: 700;
        }}
        
        .pivot-row.nearest-resistance {{
            background: rgba(239, 68, 68, 0.2);
            border: 2px solid #ef4444;
        }}
        
        .pivot-row.nearest-support {{
            background: rgba(34, 197, 94, 0.2);
            border: 2px solid #22c55e;
        }}
        
        .level-name {{
            font-weight: 700;
            color: #38bdf8;
        }}
        
        .level-value {{
            font-weight: 700;
            color: #e2e8f0;
        }}
        
        .distance-value {{
            text-align: right;
            font-weight: 600;
        }}
        
        .resistance-row .distance-value {{
            color: #fca5a5;
        }}
        
        .support-row .distance-value {{
            color: #86efac;
        }}
        
        /* Strategy Cards */
        .strategies-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .strategy-card {{
            background: rgba(30, 41, 59, 0.6);
            border-radius: 12px;
            padding: 20px;
            border-left: 4px solid #38bdf8;
            transition: all 0.3s ease;
        }}
        
        .strategy-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(56, 189, 248, 0.2);
        }}
        
        .strategy-card-header {{
            margin-bottom: 15px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        }}
        
        .strategy-card-header h4 {{
            font-size: 18px;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 8px;
        }}
        
        .strategy-type-badge {{
            display: inline-block;
            background: rgba(56, 189, 248, 0.2);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: #38bdf8;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}
        
        .strategy-card-body {{
            color: #cbd5e1;
        }}
        
        .strategy-row {{
            margin-bottom: 12px;
            display: flex;
            gap: 8px;
            line-height: 1.6;
        }}
        
        .strategy-label {{
            font-weight: 700;
            color: #94a3b8;
            min-width: 80px;
        }}
        
        .strategy-value {{
            flex: 1;
            color: #e2e8f0;
        }}
        
        .rating-stars {{
            margin-top: 15px;
            font-size: 18px;
            text-align: center;
            padding: 10px;
            background: rgba(234, 179, 8, 0.1);
            border-radius: 8px;
        }}
        
        /* Strike Recommendations */
        .strike-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .strike-card {{
            background: rgba(30, 41, 59, 0.6);
            border-radius: 12px;
            padding: 20px;
            border-left: 4px solid {rec_badge_color};
            transition: all 0.3s ease;
        }}
        
        .strike-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(56, 189, 248, 0.2);
        }}
        
        .strike-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        }}
        
        .strike-header h4 {{
            font-size: 16px;
            font-weight: 700;
            color: #38bdf8;
        }}
        
        .strike-option-badge {{
            background: {rec_gradient};
            color: #ffffff;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
        }}
        
        .strike-details {{
            background: rgba(15, 23, 42, 0.5);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        
        .strike-detail-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px dashed rgba(148, 163, 184, 0.1);
        }}
        
        .strike-detail-row:last-child {{
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
        }}
        
        .strike-detail-label {{
            color: #94a3b8;
            font-size: 13px;
            font-weight: 600;
        }}
        
        .strike-detail-value {{
            color: #e2e8f0;
            font-weight: 700;
            font-size: 14px;
        }}
        
        .strike-detail-value.premium {{
            color: #38bdf8;
            font-size: 16px;
        }}
        
        .profit-targets {{
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(56, 189, 248, 0.05) 100%);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        
        .profit-targets h5 {{
            color: #38bdf8;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 12px;
        }}
        
        .targets-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }}
        
        .target-item {{
            background: rgba(15, 23, 42, 0.6);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        
        .target-item.target-1 {{
            border: 2px solid #22c55e;
        }}
        
        .target-item.target-2 {{
            border: 2px solid #06b6d4;
        }}
        
        .target-item.stop-loss {{
            border: 2px solid #ef4444;
        }}
        
        .target-item-label {{
            font-size: 11px;
            color: #94a3b8;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 6px;
        }}
        
        .target-item-price {{
            font-size: 16px;
            color: #e2e8f0;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        
        .target-item-profit {{
            font-size: 12px;
            font-weight: 600;
        }}
        
        .target-item.target-1 .target-item-profit,
        .target-item.target-2 .target-item-profit {{
            color: #22c55e;
        }}
        
        .target-item.stop-loss .target-item-profit {{
            color: #ef4444;
        }}
        
        .trade-example {{
            background: rgba(234, 179, 8, 0.1);
            border: 1px solid rgba(234, 179, 8, 0.3);
            border-radius: 8px;
            padding: 12px;
            font-size: 12px;
            color: #cbd5e1;
            line-height: 1.6;
        }}
        
        /* Reasons Box */
        .reasons-box {{
            background: rgba(234, 179, 8, 0.1);
            border-left: 4px solid #eab308;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .reasons-box strong {{
            color: #fbbf24;
            font-size: 16px;
        }}
        
        .reasons-box ul {{
            list-style: none;
            margin-top: 15px;
        }}
        
        .reasons-box li {{
            padding: 10px 15px;
            background: rgba(15, 23, 42, 0.5);
            border-radius: 8px;
            margin-bottom: 10px;
            color: #e2e8f0;
            border-left: 3px solid #eab308;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 30px;
            border-top: 1px solid rgba(148, 163, 184, 0.2);
            color: #94a3b8;
            font-size: 13px;
        }}
        
        .footer p {{
            margin: 8px 0;
        }}
        
        /* Responsive Design */
        @media (max-width: 1024px) {{
            .news-feed-grid {{
                grid-template-columns: 1fr;
            }}
            
            .momentum-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 24px;
            }}
            
            .recommendation-hero h2 {{
                font-size: 28px;
            }}
            
            .momentum-value {{
                font-size: 32px;
            }}
            
            .data-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            
            .strategies-grid {{
                grid-template-columns: 1fr;
            }}
            
            .strike-grid {{
                grid-template-columns: 1fr;
            }}
            
            .targets-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        @media (max-width: 480px) {{
            .header-badges {{
                flex-direction: column;
            }}
            
            .recommendation-meta {{
                flex-direction: column;
                gap: 15px;
            }}
            
            .data-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>{title}</h1>
            <div class="header-badges">
                <div class="header-badge">⏱️ 1-HOUR TIMEFRAME</div>
                <div class="header-badge">📊 DUAL MOMENTUM ANALYSIS</div>
            </div>
            <div class="timestamp">Generated: {now_ist}</div>
        </div>
        
        <!-- Recommendation Hero Card -->
        <div class="recommendation-hero">
            <h2>{recommendation['recommendation']}</h2>
            <div class="recommendation-meta">
                <div class="meta-item">
                    <div class="meta-label">MARKET BIAS</div>
                    <div class="meta-value">{recommendation['bias']}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">CONFIDENCE</div>
                    <div class="meta-value">{recommendation['confidence']}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">CURRENT PRICE</div>
                    <div class="meta-value">₹{tech_analysis.get('current_price', 'N/A')}</div>
                </div>
            </div>
            <div class="signal-badges">
                <div class="signal-badge">🟢 Bullish Signals: {recommendation['bullish_signals']}</div>
                <div class="signal-badge">🔴 Bearish Signals: {recommendation['bearish_signals']}</div>
            </div>
        </div>
        
        <!-- Momentum Cards -->
        <div class="momentum-grid">
            <div class="momentum-card" style="--momentum-bg: {momentum_1h_colors['bg']}; --momentum-bg-dark: {momentum_1h_colors['bg_dark']}; --momentum-text: {momentum_1h_colors['text']}; --momentum-border: {momentum_1h_colors['border']};">
                <h3>⚡ 1-HOUR MOMENTUM</h3>
                <div class="momentum-value">{momentum_1h_pct:+.2f}%</div>
                <div class="momentum-signal">{momentum_1h_signal}</div>
            </div>
            <div class="momentum-card" style="--momentum-bg: {momentum_5h_colors['bg']}; --momentum-bg-dark: {momentum_5h_colors['bg_dark']}; --momentum-text: {momentum_5h_colors['text']}; --momentum-border: {momentum_5h_colors['border']};">
                <h3>📊 5-HOUR MOMENTUM</h3>
                <div class="momentum-value">{momentum_5h_pct:+.2f}%</div>
                <div class="momentum-signal">{momentum_5h_signal}</div>
            </div>
        </div>
        
        <!-- News Feed Style Layout -->
        <div class="news-feed-grid">
            <!-- Technical Analysis Section -->
            <div class="news-category">
                <div class="category-header">
                    <span class="category-icon">📈</span>
                    <h3 class="category-title">Technical Analysis</h3>
                </div>
                
                <div class="news-item">
                    <div class="news-item-header">
                        <div class="news-title">RSI & Trend Analysis</div>
                        <span class="news-source-badge">1H Chart</span>
                    </div>
                    <div class="news-description">
                        RSI(14) at {tech_analysis.get('rsi', 'N/A')} indicates {tech_analysis.get('rsi_signal', 'N/A')}. 
                        Current trend: {tech_analysis.get('trend', 'N/A')}.
                    </div>
                </div>
                
                <div class="news-item">
                    <div class="news-item-header">
                        <div class="news-title">EMA Position</div>
                        <span class="news-source-badge">Moving Avg</span>
                    </div>
                    <div class="news-description">
                        Price at ₹{tech_analysis.get('current_price', 'N/A')} vs EMA20 ₹{tech_analysis.get('ema20', 'N/A')} 
                        and EMA50 ₹{tech_analysis.get('ema50', 'N/A')}.
                    </div>
                </div>
                
                <div class="data-grid">
                    <div class="data-card">
                        <div class="data-label">Current Price</div>
                        <div class="data-value">₹{tech_analysis.get('current_price', 'N/A')}</div>
                    </div>
                    <div class="data-card">
                        <div class="data-label">RSI (14)</div>
                        <div class="data-value">{tech_analysis.get('rsi', 'N/A')}</div>
                    </div>
                </div>
            </div>
            
            <!-- Option Chain Analysis Section -->
            <div class="news-category">
                <div class="category-header">
                    <span class="category-icon">💰</span>
                    <h3 class="category-title">Option Chain Analysis</h3>
                </div>
                
                <div class="news-item">
                    <div class="news-item-header">
                        <div class="news-title">Put-Call Ratio (PCR)</div>
                        <span class="news-source-badge">Options</span>
                    </div>
                    <div class="news-description">
                        PCR at {oc_analysis.get('pcr', 'N/A')} shows {oc_analysis.get('oi_sentiment', 'N/A')} sentiment 
                        in the market based on open interest buildup.
                    </div>
                </div>
                
                <div class="news-item">
                    <div class="news-item-header">
                        <div class="news-title">Max Pain Level</div>
                        <span class="news-source-badge">OI Data</span>
                    </div>
                    <div class="news-description">
                        Maximum pain strike at ₹{oc_analysis.get('max_pain', 'N/A')} suggests potential 
                        price gravitation based on options positioning.
                    </div>
                </div>
                
                <div class="data-grid">
                    <div class="data-card">
                        <div class="data-label">Put-Call Ratio</div>
                        <div class="data-value">{oc_analysis.get('pcr', 'N/A')}</div>
                    </div>
                    <div class="data-card">
                        <div class="data-label">Max Pain</div>
                        <div class="data-value">₹{oc_analysis.get('max_pain', 'N/A')}</div>
                    </div>
                </div>
            </div>
            
            <!-- Support & Resistance Section -->
            <div class="news-category">
                <div class="category-header">
                    <span class="category-icon">🎯</span>
                    <h3 class="category-title">Key Levels</h3>
                </div>
                
                <div class="levels-grid">
                    <div class="level-box resistance">
                        <h4>🔴 Resistance Levels</h4>
                        <ul>
                            {''.join([f'<li>R{i+1}: ₹{r}</li>' for i, r in enumerate(tech_analysis.get('tech_resistances', []))])}
                        </ul>
                    </div>
                    <div class="level-box support">
                        <h4>🟢 Support Levels</h4>
                        <ul>
                            {''.join([f'<li>S{i+1}: ₹{s}</li>' for i, s in enumerate(tech_analysis.get('tech_supports', []))])}
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Pivot Points Table -->
        <div class="news-category">
            <div class="category-header">
                <span class="category-icon">📍</span>
                <h3 class="category-title">Pivot Points (Traditional - 30 Min)</h3>
            </div>
            
            <div class="table-container">
                <p style="color: #94a3b8; margin-bottom: 15px; font-size: 13px;">
                    Previous 30-min: High ₹{pivot_points.get('prev_high', 'N/A')} | 
                    Low ₹{pivot_points.get('prev_low', 'N/A')} | 
                    Close ₹{pivot_points.get('prev_close', 'N/A')}
                </p>
                <table>
                    <thead>
                        <tr>
                            <th>Level</th>
                            <th>Value</th>
                            <th>Distance</th>
                        </tr>
                    </thead>
                    <tbody>
{pivot_rows}
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Top Open Interest -->
        <div class="news-category">
            <div class="category-header">
                <span class="category-icon">🔥</span>
                <h3 class="category-title">Top Open Interest (5 CE + 5 PE)</h3>
            </div>
            
            <div class="news-feed-grid">
                <div>
                    <h4 style="color: #22c55e; margin-bottom: 15px; font-size: 16px; font-weight: 700;">📞 Top 5 Call Options (CE)</h4>
                    <div class="table-container">
                        <table>
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
                
                <div>
                    <h4 style="color: #ef4444; margin-bottom: 15px; font-size: 16px; font-weight: 700;">📉 Top 5 Put Options (PE)</h4>
                    <div class="table-container">
                        <table>
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
        
        <!-- Analysis Summary -->
        <div class="reasons-box">
            <strong>📊 Key Analysis Factors:</strong>
            <ul>
                {''.join([f'<li>{reason}</li>' for reason in recommendation.get('reasons', [])])}
            </ul>
        </div>
        
        <!-- Strike Recommendations -->
        <div class="news-category">
            <div class="category-header">
                <span class="category-icon">💰</span>
                <h3 class="category-title">Detailed Strike Recommendations</h3>
            </div>
            
            <p style="color: #94a3b8; margin-bottom: 20px; font-size: 14px;">
                Based on <strong>{recommendation['bias']}</strong> bias with current Nifty at ₹{tech_analysis.get('current_price', 0):.2f}
            </p>
            
            <div class="strike-grid">
"""
    
    # Add strike recommendations
    if strike_recommendations:
        for rec in strike_recommendations:
            html += f"""
                <div class="strike-card">
                    <div class="strike-header">
                        <h4>{rec['strategy']}</h4>
                        <span class="strike-option-badge">{rec['option_type']}</span>
                    </div>
                    
                    <div class="strike-details">
                        <div class="strike-detail-row">
                            <span class="strike-detail-label">Action:</span>
                            <span class="strike-detail-value"><strong>{rec['action']}</strong></span>
                        </div>
                        <div class="strike-detail-row">
                            <span class="strike-detail-label">Strike Price:</span>
                            <span class="strike-detail-value"><strong>₹{rec['strike']}</strong></span>
                        </div>
                        <div class="strike-detail-row">
                            <span class="strike-detail-label">Current LTP:</span>
                            <span class="strike-detail-value premium">₹{rec['ltp']:.2f}</span>
                        </div>
                        <div class="strike-detail-row">
                            <span class="strike-detail-label">Open Interest:</span>
                            <span class="strike-detail-value">{rec['oi']}</span>
                        </div>
                        <div class="strike-detail-row">
                            <span class="strike-detail-label">Volume:</span>
                            <span class="strike-detail-value">{rec['volume']}</span>
                        </div>
                    </div>
                    
                    <div class="profit-targets">
                        <h5>📊 Profit Targets & Risk Management</h5>
                        <div class="targets-grid">
                            <div class="target-item target-1">
                                <div class="target-item-label">Target 1</div>
                                <div class="target-item-price">₹{rec['target_1']}</div>
                                <div class="target-item-profit">+₹{rec['profit_at_target_1']:.2f}</div>
                            </div>
                            <div class="target-item target-2">
                                <div class="target-item-label">Target 2</div>
                                <div class="target-item-price">₹{rec['target_2']}</div>
                                <div class="target-item-profit">{f"+₹{rec['profit_at_target_2']:.2f}" if isinstance(rec['profit_at_target_2'], (int, float)) else rec['profit_at_target_2']}</div>
                            </div>
                            <div class="target-item stop-loss">
                                <div class="target-item-label">Stop Loss</div>
                                <div class="target-item-price">₹{rec['stop_loss']:.2f}</div>
                                <div class="target-item-profit">Max: ₹{rec['max_loss']:.2f}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="trade-example">
                        <strong>📝 Example Trade:</strong> If you buy 1 lot (50 qty) at LTP ₹{rec['ltp']:.2f}, 
                        investment = ₹{rec['ltp'] * 50:.0f}. 
                        At Target 1: Profit ₹{rec['profit_at_target_1'] * 50 if isinstance(rec['profit_at_target_1'], (int, float)) else 'Variable':.0f} | 
                        At Target 2: Profit ₹{rec['profit_at_target_2'] * 50 if isinstance(rec['profit_at_target_2'], (int, float)) else 'Variable':.0f}
                    </div>
                </div>
            """
    else:
        html += """
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 30px; text-align: center;">
                    <p style="color: #fca5a5; font-size: 16px;">No specific strike recommendations available at this time. Check the general strategies below.</p>
                </div>
        """
    
    html += f"""
            </div>
        </div>
        
        <!-- Options Strategies -->
        <div class="news-category">
            <div class="category-header">
                <span class="category-icon">🎯</span>
                <h3 class="category-title">Options Trading Strategies</h3>
            </div>
            
            <p style="color: #94a3b8; margin-bottom: 20px; font-size: 14px;">
                Recommended strategies based on <strong>{recommendation['bias']}</strong> market bias
            </p>
            
            <div class="strategies-grid">
{strategies_html}
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p><strong>⚠️ Disclaimer:</strong> This analysis is for educational purposes only. Trading involves substantial risk of loss.</p>
            <p><strong>📊 Analysis Methodology:</strong> Dual Momentum (1H + 5H) | RSI (Wilder's Method) | Traditional Pivot Points (30-min)</p>
            <p style="margin-top: 20px; color: #64748b;">© 2025 Nifty Trading Analyzer | Advanced Technical & Options Analysis Platform</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html
