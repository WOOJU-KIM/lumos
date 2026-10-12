# -*- coding: utf-8 -*-
import sys
import re

file_path = 'c:/Users/chabo/OneDrive/바탕 화면/lumos/dashboard_server.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

new_api = '''        if path == '/api/return_charts':
            try:
                conn = sqlite3.connect(DATA_DIR / 'live_experience.db')
                df = pd.read_sql_query("SELECT * FROM daily_portfolio_history ORDER BY date ASC", conn)
                conn.close()
                
                if df.empty:
                    self._send_json({"monthly": [], "weekly": [], "daily": []})
                    return
                
                df['date'] = pd.to_datetime(df['date'])
                
                # Daily (Last 7 days)
                import datetime
                today = df['date'].max()
                daily_labels = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
                daily_data = []
                for lbl in daily_labels:
                    row = df[df['date'] == lbl]
                    if not row.empty:
                        daily_data.append(row['return_pct_usd'].values[0])
                    else:
                        daily_data.append(0.0)
                
                # Weekly (Last 5 weeks)
                df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
                weekly_df = df.groupby('week')['return_pct_usd'].sum().reset_index()
                current_week = today.to_period('W').start_time
                weekly_labels = [(current_week - timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(4, -1, -1)]
                weekly_data = []
                for lbl in weekly_labels:
                    row = weekly_df[weekly_df['week'] == lbl]
                    if not row.empty:
                        weekly_data.append(row['return_pct_usd'].values[0])
                    else:
                        weekly_data.append(0.0)
                        
                # Monthly (Last 3 months)
                df['month'] = df['date'].dt.to_period('M').apply(lambda r: r.start_time)
                monthly_df = df.groupby('month')['return_pct_usd'].sum().reset_index()
                current_month = today.replace(day=1)
                monthly_labels = [(current_month - pd.DateOffset(months=i)).strftime('%Y-%m') for i in range(2, -1, -1)]
                monthly_data = []
                for lbl in monthly_labels:
                    row = monthly_df[monthly_df['month'] == lbl + '-01']
                    if not row.empty:
                        monthly_data.append(row['return_pct_usd'].values[0])
                    else:
                        monthly_data.append(0.0)
                        
                self._send_json({
                    "monthly": {"labels": monthly_labels, "data": monthly_data},
                    "weekly": {"labels": weekly_labels, "data": weekly_data},
                    "daily": {"labels": daily_labels, "data": daily_data}
                })
            except Exception as e:
                self._send_json({"error": str(e)})
            return
            
        if path == '/api/events':'''

content = content.replace("        if path == '/api/events':", new_api)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("SUCCESS dashboard_server.py modified")
