# -*- coding: utf-8 -*-
import sys
import re

file_path = 'c:/Users/chabo/OneDrive/바탕 화면/lumos/dashboard_server.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r"        if path == '/api/return_charts':.*?return\n"

new_api = '''        if path == '/api/return_charts':
            try:
                conn = sqlite3.connect(DATA_DIR / 'live_experience.db')
                df = pd.read_sql_query("SELECT * FROM daily_portfolio_history ORDER BY date ASC", conn)
                conn.close()
                
                if df.empty:
                    self._send_json({"monthly": [], "weekly": [], "daily": []})
                    return
                
                df['date'] = pd.to_datetime(df['date'])
                
                import datetime
                today = df['date'].max()
                
                # We need balance history to calculate PnL and Returns.
                # If there are missing dates, fill them with ffill.
                full_dates = pd.date_range(start=df['date'].min() - pd.Timedelta(days=100), end=today)
                df_full = pd.DataFrame({'date': full_dates})
                df_full = pd.merge(df_full, df[['date', 'balance_usd']], on='date', how='left').ffill()
                # If still NaN at start, fill with first valid
                df_full['balance_usd'] = df_full['balance_usd'].bfill()
                
                if df_full['balance_usd'].isna().all():
                    df_full['balance_usd'] = 630.0 # fallback

                # 1. Daily (Last 7 days)
                daily_labels = []
                daily_usd = []
                daily_pct = []
                for i in range(6, -1, -1):
                    d = today - timedelta(days=i)
                    lbl = d.strftime('%Y-%m-%d')
                    daily_labels.append(lbl)
                    
                    row = df_full[df_full['date'] == d]
                    prev = df_full[df_full['date'] == d - timedelta(days=1)]
                    
                    if not row.empty and not prev.empty:
                        bal = row['balance_usd'].values[0]
                        pval = prev['balance_usd'].values[0]
                        usd_diff = bal - pval
                        pct = (usd_diff / pval * 100) if pval != 0 else 0
                        daily_usd.append(round(usd_diff, 2))
                        daily_pct.append(round(pct, 2))
                    else:
                        daily_usd.append(0.0)
                        daily_pct.append(0.0)

                # 2. Weekly (Last 5 weeks, formatted as YYYY-Wxx)
                weekly_labels = []
                weekly_usd = []
                weekly_pct = []
                
                df_full['iso_year'] = df_full['date'].dt.isocalendar().year
                df_full['iso_week'] = df_full['date'].dt.isocalendar().week
                
                # Get unique weeks ending at today's week
                current_year = today.isocalendar().year
                current_week = today.isocalendar().week
                
                weeks_list = []
                y, w = current_year, current_week
                for _ in range(6): # get 6 weeks (need previous week for first diff)
                    weeks_list.append((y, w))
                    w -= 1
                    if w == 0:
                        y -= 1
                        w = 52
                weeks_list.reverse()
                
                for i in range(1, 6):
                    y, w = weeks_list[i]
                    py, pw = weeks_list[i-1]
                    
                    lbl = f"{y}-W{w:02d}"
                    weekly_labels.append(lbl)
                    
                    # Last balance of the current week
                    cw_df = df_full[(df_full['iso_year'] == y) & (df_full['iso_week'] == w)]
                    pw_df = df_full[(df_full['iso_year'] == py) & (df_full['iso_week'] == pw)]
                    
                    if not cw_df.empty and not pw_df.empty:
                        bal = cw_df.iloc[-1]['balance_usd']
                        pval = pw_df.iloc[-1]['balance_usd']
                        usd_diff = bal - pval
                        pct = (usd_diff / pval * 100) if pval != 0 else 0
                        weekly_usd.append(round(usd_diff, 2))
                        weekly_pct.append(round(pct, 2))
                    else:
                        weekly_usd.append(0.0)
                        weekly_pct.append(0.0)

                # 3. Monthly (Last 3 months)
                monthly_labels = []
                monthly_usd = []
                monthly_pct = []
                
                df_full['year'] = df_full['date'].dt.year
                df_full['month'] = df_full['date'].dt.month
                
                cy, cm = today.year, today.month
                months_list = []
                for _ in range(4): # need prev month
                    months_list.append((cy, cm))
                    cm -= 1
                    if cm == 0:
                        cy -= 1
                        cm = 12
                months_list.reverse()
                
                for i in range(1, 4):
                    y, m = months_list[i]
                    py, pm = months_list[i-1]
                    
                    lbl = f"{y}-{m:02d}"
                    monthly_labels.append(lbl)
                    
                    cm_df = df_full[(df_full['year'] == y) & (df_full['month'] == m)]
                    pm_df = df_full[(df_full['year'] == py) & (df_full['month'] == pm)]
                    
                    if not cm_df.empty and not pm_df.empty:
                        bal = cm_df.iloc[-1]['balance_usd']
                        pval = pm_df.iloc[-1]['balance_usd']
                        usd_diff = bal - pval
                        pct = (usd_diff / pval * 100) if pval != 0 else 0
                        monthly_usd.append(round(usd_diff, 2))
                        monthly_pct.append(round(pct, 2))
                    else:
                        monthly_usd.append(0.0)
                        monthly_pct.append(0.0)

                self._send_json({
                    "monthly": {"labels": monthly_labels, "usd": monthly_usd, "pct": monthly_pct},
                    "weekly": {"labels": weekly_labels, "usd": weekly_usd, "pct": weekly_pct},
                    "daily": {"labels": daily_labels, "usd": daily_usd, "pct": daily_pct}
                })
            except Exception as e:
                self._send_json({"error": str(e)})
            return
'''

content = re.sub(pattern, new_api, content, flags=re.DOTALL)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("SUCCESS dashboard_server.py combination")
