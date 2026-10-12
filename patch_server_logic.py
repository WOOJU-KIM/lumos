# -*- coding: utf-8 -*-
import sys
import re

file_path = 'c:/Users/chabo/OneDrive/바탕 화면/lumos/dashboard_server.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the daily logic
old_daily = '''                    if not row.empty and not prev.empty:
                        bal = row['balance_usd'].values[0]
                        pval = prev['balance_usd'].values[0]
                        usd_diff = bal
                        pct = (usd_diff / pval * 100) if pval != 0 else 0
                        daily_usd.append(round(usd_diff, 2))
                        daily_pct.append(round(pct, 2))
                    else:
                        daily_usd.append(0.0)
                        daily_pct.append(0.0)'''
new_daily = '''                    if not row.empty:
                        bal = row['balance_usd'].values[0]
                        pval = prev['balance_usd'].values[0] if not prev.empty else bal
                        pct = ((bal - pval) / pval * 100) if pval != 0 and not prev.empty else 0
                        daily_usd.append(round(bal, 2))
                        daily_pct.append(round(pct, 2))
                    else:
                        daily_usd.append(0.0)
                        daily_pct.append(0.0)'''
content = content.replace(old_daily, new_daily)

# Replace the weekly logic
old_weekly = '''                    if not cw_df.empty and not pw_df.empty:
                        bal = cw_df.iloc[-1]['balance_usd']
                        pval = pw_df.iloc[-1]['balance_usd']
                        usd_diff = bal
                        pct = (usd_diff / pval * 100) if pval != 0 else 0
                        weekly_usd.append(round(usd_diff, 2))
                        weekly_pct.append(round(pct, 2))
                    else:
                        weekly_usd.append(0.0)
                        weekly_pct.append(0.0)'''
new_weekly = '''                    if not cw_df.empty:
                        bal = cw_df.iloc[-1]['balance_usd']
                        pval = pw_df.iloc[-1]['balance_usd'] if not pw_df.empty else bal
                        pct = ((bal - pval) / pval * 100) if pval != 0 and not pw_df.empty else 0
                        weekly_usd.append(round(bal, 2))
                        weekly_pct.append(round(pct, 2))
                    else:
                        weekly_usd.append(0.0)
                        weekly_pct.append(0.0)'''
content = content.replace(old_weekly, new_weekly)

# Replace the monthly logic
old_monthly = '''                    if not cm_df.empty and not pm_df.empty:
                        bal = cm_df.iloc[-1]['balance_usd']
                        pval = pm_df.iloc[-1]['balance_usd']
                        usd_diff = bal
                        pct = (usd_diff / pval * 100) if pval != 0 else 0
                        monthly_usd.append(round(usd_diff, 2))
                        monthly_pct.append(round(pct, 2))
                    else:
                        monthly_usd.append(0.0)
                        monthly_pct.append(0.0)'''
new_monthly = '''                    if not cm_df.empty:
                        bal = cm_df.iloc[-1]['balance_usd']
                        pval = pm_df.iloc[-1]['balance_usd'] if not pm_df.empty else bal
                        pct = ((bal - pval) / pval * 100) if pval != 0 and not pm_df.empty else 0
                        monthly_usd.append(round(bal, 2))
                        monthly_pct.append(round(pct, 2))
                    else:
                        monthly_usd.append(0.0)
                        monthly_pct.append(0.0)'''
content = content.replace(old_monthly, new_monthly)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("SUCCESS dashboard_server.py logic fix")
