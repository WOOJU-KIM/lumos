# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# DB Load
conn = sqlite3.connect('c:/Users/chabo/OneDrive/바탕 화면/lumos/data/live_experience.db')
df = pd.read_sql_query('SELECT * FROM live_trades ORDER BY entry_time ASC', conn)
conn.close()

df['entry_time'] = pd.to_datetime(df['entry_time'])
df_oct = df[df['entry_time'].dt.month == 10].copy()

# KST to NYT
df_oct['nyt_time'] = df_oct['entry_time'] - pd.Timedelta(hours=13)
df_oct['date_nyt'] = df_oct['nyt_time'].dt.date

daily_pnl = df_oct.groupby('date_nyt')['pnl_usd'].sum().reset_index()

# Calculate Balance and Yield
starting_balance = 624.0
daily_pnl['prev_balance'] = starting_balance
daily_pnl['balance'] = starting_balance

for i in range(len(daily_pnl)):
    if i == 0:
        daily_pnl.loc[i, 'prev_balance'] = starting_balance
    else:
        daily_pnl.loc[i, 'prev_balance'] = daily_pnl.loc[i-1, 'balance']
    daily_pnl.loc[i, 'balance'] = daily_pnl.loc[i, 'prev_balance'] + daily_pnl.loc[i, 'pnl_usd']
    
daily_pnl['yield_pct'] = (daily_pnl['pnl_usd'] / daily_pnl['prev_balance']) * 100

# Draw Chart
fig, ax1 = plt.subplots(figsize=(10, 6))

color_bar = '#3498db'
ax1.set_xlabel('Date (NYT)')
ax1.set_ylabel('Total Balance (USD)', color=color_bar)
# Plot Balance
bars = ax1.bar(daily_pnl['date_nyt'].astype(str), daily_pnl['balance'], color=color_bar, alpha=0.7, label='Total Balance (USD)')
ax1.tick_params(axis='y', labelcolor=color_bar)

# Set Y limit to start not from 0 to make differences visible if needed, but starting from 0 is standard for bar charts.
ax1.set_ylim(0, max(daily_pnl['balance']) * 1.1)

ax2 = ax1.twinx()
color_line = '#e74c3c'
ax2.set_ylabel('Daily Yield (%)', color=color_line)
lines = ax2.plot(daily_pnl['date_nyt'].astype(str), daily_pnl['yield_pct'], color=color_line, marker='o', linewidth=2, label='Yield (%)')
ax2.tick_params(axis='y', labelcolor=color_line)

# Add values on bars (Total Balance)
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 5, f'', ha='center', va='bottom', color='black', fontweight='bold')

# Add values on line (Yield)
for i, txt in enumerate(daily_pnl['yield_pct']):
    ax2.text(i, txt + (0.05 if txt > 0 else -0.1), f'{txt:.2f}%', ha='center', va='bottom' if txt > 0 else 'top', color=color_line, fontweight='bold')

plt.title('Daily Total Balance & Yield (October 2026)')
fig.tight_layout()

# Save
out_path = r'C:\Users\chabo\.gemini\antigravity\brain\0a9a5fc0-a59c-41c7-bd00-7860a7191e8f\oct_trades_chart_balance.png'
plt.savefig(out_path, dpi=150)
print("SUCCESS")
