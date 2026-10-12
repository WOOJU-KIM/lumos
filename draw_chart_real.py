# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# DB Load
conn = sqlite3.connect('c:/Users/chabo/OneDrive/바탕 화면/lumos/data/live_experience.db')
df = pd.read_sql_query("SELECT * FROM daily_portfolio_history WHERE date >= '2026-09-30' ORDER BY date ASC", conn)
conn.close()

# Draw Chart
fig, ax1 = plt.subplots(figsize=(10, 6))

color_bar = '#3498db'
ax1.set_xlabel('Date (NYT)')
ax1.set_ylabel('Total Balance (USD)', color=color_bar)

# Plot Balance (Bar Chart) starting from 620 visually to emphasize growth
bars = ax1.bar(df['date'].astype(str), df['balance_usd'], color=color_bar, alpha=0.7, label='Total Balance (USD)')
ax1.tick_params(axis='y', labelcolor=color_bar)
ax1.set_ylim(625, max(df['balance_usd']) + 5)

ax2 = ax1.twinx()
color_line = '#e74c3c'
ax2.set_ylabel('Daily Yield (%)', color=color_line)
lines = ax2.plot(df['date'].astype(str), df['return_pct_usd'], color=color_line, marker='o', linewidth=2, label='Yield (%)')
ax2.tick_params(axis='y', labelcolor=color_line)

# Add values on bars (Total Balance)
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'', ha='center', va='bottom', color='black', fontweight='bold')

# Add values on line (Yield)
for i, txt in enumerate(df['return_pct_usd']):
    ax2.text(i, txt + (0.05 if txt > 0 else -0.05), f'{txt:.2f}%', ha='center', va='bottom' if txt >= 0 else 'top', color=color_line, fontweight='bold')

plt.title('Daily Total Balance & Yield (Sep 30 - Oct 2026)')
fig.tight_layout()

# Save
out_path = r'C:\Users\chabo\.gemini\antigravity\brain\0a9a5fc0-a59c-41c7-bd00-7860a7191e8f\oct_trades_chart_real_db.png'
plt.savefig(out_path, dpi=150)
print("SUCCESS")
