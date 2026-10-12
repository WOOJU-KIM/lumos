import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Load background confidence data
df_bg = pd.read_csv('data/oct_gbdt_conf.csv')
df_bg['datetime'] = pd.to_datetime(df_bg['datetime'])
df_bg['datetime'] = df_bg['datetime'].dt.tz_localize('Asia/Seoul').dt.tz_convert('America/New_York')

# Load actual trades
df_trades = pd.read_csv('data/live_trades.csv', on_bad_lines='skip')
df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
df_oct_trades = df_trades[df_trades['entry_time'] >= '2026-10-01'].copy()
df_oct_trades['entry_time'] = df_oct_trades['entry_time'].dt.tz_localize('Asia/Seoul').dt.tz_convert('America/New_York')
df_oct_trades['gbdt_confidence_pct'] = df_oct_trades['gbdt_confidence'] * 100

plt.figure(figsize=(12, 6))

# Plot all confidence points as small gray scatter dots (instead of lines)
# The user said "라인챠트에서 스캐터로 바꿔서. 0은 점 안찍어도 돼"
# We will filter out any 0 confidence if they exist.
df_bg = df_bg[df_bg['confidence'] > 0]
plt.scatter(df_bg['datetime'], df_bg['confidence'], color='gray', s=10, alpha=0.3, label='Other Confidence Logs')

# Plot trades
# 익절 (Red), 손절 (Blue)
wins = df_oct_trades[df_oct_trades['pnl_pct'] > 0]
losses = df_oct_trades[df_oct_trades['pnl_pct'] < 0]

plt.scatter(wins['entry_time'], wins['gbdt_confidence_pct'], color='red', s=60, edgecolors='black', label='Profit (Win)', zorder=5)
plt.scatter(losses['entry_time'], losses['gbdt_confidence_pct'], color='blue', s=60, edgecolors='black', label='Loss', zorder=5)

# Add threshold lines
plt.axhline(y=70.4, color='red', linestyle='--', linewidth=1.5, label='Entry Threshold (70.4%)')
plt.axhline(y=60.0, color='orange', linestyle='--', linewidth=1.5, label='60.0% Line')

# Formatting
plt.title('GBDT Output Values & Trades - October 2026', fontsize=14)
plt.xlabel('Date / Time (US Eastern)', fontsize=12)
plt.ylabel('Confidence (%)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)

# Format x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.xticks(rotation=45)

plt.legend()
plt.tight_layout()

plt.savefig('data/oct_gbdt_chart.png', dpi=150)
print("Chart saved to data/oct_gbdt_chart.png")
