import pandas as pd
import matplotlib.pyplot as plt
import json
import os

df_704 = pd.read_csv('data/backtest_wfa_704_trades.csv')
df_650 = pd.read_csv('data/backtest_wfa_650_trades.csv')

df_704['datetime'] = pd.to_datetime(df_704['datetime'])
df_650['datetime'] = pd.to_datetime(df_650['datetime'])

df_704 = df_704.set_index('datetime')
df_650 = df_650.set_index('datetime')

# Resample to daily to make the chart smooth and comparable
daily_704 = df_704['ending_capital'].resample('D').last().fillna(method='ffill')
daily_650 = df_650['ending_capital'].resample('D').last().fillna(method='ffill')

# Fill missing start/end dates
start_date = min(daily_704.index.min(), daily_650.index.min())
end_date = max(daily_704.index.max(), daily_650.index.max())

idx = pd.date_range(start_date, end_date)
daily_704 = daily_704.reindex(idx).fillna(method='ffill').fillna(10000000)
daily_650 = daily_650.reindex(idx).fillna(method='ffill').fillna(10000000)

plt.figure(figsize=(12, 6))
plt.plot(daily_650.index, daily_650.values, label='Threshold 65.0%', color='red', alpha=0.7)
plt.plot(daily_704.index, daily_704.values, label='Threshold 70.4%', color='blue', linewidth=2)

plt.title('Equity Curve Comparison: 65.0% vs 70.4% Threshold (2026 WFA)', fontsize=14)
plt.ylabel('Capital (KRW)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save chart
out_img = 'C:\\Users\\chabo\\.gemini\\antigravity\\brain\\0a1158f2-2464-4301-863e-d8cfee9b5aac\\comparison_chart.png'
plt.savefig(out_img)
print(f"Chart saved to {out_img}")

# Compare Summaries
with open('data/backtest_wfa_650_summary.json', 'r', encoding='utf-8') as f:
    sum_650 = json.load(f)

with open('data/backtest_weekly_rolling_wfa_summary.json', 'r', encoding='utf-8') as f:
    sum_704 = json.load(f)

print("=== SUMMARY METRICS ===")
print("Metric | 65.0% Threshold | 70.4% Threshold")
print("--- | --- | ---")
print(f"Total Return | {sum_650['total_stats']['ret_pct']}% | {sum_704['total_stats']['ret_pct']}%")
print(f"Final Capital | {sum_650['total_stats']['end_cap']:,} | {sum_704['total_stats']['end_cap']:,}")
print(f"Max Drawdown | {sum_650['mdd_pct']}% | {sum_704['mdd_pct']}%")
print(f"Win Rate | {sum_650['total_stats']['win_rate']}% | {sum_704['total_stats']['win_rate']}%")
print(f"Total Trades | {sum_650['total_stats']['trades']} | {sum_704['total_stats']['trades']}")
print(f"Profit Factor | {sum_650['total_stats']['pf']} | {sum_704['total_stats']['pf']}")
