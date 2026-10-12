import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

df = pd.read_csv('data/backtest_weekly_rolling_wfa_trades.csv')
weekly = df.groupby('week_id').agg(
    start_cap=('ending_capital', lambda x: x.iloc[0] - df.loc[x.index[0], 'pnl_krw']), 
    end_cap=('ending_capital', 'last')
)
weekly['ret_pct'] = (weekly['end_cap'] / weekly['start_cap'] - 1) * 100

weeks = []
min_w = weekly.index.min()
max_w = weekly.index.max()
y, w = int(min_w[:4]), int(min_w[5:])
max_y, max_w_idx = int(max_w[:4]), int(max_w[5:])
while (y < max_y) or (y == max_y and w <= max_w_idx):
    weeks.append(f'{y}-{w:02d}')
    w += 1
    if w > 52: y += 1; w = 1

weekly = weekly.reindex(weeks).fillna({'ret_pct': 0})
weekly['ma_12'] = weekly['ret_pct'].rolling(window=12, min_periods=1).mean()

# Calculate linear trendline
x = np.arange(len(weekly))
y_vals = weekly['ret_pct'].values
slope, intercept, r_value, p_value, std_err = linregress(x, y_vals)
trendline = slope * x + intercept

plt.figure(figsize=(15, 6))

# Plot raw weekly returns as faded scatter dots to reduce noise
plt.scatter(weekly.index, weekly['ret_pct'], color='gray', alpha=0.3, s=15, label='Raw Weekly Return (%)')

# Plot the 12-week moving average (the "Trend" of the returns)
plt.plot(weekly.index, weekly['ma_12'], color='#2980b9', linewidth=2.5, label='12-Week Moving Average (Return Power)')

# Plot the absolute linear trendline
plt.plot(weekly.index, trendline, color='#c0392b', linewidth=2.5, linestyle='--', label=f'Linear Trendline (Slope: {slope:.4f})')

plt.axhline(0, color='black', linewidth=1)
plt.title('Lumos V5: Evolution of Earning Power (Weekly Return % Trend)', fontsize=14, fontweight='bold')
plt.xlabel('Week (YYYY-WW)')
plt.ylabel('Weekly Return (%)')

tick_locs = range(0, len(weekly.index), 10)
plt.xticks(tick_locs, [weekly.index[i] for i in tick_locs], rotation=45)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.tight_layout()

plt.savefig('C:/Users/chabo/.gemini/antigravity/brain/b3d88781-9624-404e-9086-59bbf584c9a8/weekly_returns_trendline.png')
print("Trendline chart generated.")
