import pandas as pd
import matplotlib.pyplot as plt

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
    if w > 52:
        y += 1
        w = 1

weekly = weekly.reindex(weeks).fillna({'ret_pct': 0})

plt.figure(figsize=(15, 6))

# Plot Line Chart
plt.plot(weekly.index, weekly['ret_pct'], color='#3498db', marker='o', markersize=3, linewidth=1.5, label='Weekly Return')

# Fill the area above and below zero
plt.fill_between(weekly.index, weekly['ret_pct'], 0, where=(weekly['ret_pct'] >= 0), color='#2ecc71', alpha=0.3, interpolate=True)
plt.fill_between(weekly.index, weekly['ret_pct'], 0, where=(weekly['ret_pct'] < 0), color='#e74c3c', alpha=0.3, interpolate=True)

plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
plt.title('Lumos V5: Weekly Return Trend (%) [2022-08 ~ 2026-10]', fontsize=14, fontweight='bold')
plt.xlabel('Week (YYYY-WW)')
plt.ylabel('Weekly Return (%)')

tick_locs = range(0, len(weekly.index), 10)
plt.xticks(tick_locs, [weekly.index[i] for i in tick_locs], rotation=45)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig('C:/Users/chabo/.gemini/antigravity/brain/b3d88781-9624-404e-9086-59bbf584c9a8/weekly_returns_line.png')
print("Line chart generated successfully.")
