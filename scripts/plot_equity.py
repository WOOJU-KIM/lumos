import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/backtest_weekly_rolling_wfa_trades.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

plt.figure(figsize=(15, 6))

# Plot the equity curve
plt.plot(df['datetime'], df['ending_capital'], color='#2c3e50', linewidth=1.5, label='Account Balance')

# Initial capital line
plt.axhline(10000000, color='#e74c3c', linestyle='--', linewidth=1.5, label='Initial Capital (10,000,000 KRW)')

# Fill the area above and below initial capital
plt.fill_between(df['datetime'], df['ending_capital'], 10000000, where=(df['ending_capital'] >= 10000000), color='#2ecc71', alpha=0.2, interpolate=True)
plt.fill_between(df['datetime'], df['ending_capital'], 10000000, where=(df['ending_capital'] < 10000000), color='#e74c3c', alpha=0.2, interpolate=True)

plt.title('Lumos V5: Cumulative Equity Curve [2022-08 ~ 2026-10]', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Account Balance (KRW)')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.tight_layout()

plt.savefig('C:/Users/chabo/.gemini/antigravity/brain/b3d88781-9624-404e-9086-59bbf584c9a8/cumulative_equity.png')
print("Cumulative equity chart generated.")
