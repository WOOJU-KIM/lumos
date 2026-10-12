import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

with open('data/backtest_5m_precision_rolling_wfa.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

months = [item['month'] for item in data['monthly']]
equity = [item['end_cap'] for item in data['monthly']]

# Add starting point
months.insert(0, "2022-08-01")  # dummy start before first trade
equity.insert(0, data['initial_capital'])

df = pd.DataFrame({'month': months, 'equity': equity})
# Ensure month is proper datetime for plotting, some are YYYY-MM
df['month'] = pd.to_datetime(df['month'])

plt.figure(figsize=(10, 5))
plt.plot(df['month'], df['equity'], label='Lumos V5 5m Precision WFA', color='blue', linewidth=2)
plt.title('Lumos V5: 5m Precision Backtest (Strict Config)')
plt.xlabel('Date')
plt.ylabel('Equity (KRW)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(r'C:\Users\chabo\.gemini\antigravity\brain\ec99a813-00c6-450a-a1d9-4f8e458c57e5\backtest_5m_strict_chart.png')
print("Chart saved.")
