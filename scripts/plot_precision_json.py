import json
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("data")
json_file = data_dir / "backtest_5m_precision_rolling_wfa.json"

with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

months = []
caps = []

for m in data.get('monthly', []):
    months.append(m['month'])
    caps.append(m['end_cap'])

plt.figure(figsize=(12, 6))
plt.plot(months, caps, label='Capital (KRW)', color='blue', marker='o', markersize=3)
plt.title('5m Precision Backtest - Capital Growth (Monthly)')
plt.xlabel('Month')
plt.ylabel('Capital (KRW)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

chart_path = data_dir / "backtest_5m_precision_chart.png"
plt.savefig(chart_path)
print(f"Chart saved to {chart_path}")
