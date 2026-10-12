import pandas as pd
import matplotlib.pyplot as plt
import os

true_csv = "data/equity_curve_true.csv"
false_csv = "data/equity_curve_false.csv"

if not os.path.exists(true_csv) or not os.path.exists(false_csv):
    print("CSV files missing.")
    exit(1)

df_true = pd.read_csv(true_csv)
df_false = pd.read_csv(false_csv)

df_true['date'] = pd.to_datetime(df_true['date'])
df_false['date'] = pd.to_datetime(df_false['date'])

plt.figure(figsize=(14, 7))
plt.plot(df_false['date'], df_false['capital_after'], label='Filters OFF (False)', color='gray', alpha=0.7)
plt.plot(df_true['date'], df_true['capital_after'], label='Filters ON (True)', color='blue', linewidth=2)

plt.title('Lumos V5 Full DB Backtest Comparison: Veto & Trend Filters')
plt.xlabel('Date')
plt.ylabel('Capital (USD)')
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = "data/comparison_chart.png"
plt.savefig(plot_path)
print(f"Chart saved to {plot_path}")
