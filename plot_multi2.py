import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

sym = sys.argv[1]
filename = f'data/backtest_trades_{sym}.csv' if sym != 'TQQQ' else 'data/backtest_weekly_rolling_wfa_trades.csv'

try:
    df = pd.read_csv(filename)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df['ending_capital'].plot(title=f'{sym} Cumulative Return (10M KRW start)', figsize=(10,6), color='blue')
    plt.ylabel('Capital (KRW)')
    plt.grid(True)
    plt.tight_layout()
    
    out_dir = r"C:\Users\chabo\.gemini\antigravity\brain\d51baff8-e911-4c23-b9b2-df7dc7b90e2f\scratch"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'chart_{sym}.png')
    
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")
except Exception as e:
    print("Error:", e)
