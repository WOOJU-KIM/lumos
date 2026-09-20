import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json

def plot_wfa_results():
    data_dir = Path("data")
    csv_file = data_dir / "backtest_weekly_rolling_wfa_trades.csv"
    
    if not csv_file.exists():
        print("CSV file not found!")
        return

    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values('datetime', inplace=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['ending_capital'], label='Capital (KRW)', color='blue')
    plt.title('Weekly Rolling WFA Backtest - Capital Growth')
    plt.xlabel('Date')
    plt.ylabel('Capital (KRW)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    chart_path = data_dir / "backtest_weekly_rolling_wfa_chart.png"
    plt.savefig(chart_path)
    print(f"Chart saved to {chart_path}")

if __name__ == '__main__':
    plot_wfa_results()
