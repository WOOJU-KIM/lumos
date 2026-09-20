import pandas as pd
import glob

files = {
    'Original (None)': 'data/wfa_trades_03.csv',
    'Regime (10, 30)': 'data/wfa_regime_test_10_30.csv',
    'Regime (20, 50)': 'data/wfa_regime_test_trades.csv',
    'Regime (20, 80)': 'data/wfa_regime_test_20_80.csv',
    'Regime (50, 120)': 'data/wfa_regime_test_50_120.csv'
}

print(f"{'Model':<18} | {'Trades':<6} | {'WinRate':<7} | {'Return':<7}")
print('-'*50)
for name, f in files.items():
    try:
        df = pd.read_csv(f)
        if len(df) == 0:
            print(f"{name:<18} | {'0':<6} | {'0%':<7} | {'0%':<7}")
            continue
        wins = len(df[df['net_ret_pct'] > 0])
        win_rate = wins / len(df) * 100
        cum_ret = (df['net_ret_pct']/100 + 1).prod() - 1
        print(f"{name:<18} | {len(df):<6} | {win_rate:>5.1f}% | {cum_ret*100:>6.1f}%")
    except Exception as e:
        print(f"{name:<18} | Error: {e}")
